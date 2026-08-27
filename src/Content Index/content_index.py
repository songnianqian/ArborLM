"""
content_index.py — Frozen runtime Content Index (design note v3).

ONE JOB (design §2, §31): divide runtime contexts into a stable, balanced,
confident partition so each specialist receives a consistent disjoint slice. It is
NOT required to discover semantic domains. There is NO LM-performance feedback into
this module (design §1, §12, §30).

Pipeline (cheapest-first, design §6 / §28):

    runtime text/context
        -> frozen semantic encoder          (ContentEncoder)
        -> TOKENIZER-EXACT overlapping chunks (design §7)   [#5]
        -> mean pool + L2-normalize          -> content vector z
        -> balanced clustering               (BalancedCluster: cosine to K centroids)
        -> confidence + sticky/hysteresis routing (ContentIndex.route)
        -> Content ID k   (or GENERAL fallback)

The HyperNet Content Transformer is NOT built here (design §6, §8, §25): it is a
conditional upgrade behind the SAME route() contract, added only if the go/no-go
gate fails.

REVIEW FIXES in this file:
  #1  BalancedCluster.fit_balanced recomputes inertia against the FINAL centroids
      (final Lloyd reassignment) and returns BOTH the constrained fit assignment and
      the runtime argmax assignment, so callers can evaluate on the runtime partition.
  #5  ContentEncoder chunks on the ENCODER's own tokenizer (exact <=CHUNK_TOKENS).
  #6  the artifact stores the ENTIRE embedding pipeline (backend, revision, caps,
      normalization) and load() reconstructs from the artifact, not live config.
      POOL is now actually applied on load (old setattr bug fixed).

Two invariants this file exists to guarantee (design §30):
  * every parameter here is frozen (requires_grad == False) after load;
  * cluster IDs are computed LIVE from context, never baked into the dataset.
Enforced by assert_frozen() and attach_to_optimizer_guard() (design §12).
"""
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F

import content_index_config as CIC


# ----------------------------------------------------------------------
# Route result — the external API boundary the LM sees (design §13, §29).
# ----------------------------------------------------------------------
@dataclass
class RouteResult:
    cluster_id: Optional[int]   # chosen specialist k, or None on GENERAL fallback
    confidence: float           # calibrated confidence in [0,1]
    top1_similarity: float      # s1 = highest cosine to a centroid
    top2_similarity: float      # s2 = second-highest
    margin: float               # s1 - s2
    fallback: bool              # True => GENERAL only, alpha = 0 (design §17)
    candidate_cluster: int      # nearest cluster BEFORE fallback/hysteresis
    switched: bool = False      # True if this turn changed the incumbent path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id, "confidence": self.confidence,
            "top1_similarity": self.top1_similarity,
            "top2_similarity": self.top2_similarity, "margin": self.margin,
            "fallback": self.fallback, "candidate_cluster": self.candidate_cluster,
            "switched": self.switched,
        }


# ======================================================================
# Frozen semantic encoder + TOKENIZER-EXACT chunking + pooling (design §6, §7) [#5]
# ======================================================================
class ContentEncoder:
    """Frozen pretrained sentence encoder with tokenizer-exact overlapping-chunk
    mean pooling. Not a trainable module — it wraps a frozen model and produces z
    from raw text OR a token-id context. Short input -> one embedding (no padding);
    long input -> overlapping <=CHUNK_TOKENS-token windows -> mean pool (design §7).
    The WHOLE context is represented, never truncated to the first chunk.

    [#6] The backend ("st" | "hf") and encoder revision are RESOLVED at load time and
    can be pinned, so the same artifact embeds identically on any machine.
    """
    # fields that define embedding behaviour and MUST match the artifact (design §26)
    PIPELINE_KEYS = ("name", "revision", "backend", "dim", "chunk_tokens",
                     "chunk_overlap", "max_chunks", "encoder_max_tokens",
                     "pool", "normalize_z")

    def __init__(self, cfg=CIC):
        self.cfg = cfg
        self.name = cfg.ENCODER_NAME
        self.revision = cfg.ENCODER_REVISION
        self.backend = cfg.ENCODER_BACKEND        # "auto"|"st"|"hf"; resolved on load
        self.dim = cfg.ENCODER_DIM
        self.chunk_tokens = cfg.CHUNK_TOKENS
        self.chunk_overlap = cfg.CHUNK_OVERLAP
        self.max_chunks = cfg.MAX_CHUNKS
        self.encoder_max_tokens = cfg.ENCODER_MAX_TOKENS
        self.pool = cfg.POOL
        self.normalize_z = cfg.NORMALIZE_Z
        self._device = cfg.ENCODER_DEVICE
        self._model = None
        self._tok = None                          # the ENCODER's tokenizer (for #5)
        self.resolved_backend = None
        self.resolved_revision = None

    # -- pipeline descriptor: exactly what determines z (design §26) --
    def pipeline_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.resolved_revision or self.revision,
            "backend": self.resolved_backend or self.backend,
            "dim": self.dim, "chunk_tokens": self.chunk_tokens,
            "chunk_overlap": self.chunk_overlap, "max_chunks": self.max_chunks,
            "encoder_max_tokens": self.encoder_max_tokens,
            "pool": self.pool, "normalize_z": self.normalize_z,
        }

    @staticmethod
    def from_spec(spec: Dict[str, Any], cfg=CIC, device: Optional[str] = None
                  ) -> "ContentEncoder":
        """[#6] Rebuild an encoder whose embedding behaviour is fully determined by
        the saved spec, NOT by whatever content_index_config currently says."""
        enc = ContentEncoder(cfg)
        enc.name = spec["name"]
        enc.revision = spec.get("revision")
        enc.backend = spec.get("backend", "auto")
        enc.dim = spec["dim"]
        enc.chunk_tokens = spec["chunk_tokens"]
        enc.chunk_overlap = spec["chunk_overlap"]
        enc.max_chunks = spec["max_chunks"]
        enc.encoder_max_tokens = spec["encoder_max_tokens"]
        enc.pool = spec["pool"]
        enc.normalize_z = spec["normalize_z"]
        if device:
            enc._device = device
        return enc

    # -- lazy load + FREEZE (design §12: not just eval; grad disabled) --
    def _ensure(self):
        if self._model is not None:
            return
        want = self.backend
        loaded = None
        if want in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                kw = {}
                if self.revision:
                    kw["revision"] = self.revision
                m = SentenceTransformer(self.name, **kw)
                self._model = m
                # sentence-transformers exposes the underlying HF tokenizer; use it
                # so chunking is EXACT on the same tokens the model will see (#5).
                self._tok = m.tokenizer
                loaded = "st"
            except Exception:
                if want == "st":
                    raise
        if loaded is None:      # want == "hf" or auto-fallback
            from transformers import AutoModel, AutoTokenizer
            kw = {}
            if self.revision:
                kw["revision"] = self.revision
            self._tok = AutoTokenizer.from_pretrained(self.name, **kw)
            self._model = AutoModel.from_pretrained(self.name, **kw)
            loaded = "hf"
        self.resolved_backend = loaded
        # record the resolved revision if the hub exposes it
        try:
            self.resolved_revision = getattr(
                getattr(self._model, "config", None), "_commit_hash", None
            ) or self.revision
        except Exception:
            self.resolved_revision = self.revision
        # device + freeze
        try:
            self._model.to(self._device)
        except Exception:
            self._device = "cpu"
            self._model.to("cpu")
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        # verify / correct width
        probe = self._encode_texts(["_"])
        if probe.shape[-1] != self.dim:
            self.dim = probe.shape[-1]

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """[N, dim] chunk embeddings for a list of already-token-bounded chunks."""
        self._ensure()
        if self.resolved_backend == "st":
            emb = self._model.encode(
                texts, batch_size=self.cfg.ENCODER_BATCH,
                convert_to_tensor=True, normalize_embeddings=False,
                show_progress_bar=False)
            return emb.float().cpu()
        outs = []
        for i in range(0, len(texts), self.cfg.ENCODER_BATCH):
            batch = texts[i:i + self.cfg.ENCODER_BATCH]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=self.encoder_max_tokens,
                            return_tensors="pt").to(self._device)
            h = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            outs.append(pooled.float().cpu())
        return torch.cat(outs, dim=0)

    # -- TOKENIZER-EXACT chunking (design §7) [#5] --
    def _token_windows(self, text: str) -> List[str]:
        """Slice `text` into overlapping windows of <= chunk_tokens ENCODER tokens,
        decoding each window back to a string. Every window is provably within the
        encoder's token budget and the overlap is exact, independent of vocabulary
        (fixes the whitespace-word approximation)."""
        self._ensure()
        # encode without special tokens so window boundaries are content tokens
        try:
            ids = self._tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n == 0:
            return [text if text.strip() else " "]

        # Reserve room for encoder-added special tokens such as [CLS] / [SEP].
        special = 0
        if hasattr(self._tok, "num_special_tokens_to_add"):
            special = int(self._tok.num_special_tokens_to_add(pair=False))

        model_limit = self.encoder_max_tokens

        # SentenceTransformer may have its own effective sequence limit.
        if self.resolved_backend == "st":
            st_limit = getattr(self._model, "max_seq_length", None)
            if st_limit is not None:
                model_limit = min(model_limit, int(st_limit))

        L = min(self.chunk_tokens, model_limit - special)

        if L <= 0:
            raise RuntimeError(
                f"invalid encoder token budget: model_limit={model_limit}, "
                f"special_tokens={special}"
            )

        if n <= L:
            return [text]

        step = max(1, int(round(L * (1.0 - self.chunk_overlap))))
        
        windows = []
        for start in range(0, n, step):
            piece = ids[start:start + L]
            if not piece:
                continue
            try:
                s = self._tok.decode(piece, skip_special_tokens=True)
            except Exception:
                s = self._tok.decode(piece)
            if s.strip():
                windows.append(s)
            if len(windows) >= self.max_chunks:
                break
            if start + L >= n:
                break
        return windows or [text]

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        chunks = self._token_windows(text)
        emb = self._encode_texts(chunks)          # [n_chunks, dim]
        return self._pool(emb)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self.embed_text(t) for t in texts], dim=0)

    def _pool(self, chunk_emb: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            z = chunk_emb.mean(dim=0)
        elif self.pool == "max":
            z = chunk_emb.max(dim=0).values
        else:
            raise ValueError(f"unknown pool mode {self.pool!r}")
        if self.normalize_z:
            z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        return z

    @torch.no_grad()
    def embed_context(self, current_turn: str, history: str = "") -> torch.Tensor:
        joined = (history + "\n" + current_turn).strip() if history else current_turn
        return self.embed_text(joined)


# ======================================================================
# Multi-prototype cluster — S_k(z) = max_m cos(z, c_{k,m})  (design §22, §23)
# ======================================================================
# Produced by content_index_trainer.py (round >= 2). The runtime scores every
# prototype and MAX-pools within each specialist, so a specialist can cover
# nonlinear / multimodal / disconnected regions (§22) without changing route()'s
# contract: similarities(z) still returns [N, K] specialist scores. A single-
# prototype-per-specialist MultiPrototypeCluster is numerically identical to a
# BalancedCluster, so nothing downstream needs to special-case it.
class MultiPrototypeCluster:
    def __init__(self, prototypes: torch.Tensor, proto_specialist: torch.Tensor):
        """prototypes: [P, dim] unit vectors (P = total prototypes across specialists).
        proto_specialist: [P] long, giving each prototype's specialist id in [0,K)."""
        self.prototypes = F.normalize(prototypes.float(), p=2, dim=-1, eps=1e-8)
        self.proto_specialist = proto_specialist.long()
        self.K = int(self.proto_specialist.max().item()) + 1
        self.P = self.prototypes.shape[0]
        self.dim = self.prototypes.shape[1]
        # boolean [K, P] membership mask for scatter-free max-pool
        self._mask = torch.zeros(self.K, self.P, dtype=torch.bool)
        self._mask[self.proto_specialist, torch.arange(self.P)] = True

    @property
    def centroids(self) -> torch.Tensor:
        """Alias so assert_frozen / optimizer_param_ids / save treat prototypes as
        THE frozen tensor, exactly like BalancedCluster.centroids."""
        return self.prototypes

    @torch.no_grad()
    def similarities(self, z: torch.Tensor) -> torch.Tensor:
        """[N, K] specialist scores via max over each specialist's prototypes (§22)."""
        z = F.normalize(z.float(), p=2, dim=-1, eps=1e-8)
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        proto_sim = z @ self.prototypes.t()                    # [N, P]
        N = proto_sim.shape[0]
        out = proto_sim.new_full((N, self.K), -1.0)
        for k in range(self.K):
            cols = self._mask[k]
            if cols.any():
                out[:, k] = proto_sim[:, cols].max(dim=1).values
        return out.squeeze(0) if squeeze else out

    @torch.no_grad()
    def runtime_assign(self, Z: torch.Tensor) -> torch.Tensor:
        return self.similarities(Z).argmax(dim=-1)

    @torch.no_grad()
    def nearest_prototype(self, Z: torch.Tensor, k: int) -> torch.Tensor:
        """Within specialist k, index (into that specialist's prototypes) of the
        nearest prototype for each row of Z. Used by the trainer's M-step."""
        z = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)
        cols = torch.where(self._mask[k])[0]
        sub = z @ self.prototypes[cols].t()                    # [N, m_k]
        return sub.argmax(dim=-1)


# ======================================================================
# Balanced clustering model — cosine to K frozen centroids (design §10, §13)
# ======================================================================
class BalancedCluster:
    def __init__(self, centroids: torch.Tensor):
        self.centroids = F.normalize(centroids.float(), p=2, dim=-1, eps=1e-8)
        self.K = self.centroids.shape[0]
        self.dim = self.centroids.shape[1]

    @torch.no_grad()
    def similarities(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z.float(), p=2, dim=-1, eps=1e-8)
        return z @ self.centroids.t()

    @torch.no_grad()
    def runtime_assign(self, Z: torch.Tensor) -> torch.Tensor:
        """[#1] The partition specialists ACTUALLY see: plain nearest-centroid
        argmax, no capacity constraint. This is what evaluation must use."""
        return self.similarities(Z).argmax(dim=-1)

    # ---- offline fit (used by build_content_index.py, NOT at runtime) ----
    @staticmethod
    def fit_balanced(Z: torch.Tensor, K: int, cfg=CIC,
                     source_ids: Optional[List] = None,
                     source_pressure: float = 0.0
                     ) -> Tuple["BalancedCluster", torch.Tensor, torch.Tensor, float]:
        """Balanced / capacity-constrained k-means on unit vectors (cosine).

        Returns (cluster, fit_assign, runtime_assign, inertia):
          fit_assign     : the capacity-constrained assignment (TRAINS centroids)
          runtime_assign : nearest-centroid argmax against the FINAL centroids [#1]
          inertia        : recomputed against the FINAL centroids + runtime_assign
                           (final Lloyd reassignment, [smaller-review])
        Callers evaluate on runtime_assign; fit_assign is only the training vehicle.

        SOURCE-DIVERSE SEEDING (initial build only; design §9). When source_ids is
        given and source_pressure > 0, k-means++ seed selection is biased so each
        new seed prefers a point from a source NOT yet among the chosen seeds —
        spreading initial centroids across sources. Unlike rarity weighting, this
        WORKS ON A SOURCE-BALANCED DATASET (equal per-source counts), which is what
        build_local_dataset.py produces. It touches ONLY where the fit starts — the
        capacity-constrained Lloyd iterations below still move centroids on GEOMETRY
        alone, and runtime routing never sees source. Source stays metadata, never a
        routing target (§9). source_ids=None or source_pressure=0.0 reproduces the
        original seeding byte-for-byte. The round-level trainer never passes these
        (subsequent rounds are PPL-driven)."""
        g = torch.Generator().manual_seed(cfg.KMEANS_SEED)
        Zc = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)
        N = Zc.shape[0]
        capacity = math.ceil(cfg.CAPACITY_FACTOR * N / K)

        # SOURCE-DIVERSE SEEDING (issue #1 fix; design §9). The old rarity weighting
        # was a NO-OP on a source-BALANCED dataset (equal per-source counts -> all
        # weights 1.0). Instead we bias k-means++ so each NEW seed prefers a point
        # from a source NOT yet represented among the chosen seeds. This spreads the
        # initial centroids across sources regardless of balance.
        #
        # HONEST LIMITATION (ratified): this is SEED-ONLY. On well-separated data the
        # capacity-constrained Lloyd iterations + KMEANS_RESTARTS can converge to the
        # same optimum regardless of initialization, so the effect may WASH OUT — it
        # is a nudge, not a guarantee. We keep it seed-only ON PURPOSE: pushing source
        # into the Lloyd objective would make source a soft routing signal and break
        # §9 (source is metadata, never a routing target). Where seeding survives it
        # helps; where Lloyd converges regardless it's a harmless no-op. It never
        # touches runtime routing. source_ids=None or source_pressure=0.0 reproduces
        # the original seeding byte-for-byte.
        use_src = (source_ids is not None and source_pressure > 0.0
                   and len(source_ids) == N)
        src_idx = None
        if use_src:
            # map each point to an integer source id
            uniq = {}
            src_idx = torch.empty(N, dtype=torch.long)
            for i, s in enumerate(source_ids):
                key = str(s)
                if key not in uniq:
                    uniq[key] = len(uniq)
                src_idx[i] = uniq[key]

        best = None
        for _ in range(cfg.KMEANS_RESTARTS):
            idx0 = torch.randint(0, N, (1,), generator=g).item()
            cents = [Zc[idx0]]
            seed_sources = set()
            if use_src:
                seed_sources.add(int(src_idx[idx0]))
            for _ in range(1, K):
                sims = torch.stack([Zc @ c for c in cents], dim=1)
                dmin = 1.0 - sims.max(dim=1).values
                probs = torch.clamp(dmin, min=0) + 1e-9
                if use_src:
                    # boost points whose source isn't among the seeds yet, so the
                    # seed set spreads across sources. Multiplicative, bounded.
                    novel = torch.tensor(
                        [0.0 if int(src_idx[i]) in seed_sources else 1.0
                         for i in range(N)])
                    probs = probs * (1.0 + source_pressure * novel)
                probs = probs / probs.sum()
                nxt = torch.multinomial(probs, 1, generator=g).item()
                cents.append(Zc[nxt])
                if use_src:
                    seed_sources.add(int(src_idx[nxt]))
            C = F.normalize(torch.stack(cents, dim=0), p=2, dim=-1, eps=1e-8)

            fit_assign = torch.full((N,), -1, dtype=torch.long)
            for _ in range(cfg.KMEANS_ITERS):
                sims = Zc @ C.t()
                fit_assign = BalancedCluster._capacitated_assign(sims, capacity)
                newC = []
                for k in range(K):
                    members = Zc[fit_assign == k]
                    if members.numel() == 0:
                        worst = (Zc @ C.t()).max(dim=1).values.argmin().item()
                        newC.append(Zc[worst])
                    else:
                        newC.append(members.mean(dim=0))
                newC = F.normalize(torch.stack(newC, dim=0), p=2, dim=-1, eps=1e-8)
                if torch.allclose(newC, C, atol=1e-5):
                    C = newC
                    break
                C = newC
            # [#1 + smaller-review] score on the RUNTIME partition against FINAL C
            runtime_assign = (Zc @ C.t()).argmax(dim=1)
            assigned_sim = (Zc @ C.t()).gather(
                1, runtime_assign.unsqueeze(1)).squeeze(1)
            inertia = float((1.0 - assigned_sim).sum())
            if best is None or inertia < best[3]:
                best = (C, fit_assign.clone(), runtime_assign.clone(), inertia)
        C, fit_assign, runtime_assign, inertia = best
        return BalancedCluster(C), fit_assign, runtime_assign, inertia

    @staticmethod
    def _capacitated_assign(sims: torch.Tensor, capacity: int) -> torch.Tensor:
        N, K = sims.shape
        counts = [0] * K
        assign = torch.full((N,), -1, dtype=torch.long)
        order = torch.argsort(sims.reshape(-1), descending=True)
        remaining = N
        for pos in order.tolist():
            if remaining == 0:
                break
            p, k = divmod(pos, K)
            if assign[p] >= 0 or counts[k] >= capacity:
                continue
            assign[p] = k
            counts[k] += 1
            remaining -= 1
        if remaining:
            for p in range(N):
                if assign[p] < 0:
                    k = int(torch.tensor(counts).argmin().item())
                    assign[p] = k
                    counts[k] += 1
        return assign


# ======================================================================
# The frozen runtime Content Index (design §13-§17, §29, §30)
# ======================================================================
class ContentIndex:
    def __init__(self, encoder: ContentEncoder, cluster: BalancedCluster,
                 thresholds: Dict[str, float], version_id: str,
                 meta: Optional[Dict[str, Any]] = None):
        self.encoder = encoder
        self.cluster = cluster
        self.K = cluster.K
        self.version_id = version_id
        self.meta = meta or {}
        self.t_margin = float(thresholds["T_MARGIN_FALLBACK"])
        self.t_sim = float(thresholds["T_SIM_FALLBACK"])
        self.t_switch = float(thresholds["T_SWITCH"])

    @torch.no_grad()
    def score_z(self, z: torch.Tensor):
        sims = self.cluster.similarities(z)
        top2 = torch.topk(sims, k=min(2, self.K))
        s1 = float(top2.values[0])
        s2 = float(top2.values[1]) if self.K > 1 else -1.0
        k1 = int(top2.indices[0])
        return k1, s1, s2, s1 - s2, sims

    def _confidence(self, s1: float, margin: float) -> float:
        a = max(0.0, min(1.0, (s1 - self.t_sim) / max(1e-6, 1.0 - self.t_sim)))
        b = max(0.0, min(1.0, margin / max(1e-6, 2.0 * self.t_margin)))
        return float(0.5 * a + 0.5 * b)

    @torch.no_grad()
    def route_z(self, z: torch.Tensor, previous_cluster: Optional[int] = None
                ) -> RouteResult:
        k1, s1, s2, margin, sims = self.score_z(z)
        conf = self._confidence(s1, margin)
        uncertain = (s1 < self.t_sim) or (margin < self.t_margin)
        # NO GENERAL path (6 specialists only): uncertain contexts still route to
        # top-1. `fallback` is now a DIAGNOSTIC flag (low-confidence marker) only —
        # it is never a routing target and cluster_id is never None.
        chosen, switched = k1, False
        if not uncertain and previous_cluster is not None and previous_cluster != k1:
            s_inc = float(sims[previous_cluster])
            s_new = float(sims[k1])
            if s_new > s_inc + self.t_switch:
                chosen, switched = k1, True
            else:
                chosen, switched = previous_cluster, False
        return RouteResult(chosen, conf, s1, s2, margin, uncertain, k1, switched)

    @torch.no_grad()
    def route(self, context, previous_cluster: Optional[int] = None) -> RouteResult:
        if isinstance(context, torch.Tensor):
            z = context
        elif isinstance(context, (tuple, list)) and len(context) == 2:
            z = self.encoder.embed_context(context[0], context[1])
        else:
            z = self.encoder.embed_text(str(context))
        return self.route_z(z, previous_cluster)

    @torch.no_grad()
    def route_batch_z(self, Z: torch.Tensor,
                      previous: Optional[List[Optional[int]]] = None
                      ) -> List[RouteResult]:
        out = []
        for i in range(Z.shape[0]):
            prev = None if previous is None else previous[i]
            out.append(self.route_z(Z[i], prev))
        return out

    # ---------- frozen-boundary enforcement (design §12, §30) ----------
    def assert_frozen(self):
        bad = []
        if getattr(self.encoder, "_model", None) is not None:
            for n, p in self.encoder._model.named_parameters():
                if p.requires_grad:
                    bad.append(f"encoder.{n}")
        if self.cluster.centroids.requires_grad:
            bad.append("cluster.centroids")
        if bad:
            raise RuntimeError(
                "Content Index NOT frozen — trainable tensors: "
                + ", ".join(bad[:8]) + ("..." if len(bad) > 8 else "")
                + "\nLM training must stop (design §12, §30).")
        return True

    def optimizer_param_ids(self):
        ids = {id(self.cluster.centroids)}
        if getattr(self.encoder, "_model", None) is not None:
            for p in self.encoder._model.parameters():
                ids.add(id(p))
        return ids

    def attach_to_optimizer_guard(self, optimizer):
        ci_ids = self.optimizer_param_ids()
        for gi, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                if id(p) in ci_ids:
                    raise RuntimeError(
                        f"Content Index parameter leaked into LM optimizer "
                        f"(param_group {gi}). Aborting (design §12, §30).")
        self.assert_frozen()
        return True

    # ---------- artifact save / load (design §26, §27) [#6] ----------
    def save(self, path: str):
        import os
        d = os.path.dirname(path)
        if d:                              # bare filename -> dirname '' -> skip mkdir
            os.makedirs(d, exist_ok=True)
        blob = {
            "version_id": self.version_id,
            # [#6] the ENTIRE embedding pipeline, not a subset
            "pipeline": self.encoder.pipeline_spec(),
            "K": self.K,
            "centroids": self.cluster.centroids.cpu(),
            "thresholds": {
                "T_MARGIN_FALLBACK": self.t_margin,
                "T_SIM_FALLBACK": self.t_sim,
                "T_SWITCH": self.t_switch,
            },
            "meta": self.meta,                 # includes sweep, OOD stats, calib info
            "saved_at": time.time(),
            "format": 2,
            # ci_prefix_patch: record the routing prefix the index was BUILT on, so
            # runtime can prove the artifact matches the LM's causal routing view.
            "route_prefix_tokens": getattr(self.encoder.cfg,
                                           "ROUTE_PREFIX_TOKENS", None),
        }
        # multi-prototype artifacts (trainer output, §22) also record the specialist
        # id of each prototype; single-centroid artifacts omit this and load as before.
        if isinstance(self.cluster, MultiPrototypeCluster):
            blob["proto_specialist"] = self.cluster.proto_specialist.cpu()
        tmp = path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str, cfg=CIC, device: Optional[str] = None) -> "ContentIndex":
        blob = torch.load(path, map_location="cpu")
        enc = ContentEncoder.from_spec(blob["pipeline"], cfg, device)
        if "proto_specialist" in blob:
            cluster = MultiPrototypeCluster(blob["centroids"],
                                            blob["proto_specialist"])
        else:
            cluster = BalancedCluster(blob["centroids"])
        ci = ContentIndex(enc, cluster, blob["thresholds"],
                          blob["version_id"], blob.get("meta", {}))
        # ci_prefix_patch: carry the built-on routing prefix, and warn if it differs
        # from the LM's CI_ROUTE_PREFIX (the index would be calibrated on a different
        # routing view than the LM uses). A hard assertion lives in
        # verify_ci_prefix_match.py; this is the load-time heads-up.
        saved_pref = blob.get("route_prefix_tokens")
        ci.route_prefix_tokens = saved_pref
        if saved_pref is not None:
            try:
                import config as _LMC
                if saved_pref != getattr(_LMC, "CI_ROUTE_PREFIX", saved_pref):
                    print(f"[CI] WARNING: artifact route_prefix_tokens={saved_pref} "
                          f"!= LM CI_ROUTE_PREFIX={_LMC.CI_ROUTE_PREFIX}. The index "
                          f"was calibrated on a different routing view than the LM "
                          f"uses.")
            except Exception:
                pass
        return ci
