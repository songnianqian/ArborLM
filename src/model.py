"""
model.py — Project B v3.1 model: a shared TRUNK feeding six independent SPECIALIST
continuations, routed by a FROZEN Content Index (content_index.py). No GENERAL
branch, no fusion (design note v3.1 §2/§3/§19).

    input -> TRUNK (8, HyperNet A[0:4] / B[4:8]) -> h_trunk
                                                     |
                                     +---------------+---------------+
                                     |               |               |
                               SPECIALIST P0 ...  SPECIALIST Pk ...  P5
                                     |               |               |
                                     +---------------+---------------+
                                                     |
                                     h_p -> LN_f -> LM head -> logits

For specialist p:
    h_trunk = Trunk(x)
    h_p     = Specialist_p(h_trunk)
    logits  = LMHead(LN_f(h_p))
The selected specialist is a COMPLETE upper continuation from h_trunk, not a
residual correction to a moving GENERAL branch.

Routing (frozen Content Index, no LM gradient — design §6/§10):
  * training : CI top-CI_TRAIN_TOPN specialists; the two are scored SEPARATELY and
    the per-sample loss is mean(L_Pa, L_Pb). They are NOT fused.
  * inference: CI top-1 specialist executes; head(LN_f(h_p)).

v3.1 delta vs v3 (GENERAL-always-fused): self.general, self.general_hypernet,
self.fusion_alpha, _fuse(), run_general(), L_general and every fusion diagnostic
are removed. teacher_sweep / fresh_gains remain as no-grad DIAGNOSTIC measurements
(now per-specialist, un-fused) but drive nothing.
"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config

import config as C
from clustered_hypernet import ClusteredHypernetwork, HyperNetBlock


def build_gpt2_config():
    cfg = GPT2Config(
        vocab_size=C.VOCAB_SIZE, n_positions=C.N_POSITIONS,
        n_embd=C.D_MODEL, n_layer=C.N_LAYERS_EFFECTIVE, n_head=C.N_HEADS)
    cfg.gate_hidden = C.GATE_HIDDEN
    cfg.use_cache = False
    cfg.score_from = C.SCORE_FROM
    return cfg


class GeneralistSpecialistModel(nn.Module):
    def __init__(self, gpt2_config=None):
        super().__init__()
        cfg = gpt2_config or build_gpt2_config()
        self.config = cfg
        d = C.D_MODEL

        self.wte = nn.Embedding(C.VOCAB_SIZE, d)
        self.wpe = nn.Embedding(C.N_POSITIONS, d)
        self.drop = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(d, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(d, C.VOCAB_SIZE, bias=False)

        # ---- trunk (shared): hypernet placement per PoC config ----
        # HYPERNET_TRUNK_A gates layers [0,split); HYPERNET_TRUNK_B gates [split,L).
        # A gated-off region gets hypernet=None on its blocks (MoE-only, no cost).
        self.trunk_split_at = C.TRUNK_SPLIT_AT
        self.trunk_hypernet_a = ClusteredHypernetwork(
            cfg, num_centroids=C.HYPERNET_CENTROIDS_TRUNK_A,
            temperature=C.HYPERNET_TEMPERATURE, balance_mode=C.HYPERNET_BALANCE_MODE
        ) if C.HYPERNET_TRUNK_A else None
        self.trunk_hypernet_b = ClusteredHypernetwork(
            cfg, num_centroids=C.HYPERNET_CENTROIDS_TRUNK_B,
            temperature=C.HYPERNET_TEMPERATURE, balance_mode=C.HYPERNET_BALANCE_MODE
        ) if C.HYPERNET_TRUNK_B else None
        self.trunk = nn.ModuleList([
            HyperNetBlock(cfg, i,
                          self.trunk_hypernet_a if i < self.trunk_split_at
                          else self.trunk_hypernet_b)
            for i in range(C.TRUNK_LAYERS)])

        # ---- specialists read h_trunk directly (v3.1: no GENERAL branch) ----
        # design note §2/§3/§19: the selected specialist IS the upper continuation
        # from h_trunk; there is no always-resident GENERAL path and no fusion.
        base_idx = C.TRUNK_LAYERS
        if C.HYPERNET_SPEC:
            self.spec_hypernets = nn.ModuleList([
                ClusteredHypernetwork(cfg, num_centroids=C.HYPERNET_CENTROIDS_SPEC,
                                      temperature=C.HYPERNET_TEMPERATURE,
                                      balance_mode=C.HYPERNET_BALANCE_MODE)
                for _ in range(C.N_SPECIALISTS)])
        else:
            self.spec_hypernets = [None] * C.N_SPECIALISTS
        self.specialists = nn.ModuleList([
            nn.ModuleList([HyperNetBlock(cfg, base_idx + j, self.spec_hypernets[p])
                           for j in range(C.SPECIALIST_LAYERS)])
            for p in range(C.N_SPECIALISTS)])

        # v3.1: no fusion alpha (GENERAL/fusion removed — design note §2/§19).

        self.apply(self._init_weights)
        self.lm_head.weight = self.wte.weight
        # (simplified hypernet: additive near-zero-init correction, no wide
        # reason_proj in this block design — nothing extra to reinit here)

        # ---- frozen Content Index router (attached at runtime) ----
        # NOT an nn.Module and NOT registered — it must never appear in state_dict
        # or the optimizer. Stored as a plain attribute (design §12/§30).
        self._ci = None
        # v3.1: no GENERAL destination — CI ranks specialists P0..P{N-1} only.

    # ---- Content Index attachment (design §12/§26/§30) ----
    def attach_content_index(self, ci):
        """Attach the frozen Content Index. Verifies its K matches N_SPECIALISTS and
        that it is frozen. Does NOT register it (stays out of state_dict/optimizer)."""
        if ci.K != C.N_SPECIALISTS:
            raise RuntimeError(
                f"Content Index K ({ci.K}) != N_SPECIALISTS ({C.N_SPECIALISTS}). "
                f"Rebuild the index at K={C.N_SPECIALISTS} or change the path count "
                f"(design §9 — this must be an explicit decision, not silent).")
        ci.assert_frozen()
        self._ci = ci
        return self

    @property
    def content_index(self):
        if self._ci is None:
            raise RuntimeError("No Content Index attached — call "
                               "attach_content_index(ci) after loading the artifact.")
        return self._ci

    def assert_router_out_of_optimizer(self, optimizer):
        """Startup guard (design §12/§30): assert no Content Index tensor is in the
        LM optimizer. Call right after building the optimizer, before step 1."""
        return self.content_index.attach_to_optimizer_guard(optimizer)

    # ---- init ----
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_(); m.weight.data.fill_(1.0)

    def _all_blocks(self):
        blocks = list(self.trunk)
        for path in self.specialists:
            blocks += list(path)
        return blocks

    @contextlib.contextmanager
    def count_mode(self, mode):
        prev = []
        for _nm, hn in self.named_hypernets():
            if hn is None:
                continue
            prev.append((hn, getattr(hn, "count_mode", "train")))
            hn.count_mode = mode
        try:
            yield
        finally:
            for hn, m in prev:
                hn.count_mode = m

    @contextlib.contextmanager
    def dropout_off(self):
        flipped = []
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                              nn.Dropout3d, nn.AlphaDropout,
                              nn.FeatureAlphaDropout)) and m.training:
                m.training = False
                flipped.append(m)
        try:
            yield
        finally:
            for m in flipped:
                m.training = True

    # ---- freeze control (design §5 late option / §9) ----
    # v3.1: the "base" is TRUNK-only (no GENERAL branch). Freezing the base leaves
    # the specialists (+ their HyperNets) trainable.
    def base_reference_params(self):
        named = []
        for i, blk in enumerate(self.trunk):
            named += [(f"trunk.{i}.{n}", p) for n, p in blk.named_parameters()]
        for _hn_name, _hn in (("trunk_hypernet_a", self.trunk_hypernet_a),
                              ("trunk_hypernet_b", self.trunk_hypernet_b)):
            if _hn is not None:
                named += [(f"{_hn_name}.{n}", p) for n, p in _hn.named_parameters()]
        named += [(f"wte.{n}", p) for n, p in self.wte.named_parameters()]
        named += [(f"wpe.{n}", p) for n, p in self.wpe.named_parameters()]
        named += [(f"ln_f.{n}", p) for n, p in self.ln_f.named_parameters()]
        named += [(f"lm_head.{n}", p) for n, p in self.lm_head.named_parameters()
                  if p is not self.wte.weight]
        return named

    # back-compat alias (older callers referenced general_reference_params)
    general_reference_params = base_reference_params

    def set_base_frozen(self, frozen: bool):
        for _name, p in self.base_reference_params():
            p.requires_grad = not frozen

    @torch.no_grad()
    def assert_reference_frozen(self):
        live = [name for name, p in self.base_reference_params()
                if p.requires_grad]
        if live:
            raise RuntimeError(
                "headline eval requires a FULLY frozen base (trunk) reference, but "
                f"these reference params are still trainable: {live[:12]}"
                + (" ..." if len(live) > 12 else ""))
        return True

    # ---- forward pieces ----
    def _embed(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        return self.drop(self.wte(input_ids) + self.wpe(pos))

    def run_trunk(self, input_ids, attention_mask):
        h = self._embed(input_ids)
        bal_a, n_a, bal_b, n_b = 0.0, 0, 0.0, 0
        moe_bal, n_moe = 0.0, 0
        for i, blk in enumerate(self.trunk):
            h, info = blk(h, attention_mask=attention_mask)[:2]
            if "balance_loss" in info:
                if i < self.trunk_split_at:
                    bal_a = bal_a + info["balance_loss"]; n_a += 1
                else:
                    bal_b = bal_b + info["balance_loss"]; n_b += 1
            if "moe_balance_loss" in info:
                moe_bal = moe_bal + info["moe_balance_loss"]; n_moe += 1
        groups = []
        if n_a > 0:
            groups.append(bal_a / n_a)
        if n_b > 0:
            groups.append(bal_b / n_b)
        trunk_balance = (sum(groups) / len(groups)) if groups else h.new_zeros(())
        moe_balance = (moe_bal / n_moe) if n_moe > 0 else h.new_zeros(())
        return h, trunk_balance, moe_balance

    def _run_stack(self, blocks, h, attention_mask):
        bal, n = 0.0, 0
        moe_bal, n_moe = 0.0, 0
        for blk in blocks:
            h, info = blk(h, attention_mask=attention_mask)[:2]
            if "balance_loss" in info:
                bal = bal + info["balance_loss"]; n += 1
            if "moe_balance_loss" in info:
                moe_bal = moe_bal + info["moe_balance_loss"]; n_moe += 1
        bal = (bal / n) if n > 0 else h.new_zeros(())
        moe_balance = (moe_bal / n_moe) if n_moe > 0 else h.new_zeros(())
        return h, bal, moe_balance

    def run_specialist(self, h_trunk, p, attention_mask):
        return self._run_stack(self.specialists[p], h_trunk, attention_mask)

    # ---- scoring (causal crop) ----
    def _token_ce(self, hidden, input_ids, attention_mask, topk=0, score_from=None):
        """Per-token CE over the scored region. If topk>0 (training only), compute a
        SAMPLED top-k softmax: restrict the softmax denominator to the top-k highest-
        logit vocab entries UNION the true target (target always included so its
        probability is never dropped). This approximates the full-vocab loss and is
        cheaper in the CE/softmax step. topk=0 -> exact full-vocab CE.

        IMPORTANT: top-k CE is an APPROXIMATION of the loss (different gradient than
        full-vocab). It is training-only; every REPORTED PPL/ACC must call this with
        topk=0 (full vocab). See eval.py / fresh_gains (all use topk=0)."""
        h = self.ln_f(hidden)
        logits = self.lm_head(h)
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        start = max(sf - 1, 0)
        sl = logits[:, start:-1, :]
        st = input_ids[:, start + 1:]
        flat_logits = sl.reshape(-1, sl.size(-1))       # [T, V]
        flat_tgt = st.reshape(-1)                        # [T]

        if topk and 0 < int(topk) < flat_logits.size(-1):
            k = int(topk)
            # top-k logits per token, then force the true target's logit in
            topk_vals, topk_idx = torch.topk(flat_logits, k=k, dim=-1)   # [T,k]
            tgt_logit = flat_logits.gather(1, flat_tgt.unsqueeze(1))     # [T,1]
            # is the target already in the top-k? if not, append it as a k+1th column
            tgt_in = (topk_idx == flat_tgt.unsqueeze(1)).any(dim=1, keepdim=True)  # [T,1]
            cand_logits = torch.cat([topk_vals, tgt_logit], dim=1)      # [T,k+1]
            # target's position in the candidate set: last column when not already in,
            # else its existing top-k slot. Build a label vector for the sampled CE.
            # We place target logit in the last column and mark it as the class; when
            # the target WAS in top-k it appears twice (harmless: logsumexp double-
            # counts one duplicate by <=log2 nats; we mask that by -inf'ing the dup).
            dup_mask = (topk_idx == flat_tgt.unsqueeze(1))               # [T,k]
            neg_inf = torch.finfo(cand_logits.dtype).min
            masked_topk = topk_vals.masked_fill(dup_mask, neg_inf)       # remove dup
            cand_logits = torch.cat([masked_topk, tgt_logit], dim=1)     # [T,k+1]
            labels = torch.full((cand_logits.size(0),), k,
                                device=cand_logits.device, dtype=torch.long)  # last col
            ce = F.cross_entropy(cand_logits, labels, reduction="none")
            ce = ce.view(st.size(0), st.size(1))
        else:
            ce = F.cross_entropy(flat_logits, flat_tgt,
                                 reduction="none").view(st.size(0), st.size(1))

        m = None
        if attention_mask is not None:
            m = attention_mask[:, start + 1:].float()
        return ce, m

    def _seq_nll(self, hidden, input_ids, attention_mask, topk=0, score_from=None):
        ce, m = self._token_ce(hidden, input_ids, attention_mask, topk=topk,
                               score_from=score_from)
        if m is not None:
            return (ce * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return ce.mean(dim=1)

    # ---- VOCAB_TOPK candidate cache (per-sample two-path LM-head saving) ----
    # The two CI specialists that score one sample predict the SAME target tokens.
    # PATH 1 (cache miss) computes the full shared LM head and records, per scored
    # token, its top-K vocab ids. PATH 2 (cache hit) reuses those ids and projects
    # ONLY the K (+target) corresponding LM-head rows — replacing a [T,V] matmul with
    # a [T,K] gather+dot, a genuine head-compute saving. The candidate set is shared
    # across the two paths (path-2's loss is top-K of PATH-1's logits ∪ target); this
    # is a deliberate training-only approximation, like the existing top-k CE. The
    # cache lives only inside one training_forward call (one micro-batch): a plain
    # local dict, reset every micro-batch and never checkpointed.
    def _score_full_capture(self, hidden, input_ids, attention_mask, k, score_from=None):
        """PATH-1 (cache miss): full-vocab top-k sampled CE, ALSO returning the per-
        token top-K candidate ids [n, T, K] to seed the cache. Semantics match
        _token_ce(topk=k): candidate set = top-k logits with the true target forced
        in. Returns (seq_nll[n], cand_ids[n,T,K]).

        CROP-BEFORE-HEAD: the model only scores positions [start:-1] (after the
        prefix R / SCORE_FROM), so we crop hidden to that region BEFORE ln_f/lm_head.
        The full-vocab projection then runs on ~T positions instead of all S — ~25%
        less LM-head compute AND retained activation. Numerically identical to
        projecting all positions then slicing."""
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        start = max(sf - 1, 0)
        h = self.ln_f(hidden[:, start:-1, :])                   # [n, T, d] scored region only
        sl = self.lm_head(h)                                    # [n, T, V] (cropped proj)
        st = input_ids[:, start + 1:]                           # [n, T]
        n, T, V = sl.shape
        flat_logits = sl.reshape(-1, V)                          # [nT, V]
        flat_tgt = st.reshape(-1)                                # [nT]
        k = int(min(k, V))
        topk_vals, topk_idx = torch.topk(flat_logits, k=k, dim=-1)      # [nT,k]
        tgt_logit = flat_logits.gather(1, flat_tgt.unsqueeze(1))        # [nT,1]
        dup_mask = (topk_idx == flat_tgt.unsqueeze(1))                  # [nT,k]
        neg_inf = torch.finfo(topk_vals.dtype).min
        masked_topk = topk_vals.masked_fill(dup_mask, neg_inf)
        cand_logits = torch.cat([masked_topk, tgt_logit], dim=1)       # [nT,k+1]
        labels = torch.full((cand_logits.size(0),), k,
                            device=cand_logits.device, dtype=torch.long)
        ce = F.cross_entropy(cand_logits, labels, reduction="none").view(n, T)
        seq_nll = self._reduce_ce(ce, attention_mask, start)
        cand_ids = topk_idx.view(n, T, k).detach()             # cache top-K ids [n,T,k]
        return seq_nll, cand_ids

    def _score_cached(self, hidden, input_ids, attention_mask, cand_ids, score_from=None):
        """PATH-2 (cache hit): project ONLY the cached K candidate rows (∪ target) of
        the LM head — no [T,V] matmul. cand_ids: [n,T,K] vocab ids from PATH-1.
        Returns seq_nll[n]. Gradient flows through hidden and the gathered head rows
        (tied to wte), so PATH-2 still trains.

        MEMORY: the gathered weight tensor W[cand] is [n,T,K+1,d] — at n=2,T=768,
        K=512,d=768 that is ~1.1 GiB bf16. Two mechanisms bound it:
          (1) CHUNK over the time axis (VOCAB_TOPK_CHUNK positions) so each chunk's
              forward gather is [n,chunk,K+1,d] (~192 MB at chunk=128).
          (2) RECOMPUTE (VOCAB_TOPK_RECOMPUTE, default on): wrap each chunk in
              gradient checkpointing so its gathered Wc / candidate logits are NOT
              retained in the autograd graph — they are recomputed in backward. Without
              this, autograd holds every chunk's Wc simultaneously and the chunked peak
              (~sum of chunks) approaches the un-chunked ~1.1 GiB. With it, retained
              memory is ~one chunk. Costs one extra gather+einsum per chunk in backward.
        Math is identical to the un-chunked path in all cases; only allocation differs."""
        import torch.utils.checkpoint as _ckpt
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        start = max(sf - 1, 0)
        h = self.ln_f(hidden[:, start:-1, :])                   # [n,T,d] scored region only
        st = input_ids[:, start + 1:]                           # [n,T]
        n, T, d = h.shape
        K = cand_ids.size(-1)
        W = self.lm_head.weight                                  # [V,d] (tied to wte)
        neg_inf = torch.finfo(h.dtype).min
        chunk = int(getattr(C, "VOCAB_TOPK_CHUNK", 128)) or T
        recompute = (bool(getattr(C, "VOCAB_TOPK_RECOMPUTE", True))
                     and torch.is_grad_enabled() and h.requires_grad)

        def _chunk_ce(hc, sc, ic):
            # hc [n,c,d], sc [n,c] target ids, ic [n,c,K] candidate ids
            cand = torch.cat([ic, sc.unsqueeze(-1)], dim=-1)     # [n,c,K+1] target last
            Wc = W[cand]                                         # [n,c,K+1,d] (recomputed in bwd)
            cl = torch.einsum("ncd,nckd->nck", hc, Wc)          # [n,c,K+1]
            dup = (cand[..., :K] == sc.unsqueeze(-1))           # [n,c,K]
            cl = torch.cat([cl[..., :K].masked_fill(dup, neg_inf),
                            cl[..., K:]], dim=-1)                # keep target col
            c = cl.size(1)
            lab = torch.full((n, c), K, device=hc.device, dtype=torch.long)
            return F.cross_entropy(cl.reshape(n * c, K + 1),
                                   lab.reshape(-1), reduction="none").view(n, c)

        ce_parts = []
        for s0 in range(0, T, chunk):
            s1 = min(s0 + chunk, T)
            hc = h[:, s0:s1, :]
            sc = st[:, s0:s1]
            ic = cand_ids[:, s0:s1, :]
            if recompute:
                # checkpoint needs a tensor input that requires grad to trigger
                # recompute; hc carries the graph. ic/sc are non-diff (ints) — pass
                # through as-is; only hc (and W, a leaf) participate in backward.
                ce_c = _ckpt.checkpoint(_chunk_ce, hc, sc, ic, use_reentrant=False)
            else:
                ce_c = _chunk_ce(hc, sc, ic)
            ce_parts.append(ce_c)
        ce = torch.cat(ce_parts, dim=1)                          # [n,T]
        return self._reduce_ce(ce, attention_mask, start)

    def _reduce_ce(self, ce, attention_mask, start):
        """Mask + mean per sequence over the scored region -> [n]."""
        if attention_mask is not None:
            m = attention_mask[:, start + 1:].float()
            return (ce * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return ce.mean(dim=1)

    def _mean_nll(self, hidden, input_ids, attention_mask, topk=0):
        return self._seq_nll(hidden, input_ids, attention_mask, topk=topk).mean()

    @torch.no_grad()
    def _seq_acc(self, hidden, input_ids, attention_mask, score_from=None):
        """Per-sequence next-token accuracy over the scored region (full-vocab
        argmax == target). Returns [B]. Reporting-only (always full-vocab)."""
        h = self.ln_f(hidden)
        logits = self.lm_head(h)
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        start = max(sf - 1, 0)
        pred = logits[:, start:-1, :].argmax(dim=-1)          # [B, T]
        tgt = input_ids[:, start + 1:]                        # [B, T]
        correct = (pred == tgt).float()
        if attention_mask is not None:
            m = attention_mask[:, start + 1:].float()
            return (correct * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return correct.mean(dim=1)

    # ---- teacher sweep: DIAGNOSTIC ONLY now (design health check) ----
    @torch.no_grad()
    def teacher_sweep(self, input_ids, attention_mask=None):
        """DIAGNOSTIC per-specialist NLL measurement (design note training-health
        check). Deterministic (dropout off), detached, from a DETACHED h_trunk so it
        never updates the base. Returns:
            spec_nll [B, n_spec]   per-specialist next-token NLL (un-fused; each
                                   specialist is a complete continuation of h_trunk)
        Routing is the frozen CI's job, not this sweep's. This drives NOTHING except
        diagnostics."""
        was_training = self.training
        self.eval()
        try:
            with self.count_mode("sweep"):
                h_trunk, _, _ = self.run_trunk(input_ids, attention_mask)
                h_trunk_det = h_trunk.detach()
                nlls = []
                for p in range(C.N_SPECIALISTS):
                    hp, _, _ = self.run_specialist(h_trunk_det, p, attention_mask)
                    nlls.append(self._seq_nll(hp, input_ids, attention_mask))
                spec_nll = torch.stack(nlls, dim=1)
        finally:
            if was_training:
                self.train()
        return spec_nll.detach()

    # ---- KD hooks (auxiliary) ----
    def _training_aux_prepare(self, h_trunk, input_ids, attention_mask, score_from):
        """Base-class stub; overridden in multipath_model to provide teacher hidden."""
        return None

    def _training_aux_group(
        self, aux_ctx, hp, idx, ids_p, am_p, path_id, score_from
    ):
        """Base-class stub; overridden in multipath_model to compute KD."""
        return None

    # ---- training_forward (modified to accept KD hooks) ----
    def training_forward(self, input_ids, attention_mask, per_sample_dests, topk=0,
                         score_from=None):
        """One micro-batch (B samples) training forward with PER-SAMPLE routing and
        GROUP-BY-DESTINATION execution (design note v3.1 §6/§10/§20).

        per_sample_dests: list length B; per_sample_dests[i] is the list of specialist
        ids (P0..P5) that sample i routes to (its CI top-N). There is NO GENERAL
        destination and NO fusion — each routed specialist is a COMPLETE upper
        continuation of h_trunk, scored on its own.

        score_from: the causal boundary R for THIS micro-batch (variable-prefix
        training). The LM scores next-token targets from position R onward; routing
        (done by the caller on tokens[:R]) never saw token R or later. Defaults to the
        fixed C.SCORE_FROM when None.

        Execution: trunk runs ONCE over the full batch. Then for each specialist p we
        gather the rows that routed to p (across all samples' top-N), run that
        specialist ONCE on the sub-batch (one forward per active specialist, not per
        sample), and score its output directly (top-k CE if topk>0). Each sample's
        loss is the MEAN over ITS routed specialists' separate NLLs
        (L_sample = mean(L_Pa, L_Pb)); the micro-batch loss is the mean over samples.
        Both routed specialists' losses backprop into the shared trunk.

        Returns L_final plus the bookkeeping train.py needs."""
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        B = input_ids.size(0)
        h_trunk, trunk_bal, trunk_moe_bal = self.run_trunk(input_ids, attention_mask)

        # ---- KD auxiliary preparation (teacher hidden) ----
        aux_ctx = self._training_aux_prepare(
            h_trunk, input_ids, attention_mask, sf
        )
        aux_sum = h_trunk.new_zeros(())
        aux_tokens = h_trunk.new_zeros(())

        # invert per-sample -> per-specialist row lists
        rows_for_spec = {p: [] for p in range(C.N_SPECIALISTS)}
        for i, dests in enumerate(per_sample_dests):
            for d in dests:
                d = int(d)
                if 0 <= d < C.N_SPECIALISTS:     # specialists only (defensive)
                    rows_for_spec[d].append(i)

        per_sample_nll = [None] * B
        per_sample_k = [0] * B

        # Preserve CI rank identity:
        # per_sample_rank_nll[i][0] = CI rank-1 NLL
        # per_sample_rank_nll[i][1] = CI rank-2 NLL
        per_sample_rank_nll = [
            [None] * C.CI_TRAIN_TOPN
            for _ in range(B)
        ]

        spec_balances = []
        spec_moe_balances = []
        dest_counts = torch.zeros(C.N_SPECIALISTS)

        # ---- VOCAB_TOPK per-sample candidate cache (this micro-batch only) ----
        # row index -> per-token top-K vocab ids [T,K] captured by the sample's
        # RANK-1 CI path. Only used when topk>0. Local => reset every micro-batch,
        # never checkpointed.
        use_cache = bool(topk) and 0 < int(topk) < C.VOCAB_SIZE
        cand_cache = {}

        def _score_group(p, rows):
            """Run specialist p once over `rows`; return its per-row seq_nll aligned to
            `rows`. Uses/ seeds the candidate cache when use_cache. Also records this
            group's balance + moe-balance + dest counts (called once per (phase,p))."""
            nonlocal aux_sum, aux_tokens
            idx = torch.tensor(rows, device=h_trunk.device, dtype=torch.long)
            ht = h_trunk.index_select(0, idx)
            am_p = attention_mask.index_select(0, idx) if attention_mask is not None else None
            ids_p = input_ids.index_select(0, idx)
            hp, bal_p, moe_bal_p = self.run_specialist(ht, p, am_p)
            spec_balances.append(bal_p)
            spec_moe_balances.append(moe_bal_p)
            dest_counts[p] += len(rows)

            # ---- KD auxiliary per-group call ----
            aux = self._training_aux_group(
                aux_ctx, hp, idx, ids_p, am_p, p, sf
            )
            if aux is not None:
                a_sum, a_tokens = aux
                aux_sum = aux_sum + a_sum
                aux_tokens = aux_tokens + a_tokens

            if not use_cache:
                return self._seq_nll(hp, ids_p, am_p, topk=topk, score_from=sf)

            miss_j = [j for j, i in enumerate(rows) if i not in cand_cache]
            hit_j = [j for j, i in enumerate(rows) if i in cand_cache]
            nll_by_j = {}
            if miss_j:
                sub = torch.tensor(miss_j, device=hp.device, dtype=torch.long)
                am_s = am_p.index_select(0, sub) if am_p is not None else None
                nll_m, cand_m = self._score_full_capture(
                    hp.index_select(0, sub), ids_p.index_select(0, sub), am_s,
                    int(topk), score_from=sf)
                for t, j in enumerate(miss_j):
                    nll_by_j[j] = nll_m[t]
                    cand_cache[rows[j]] = cand_m[t]              # seed [T,K] for rank-2
            if hit_j:
                sub = torch.tensor(hit_j, device=hp.device, dtype=torch.long)
                am_s = am_p.index_select(0, sub) if am_p is not None else None
                cand_h = torch.stack([cand_cache[rows[j]] for j in hit_j], dim=0)
                nll_h = self._score_cached(
                    hp.index_select(0, sub), ids_p.index_select(0, sub), am_s, cand_h,
                    score_from=sf)
                for t, j in enumerate(hit_j):
                    nll_by_j[j] = nll_h[t]
            return [nll_by_j[j] for j in range(len(rows))]

        def _accumulate(p, rows, nll_list):
            for j, i in enumerate(rows):
                v = nll_list[j]

                rank = next(
                    (
                        r for r, d in enumerate(per_sample_dests[i])
                        if int(d) == int(p)
                    ),
                    None
                )

                if rank is None or rank >= C.CI_TRAIN_TOPN:
                    raise RuntimeError(
                        f"routed specialist P{p} missing from sample "
                        f"{i} ranked destinations"
                    )

                if per_sample_rank_nll[i][rank] is not None:
                    raise RuntimeError(
                        f"duplicate NLL for sample {i}, "
                        f"CI rank {rank}, specialist P{p}"
                    )

                per_sample_rank_nll[i][rank] = v

                # Keep legacy equal-top2 aggregate so rescue OFF
                # reproduces the old training objective exactly.
                per_sample_nll[i] = (
                    v if per_sample_nll[i] is None
                    else per_sample_nll[i] + v
                )
                per_sample_k[i] += 1

        if not use_cache:
            # original grouped-by-specialist execution (cache off => order irrelevant)
            for p, rows in rows_for_spec.items():
                if rows:
                    _accumulate(p, rows, _score_group(p, rows))
        else:
            # CACHE ON: to avoid path-ID seeding bias, execute ALL rank-1 groups
            # (which seed the full-vocab candidate cache) BEFORE any rank-2 group
            # (which reads it). Rank is per-sample position in per_sample_dests:
            # index 0 = CI rank-1, index 1 = CI rank-2. Within each phase we still
            # group by specialist so each specialist runs once per phase.
            rank1 = {p: [] for p in range(C.N_SPECIALISTS)}
            rank2 = {p: [] for p in range(C.N_SPECIALISTS)}
            for i, dests in enumerate(per_sample_dests):
                for rank, d in enumerate(dests):
                    d = int(d)
                    if not (0 <= d < C.N_SPECIALISTS):
                        continue
                    (rank1 if rank == 0 else rank2)[d].append(i)
            for phase in (rank1, rank2):          # rank-1 seeds, rank-2 reads
                for p, rows in phase.items():
                    if rows:
                        _accumulate(p, rows, _score_group(p, rows))

        # Verify that each sample has all rank NLLs
        rank_rows = []
        for i in range(B):
            if any(v is None for v in per_sample_rank_nll[i]):
                raise RuntimeError(
                    f"sample {i} missing one or more "
                    f"CI top-{C.CI_TRAIN_TOPN} NLLs"
                )

            rank_rows.append(
                torch.stack(per_sample_rank_nll[i], dim=0)
            )
        rank_nll = torch.stack(rank_rows, dim=0)
        # shape: [B, 2]

        # Legacy equal-top2 loss (used when rescue is off)
        losses = []
        for i in range(B):
            if per_sample_nll[i] is not None and per_sample_k[i] > 0:
                losses.append(per_sample_nll[i] / per_sample_k[i])
        L_final = torch.stack(losses).mean() if losses else h_trunk.new_zeros(())

        spec_balance = (sum(spec_balances) / len(spec_balances)) if spec_balances \
            else h_trunk.new_zeros(())
        # MoE gate-balance: mean of trunk-B MoE balance and the routed specialists'
        # MoE balance (design note item 2). Averaged so the coefficient scale is
        # independent of how many specialists were active this micro-batch.
        moe_terms = [trunk_moe_bal] + spec_moe_balances
        moe_balance = sum(moe_terms) / len(moe_terms) if moe_terms \
            else h_trunk.new_zeros(())

        return {
            "L_final": L_final,
            "rank_nll": rank_nll,
            "trunk_balance": trunk_bal,
            "general_balance": h_trunk.new_zeros(()),   # kept for caller compat
            "spec_balance": spec_balance,
            "moe_balance": moe_balance,
            "dest_counts": dest_counts,
            "kd_loss": (
                aux_sum / aux_tokens.clamp_min(1.0)
                if aux_ctx is not None
                else h_trunk.new_zeros(())
            )
        }

    # ---- causal routing helper (shared by infer / fresh_gains) ----
    @torch.no_grad()
    def _route_topk_causal(self, input_ids, tokenizer, texts=None, k=1, route_prefix=None):
        """Top-k CI SPECIALIST ids per input (P0..P5 only — GENERAL is never a
        destination), routing ONLY on the causal prefix (first `route_prefix` tokens,
        default C.CI_ROUTE_PREFIX). Returns a LongTensor [B, k] of specialist ids,
        ranked best->worst by the CI's cosine similarity. Provide a tokenizer
        (preferred) or pre-truncated `texts`. Matches train._route_topn exactly."""
        ci = self.content_index
        B = input_ids.size(0)
        k = min(int(k), C.N_SPECIALISTS)
        rp = int(C.CI_ROUTE_PREFIX if route_prefix is None else route_prefix)
        rows = []
        for b in range(B):
            if tokenizer is not None:
                pref = input_ids[b, :rp]
                txt = tokenizer.decode(
                    pref.tolist(), skip_special_tokens=C.CI_DECODE_SKIP_SPECIAL)
            else:
                txt = texts[b]
            z = ci.encoder.embed_text(txt)
            _c, _s1, _s2, _m, sims = ci.score_z(z)
            order = torch.argsort(sims, descending=True).tolist()[:k]
            rows.append([int(x) for x in order])
        return torch.tensor(rows, device=input_ids.device)      # [B, k]

    # ---- inference (design §7/§13): frozen CI routes ONCE, top-1 specialist ----
    @torch.no_grad()
    def infer(self, input_ids, tokenizer=None, texts=None, attention_mask=None):
        """Route ONCE via the frozen Content Index (top-1 SPECIALIST) and run ONLY
        that specialist as the complete upper continuation of h_trunk (the memory
        win — one 4-layer path executes). No GENERAL branch, no fusion. Routing is
        CAUSAL (first C.CI_ROUTE_PREFIX tokens)."""
        with self.count_mode("sweep"):
            h_trunk, _, _ = self.run_trunk(input_ids, attention_mask)
            top1 = self._route_topk_causal(input_ids, tokenizer, texts, k=1)[:, 0]  # [B]

            out = h_trunk.clone()
            for p in range(C.N_SPECIALISTS):
                mask = (top1 == p)
                if not bool(mask.any()):
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                ht = h_trunk[idx]
                mp = attention_mask[idx] if attention_mask is not None else None
                hp, _, _ = self.run_specialist(ht, p, mp)
                out[idx] = hp
            logits = self.lm_head(self.ln_f(out))
        return logits, top1

    # ---- FROZEN-BASE headline measurement (design §15.3): per-path metrics ----
    @torch.no_grad()
    def fresh_gains(self, input_ids, tokenizer=None, texts=None, attention_mask=None,
                    score_from=None):
        """Fresh full-vocab per-path measurement (dropout off, no grad). Computes, for
        each specialist p, its OWN (un-fused) per-sequence NLL and ACC over the batch
        — each specialist is a complete continuation of h_trunk — plus the CI's exact
        top-2 specialist ids per input (causal-prefix routed).

        score_from: causal boundary R. Routing uses tokens[:R] and scoring starts at
        target position R. Defaults to the fixed C.SCORE_FROM (the exact-256 reference)
        — so unbucketed eval is unchanged. Bucketed eval passes each bucket's R.

        v3.1 semantics: no GENERAL branch, no fusion. Oracle is the best of the six
        specialists per doc. Returns spec_nll/spec_acc [B,n_spec], ci_top2 [B,2],
        L_selected/L_top2/L_oracle [B], oracle_path [B]."""
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        was_training = self.training
        self.eval()
        try:
            with self.count_mode("sweep"):
                h_trunk, _, _ = self.run_trunk(input_ids, attention_mask)
                nlls, accs = [], []
                for p in range(C.N_SPECIALISTS):
                    hp, _, _ = self.run_specialist(h_trunk, p, attention_mask)
                    nlls.append(self._seq_nll(hp, input_ids, attention_mask, score_from=sf))
                    accs.append(self._seq_acc(hp, input_ids, attention_mask, score_from=sf))
                spec_nll = torch.stack(nlls, dim=1)                 # [B, n_spec]
                spec_acc = torch.stack(accs, dim=1)                 # [B, n_spec]
                ci_top2 = self._route_topk_causal(input_ids, tokenizer, texts, k=2,
                                                  route_prefix=sf)   # [B,2]
        finally:
            if was_training:
                self.train()

        top1 = ci_top2[:, 0]
        top2 = ci_top2[:, 1] if ci_top2.size(1) > 1 else ci_top2[:, 0]
        L_selected = spec_nll.gather(1, top1.unsqueeze(1)).squeeze(1)      # [B]
        nll_top2 = spec_nll.gather(1, top2.unsqueeze(1)).squeeze(1)        # [B]
        L_top2 = torch.minimum(L_selected, nll_top2)                       # EXACT CI-top2-best
        L_oracle, oracle_path = spec_nll.min(dim=1)                        # specialists only
        return {"spec_nll": spec_nll, "spec_acc": spec_acc,
                "ci_top2": ci_top2, "selected": top1,
                "L_selected": L_selected, "L_top2": L_top2,
                "L_oracle": L_oracle, "oracle_path": oracle_path}

    # ---- rejuvenation (design §6 defense order) — HyperNet centroids only ----
    @torch.no_grad()
    def _rejuvenate_one_hypernet(self, hyper, optimizer=None, noise_std=0.01,
                                 donor_frac=2.0 / 3.0):
        cents = hyper.centroids
        counts = hyper.train_usage_window
        dead = (counts == 0).nonzero(as_tuple=False).flatten()
        alive = (counts > 0).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0 or alive.numel() == 0:
            return []
        m = dead.numel()
        alive_sorted = alive[torch.argsort(counts[alive], descending=True)]
        m_donors = int(round(m * donor_frac))
        donor_k = min(m_donors, alive_sorted.numel())
        if donor_k == 0:
            m_donors = 0
        revived = []
        for i in range(m_donors):
            j = int(dead[i]); src = int(alive_sorted[i % donor_k])
            cents[j].copy_(cents[src])
            if noise_std > 0:
                cents[j].add_(torch.randn_like(cents[j]) * noise_std)
                cents[src].add_(torch.randn_like(cents[src]) * noise_std * 0.25)
            revived.append(j)
        for i in range(m_donors, m):
            j = int(dead[i])
            cents[j].copy_(torch.randn_like(cents[j]) * 0.02)
            if noise_std > 0:
                cents[j].add_(torch.randn_like(cents[j]) * noise_std * 0.5)
            revived.append(j)
        if optimizer is not None and revived:
            st = optimizer.state.get(cents)
            if st:
                rows = torch.tensor(revived, device=cents.device, dtype=torch.long)
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        st[key][rows] = 0.0
        return revived

    @torch.no_grad()
    def rejuvenate(self, step, optimizer=None):
        """Early/mid-only HyperNet-centroid rejuvenation (design §6). The routing-
        PROTOTYPE rejuvenation is GONE — there are no learned routing prototypes
        anymore (the CI is frozen). Only trunk A/B + specialist centroids."""
        if not C.REJUV_ENABLED or step > C.REJUV_STOP_STEP:
            return {}
        report = {}
        for name, hyper in self.named_hypernets():
            if int(hyper.train_usage_lifetime.sum()) < C.REJUV_MIN_EXPOSURE:
                continue
            revived = self._rejuvenate_one_hypernet(
                hyper, optimizer=optimizer,
                noise_std=C.REJUV_NOISE_STD, donor_frac=C.REJUV_DONOR_FRAC)
            if revived:
                report[name] = len(revived)
            hyper.train_usage_window.zero_()
        return report

    # ---- diagnostics passthroughs ----
    def named_hypernets(self):
        """(name, hypernet) pairs, EXCLUDING placements that are None (PoC placement
        leaves some regions hypernet-free)."""
        pairs = [("trunk_a", self.trunk_hypernet_a),
                 ("trunk_b", self.trunk_hypernet_b)]
        pairs += [(f"spec{p}", self.spec_hypernets[p])
                  for p in range(C.N_SPECIALISTS)]
        return [(nm, hn) for nm, hn in pairs if hn is not None]

    def get_centroid_usage(self):
        return {nm: hn.centroid_usage_counts.clone()
                for nm, hn in self.named_hypernets()}

    def get_expert_usage(self):
        """DEPRECATED aggregate-by-kind view (collapses the two perc4 slots). Kept
        for back-compat; prefer get_moe_slot_usage() which preserves E0..E4."""
        slot = self.get_moe_slot_usage()
        # fold back to kind-percentages from the slot gate-mass shares
        kinds = list(C.OURS_EXPERT_KINDS)
        out = {}
        for e in slot["experts"]:
            out[e["kind"]] = out.get(e["kind"], 0.0) + e["gate_mass_pct"]
        return {k: out.get(k, 0.0) for k in dict.fromkeys(kinds)}

    @torch.no_grad()
    def get_moe_slot_usage(self):
        """Slot-level soft-MoE usage (design note item 6): sums usage_counts (soft
        gate mass) and selection_counts (argmax/top-1) across ALL blocks, per SLOT
        E0..E4 — the two perc4 slots stay SEPARATE. Also returns mean gate entropy.
        All buffers are accumulated on-GPU during training; converted here once."""
        n_slots = C.num_mlps()
        gate_mass = torch.zeros(n_slots)
        top1 = torch.zeros(n_slots)
        ent_sum, ent_n = 0.0, 0
        for blk in self._all_blocks():
            moe = getattr(blk, "moe", None)
            if moe is None or not hasattr(moe, "usage_counts"):
                continue
            if moe.usage_counts.numel() != n_slots:
                continue
            gate_mass += moe.usage_counts.detach().cpu()
            top1 += moe.selection_counts.detach().cpu()
            d = moe.gate_diag()          # GPU->python once (report time only)
            ent_sum += d["gate_entropy"]; ent_n += 1
        gm_tot = float(gate_mass.sum()) or 1.0
        t1_tot = float(top1.sum()) or 1.0
        kinds = list(C.OURS_EXPERT_KINDS)
        experts = []
        for i in range(n_slots):
            experts.append({
                "slot": i,
                "kind": kinds[i] if i < len(kinds) else f"E{i}",
                "gate_mass_pct": 100.0 * float(gate_mass[i]) / gm_tot,
                "top1_pct": 100.0 * float(top1[i]) / t1_tot,
            })
        dead = [e["slot"] for e in experts if e["gate_mass_pct"] < 1e-3]
        return {"experts": experts,
                "gate_entropy": (ent_sum / ent_n) if ent_n else 0.0,
                "dead_experts": dead}

    @torch.no_grad()
    def get_hypernet_centroid_diag(self, dead_frac_of_mean=0.02):
        """Per-HyperNet centroid diagnostics (design note item 5). Uses the
        TRAINING-ONLY cumulative counter train_usage_lifetime (NOT the generic
        centroid_usage_counts, which sweep/eval forwards pollute). Reports per
        HyperNet: used/total, dead/total, top-5 shares, usage entropy, effective
        centroid count (exp(entropy)), and current HyperNet gain."""
        import math as _m
        report = {}
        for name, hn in self.named_hypernets():
            counts = hn.train_usage_lifetime.detach().float().cpu()
            total = counts.numel()
            tot = float(counts.sum())
            if tot <= 0:
                report[name] = {"total": total, "used": 0, "dead": total,
                                "top5": [], "entropy": 0.0, "eff": 0.0,
                                "gain": self._hn_gain(hn), "exposure": 0.0}
                continue
            share = counts / tot
            used = int((counts > 0).sum())
            # "dead" = negligible relative to uniform mass (persistently unused)
            thresh = dead_frac_of_mean * (tot / total)
            dead = int((counts <= thresh).sum())
            topv, topi = torch.topk(share, k=min(5, total))
            top5 = [(int(topi[j]), float(topv[j]) * 100.0) for j in range(topi.numel())]
            p = share.clamp_min(1e-12)
            entropy = float(-(p * p.log()).sum())
            eff = float(_m.exp(entropy))
            report[name] = {"total": total, "used": used, "dead": dead,
                            "top5": top5, "entropy": entropy, "eff": eff,
                            "gain": self._hn_gain(hn), "exposure": tot}
        return report

    @staticmethod
    def _hn_gain(hn):
        import torch.nn.functional as _F
        if hasattr(hn, "hyper_gain_raw") and hasattr(hn, "min_gain"):
            return float(hn.min_gain + _F.softplus(hn.hyper_gain_raw.detach()))
        return float("nan")

    def reset_expert_usage(self):
        for blk in self._all_blocks():
            moe = getattr(blk, "moe", None)
            if moe is not None and hasattr(moe, "usage_counts"):
                moe.usage_counts.zero_()
                if hasattr(moe, "selection_counts"):
                    moe.selection_counts.zero_()
                if hasattr(moe, "reset_gate_diag"):
                    moe.reset_gate_diag()