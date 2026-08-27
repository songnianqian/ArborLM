"""
build_content_index.py — Offline Content Index training + GO/NO-GO gate (design §28).

Run ONCE, standalone, BEFORE any LM training (design §1, §3).

  Step 1  collect runtime-context vectors z from the LM's own streamed corpus,
          in THREE disjoint splits: FIT / CALIBRATION / GATE  [#3]           §6,§7
  Step 3  K sweep over {3..10}, evaluated on the RUNTIME nearest-centroid
          partition (NOT the constrained fit assignment)       [#1]          §9
  Step 2b HARD STOP if chosen K != N_SPECIALISTS unless FORCE_K resolves it  [#2] §9
  Step 4  fit BALANCED clustering at the chosen K (capacity-constrained fit)  §10
  Step-   calibrate confidence + hysteresis thresholds from the CALIBRATION
          split; T_SWITCH from same-state vs topic-shift regimes [#4]        §14,§15
  Step 7  GATE diagnostics on the untouched GATE split:
            - same-state STABILITY (low switching)             [#4]          §19,§24.4
            - short/long HISTORY consistency (not first-third) [#4/small]    §11,§24.5
            - TOPIC-SHIFT responsiveness (must switch)         [#4]          §18
            - balance / margin / coverage on runtime partition [#1]          §20,§24
            - OOD separation statistics                        [+review]     §26
  Step 8  GO / NO-GO decision
  Step 10 freeze assertions, then save the full-pipeline artifact            §12,§26,§30

If GO: stop — do NOT build the HyperNet Content Transformer (§24, §25).

Usage:
    python build_content_index.py            # full build + gate + save
    python build_content_index.py --sweep    # only the K sweep table
    python build_content_index.py --dry      # build + gate but do NOT save
"""
import sys
import math
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

import content_index_config as CIC
from content_index import ContentEncoder, BalancedCluster, ContentIndex

try:
    import config as C
    import data as D
    _HAVE_LM_DATA = True
except Exception:
    _HAVE_LM_DATA = False


# ======================================================================
# Step 1 — collect z vectors (+ raw text kept for perturbation trials)
# ======================================================================
def _decode_segment(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

def collect_multiview_fit(
    encoder: ContentEncoder,
    n: int,
    skip: int,
) -> Tuple[torch.Tensor, List, Dict]:
    """Ratified D1=B — MULTI-VIEW FIT augmentation. Collect `n` SEQUENCES and, for
    each, emit one content vector PER natural-break prefix view (32..256 via
    prefix_views.py, LM-token space, dirty/hard-cut training views dropped, min 32).
    The returned Z is [total_views, dim] and feeds ONLY the FIT clustering — so the
    centroids see short prefixes and short-prefix routing becomes robust.

    IMPORTANT (D1=B scope): CALIB, GATE, and the headline GO/NO-GO stay on the pure
    256 runtime view (collect_content_vectors) — the LM deploys at 256, so the
    thresholds and the gate must describe THAT distribution, not the augmented fit
    cloud. Yes, this means the fit distribution and the calibration distribution
    differ; that is a DELIBERATE, ratified choice (calibrate on what's deployed),
    not the silent view-mismatch the -p256 patch guarded against.

    Returns (Z_views, sources_views, info). sources_views[i] is the SOURCE of the
    sequence view i came from (each view inherits its sequence's source) so the
    source-diverse seeding can spread centroids across sources. info carries the
    per-target view counts for the build log."""
    if not _HAVE_LM_DATA:
        raise RuntimeError("config.py / data.py not importable — run from projectB.")
    try:
        from prefix_views import prefix_views_for_sequence
    except Exception as e:
        raise RuntimeError(f"prefix_views.py required for multi-view FIT: {e}")

    tokenizer = D.get_tokenizer()
    targets = list(getattr(CIC, "ROUTE_PREFIX_TOKENS_SWEEP", [32, 64, 128, 256]))
    runtime_view = CIC.ROUTE_PREFIX_TOKENS
    if runtime_view not in targets:
        targets.append(runtime_view)

    Zv: List[torch.Tensor] = []
    src_v: List = []
    per_target = {t: 0 for t in sorted(set(targets))}
    it = D.batch_iterator(tokenizer, skip_sequences=skip)
    n_seq = 0
    for input_ids, srcs, _c, _s in it:
        for b in range(input_ids.shape[0]):
            views = prefix_views_for_sequence(
                tokenizer, input_ids[b], targets=targets, runtime_view=runtime_view)
            if not views:
                continue
            for v in views:
                if not v.text.strip():
                    continue
                Zv.append(encoder.embed_text(v.text))
                src_v.append(srcs[b])
                per_target[v.target_len] = per_target.get(v.target_len, 0) + 1
            n_seq += 1
            if n_seq >= n:
                break
        if n_seq >= n:
            break

    if not Zv:
        raise RuntimeError("multi-view FIT collected 0 views — check DATA_SOURCE.")
    info = {"n_sequences": n_seq, "n_views": len(Zv), "per_target": per_target}
    return torch.stack(Zv, dim=0), src_v, info


def collect_content_vectors(
    encoder: ContentEncoder,
    n: int,
    skip: int,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Collect n content vectors from the streamed LM corpus.

    Collection begins after `skip` sequences.  FIT / CALIB / GATE separation
    is controlled by the different skip offsets used by the caller.
    """
    if not _HAVE_LM_DATA:
        raise RuntimeError(
            "config.py / data.py not importable — run from the projectB folder so "
            "the Content Index fits on the LM's corpus (design §10)."
        )

    tokenizer = D.get_tokenizer()

    Z = []
    texts = []
    prefix_texts = []
    sources = []

    it = D.batch_iterator(
        tokenizer,
        skip_sequences=skip,
    )

    seen = 0

    for input_ids, srcs, consumed, stream in it:
        for b in range(input_ids.shape[0]):
            full_txt = _decode_segment(tokenizer, input_ids[b])

            if not full_txt.strip():
                continue

            # ROUTE-MATCH (ci_prefix_patch): z is built from the LM's causal routing
            # prefix ONLY (first ROUTE_PREFIX_TOKENS tokens), decoded with the SAME
            # (LM/data) tokenizer the router uses at train/inference time — the exact
            # view model._route_top1_causal / train._decode_route_prefix produce.
            # `texts` keeps the FULL segment: the gate's short/long + perturbation
            # trials deliberately exercise truncation and need the full text as their
            # "long" reference.
            prefix_ids = input_ids[b, :CIC.ROUTE_PREFIX_TOKENS]
            prefix_txt = _decode_segment(tokenizer, prefix_ids)
            if not prefix_txt.strip():
                continue

            Z.append(encoder.embed_text(prefix_txt))   # prefix z (was full-segment z)
            texts.append(full_txt)                     # full text kept for gate trials
            prefix_texts.append(prefix_txt)            # prefix view for runtime calib/gates
            sources.append(srcs[b])

            seen += 1

            if seen >= n:
                break

        if seen >= n:
            break

    if not Z:
        raise RuntimeError(
            "collected 0 content vectors — check DATA_SOURCE."
        )

    return torch.stack(Z, dim=0), texts, prefix_texts, sources

# ======================================================================
# Evaluation helpers — ALL on the RUNTIME nearest-centroid partition [#1]
# ======================================================================
def _balance_stats(assign: torch.Tensor, K: int) -> Dict[str, float]:
    counts = torch.bincount(assign, minlength=K).float()
    share = counts / counts.sum().clamp_min(1)
    equal = 1.0 / K
    return {
        "min_share": float(share.min()), "max_share": float(share.max()),
        "starved": float((share < CIC.MIN_CLUSTER_SHARE * equal).sum()),
        "imbalance": float(share.max() / share.min().clamp_min(1e-9)),
        "empty": int((counts == 0).sum()),
    }


def _margins(cluster: BalancedCluster, Z: torch.Tensor) -> torch.Tensor:
    sims = cluster.similarities(Z)
    if cluster.K == 1:
        return torch.ones(Z.shape[0])
    top2 = torch.topk(sims, k=2, dim=1)
    return top2.values[:, 0] - top2.values[:, 1]


def _silhouette_cosine(Z: torch.Tensor, assign: torch.Tensor, K: int,
                       sample: int = 2000) -> float:
    N = Z.shape[0]
    idx = torch.randperm(N)[:min(sample, N)]
    Zs, As = F.normalize(Z[idx], p=2, dim=-1), assign[idx]
    dist = 1.0 - (Zs @ Zs.t())
    m = Zs.shape[0]
    sil = []
    for i in range(m):
        same = (As == As[i]); same[i] = False
        if same.sum() == 0:
            continue
        a = float(dist[i][same].mean())
        b = math.inf
        for k in range(K):
            if k == int(As[i]):
                continue
            other = (As == k)
            if other.sum() == 0:
                continue
            b = min(b, float(dist[i][other].mean()))
        if b == math.inf:
            continue
        sil.append((b - a) / max(a, b, 1e-9))
    return float(sum(sil) / len(sil)) if sil else 0.0


def _cosine_noise(Z: torch.Tensor, target_cos: float) -> torch.Tensor:
    """Perturb unit vectors so each keeps ~= target_cos with its original
    ([smaller-review]: parametrize by cosine, not a raw std that is huge in 384-d)."""
    Zc = F.normalize(Z, p=2, dim=-1)
    noise = torch.randn_like(Zc)
    # remove the component along Z so the noise is tangent, then scale to hit cos
    noise = noise - (noise * Zc).sum(-1, keepdim=True) * Zc
    noise = F.normalize(noise, p=2, dim=-1)
    theta = math.acos(max(-1.0, min(1.0, target_cos)))
    return F.normalize(math.cos(theta) * Zc + math.sin(theta) * noise, p=2, dim=-1)


def _embedding_stability(cluster: BalancedCluster, Z: torch.Tensor,
                         target_cos: float, trials: int) -> float:
    base = cluster.runtime_assign(Z)
    agree, total = 0, 0
    for _ in range(trials):
        Zp = _cosine_noise(Z, target_cos)
        agree += int((cluster.runtime_assign(Zp) == base).sum())
        total += base.numel()
    return agree / max(1, total)


# ======================================================================
# Step 3 — K sweep on the RUNTIME partition [#1] (design §9)
# ======================================================================
def k_sweep(Z_fit: torch.Tensor, Z_eval: torch.Tensor) -> List[Dict]:
    """Fit centroids on Z_fit (capacity-balanced), but report every §9 metric on the
    RUNTIME nearest-centroid partition of Z_eval — the partition specialists see."""
    rows = []
    for K in CIC.K_SWEEP:
        cluster, _fit_assign, _run_fit, inertia = BalancedCluster.fit_balanced(
            Z_fit, K, CIC)
        run_assign = cluster.runtime_assign(Z_eval)           # [#1]
        bal = _balance_stats(run_assign, K)
        marg = _margins(cluster, Z_eval)
        row = {
            "K": K, "inertia": inertia,
            "min_share": bal["min_share"], "max_share": bal["max_share"],
            "imbalance": bal["imbalance"], "starved": int(bal["starved"]),
            "empty": bal["empty"],
            "margin_mean": float(marg.mean()),
            "margin_p10": float(torch.quantile(marg, 0.10)),
            "silhouette": _silhouette_cosine(Z_eval, run_assign, K),
            "stability": _embedding_stability(cluster, Z_eval, CIC.PERTURB_COSINE,
                                              CIC.STABILITY_TRIALS),
        }
        rows.append(row)
        print(f"  K={K:2d} | RUNTIME bal[min={row['min_share']:.3f} "
              f"max={row['max_share']:.3f} imbal={row['imbalance']:.2f} "
              f"starved={row['starved']} empty={row['empty']}] "
              f"| margin[mean={row['margin_mean']:.3f} p10={row['margin_p10']:.3f}] "
              f"| sil={row['silhouette']:+.3f} stab={row['stability']:.3f}")
    return rows


def choose_K(rows: List[Dict]) -> int:
    def score(r):
        s = r["silhouette"] + r["margin_mean"] + r["stability"]
        s -= 2.0 * r["starved"] + 2.0 * r["empty"]
        s -= 0.2 * max(0.0, r["imbalance"] - 2.0)
        s -= 0.02 * abs(r["K"] - CIC.K_DEFAULT)
        return s
    return max(rows, key=score)["K"]


# ======================================================================
# Threshold calibration on the CALIBRATION split [#3] (design §14, §15)
# ======================================================================
def calibrate_fallback(cluster: BalancedCluster, Z_calib: torch.Tensor
                       ) -> Dict[str, float]:
    sims = cluster.similarities(Z_calib)
    top2 = torch.topk(sims, k=min(2, cluster.K), dim=1)
    s1 = top2.values[:, 0]
    margins = (top2.values[:, 0] - top2.values[:, 1]) if cluster.K > 1 \
        else torch.ones_like(s1)
    t_margin = float(torch.quantile(margins, CIC.CAL_MARGIN_QUANTILE))
    t_sim = float(torch.quantile(s1, CIC.CAL_SIM_QUANTILE))
    return {"T_MARGIN_FALLBACK": t_margin, "T_SIM_FALLBACK": t_sim,
            "_calib_margin_mean": float(margins.mean()),
            "_calib_s1_mean": float(s1.mean())}


def calibrate_switch(cluster: BalancedCluster, texts_calib: List[str],
                     encoder: ContentEncoder,
                     prefix_texts_calib: List[str] = None) -> Dict[str, float]:
    """[#4] Calibrate T_SWITCH between two MEASURED regimes:
      same-state gaps  : incumbent-vs-challenger score gap when only the VIEW of the
                         same state changes (should NOT switch) -> set T_SWITCH above
                         the 90th pct of these so normal jitter never flips a path.
      topic-shift gaps : score gap when the content genuinely changes clusters
                         (SHOULD switch) -> verify T_SWITCH sits below these.
    Returns the chosen T_SWITCH plus both distributions' summaries for the artifact.

    ci_prefix_patch: BOTH regimes now route on the PREFIX view the LM uses.
    topic-shift embeds a different context's prefix; same-state measures jitter of
    the SAME prefix under overlap perturbation (matching same_state_switch_rate) —
    not history-length variation — because T_SWITCH is the runtime hysteresis
    threshold and the LM routes on the prefix with no separate history mechanism.
    """
    pref = prefix_texts_calib if prefix_texts_calib is not None else texts_calib
    same_gaps, shift_gaps = [], []
    n = min(len(texts_calib), 400)
    idx = random.sample(range(len(texts_calib)), n)
    base_overlap = encoder.chunk_overlap
    try:
        for i in idx:
            # ci_prefix_patch: T_SWITCH is the RUNTIME hysteresis threshold, and the
            # LM routes on the 256-token PREFIX with no separate history mechanism.
            # So same-state jitter must be measured in the PREFIX routing view (the
            # same perturbation same_state_switch_rate / same_state_stability use —
            # overlap jitter on the same prefix), NOT history-length variation.
            prefix_txt = pref[i]
            if not prefix_txt.strip():
                continue

            # Incumbent: prefix routed at base overlap.
            encoder.chunk_overlap = base_overlap
            z0 = encoder.embed_text(prefix_txt)
            a0 = int(cluster.similarities(z0).argmax())

            # Perturbed SAME-state view: same prefix, jittered overlap.
            encoder.chunk_overlap = random.choice([0.0, 0.15, 0.25, 0.4])
            zv = encoder.embed_text(prefix_txt)
            sims_v = cluster.similarities(zv)

            k1 = int(sims_v.argmax())
            gap = float(sims_v[k1] - sims_v[a0])
            if gap > 0:
                same_gaps.append(gap)

            # topic-shift view: a DIFFERENT context, routed on the PREFIX the LM uses
            encoder.chunk_overlap = base_overlap
            j = random.choice(idx)
            if j != i:
                zj = encoder.embed_text(pref[j])
                aj = int(cluster.similarities(zj).argmax())
                if aj != a0:
                    sims_j = cluster.similarities(zj)
                    shift_gaps.append(float(sims_j[aj] - sims_j[a0]))
    finally:
        encoder.chunk_overlap = base_overlap
    same_t = torch.tensor(same_gaps) if same_gaps else torch.tensor([0.0])
    shift_t = torch.tensor(shift_gaps) if shift_gaps else torch.tensor([1.0])
    same_q = float(torch.quantile(same_t, CIC.CAL_SWITCH_SAMESTATE_Q))
    shift_q10 = float(torch.quantile(shift_t, 0.10))
    # T_SWITCH above same-state jitter; warn if it's not comfortably below shifts
    t_switch = max(same_q, 1e-4)
    note = "ok"
    if t_switch >= shift_q10:
        note = (f"WARNING: T_SWITCH ({t_switch:.4f}) >= 10th-pct topic-shift gap "
                f"({shift_q10:.4f}) — hysteresis may suppress real topic switches.")
    return {"T_SWITCH": t_switch, "_same_state_q": same_q,
            "_shift_q10": shift_q10, "_switch_note": note,
            "_n_same": len(same_gaps), "_n_shift": len(shift_gaps)}


# ======================================================================
# OOD calibration statistics [+review] (design §26)
# ======================================================================
def ood_stats(cluster: BalancedCluster, Z_id: torch.Tensor,
              texts_id: List[str], encoder: ContentEncoder) -> Dict[str, float]:
    """Record the ID top1-similarity distribution and, if enabled, its separation
    from a shuffled-token OOD proxy. GENERAL fallback is then a geometry-grounded
    'low ID confidence', with a recorded OOD reference."""
    id_s1 = cluster.similarities(Z_id).max(dim=1).values
    stats = {
        "id_s1_mean": float(id_s1.mean()), "id_s1_std": float(id_s1.std()),
        "id_s1_p05": float(torch.quantile(id_s1, 0.05)),
        "id_s1_p50": float(torch.quantile(id_s1, 0.50)),
    }
    if CIC.OOD_ENABLED and texts_id:
        n = min(CIC.OOD_PROXY_SAMPLES, len(texts_id))
        proxies = []
        for t in random.sample(texts_id, n):
            w = t.split()
            random.shuffle(w)                       # shuffled-token OOD proxy
            proxies.append(" ".join(w))
        Zo = encoder.embed_texts(proxies)
        ood_s1 = cluster.similarities(Zo).max(dim=1).values
        stats.update({
            "ood_s1_mean": float(ood_s1.mean()),
            "ood_s1_p50": float(torch.quantile(ood_s1, 0.50)),
            "id_ood_separation": float(id_s1.mean() - ood_s1.mean()),
        })
    return stats


# ======================================================================
# GATE diagnostics on the untouched GATE split [#3] (design §18,§19,§24)
# ======================================================================
def short_long_history_consistency(encoder, cluster, texts, n=300) -> float:
    """[#4/small] SAME current state, full vs shortened HISTORY (design §11, §24.5).
    We treat the last third of the text as the 'current turn' and the rest as
    'history'; full-history vs shortened-history must route the same. This is NOT
    first-third-of-document truncation (which can legitimately be a different topic)."""
    idx = random.sample(range(len(texts)), min(n, len(texts)))
    agree = 0
    for i in idx:
        words = texts[i].split()
        if len(words) < 6:
            agree += 1
            continue
        cut = max(1, int(len(words) * 2 / 3))
        current = " ".join(words[cut:])
        full_hist = " ".join(words[:cut])
        short_hist = " ".join(words[max(0, cut - len(words)//4):cut])
        zf = encoder.embed_context(current, full_hist)
        zs = encoder.embed_context(current, short_hist)
        agree += int(int(cluster.similarities(zf).argmax())
                     == int(cluster.similarities(zs).argmax()))
    return agree / max(1, len(idx))


def same_state_stability(encoder, cluster, texts, n=300) -> float:
    """[#4] Different VIEWS of the same state (overlap change) -> same cluster.
    Restores encoder.chunk_overlap with try/finally ([smaller-review])."""
    idx = random.sample(range(len(texts)), min(n, len(texts)))
    base_overlap = encoder.chunk_overlap
    agree, total = 0, 0
    try:
        for i in idx:
            z0 = encoder.embed_text(texts[i])
            k0 = int(cluster.similarities(z0).argmax())
            for _ in range(CIC.STABILITY_TRIALS):
                encoder.chunk_overlap = random.choice([0.0, 0.15, 0.25, 0.4])
                zp = encoder.embed_text(texts[i])
                agree += int(int(cluster.similarities(zp).argmax()) == k0)
                total += 1
    finally:
        encoder.chunk_overlap = base_overlap
    return agree / max(1, total)


def topic_shift_responsiveness(ci: ContentIndex, cluster: BalancedCluster,
                               encoder: ContentEncoder, texts, n=200) -> float:
    """[#4] When the content genuinely moves to a STRONGLY different cluster, the
    router (with hysteresis) SHOULD switch. Returns the switch rate on genuine
    strong shifts — HIGHER is better (design §18)."""
    idx = random.sample(range(len(texts)), min(n, len(texts)))
    switched, eligible = 0, 0
    for i in idx:
        z0 = encoder.embed_text(texts[i])
        a0 = int(cluster.similarities(z0).argmax())
        # find a text that lands in a different cluster with a clear margin
        for _ in range(6):
            j = random.choice(idx)
            zj = encoder.embed_text(texts[j])
            sims_j = cluster.similarities(zj)
            aj = int(sims_j.argmax())
            marg = float(torch.topk(sims_j, 2).values[0]
                         - torch.topk(sims_j, 2).values[1])
            if aj != a0 and marg > ci.t_margin:      # a genuine, confident shift
                eligible += 1
                res = ci.route_z(zj, previous_cluster=a0)
                switched += int(res.switched or res.cluster_id == aj)
                break
    return switched / max(1, eligible)


def same_state_switch_rate(ci: ContentIndex, encoder, texts, n=200) -> float:
    """[#4] Repeated SAME-state views through route() must almost never switch
    (design §18 unstable-oscillation). LOWER is better."""
    idx = random.sample(range(len(texts)), min(n, len(texts)))
    switched, transitions = 0, 0
    base_overlap = encoder.chunk_overlap
    try:
        for i in idx:
            prev = None
            for _ in range(CIC.STABILITY_TRIALS + 1):
                encoder.chunk_overlap = random.choice([0.0, 0.15, 0.25, 0.4])
                res = ci.route(texts[i], previous_cluster=prev)
                if prev is not None:
                    transitions += 1
                    switched += int(res.switched)
                prev = res.cluster_id if res.cluster_id is not None else prev
    finally:
        encoder.chunk_overlap = base_overlap
    return switched / max(1, transitions)


def coverage_on(ci: ContentIndex, Z: torch.Tensor) -> Dict[str, float]:
    res = ci.route_batch_z(Z)

    counts = torch.zeros(ci.K, dtype=torch.long)
    fallback = 0

    for r in res:
        if r.cluster_id is None:
            fallback += 1
        else:
            counts[r.cluster_id] += 1

    total = max(1, len(res))
    used = len(res) - fallback

    # Actual fraction of ALL inputs that each specialist receives.
    share_all = counts.float() / total

    return {
        "coverage": used / total,
        "fallback": fallback / total,
        "effective_counts": counts.tolist(),
        "effective_shares": share_all.tolist(),
        "effective_min_share": float(share_all.min()),
        "effective_max_share": float(share_all.max()),
        "effective_imbalance": float(
            share_all.max() / share_all.min().clamp_min(1e-9)
        ),
    }

def go_no_go(ci, cluster, encoder, Z_gate, texts_gate,
             prefix_texts_gate=None) -> Tuple[bool, Dict]:
    K = cluster.K
    # runtime-relevant gates (ci_prefix_patch): simulate what the LM routes on — the
    # 256-token prefix — so stability/switch/responsiveness certify the actual
    # routing view. history-variation gates (short/long) deliberately need the FULL
    # text as their "long" reference and stay on texts_gate.
    pref = prefix_texts_gate if prefix_texts_gate is not None else texts_gate
    run_assign = cluster.runtime_assign(Z_gate)            # [#1]
    bal = _balance_stats(run_assign, K)
    marg = _margins(cluster, Z_gate)
    cov = coverage_on(ci, Z_gate)
    stab = same_state_stability(encoder, cluster, pref)         # runtime view
    sl = short_long_history_consistency(encoder, cluster, texts_gate)  # full (by design)
    same_sw = same_state_switch_rate(ci, encoder, pref)        # runtime view
    shift_resp = topic_shift_responsiveness(ci, cluster, encoder, pref)  # runtime view

    equal = 1.0 / K
    checks = {
        # Candidate nearest-centroid partition.
        "no_starvation":
            bal["min_share"] >= CIC.MIN_CLUSTER_SHARE * equal,

        "no_empty_cluster":
            bal["empty"] == 0,

        "acceptable_balance":
            bal["imbalance"] <= (1.0 / CIC.MIN_CLUSTER_SHARE) + 0.5,

        # Actual specialist exposure AFTER confidence fallback.
        "effective_no_starvation":
            cov["effective_min_share"] >= CIC.MIN_CLUSTER_SHARE * equal,

        "effective_balance":
            cov["effective_imbalance"]
            <= (1.0 / CIC.MIN_CLUSTER_SHARE) + 0.5,

        "coverage_ok":
            cov["coverage"] >= CIC.MIN_SPECIALIST_COVERAGE,

        "useful_margins":
            float(marg.mean()) > ci.t_margin,

        "reproducible":
            stab >= CIC.STABILITY_MIN_SAMERATE,

        "short_long":
            sl >= CIC.SHORTLONG_MIN_AGREE,

        "low_samestate_switch":
            same_sw <= CIC.SAMESTATE_MAX_SWITCHRATE,

        "topicshift_responsive":
            shift_resp >= CIC.TOPICSHIFT_MIN_SWITCHRATE,
    }
    passed = all(checks.values())
    report = {
        "K": K, "balance": bal, "margin_mean": float(marg.mean()),
        "coverage": cov, "stability": stab, "short_long": sl,
        "samestate_switch": same_sw, "topicshift_switch": shift_resp,
        "checks": checks, "passed": passed,
    }
    return passed, report


def _print_gate(report: Dict):
    print("\n" + "=" * 64)
    print("CONTENT INDEX GO / NO-GO GATE (design §24) — RUNTIME partition")
    print("=" * 64)
    b = report["balance"]
    print(f"K                : {report['K']}")
    print(f"balance          : min={b['min_share']:.3f} max={b['max_share']:.3f} "
          f"imbal={b['imbalance']:.2f} starved={int(b['starved'])} empty={b['empty']}")
    print(f"margin (mean)    : {report['margin_mean']:.4f}")
    print(f"coverage         : spec={report['coverage']['coverage']:.3f} "
          f"fallback={report['coverage']['fallback']:.3f}")
    
    c = report["coverage"]

    print(f"effective balance: min={c['effective_min_share']:.3f} "
        f"max={c['effective_max_share']:.3f} "
        f"imbal={c['effective_imbalance']:.2f}")

    print(
        "effective shares : "
        + " ".join(
            f"P{i}={x:.3f}"
            for i, x in enumerate(c["effective_shares"])
        )
    )
    print(f"same-state stab  : {report['stability']:.3f} "
          f"(need >= {CIC.STABILITY_MIN_SAMERATE})")
    print(f"short/long hist  : {report['short_long']:.3f} "
          f"(need >= {CIC.SHORTLONG_MIN_AGREE})")
    print(f"same-state switch: {report['samestate_switch']:.3f} "
          f"(need <= {CIC.SAMESTATE_MAX_SWITCHRATE})  [oscillation]")
    print(f"topic-shift switch: {report['topicshift_switch']:.3f} "
          f"(need >= {CIC.TOPICSHIFT_MIN_SWITCHRATE})  [responsiveness]")
    print("-" * 64)
    for name, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("-" * 64)
    verdict = ("GO — ship the simple index; do NOT build the Content Transformer (§24/§25)."
               if report["passed"] else
               "NO-GO — investigate the HyperNet Content Transformer (§25).")
    print(f"VERDICT: {verdict}")
    print("=" * 64 + "\n")


def multiview_selfconsistency(ci, encoder: ContentEncoder,
                              n_samples: int, skip: int) -> Dict:
    """BUILD-TIME multi-view GEOMETRY SELF-CONSISTENCY pass (not PPL accuracy).

    For a sample of sequences, cut natural-break prefixes at ROUTE_PREFIX_TOKENS_SWEEP
    lengths (via prefix_views.py, in LM-token space), route each view through the
    FROZEN ci, and measure how often a shorter view's route agrees with the SAME
    sequence's route at the 256 runtime view (the label). This answers "from how
    short a prefix does the CI's own routing stabilize?" — a geometry diagnostic.
    It does NOT use LM P0..P5 NLL and is NOT routing accuracy/regret; the harness
    measures those later against the real LM. Never gates GO/NO-GO.

    Reported per prefix length: agree-rate with the 256 route (top-1), top-2
    agreement (256 route in the view's top-2 centroids), a confusion matrix
    (256-route cluster -> view-route cluster), and the count of sequences that
    actually produced a view at that length (short sequences yield fewer views)."""
    try:
        from prefix_views import prefix_views_for_sequence
    except Exception as e:
        print(f"[multiview] prefix_views.py not importable ({e}); skipping.")
        return {"enabled": False, "reason": "prefix_views import failed"}
    if not _HAVE_LM_DATA:
        print("[multiview] no LM data stream; skipping.")
        return {"enabled": False, "reason": "no data.py"}

    K = ci.cluster.K
    targets = list(getattr(CIC, "ROUTE_PREFIX_TOKENS_SWEEP", [32, 64, 128, 256]))
    runtime_view = CIC.ROUTE_PREFIX_TOKENS
    if runtime_view not in targets:
        targets.append(runtime_view)
    targets = sorted(set(targets))

    tokenizer = D.get_tokenizer()
    it = D.batch_iterator(tokenizer, skip_sequences=skip)

    agree1 = {t: 0 for t in targets}
    agree2 = {t: 0 for t in targets}
    seen = {t: 0 for t in targets}
    confusion = {t: torch.zeros(K, K, dtype=torch.long) for t in targets}
    n_seq = 0

    for input_ids, srcs, _c, _s in it:
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            views = prefix_views_for_sequence(tokenizer, ids, targets=targets,
                                              runtime_view=runtime_view)
            if not views:
                continue
            label_view = next((v for v in views if v.target_len == runtime_view), None)
            if label_view is None:
                continue                       # sequence too short to reach 256
            z_label = encoder.embed_text(label_view.text)
            label = int(ci.cluster.runtime_assign(z_label.unsqueeze(0))[0])

            for v in views:
                zv = encoder.embed_text(v.text)
                sims = ci.cluster.similarities(zv.unsqueeze(0))[0]
                order = torch.argsort(sims, descending=True)
                top1 = int(order[0])
                top2 = {int(order[0]), int(order[1])} if K > 1 else {int(order[0])}
                seen[v.target_len] += 1
                if top1 == label:
                    agree1[v.target_len] += 1
                if label in top2:
                    agree2[v.target_len] += 1
                confusion[v.target_len][label, top1] += 1

            n_seq += 1
            if n_seq >= n_samples:
                break
        if n_seq >= n_samples:
            break

    report = {"enabled": True, "n_sequences": n_seq, "runtime_view": runtime_view,
              "note": ("geometry self-consistency vs the 256 route; NOT PPL "
                       "accuracy/regret (harness measures those vs LM NLL)"),
              "per_length": {}}
    print("\n" + "=" * 64)
    print("MULTI-VIEW SELF-CONSISTENCY (build-time geometry diagnostic)")
    print("  agree = shorter view's route matches THIS sequence's 256-view route")
    print("  (NOT routing accuracy vs LM PPL — that's the harness's job)")
    print("=" * 64)
    print(f"{'prefix':>7} | {'n_seq':>6} | {'top1_agree':>10} | {'top2_agree':>10}")
    for t in targets:
        n = max(1, seen[t])
        a1 = agree1[t] / n
        a2 = agree2[t] / n
        report["per_length"][t] = {
            "n": seen[t], "top1_agree": a1, "top2_agree": a2,
            "confusion": confusion[t].tolist(),
        }
        tag = "  <- runtime view (self, =1.0 by construction)" if t == runtime_view else ""
        print(f"{t:>7} | {seen[t]:>6} | {a1:>10.3f} | {a2:>10.3f}{tag}")
    print("=" * 64)
    print("Read the curve: top1_agree should RISE toward the 256 view. A low value")
    print("at 32/64 means short-prefix routing is geometrically unstable — expected,")
    print("and exactly what the later PPL harness will quantify against the real LM.")
    return report


# ======================================================================
# main
# ======================================================================
def main():
    args = set(sys.argv[1:])
    # PPL-feedback refit is a permanent generic MODE (not round-specific): refit the
    # frozen CI against any LM checkpoint's six-path PPL. Delegated to ppl_refit.py,
    # which reuses this file's encoder/cluster/save conventions.
    if "--ppl-refit" in args:
        import ppl_refit
        # pass through everything except the mode flag itself
        ppl_refit.run([a for a in sys.argv[1:] if a != "--ppl-refit"])
        return
    sweep_only = "--sweep" in args
    dry = "--dry" in args
    random.seed(CIC.BUILD_SEED)
    torch.manual_seed(CIC.BUILD_SEED)

    # ci_prefix_patch — HARD prefix match. The Content Index MUST fit/calibrate on
    # the SAME routing view the LM queries (its causal CI_ROUTE_PREFIX). A mismatch
    # would certify thresholds the LM never uses, so refuse to build rather than ship
    # a silently-miscalibrated artifact. (The config comment alone isn't enforcement.)
    if _HAVE_LM_DATA:
        lm_prefix = getattr(C, "CI_ROUTE_PREFIX", None)
        ci_prefix = getattr(CIC, "ROUTE_PREFIX_TOKENS", None)
        if lm_prefix is not None and ci_prefix is not None and lm_prefix != ci_prefix:
            raise SystemExit(
                f"[FATAL] ROUTE_PREFIX_TOKENS ({ci_prefix}) != LM CI_ROUTE_PREFIX "
                f"({lm_prefix}). The index would be calibrated on a different routing "
                f"view than the LM uses. Set content_index_config.ROUTE_PREFIX_TOKENS "
                f"= config.CI_ROUTE_PREFIX and rebuild.")

    print("=" * 64)
    print(f"BUILD CONTENT INDEX — {CIC.VERSION_ID}")
    print(f"encoder={CIC.ENCODER_NAME} backend={CIC.ENCODER_BACKEND} "
          f"chunk={CIC.CHUNK_TOKENS}tok overlap={CIC.CHUNK_OVERLAP} pool={CIC.POOL}")
    print("=" * 64)

    encoder = ContentEncoder(CIC)

    # Step 1 — THREE disjoint splits [#3]. The inter-split GAP keeps splits from
    # overlapping on the STREAMED corpus (skip past already-seen region). On a
    # finite local dataset the splits are placed back-to-back (gap 0) since the
    # frozen array is already disjoint by offset. Configurable via BUILD_SPLIT_GAP.
    gap = getattr(CIC, "BUILD_SPLIT_GAP", 10000)
    print(f"[collect] FIT={CIC.FIT_SAMPLES} CALIB={CIC.CALIB_SAMPLES} "
          f"GATE={CIC.GATE_SAMPLES} gap={gap} (disjoint)")

    # D1=B — FIT trains on the 32..256 MULTI-VIEW cloud (augmentation). CALIB and
    # GATE stay on the pure 256 runtime view: the LM deploys at 256, so thresholds
    # and the GO gate describe the DEPLOYED distribution (ratified override).
    Z_fit, sources_fit, mv_fit_info = collect_multiview_fit(
        encoder, CIC.FIT_SAMPLES, CIC.BUILD_SKIP)
    print(f"[collect] FIT multi-view: {mv_fit_info['n_sequences']} sequences -> "
          f"{mv_fit_info['n_views']} views  per_target={mv_fit_info['per_target']}")

    off1 = CIC.BUILD_SKIP + CIC.FIT_SAMPLES + gap

    Z_cal, texts_cal, pref_cal, _ = collect_content_vectors(
        encoder, CIC.CALIB_SAMPLES, off1)

    off2 = off1 + CIC.CALIB_SAMPLES + gap

    Z_gate, texts_gate, pref_gate, _ = collect_content_vectors(
        encoder, CIC.GATE_SAMPLES, off2)

    print(f"[collect] Z_fit(multiview)={tuple(Z_fit.shape)} "
          f"Z_cal(256)={tuple(Z_cal.shape)} Z_gate(256)={tuple(Z_gate.shape)}")
    
    spec = encoder.pipeline_spec()

    print(f"[encoder] resolved_backend={spec['backend']} "
        f"resolved_revision={spec['revision']}")

    if spec["revision"] is None:
        print("[encoder] WARNING: model revision was not resolved; "
            "set ENCODER_REVISION explicitly for a fully pinned artifact.")

    # Step 3 — K sweep, evaluated on the RUNTIME partition of the CALIB split [#1]
    print("\n[sweep] K in", CIC.K_SWEEP, "(fit on FIT, eval RUNTIME argmax on CALIB)")
    rows = k_sweep(Z_fit, Z_cal)
    chosen = choose_K(rows)
    K = CIC.FORCE_K if CIC.FORCE_K is not None else chosen
    if not hasattr(C, "N_SPECIALISTS"):
        raise RuntimeError(
            "config.py must define N_SPECIALISTS explicitly. "
            "Content Index build cannot verify K/path compatibility."
        )

    n_spec = int(C.N_SPECIALISTS)
    print(f"\n[sweep] sweep winner K={chosen}; "
          f"{'FORCE_K=' + str(CIC.FORCE_K) if CIC.FORCE_K is not None else 'using winner'} "
          f"-> K={K}   (N_SPECIALISTS={n_spec})")

    if sweep_only:
        return

    # Step 2b — HARD STOP on K mismatch [#2] (design §9)
    if K != n_spec and CIC.REQUIRE_K_EQUALS_SPECIALISTS:
        print("\n" + "!" * 64)
        print(f"HARD STOP: chosen K ({K}) != N_SPECIALISTS ({n_spec}).")
        print("The design requires an EXPLICIT architecture decision here (§9):")
        print(f"  A. change N_SPECIALISTS to {K} in config.py, OR")
        print(f"  B. keep {n_spec} paths and set FORCE_K={n_spec} in "
              f"content_index_config.py to tolerate redundant clusters.")
        print("Refusing to save a GO artifact until this is resolved.")
        print("(If K > N_SPECIALISTS the CI would emit cluster IDs with no path;")
        print(" if K < N_SPECIALISTS some specialists would get no partition.)")
        print("!" * 64)
        return

    # Step 4 — fit BALANCED clustering at K on the FIT split (design §10).
    # WEAK SOURCE PRESSURE (initial build only, §9): bias the k-means++ SEED across
    # sources so no dominant source starves a cluster. Geometry alone drives the
    # Lloyd iterations; runtime routing never sees source. The K sweep above stays
    # source-FREE (pressure there would confound the sweep). Subsequent CI rounds
    # (the trainer) are PPL-driven and pass no source pressure.
    src_w = getattr(CIC, "SOURCE_PRESSURE_WEIGHT", 0.0)
    print(f"\n[fit] balanced clustering K={K} (capacity_factor={CIC.CAPACITY_FACTOR}"
          + (f", source_pressure={src_w} on {len(set(map(str, sources_fit)))} sources)"
             if src_w > 0 else ")"))
    cluster, fit_assign, run_assign, inertia = BalancedCluster.fit_balanced(
        Z_fit, K, CIC, source_ids=sources_fit, source_pressure=src_w)
    rb = _balance_stats(run_assign, K)              # RUNTIME balance on the fit split
    print(f"[fit] RUNTIME shares min={rb['min_share']:.3f} max={rb['max_share']:.3f} "
          f"imbalance={rb['imbalance']:.2f} starved={int(rb['starved'])} "
          f"empty={rb['empty']}")

    # thresholds — CALIB split only [#3]
    fb = calibrate_fallback(cluster, Z_cal)
    sw = calibrate_switch(cluster, texts_cal, encoder, prefix_texts_calib=pref_cal)
   
    if sw["_n_same"] < 20 or sw["_n_shift"] < 20:
        sw["T_SWITCH"] = CIC.CAL_SWITCH_OVER_MARGIN * max(
            fb["T_MARGIN_FALLBACK"], 1e-4
        )
        sw["_switch_note"] = (
            "heuristic (insufficient same-state or topic-shift gaps)"
        )

    thr = {
        "T_MARGIN_FALLBACK": fb["T_MARGIN_FALLBACK"],
        "T_SIM_FALLBACK": fb["T_SIM_FALLBACK"],
        "T_SWITCH": sw["T_SWITCH"],
    }

    print(f"[calibrate] T_MARGIN_FALLBACK={thr['T_MARGIN_FALLBACK']:.4f} "
          f"T_SIM_FALLBACK={thr['T_SIM_FALLBACK']:.4f} "
          f"T_SWITCH={thr['T_SWITCH']:.4f}")
    print(f"[calibrate] switch: same_state_q={sw['_same_state_q']:.4f} "
          f"shift_q10={sw['_shift_q10']:.4f} n_same={sw['_n_same']} "
          f"n_shift={sw['_n_shift']} -> {sw['_switch_note']}")

    # OOD stats on CALIB [+review]
    ood = ood_stats(cluster, Z_cal, pref_cal, encoder)
    print(f"[ood] id_s1_mean={ood['id_s1_mean']:.3f} "
          + (f"ood_s1_mean={ood.get('ood_s1_mean', float('nan')):.3f} "
             f"sep={ood.get('id_ood_separation', float('nan')):.3f}"
             if 'ood_s1_mean' in ood else "(proxy off)"))

    ci = ContentIndex(encoder, cluster, thr, CIC.VERSION_ID, meta={
        "chosen_K": K, "sweep_winner": chosen, "n_specialists": n_spec,
        "fit_inertia": inertia, "sweep": rows,
        "calibration": {**fb, **sw}, "ood": ood,
        "splits": {"fit": CIC.FIT_SAMPLES, "calib": CIC.CALIB_SAMPLES,
                   "gate": CIC.GATE_SAMPLES},
        # runtime routing view (deployment) — the LM queries the first 256 tokens
        # verbatim; CALIB/GATE/thresholds are all measured on THIS view.
        "route_prefix_tokens": CIC.ROUTE_PREFIX_TOKENS,
        # D1=B provenance: FIT trained on the multi-view cloud; record which views
        # and how many, so the artifact honestly reflects the fit distribution.
        "fit_multiview": {
            "sweep": list(getattr(CIC, "ROUTE_PREFIX_TOKENS_SWEEP",
                                  [32, 64, 128, 256])),
            "per_target_views": mv_fit_info.get("per_target"),
            "n_sequences": mv_fit_info.get("n_sequences"),
            "n_views": mv_fit_info.get("n_views"),
            "note": ("centroids fit on 32..256 views (32 forward-snapped, 64/128 "
                     "backward-snapped, 256 exact); CALIB/GATE on 256 only (D1=B)"),
        },
        "source_pressure_weight": getattr(CIC, "SOURCE_PRESSURE_WEIGHT", 0.0),
    })

    # Step 7 + 8 — GATE diagnostics on the UNTOUCHED gate split [#3] + GO/NO-GO
    passed, report = go_no_go(ci, cluster, encoder, Z_gate, texts_gate,
                              prefix_texts_gate=pref_gate)
    _print_gate(report)

    # Step 10 — freeze assertions (design §12, §30)
    ci.assert_frozen()
    print("[freeze] assert_frozen() OK — no trainable Content Index tensors.")

    if not passed:
        print("[build] gate FAILED — not saving a GO artifact. The FAILED checks "
              "indicate whether mean pooling is the bottleneck (design §25) before "
              "adding the Content Transformer.")
        if not dry:
            nogo = CIC.ARTIFACT.replace(".pt", ".NOGO.pt")
            ci.save(nogo)
            print(f"[build] diagnostics artifact saved: {nogo}")
        return
    # Multi-view self-consistency (build-time geometry diagnostic; never gates).
    # Sample sequences PAST the gate split so it doesn't reuse gate contexts.
    if getattr(CIC, "MULTIVIEW_EVAL_ENABLED", False):
        mv_skip = off2 + CIC.GATE_SAMPLES + gap
        mv = multiview_selfconsistency(
            ci, encoder, getattr(CIC, "MULTIVIEW_EVAL_SAMPLES", 2000), mv_skip)
        ci.meta["multiview_selfconsistency"] = mv

    if dry:
        print("[build] --dry: gate PASSED but not saving.")
        return

    path = ci.save(CIC.ARTIFACT)
    print(f"[build] GO — saved frozen Content Index artifact: {path}")
    print(f"[build] LM checkpoints must record version_id = {ci.version_id} (§26).")


if __name__ == "__main__":
    main()
