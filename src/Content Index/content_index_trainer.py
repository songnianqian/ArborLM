"""
content_index_trainer.py — Round-level Content Index UPDATE (design §4, §13, §18,
§22-§26, §30). The co-training loop's "Content Index update" half.

    prior artifact  +  ppl_round.json (measured per-specialist NLLs, NO GENERAL)
        -> soft PPL responsibilities  q_d(x) ∝ exp(-L_d/τ)          §13
        -> contested-only relabel vs carried soft targets           §18
        -> responsibility-weighted multi-prototype M-step           §13, §22
        -> dead-prototype detection + PPL-aware rejuvenation        §23-§26
        -> anti-oscillation damping                                 §13
        -> threshold re-calibration for the new geometry            §14, §15
        -> freeze + save NEXT artifact + carried soft targets       §26, §30

INPUTS
  prior artifact : content_index.pt (or a previous round's .rN.pt) — read-only.
  ppl JSON       : the harness output (ppl_harness.py). Per-context records with
                   nll{0..K-1} (per specialist, NO GENERAL), route, gains, margins.
  prior soft     : optional soft_targets.r{N-1}.json for contested-only carry (§18).

OUTPUTS
  content_index.r{N}.pt          : the NEXT frozen artifact (multi-prototype).
  soft_targets.r{N}.json         : per-context soft targets, so round N+1 can carry.

WHAT THIS TRAINER DOES NOT DO
  * It never trains the LM and never sees an LM gradient (§1, §2, §30).
  * It never uses source id as a routing target — source is metadata only (§9).
  * It does not run the LM; it consumes the PPL JSON the harness already wrote.

The output artifact is loadable by the UNCHANGED frozen runtime (content_index.py):
ContentIndex.load auto-detects the multi-prototype layout; route()'s contract is
identical (similarities -> [N,K] -> argmax + hysteresis).

Usage:
    python content_index_trainer.py \
        --artifact content_index.pt \
        --ppl ppl_round.json \
        [--prior-soft soft_targets.r1.json] \
        [--out content_index.r2.pt] \
        [--round 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import content_index_config as CIC
import content_trainer_config as TC
from content_index import (ContentEncoder, ContentIndex,
                           BalancedCluster, MultiPrototypeCluster)


# ======================================================================
# Load the measured PPL round (harness output) + prior soft targets
# ======================================================================
def load_ppl(ppl_path: str) -> Dict:
    with open(ppl_path) as f:
        blob = json.load(f)
    if blob.get("schema") not in ("projectB.ppl.v1", "projectB.ppl_round1.v1"):
        raise RuntimeError(f"unexpected PPL schema: {blob.get('schema')!r}")
    if not blob.get("records"):
        raise RuntimeError("PPL JSON has no records.")
    return blob


def load_prior_soft(path: Optional[str]) -> Dict[str, Dict]:
    """Map sample_id -> prior soft-target dict, or {} if none (bootstrap round)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        blob = json.load(f)
    return {r["sample_id"]: r for r in blob.get("records", [])}


# ======================================================================
# Step 1 — soft PPL responsibilities  q_d(x)  (design §13)
# ======================================================================
def _nll_vector(rec: Dict, K: int) -> List[float]:
    """[L_0..L_{K-1}] from a record's measured NLLs (NO GENERAL — 6 specialists).

    EXHAUSTIVE-ONLY (teammate note): every specialist must have a measured NLL this
    round. Top-n partial measurement (§14) fills only the predicted specialists, and
    a principled missing-loss teacher (imputing the unmeasured ones) is NOT yet
    built. So we HARD-STOP on any missing specialist rather than guess. When top-n
    is added later, replace this guard with the imputation model."""
    nll = rec["nll"]
    missing = [k for k in range(K) if str(k) not in nll]
    if missing:
        raise RuntimeError(
            f"context {rec.get('sample_id')} is missing specialist NLLs {missing}. "
            f"The round-level trainer requires EXHAUSTIVE all-path PPL (measure all "
            f"K specialists). Re-run the harness with specialists=None (the default). "
            f"Top-n partial-PPL handling is not implemented yet.")
    Ls = [float(nll[str(k)]) for k in range(K)]
    return Ls


def calibrate_tau(records: List[Dict], K: int) -> float:
    """Pick τ so the MEDIAN context's top-1 responsibility ≈ TAU_AUTO_TARGET_TOP1
    (design §13: reproduce a moderately peaked target, not near-hard, not uniform).
    Bisection on τ over the measured gain spreads."""
    if TC.TAU is not None:
        return float(TC.TAU)

    # per-context destination scores used inside the softmax (6 specialists, no
    # GENERAL). q_k ∝ exp(-L_k/τ): higher score = lower NLL = more responsibility.
    score_sets = []
    for rec in records:
        Ls = _nll_vector(rec, K)
        scores = [-Lk for Lk in Ls]
        score_sets.append(torch.tensor(scores))

    def median_top1(tau: float) -> float:
        tops = []
        for s in score_sets:
            q = torch.softmax(s / tau, dim=0)
            tops.append(float(q.max()))
        return float(torch.tensor(tops).median())

    lo, hi = TC.TAU_MIN, TC.TAU_MAX
    target = TC.TAU_AUTO_TARGET_TOP1
    # median_top1 is monoth decreasing in tau; bisect
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if median_top1(mid) > target:
            lo = mid            # too peaked -> raise tau
        else:
            hi = mid
    return max(TC.TAU_MIN, min(TC.TAU_MAX, 0.5 * (lo + hi)))


def soft_responsibilities(rec: Dict, K: int, tau: float) -> Dict[str, float]:
    """q over {P0..P{K-1}} for one context (design §13). NO GENERAL: raw softmax
    over specialist NLLs, q_k ∝ exp(-L_k/τ). Keys sum to 1 over the K specialists."""
    Ls = _nll_vector(rec, K)
    scores = torch.tensor([-Lk for Lk in Ls])
    q = torch.softmax(scores / tau, dim=0)
    out = {}
    for k in range(K):
        out[f"P{k}"] = float(q[k])
    return out


# ======================================================================
# Step 2 — contested-only relabeling (design §18)
# ======================================================================
def ppl_top1_top2_margin(rec: Dict, K: int) -> float:
    """The LM's PPL top1/top2 margin for a context: L_2nd_best − L_best (nats).
    Small => the LM is nearly indifferent between its two best specialists."""
    Ls = _nll_vector(rec, K)
    if len(Ls) < 2:
        return float("inf")
    Ls_sorted = sorted(Ls)
    return float(Ls_sorted[1] - Ls_sorted[0])


def calibrate_ppl_margin_threshold(records: List[Dict], K: int) -> float:
    """Ratified D3: contest threshold = min(quantile(margins, Q), CAP). Adaptive to
    the round (LM sharpens over rounds) but capped so a decisive round can't mark
    genuinely-separated contexts as indifferent."""
    margins = torch.tensor([ppl_top1_top2_margin(r, K) for r in records])
    margins = margins[torch.isfinite(margins)]
    if margins.numel() == 0:
        return float(TC.CONTEST_PPL_MARGIN_CAP)
    q = float(torch.quantile(margins, TC.CONTEST_PPL_MARGIN_QUANTILE))
    return float(min(q, TC.CONTEST_PPL_MARGIN_CAP))


def is_contested(rec: Dict, prior: Optional[Dict], q_new: Dict[str, float],
                 K: int, ppl_margin_thr: float) -> Tuple[bool, str]:
    """Return (contested, reason). Non-contested contexts keep their carried
    prior soft target for this round (§18)."""
    if prior is None:
        if TC.FIRST_ROUND_ALL_CONTESTED:
            return True, "no_prior"
        return True, "no_prior"

    # low Content Index top1/top2 margin (CI GEOMETRY — how confident the router is)
    if float(rec["route"]["margin"]) <= TC.CONTEST_MARGIN_MAX:
        return True, "low_margin"

    # (NO GENERAL: the GENERAL-vs-best-specialist near-tie trigger is removed —
    #  there is no GENERAL destination to tie against.)

    # Ratified D3 — LM PPL near-tie. When the LM's two best specialists are within
    # the calibrated PPL margin threshold (nats), routing barely matters / is
    # ambiguous, so re-derive this context from fresh PPL. Replaces the old
    # delta_cost/near_dcost band. This is the LM's own indifference (PPL), distinct
    # from the CI-geometry low_margin trigger above.
    if ppl_top1_top2_margin(rec, K) <= ppl_margin_thr:
        return True, "ppl_near_tie"

    # route changed vs prior artifact's recorded route
    if TC.CONTEST_ROUTE_CHANGED and "route_destination" in prior:
        cur = rec["route"]["destination"]
        if str(prior["route_destination"]) != str(cur):
            return True, "route_changed"

    # teacher disagreement: new top1 responsibility differs from carried
    prior_q = prior.get("soft", {})
    if prior_q:
        pk = max(prior_q, key=prior_q.get)
        if abs(q_new.get(pk, 0.0) - prior_q[pk]) > TC.CONTEST_TEACHER_DISAGREE:
            return True, "teacher_disagree"

    return False, "stable"


# ======================================================================
# Step 3 — responsibility-weighted multi-prototype M-step (design §13, §22)
# ======================================================================
def mstep_prototypes(
    Z: torch.Tensor,                 # [N, dim] content vectors for the round
    resp: torch.Tensor,             # [N, K] specialist responsibility (no GENERAL)
    prior_cluster,                  # prior BalancedCluster / MultiPrototypeCluster
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Soft k-means M-step, multi-prototype. For specialist k, every context pulls
    on k with weight resp[:,k]; within k the context routes to its nearest of M
    prototypes and updates it by the responsibility-weighted mean. Returns
    (prototypes [P,dim], proto_specialist [P], usage_info)."""
    dim = Z.shape[1]
    M = TC.PROTOTYPES_PER_SPECIALIST
    Zc = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)

    # initialize each specialist's M prototypes from the prior geometry when possible
    protos = torch.zeros(K * M, dim)
    proto_specialist = torch.zeros(K * M, dtype=torch.long)
    for k in range(K):
        proto_specialist[k * M:(k + 1) * M] = k

    # seed: spread M prototypes around specialist k's responsibility-weighted mean
    for k in range(K):
        w = resp[:, k].clamp_min(0)
        if float(w.sum()) < TC.MSTEP_MIN_WEIGHT:
            # dead-on-arrival specialist: seed from top-responsibility contexts anyway
            top = torch.topk(resp[:, k], min(M, Z.shape[0])).indices
            seed = Zc[top]
        else:
            mean = F.normalize((Zc * w.unsqueeze(1)).sum(0) / w.sum(), dim=-1)
            # k-means++-ish spread: mean + its M-1 farthest weighted contexts
            sims_to_mean = Zc @ mean
            far = torch.topk(-(sims_to_mean) * (w > 0).float(),
                             min(M - 1, Z.shape[0])).indices
            seed = torch.cat([mean.unsqueeze(0), Zc[far]], dim=0)
        if seed.shape[0] < M:                       # pad if too few contexts
            pad = seed[-1:].repeat(M - seed.shape[0], 1)
            seed = torch.cat([seed, pad], dim=0)
        protos[k * M:(k + 1) * M] = F.normalize(seed[:M], dim=-1)

    # Lloyd iterations: assign-to-nearest-prototype-within-specialist, weighted mean
    usage = torch.zeros(K * M)
    for _ in range(TC.MSTEP_ITERS):
        new = protos.clone()
        usage = torch.zeros(K * M)
        for k in range(K):
            cols = slice(k * M, (k + 1) * M)
            Pk = protos[cols]                        # [M, dim]
            w = resp[:, k].clamp_min(0)              # [N]
            active = w > TC.MSTEP_MIN_WEIGHT
            if active.sum() == 0:
                continue
            sub = Zc[active] @ Pk.t()                # [n_k, M]
            nearest = sub.argmax(dim=1)              # which prototype each ctx feeds
            wk = w[active]
            for m in range(M):
                sel = nearest == m
                usage[k * M + m] += float(wk[sel].sum())
                if sel.any() and float(wk[sel].sum()) > TC.MSTEP_MIN_WEIGHT:
                    num = (Zc[active][sel] * wk[sel].unsqueeze(1)).sum(0)
                    new[k * M + m] = F.normalize(num, dim=-1)
        if torch.allclose(new, protos, atol=1e-5):
            protos = new
            break
        protos = new

    return protos, proto_specialist, {"usage": usage.tolist(), "M": M}


# ======================================================================
# Step 4 — anti-oscillation damping (design §13)
# ======================================================================
def damp_prototypes(new_protos: torch.Tensor, proto_specialist: torch.Tensor,
                    prior_cluster, exempt_mask: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
    """c := (1-DAMP)*c_new + DAMP*c_prior, matched per specialist by nearest prior
    prototype. Rejuvenated prototypes (exempt_mask True) are NOT damped — they must
    move freely to their new region (§24/§25)."""
    damp = TC.CENTROID_DAMP
    if damp <= 0.0 or prior_cluster is None:
        return F.normalize(new_protos, dim=-1)
    prior = getattr(prior_cluster, "prototypes", None)
    if prior is None:
        prior = prior_cluster.centroids
    prior_spec = getattr(prior_cluster, "proto_specialist", None)

    out = new_protos.clone()
    for i in range(new_protos.shape[0]):
        if exempt_mask is not None and bool(exempt_mask[i]):
            continue
        k = int(proto_specialist[i])
        if prior_spec is not None:
            cand = prior[prior_spec == k]
        else:
            cand = prior[k:k + 1] if k < prior.shape[0] else prior
        if cand.numel() == 0:
            continue
        j = (F.normalize(new_protos[i:i + 1], dim=-1) @ cand.t()).argmax()
        out[i] = F.normalize((1 - damp) * new_protos[i] + damp * cand[j], dim=-1)
    return F.normalize(out, dim=-1)


# ======================================================================
# Step 5 — dead-prototype detection + PPL-aware rejuvenation (§23-§26)
# ======================================================================
def rejuvenate(protos: torch.Tensor, proto_specialist: torch.Tensor,
               usage: torch.Tensor, Z: torch.Tensor, records: List[Dict],
               K: int, prior_age: Dict[str, int]
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Detect dead prototypes (usage share < DEAD_USAGE_FRAC of their specialist)
    and move each to the context maximizing R(x)=Δ_k(x)·(1-max_m cos) (§25). A
    prototype within its grace period (age < REJUV_GRACE_ROUNDS) is never killed
    (§26). Returns (protos, exempt_mask, info)."""
    M = TC.PROTOTYPES_PER_SPECIALIST
    exempt = torch.zeros(protos.shape[0], dtype=torch.bool)
    info = {"rejuvenated": [], "dead_candidates": []}
    if not TC.REJUV_ENABLED:
        return protos, exempt, info

    Zc = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)
    usage = torch.tensor(usage) if not torch.is_tensor(usage) else usage
    new_age: Dict[str, int] = {}

    for k in range(K):
        cols = list(range(k * M, (k + 1) * M))
        spec_mass = max(float(usage[cols].sum()), 1e-9)
        # per-prototype gain R(x) precompute for this specialist's best context.
        # NO GENERAL baseline: Δ_k(x) = min_{j≠k} L_j − L_k  (best-of-rest minus k).
        # Positive ⇒ specialist k is the OUTRIGHT best path for context x (beats
        # every rival), which is exactly the signal for moving a dead prototype
        # toward contexts that uniquely want k. candidate contexts: Δ_k > REJUV_MIN_GAIN
        def _delta_best_of_rest(r) -> float:
            Ls = [float(r["nll"][str(j)]) for j in range(K)]
            Lk = Ls[k]
            rest = [Ls[j] for j in range(K) if j != k]
            return (min(rest) - Lk) if rest else 0.0
        gains = torch.tensor([_delta_best_of_rest(r) for r in records])
        rejuved_here = 0
        for c in cols:
            age = prior_age.get(str(c), TC.REJUV_GRACE_ROUNDS)  # unseen -> mature
            share = float(usage[c]) / spec_mass
            in_grace = age < TC.REJUV_GRACE_ROUNDS
            if share < TC.DEAD_USAGE_FRAC and not in_grace:
                info["dead_candidates"].append({"proto": c, "k": k,
                                                "share": round(share, 4)})
                if rejuved_here >= TC.REJUV_MAX_PER_SPECIALIST:
                    new_age[str(c)] = age + 1
                    continue
                # R(x) = Δ_k(x) * (1 - max_m cos(z, c_{k,m}))
                cover = (Zc @ protos[cols].t()).max(dim=1).values   # [N]
                R = gains * (1.0 - cover)
                R[gains <= TC.REJUV_MIN_GAIN] = -1e9
                xi = int(R.argmax())
                if float(R[xi]) > 0:
                    protos[c] = Zc[xi]
                    exempt[c] = True
                    new_age[str(c)] = 0                 # reset grace (§26)
                    rejuved_here += 1
                    info["rejuvenated"].append({"proto": c, "k": k,
                                                "context": records[xi]["sample_id"],
                                                "R": round(float(R[xi]), 4)})
                else:
                    new_age[str(c)] = age + 1
            else:
                new_age[str(c)] = age + 1

    info["proto_age"] = new_age
    return F.normalize(protos, dim=-1), exempt, info


# ======================================================================
# Step 6 — threshold re-calibration for the new geometry (design §14, §15)
# ======================================================================
def recalibrate_thresholds(cluster, Z: torch.Tensor,
                           prior_thr: Dict[str, float]) -> Dict[str, float]:
    if not TC.RECALIBRATE_THRESHOLDS:
        return prior_thr
    sims = cluster.similarities(Z)
    K = cluster.K
    top2 = torch.topk(sims, k=min(2, K), dim=1)
    s1 = top2.values[:, 0]
    margins = (top2.values[:, 0] - top2.values[:, 1]) if K > 1 else torch.ones_like(s1)
    t_margin = float(torch.quantile(margins, CIC.CAL_MARGIN_QUANTILE))
    t_sim = float(torch.quantile(s1, CIC.CAL_SIM_QUANTILE))
    # keep prior T_SWITCH scaled to the new margin (switch calib needs text pairs;
    # we preserve the prior ratio rather than fabricate a weaker heuristic)
    ratio = (prior_thr["T_SWITCH"] / max(prior_thr["T_MARGIN_FALLBACK"], 1e-6))
    t_switch = float(ratio * max(t_margin, 1e-4))
    return {"T_MARGIN_FALLBACK": t_margin, "T_SIM_FALLBACK": t_sim,
            "T_SWITCH": t_switch}


# ======================================================================
# Driver
# ======================================================================
def ppl_routing_report(records: List[Dict], run_assign: torch.Tensor, K: int,
                       new_cluster=None, Z: torch.Tensor = None) -> Dict:
    """PPL-routing quality of the refit CI vs the LM's PPL-best path, SEPARATED BY
    PREFIX LENGTH when records carry one (ratified reporting request).

    Per group (and overall):
      top1_acc  : CI-routed path == PPL-best (argmin NLL) path
      top2_acc  : PPL-best path is in the CI's TOP-2 centroids by similarity. REAL
                  top-2 (Fix 3): computed from new_cluster.similarities(Z), the two
                  highest-similarity clusters for each context. Requires new_cluster
                  and Z; without them top2 is reported as None (not 0).
      regret    : NLL(CI-routed path) − NLL(PPL-best path), in nats. Report
                  mean / p90 / max — the TAIL is what hurts LM training.
      confusion : [K x K] counts, rows = PPL-best path, cols = CI-routed path.
                  Printed (Fix 3), not just stored.
      n         : contexts in the group.

    Prefix length is read from rec['prefix_len'] or rec['route'].get('prefix_len');
    otherwise everything falls into one 'all' group."""
    def prefix_of(rec):
        if "prefix_len" in rec:
            return int(rec["prefix_len"])
        r = rec.get("route", {})
        if isinstance(r, dict) and "prefix_len" in r:
            return int(r["prefix_len"])
        return "all"

    # REAL top-2 (Fix 3): the two highest-similarity clusters per context.
    top2_sets = None
    if new_cluster is not None and Z is not None and K > 1:
        sims = new_cluster.similarities(Z)                    # [N, K]
        order = torch.argsort(sims, dim=1, descending=True)   # [N, K]
        top2_sets = [ {int(order[i, 0]), int(order[i, 1])} for i in range(Z.shape[0]) ]

    groups: Dict = {}
    for i, rec in enumerate(records):
        Ls = _nll_vector(rec, K)
        best = int(min(range(K), key=lambda k: Ls[k]))     # PPL-best path
        routed = int(run_assign[i])                         # CI-routed path (refit)
        regret = float(Ls[routed] - Ls[best])               # >= 0 by construction
        g = prefix_of(rec)
        d = groups.setdefault(g, {"n": 0, "top1": 0, "top2": 0, "top2_n": 0,
                                  "regret": [],
                                  "conf": torch.zeros(K, K, dtype=torch.long)})
        d["n"] += 1
        d["top1"] += int(routed == best)
        if top2_sets is not None:
            d["top2"] += int(best in top2_sets[i])
            d["top2_n"] += 1
        d["regret"].append(regret)
        d["conf"][best, routed] += 1

    out = {}
    print("\n" + "=" * 64)
    print("PPL-ROUTING QUALITY (refit CI vs LM PPL-best), by prefix length")
    print("  regret = NLL(CI-routed) − NLL(PPL-best), nats (0 = oracle routing)")
    print("  top2 = PPL-best in CI's top-2 centroids (real, from similarities)")
    print("=" * 64)
    print(f"{'prefix':>8} | {'n':>6} | {'top1_acc':>8} | {'top2_acc':>8} | "
          f"{'regret_mean':>11} | {'regret_p90':>10} | {'regret_max':>10}")
    for g in sorted(groups, key=lambda x: (x == "all", x)):
        d = groups[g]
        reg = torch.tensor(d["regret"])
        top1 = d["top1"] / max(1, d["n"])
        top2 = (d["top2"] / d["top2_n"]) if d["top2_n"] > 0 else None
        r_mean = float(reg.mean()) if reg.numel() else 0.0
        r_p90 = float(torch.quantile(reg, 0.90)) if reg.numel() else 0.0
        r_max = float(reg.max()) if reg.numel() else 0.0
        out[str(g)] = {"n": d["n"], "top1_acc": top1, "top2_acc": top2,
                       "regret_mean": r_mean, "regret_p90": r_p90,
                       "regret_max": r_max, "confusion": d["conf"].tolist()}
        t2s = f"{top2:>8.3f}" if top2 is not None else f"{'n/a':>8}"
        print(f"{str(g):>8} | {d['n']:>6} | {top1:>8.3f} | {t2s} | "
              f"{r_mean:>11.4f} | {r_p90:>10.4f} | {r_max:>10.4f}")
    print("=" * 64)

    # Fix 3: print the confusion matrix (rows = PPL-best, cols = CI-routed), overall.
    overall = torch.zeros(K, K, dtype=torch.long)
    for d in groups.values():
        overall += d["conf"]
    print("Confusion matrix (rows = PPL-best path, cols = CI-routed path):")
    header = "        " + " ".join(f"r{c:>4}" for c in range(K))
    print(header)
    for r in range(K):
        row = " ".join(f"{int(overall[r, c]):>5}" for c in range(K))
        print(f"  best{r:>2} {row}")
    print("=" * 64)
    out["_confusion_overall"] = overall.tolist()
    return out


def train_round(artifact: str, ppl_path: str, prior_soft_path: Optional[str],
                out_path: Optional[str], round_no: int,
                soft_out: Optional[str], device: Optional[str]) -> Dict:
    print("=" * 64)
    print(f"CONTENT INDEX TRAINER — round {round_no}")
    print("=" * 64)

    prior = ContentIndex.load(artifact, CIC, device)
    prior.assert_frozen()
    K = prior.K
    ppl = load_ppl(ppl_path)
    if int(ppl["K"]) != K:
        raise RuntimeError(f"PPL K={ppl['K']} != artifact K={K}.")
    if ppl["content_index_version"] != prior.version_id:
        raise RuntimeError(
            f"VERSION MISMATCH: PPL was measured against "
            f"{ppl['content_index_version']!r} but the artifact being updated is "
            f"{prior.version_id!r}. Refusing to train — the soft responsibilities "
            f"would describe a DIFFERENT partition than the one being moved "
            f"(design §26). Re-run the harness against this exact artifact, or pass "
            f"the artifact the PPL was measured against.")
    records = ppl["records"]
    prior_soft = load_prior_soft(prior_soft_path)
    prior_age = prior.meta.get("proto_age", {})
    # Ratified D3: calibrate the PPL-margin contest threshold from THIS round's
    # measured top1/top2 margins (adaptive), capped at CONTEST_PPL_MARGIN_CAP.
    ppl_margin_thr = calibrate_ppl_margin_threshold(records, K)
    print(f"[load] K={K} records={len(records)} prior_soft={len(prior_soft)} "
          f"ppl_margin_thr={ppl_margin_thr:.4f} nat "
          f"(q={TC.CONTEST_PPL_MARGIN_QUANTILE}, cap={TC.CONTEST_PPL_MARGIN_CAP})")

    # re-embed each context so z matches the SAME frozen encoder (§26). We embed
    # from the record's decoded text via the artifact's own encoder.
    print("[embed] re-embedding contexts with the frozen encoder ...")
    enc = prior.encoder
    if getattr(enc, "_tok", None) is None:
        enc._ensure()
    # The harness recorded token ids indirectly (sample_id only), so we re-decode is
    # not possible; instead we require the PPL JSON to be produced by our harness,
    # which embeds the SAME text. We reconstruct z from the stored route+nll only if
    # text is present; otherwise we ask the harness to include text. Here we expect
    # 'text' to be present (harness includes it when --emit-text is set); fall back
    # to a stored 'z' vector if present.
    Z = _reconstruct_Z(records, enc)
    print(f"[embed] Z={tuple(Z.shape)}")

    # Step 1 — soft responsibilities
    tau = calibrate_tau(records, K)
    print(f"[soft] tau={tau:.4f} mode=raw_nll (no GENERAL; q_k ∝ exp(-L_k/τ))")
    soft_all: List[Dict] = []
    resp = torch.zeros(len(records), K)
    n_contested = 0
    for i, rec in enumerate(records):
        q_new = soft_responsibilities(rec, K, tau)
        pr = prior_soft.get(rec["sample_id"])
        contested, reason = is_contested(rec, pr, q_new, K, ppl_margin_thr)
        if contested:
            q_used = q_new
            n_contested += 1
        else:
            q_used = pr["soft"]                      # carry prior target (§18)
        soft_all.append({"sample_id": rec["sample_id"], "soft": q_used,
                         "contested": contested, "reason": reason,
                         "route_destination": rec["route"]["destination"]})
        for k in range(K):
            resp[i, k] = q_used.get(f"P{k}", 0.0)
    print(f"[relabel] contested={n_contested}/{len(records)} "
          f"({100*n_contested/len(records):.1f}%) — stable contexts carried (§18)")

    # Step 3 — responsibility-weighted multi-prototype M-step
    print(f"[mstep] {K}×{TC.PROTOTYPES_PER_SPECIALIST} prototypes, "
          f"{TC.MSTEP_ITERS} iters")
    protos, proto_spec, usage_info = mstep_prototypes(Z, resp, prior.cluster, K)
    usage = torch.tensor(usage_info["usage"])

    # Step 5 — dead-prototype detection + PPL-aware rejuvenation (before damping so
    # rejuvenated protos can be exempted from the damp toward prior geometry)
    protos, exempt, rej_info = rejuvenate(
        protos, proto_spec, usage, Z, records, K, prior_age)
    if rej_info["rejuvenated"]:
        print(f"[rejuv] rejuvenated {len(rej_info['rejuvenated'])} dead prototype(s): "
              + ", ".join(f"c{r['proto']}(P{r['k']})" for r in rej_info["rejuvenated"]))
    else:
        print(f"[rejuv] no rejuvenation "
              f"({len(rej_info['dead_candidates'])} dead candidate(s) in grace/kept)")

    # Step 4 — anti-oscillation damping (rejuvenated protos exempt)
    protos = damp_prototypes(protos, proto_spec, prior.cluster, exempt_mask=exempt)

    new_cluster = MultiPrototypeCluster(protos, proto_spec)

    # Step 6 — recalibrate thresholds for the moved geometry.
    # Fix 2 (D1=B consistency): thresholds describe the DEPLOYMENT distribution, so
    # calibrate on EXACT-256 records ONLY (prefix_len == 256), even though the M-step
    # above refit centroids on ALL multi-view records. If the harness didn't tag
    # prefix_len (older schema), fall back to all records with a warning.
    def _is_exact256(rec):
        if "prefix_len" in rec:
            return int(rec["prefix_len"]) == CIC.ROUTE_PREFIX_TOKENS
        r = rec.get("route", {})
        if isinstance(r, dict) and "prefix_len" in r:
            return int(r["prefix_len"]) == CIC.ROUTE_PREFIX_TOKENS
        return None
    flags = [_is_exact256(r) for r in records]
    if any(f is None for f in flags):
        print("[calib] WARNING: records lack prefix_len; recalibrating on ALL "
              "records (harness is not prefix-aware — thresholds may reflect the "
              "augmented multi-view distribution rather than deployment 256).")
        Z_thr = Z
    else:
        mask = torch.tensor(flags, dtype=torch.bool)
        if int(mask.sum()) == 0:
            print("[calib] WARNING: no exact-256 records found; recalibrating on ALL.")
            Z_thr = Z
        else:
            Z_thr = Z[mask]
            print(f"[calib] recalibrating thresholds on {int(mask.sum())} exact-256 "
                  f"records (of {len(records)} total multi-view; M-step used all).")
    prior_thr = {"T_MARGIN_FALLBACK": prior.t_margin,
                 "T_SIM_FALLBACK": prior.t_sim, "T_SWITCH": prior.t_switch}
    thr = recalibrate_thresholds(new_cluster, Z_thr, prior_thr)
    print(f"[calib] T_MARGIN={thr['T_MARGIN_FALLBACK']:.4f} "
          f"T_SIM={thr['T_SIM_FALLBACK']:.4f} T_SWITCH={thr['T_SWITCH']:.4f}")

    # diagnostics on the new partition (§27)
    run_assign = new_cluster.runtime_assign(Z)
    counts = torch.bincount(run_assign, minlength=K).float()
    shares = (counts / counts.sum().clamp_min(1)).tolist()
    print("[part] new specialist shares: "
          + " ".join(f"P{i}={s:.3f}" for i, s in enumerate(shares)))
    proto_usage = _prototype_usage_report(usage, K, TC.PROTOTYPES_PER_SPECIALIST)
    for k in range(K):
        print(f"       P{k} prototypes: "
              + " ".join(f"{u:.2f}" for u in proto_usage[k]))

    # PPL-routing accuracy / regret of the refit CI vs the LM PPL-best path,
    # separated by prefix length (ratified reporting request).
    ppl_report = ppl_routing_report(records, run_assign, K,
                                    new_cluster=new_cluster, Z=Z)

    # version + freeze + save
    base = prior.version_id.split("-r")[0]
    suffix = TC.NEXT_VERSION_SUFFIX or f"-r{round_no}"
    new_version = base + suffix
    new_meta = dict(prior.meta)
    new_meta.update({
        "round": round_no, "tau": tau,
        "responsibility_mode": "raw_nll_no_general",
        "contested_fraction": n_contested / len(records),
        "prototypes_per_specialist": TC.PROTOTYPES_PER_SPECIALIST,
        "prototype_usage": usage_info["usage"],
        "rejuvenation": rej_info,
        "proto_age": rej_info["proto_age"],
        "damp": TC.CENTROID_DAMP,
        "trained_from": {"artifact": artifact, "ppl": ppl_path,
                         "prior_version": prior.version_id},
        "new_shares": shares,
        "ppl_margin_threshold": ppl_margin_thr,
        "ppl_routing_report": ppl_report,
    })
    ci = ContentIndex(enc, new_cluster, thr, new_version, meta=new_meta)
    ci.assert_frozen()
    print("[freeze] assert_frozen() OK.")

    out = out_path or _sibling(artifact, f"content_index.r{round_no}.pt")
    ci.save(out)
    print(f"[save] next artifact -> {out}   version_id={new_version}")

    soft_path = soft_out or _sibling(artifact, f"soft_targets.r{round_no}.json")
    with open(soft_path, "w") as f:
        json.dump({"schema": "projectB.soft_targets.v1", "round": round_no,
                   "content_index_version": new_version, "K": K, "tau": tau,
                   "records": soft_all}, f, indent=2)
    print(f"[save] soft targets -> {soft_path}  (feeds round {round_no+1} §18 carry)")
    print("=" * 64)
    return {"artifact": out, "soft": soft_path, "version": new_version,
            "contested_fraction": n_contested / len(records),
            "rejuvenated": len(rej_info["rejuvenated"])}


# ---------- helpers ----------
def _reconstruct_Z(records: List[Dict], enc: ContentEncoder) -> torch.Tensor:
    """Get content vectors for the round. Prefer a stored 'text' per record
    (embed with the frozen encoder — §26 identical geometry); else a stored 'z'."""
    if all("text" in r for r in records):
        return torch.stack([enc.embed_text(r["text"]) for r in records], dim=0)
    if all("z" in r for r in records):
        return F.normalize(torch.tensor([r["z"] for r in records]), dim=-1)
    raise RuntimeError(
        "PPL records carry neither 'text' nor 'z'. Re-run the harness with text "
        "emission so the trainer can embed contexts with the frozen encoder "
        "(design §26). See ppl_harness emit_text option.")


def _prototype_usage_report(usage: torch.Tensor, K: int, M: int) -> List[List[float]]:
    out = []
    for k in range(K):
        block = usage[k * M:(k + 1) * M]
        s = max(float(block.sum()), 1e-9)
        out.append([float(x) / s for x in block])
    return out


def _sibling(path: str, name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="prior frozen artifact (.pt)")
    ap.add_argument("--ppl", required=True, help="ppl_round*.json from the harness")
    ap.add_argument("--prior-soft", default=None,
                    help="soft_targets.r{N-1}.json for contested-only carry (§18)")
    ap.add_argument("--out", default=None, help="output artifact path")
    ap.add_argument("--soft-out", default=None, help="output soft targets path")
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train_round(args.artifact, args.ppl, args.prior_soft, args.out,
                args.round, args.soft_out, args.device)


if __name__ == "__main__":
    main()
