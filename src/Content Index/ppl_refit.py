"""
ppl_refit.py — PPL-feedback Content Index refit (permanent, generic mode).

This is NOT round-specific. It refits the Content Index against ANY frozen LM
checkpoint at ANY training stage. The LM is read-only (six specialist paths are run
to score inputs; no LM weight is changed). Same nearest-centroid architecture as the
build — only the fit TARGET changes: content geometry (build) -> LM-PPL-best path.

Pipeline (all inside CI training; the LM is only read):
  1. COLLECT feedback inputs on the runtime routing view (256 tokens), reusing the
     build's collector so the population matches the build distribution.
  2. SCORE all six paths per input with the frozen LM -> per-input NLL[K].
  3. RAW PPL TARGETS = argmin path; report shares / winner margins / ΔNLL diagnostics.
  4. QUOTA (optional): lift any path below MIN_PATH_TARGET_SHARE using smallest-ΔNLL
     moves, never pushing a donor below the floor, optional ΔNLL cap. Report the fill
     cost. Off by default; enabled with --quota.
  5. REFIT centroids = L2-normalized mean of each target label's embeddings
     (label-mean; same BalancedCluster mechanism, no classifier head).
  6. COMPARE the current (parent) frozen CI vs the refit: current->raw-PPL and
     current->final-target transition matrices, and per-path share changes.
  7. SAVE a new artifact ONLY when --save is given, with a bumped version id and full
     provenance metadata: parent CI, teacher checkpoint/step, refit generation, quota
     settings + costs, fit method.

Reserved (NOT this mode): MAX_PATH_SHARE_DELTA drift limiting. When added later it
must be a PER-PATH SHARE-CHANGE CAP relative to the previous frozen CI on the SAME
feedback dataset — never a cap on how many individual samples may change path. This
file leaves a clearly marked hook for it and does not enforce it.

Run from the projectB folder:
    python build_content_index.py --ppl-refit --checkpoint final_weights.pt
    python build_content_index.py --ppl-refit --checkpoint ckpt.pt --quota --save
    (or directly:)  python ppl_refit.py --checkpoint final_weights.pt --quota
"""
import argparse
import time
import torch
import torch.nn.functional as F

import content_index_config as CIC
from content_index import ContentEncoder, BalancedCluster, ContentIndex


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _bar(frac, width=22):
    fill = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * fill + "." * (width - fill)


def _argmin_shares(nll):
    K = nll.size(1)
    best = nll.argmin(dim=1)
    counts = torch.bincount(best, minlength=K).float()
    return best, counts, counts / counts.sum().clamp_min(1)


def _winner_margins(nll):
    srt = nll.sort(dim=1).values
    return srt[:, 1] - srt[:, 0]


def _fill_cost(nll, best, shares, floor, path, delta_cap=None):
    """Per-path ΔNLL fill-cost diagnostic (agreed): for a below-floor `path`, sort
    the movable inputs by ΔNLL(i,path)=NLL(i,path)-min_q NLL(i,q) ascending and print
    the cumulative cost to lift `path` to the floor, so the cost knee is visible."""
    N, K = nll.shape
    cur = int((best == path).sum())
    target = int(torch.ceil(torch.tensor(floor * N)))
    need = max(0, target - cur)
    print(f"  --- fill curve P{path}: current {cur}/{N}={cur/N*100:.1f}%, "
          f"need +{need} to reach {floor*100:.0f}% ---")
    if need == 0:
        print("      already at/above floor.")
        return
    dnll = nll[:, path] - nll.min(dim=1).values
    idx = torch.nonzero(best != path, as_tuple=True)[0]
    order = idx[torch.argsort(dnll[idx])]
    if delta_cap is not None:
        order = order[dnll[order] <= delta_cap]
    take = order[:need]
    if take.numel() < need:
        print(f"      [!] only {take.numel()}/{need} movable under "
              f"{'cap '+str(delta_cap) if delta_cap is not None else 'available'} "
              f"-> PARTIAL fill.")
    if take.numel() == 0:
        return
    costs = dnll[take]; cum = torch.cumsum(costs, 0)
    med = float(costs.median())
    marks = [max(1, int(take.numel()*q))-1 for q in (0.25, 0.5, 0.75, 1.0)]
    print(f"      ΔNLL min={float(costs.min()):.4f} median={med:.4f} "
          f"max={float(costs.max()):.4f}  cumulative={float(cum[-1]):.3f}")
    print(f"      curve(moves->cumΔNLL): "
          + "  ".join(f"{m+1}:{float(cum[m]):.2f}" for m in marks))
    band = ("VERY CHEAP" if med < 0.02 else "CHEAP" if med < 0.10
            else "MODERATE" if med < 0.5 else "EXPENSIVE — likely noncompetitive")
    print(f"      read: median move ΔNLL={med:.4f} -> {band}")


def _transition_matrix(old_assign, new_assign, K):
    """M[a,b] = # inputs the parent CI put in path a that the new labels put in b."""
    M = torch.zeros(K, K, dtype=torch.long)
    for a, b in zip(old_assign.tolist(), new_assign.tolist()):
        M[int(a), int(b)] += 1
    return M


def _print_matrix(title, M):
    K = M.size(0)
    print(f"\n{title}")
    print("        " + "".join(f"  ->P{b} " for b in range(K)) + "   row=parent")
    for a in range(K):
        row = M[a]
        tot = int(row.sum())
        cells = "".join(f" {int(v):5d}" for v in row)
        print(f"  P{a} |{cells}   ({tot})")
    # diagonal retention
    diag = int(M.diag().sum()); total = int(M.sum())
    print(f"  retained on diagonal: {diag}/{total} = {100*diag/max(total,1):.1f}%")


# ----------------------------------------------------------------------
# LM scoring (read-only) — the six-path NLL that defines the PPL target
# ----------------------------------------------------------------------
@torch.no_grad()
def _six_path_nll(model, ids, score_from, C):
    """Six-path per-sequence NLL at a causal boundary. Read-only: uses public
    run_trunk / run_specialist / _seq_nll; never routes, never trains, never touches
    the CI. Scored under the training autocast/dtype (bf16 by default) so the sweep's
    VRAM matches training, not an FP32 blow-up. Returns [B, K] (float32 on CPU)."""
    import contextlib
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32}.get(getattr(C, "DTYPE", "bf16"), torch.bfloat16)
    dev = ids.device.type
    autocast = (torch.autocast(device_type="cuda", dtype=dt)
                if dev == "cuda" and dt in (torch.bfloat16, torch.float16)
                else contextlib.nullcontext())
    was_training = model.training
    model.eval()
    try:
        with model.count_mode("sweep"), autocast:
            h_trunk, _, _ = model.run_trunk(ids, None)
            h_trunk = h_trunk.detach()
            nlls = [model._seq_nll(model.run_specialist(h_trunk, p, None)[0],
                                   ids, None, score_from=score_from)
                    for p in range(C.N_SPECIALISTS)]
            spec_nll = torch.stack(nlls, dim=1)
    finally:
        if was_training:
            model.train()
    return spec_nll.float().detach().cpu()


@torch.no_grad()
def _six_path_nll_multiview(model, ids, score_froms, C):
    """Run trunk/specialists once, score the same sequences at multiple R values."""
    import contextlib

    score_froms = tuple(int(r) for r in score_froms)

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32}.get(
              getattr(C, "DTYPE", "bf16"), torch.bfloat16)

    dev = ids.device.type
    autocast = (
        torch.autocast(device_type="cuda", dtype=dt)
        if dev == "cuda" and dt in (torch.bfloat16, torch.float16)
        else contextlib.nullcontext()
    )

    out = {r: [] for r in score_froms}

    was_training = model.training
    model.eval()

    try:
        with model.count_mode("sweep"), autocast:
            h_trunk, _, _ = model.run_trunk(ids, None)
            h_trunk = h_trunk.detach()

            for p in range(C.N_SPECIALISTS):
                h_spec, _, _ = model.run_specialist(
                    h_trunk, p, None)

                for r in score_froms:
                    out[r].append(
                        model._seq_nll(
                            h_spec, ids, None, score_from=r)
                    )
    finally:
        if was_training:
            model.train()

    return {
        r: torch.stack(vals, dim=1).float().detach().cpu()
        for r, vals in out.items()
    }


def _collect_feedback(encoder, model, n, boundary, C, D, skip=0, batch_rows=4):
    """Stream n inputs via the SAME data path the build uses, beginning after `skip`
    sequences (so FEEDBACK / CALIB / GATE splits are disjoint by offset). Returns
    (Z[N,dim], nll[N,K], sources[N], prefix_texts[N], full_texts[N]) — prefix_texts is
    the 256-token routing view (for gate runtime trials), full_texts is the whole
    segment (the gate's short/long trials need it as their 'long' reference)."""
    tokenizer = D.get_tokenizer()
    device = next(model.parameters()).device

    def _decode(ids_row):
        return tokenizer.decode(ids_row.tolist(), skip_special_tokens=True)

    Z, nll_parts, sources, prefix_texts, full_texts = [], [], [], [], []
    buf_ids, buf_src, buf_ptxt, buf_ftxt = [], [], [], []
    seen = 0

    def _flush():
        if not buf_ids:
            return
        ids = torch.stack(buf_ids, dim=0).to(device)
        nll_parts.append(_six_path_nll(model, ids, boundary, C))
        for t in buf_ptxt:
            Z.append(encoder.embed_text(t))
        sources.extend(buf_src); prefix_texts.extend(buf_ptxt); full_texts.extend(buf_ftxt)
        buf_ids.clear(); buf_src.clear(); buf_ptxt.clear(); buf_ftxt.clear()

    it = D.batch_iterator(tokenizer, skip_sequences=skip)
    for input_ids, srcs, _c, _s in it:
        for b in range(input_ids.shape[0]):
            ptxt = _decode(input_ids[b, :CIC.ROUTE_PREFIX_TOKENS])
            if not ptxt.strip():
                continue
            buf_ids.append(input_ids[b]); buf_src.append(str(srcs[b]))
            buf_ptxt.append(ptxt); buf_ftxt.append(_decode(input_ids[b]))
            seen += 1
            if len(buf_ids) >= batch_rows:
                _flush()
            if seen >= n:
                break
        if seen >= n:
            break
    _flush()
    if not Z:
        raise RuntimeError("collected 0 inputs — check DATA_SOURCE.")
    return (torch.stack(Z, 0), torch.cat(nll_parts, 0), sources,
            prefix_texts, full_texts)


def _collect_feedback_multiview(
        encoder, model, n, prefixes, C, D,
        skip=0, batch_rows=4):
    """Collect embeddings and NLLs for multiple prefix lengths for each sequence."""
    tokenizer = D.get_tokenizer()
    device = next(model.parameters()).device
    prefixes = tuple(int(r) for r in prefixes)

    Z_parts = {r: [] for r in prefixes}
    nll_parts = {r: [] for r in prefixes}

    sources = []
    prefix_texts = []
    full_texts = []

    buf_ids = []
    buf_src = []
    seen = 0

    def _decode(row):
        return tokenizer.decode(row.tolist(),
                                skip_special_tokens=True)

    def _flush():
        if not buf_ids:
            return

        ids = torch.stack(buf_ids, dim=0).to(device)

        batch_nll = _six_path_nll_multiview(
            model, ids, prefixes, C)

        for r in prefixes:
            nll_parts[r].append(batch_nll[r])

        ids_cpu = ids.detach().cpu()

        for row in ids_cpu:
            view_texts = {
                r: _decode(row[:r])
                for r in prefixes
            }

            for r in prefixes:
                Z_parts[r].append(
                    encoder.embed_text(view_texts[r])
                )

            prefix_texts.append(
                view_texts[max(prefixes)])
            full_texts.append(_decode(row))

        sources.extend(buf_src)
        buf_ids.clear()
        buf_src.clear()

    it = D.batch_iterator(tokenizer, skip_sequences=skip)

    for input_ids, srcs, _c, _s in it:
        for b in range(input_ids.shape[0]):
            buf_ids.append(input_ids[b])
            buf_src.append(str(srcs[b]))
            seen += 1

            if len(buf_ids) >= batch_rows:
                _flush()

            if seen >= n:
                break

        if seen >= n:
            break

    _flush()

    if seen == 0:
        raise RuntimeError("collected 0 inputs — check DATA_SOURCE.")

    Z_views = {
        r: torch.stack(Z_parts[r], 0)
        for r in prefixes
    }

    nll_views = {
        r: torch.cat(nll_parts[r], 0)
        for r in prefixes
    }

    return (
        Z_views, nll_views,
        sources, prefix_texts, full_texts
    )


def _bucket_nll(model, D, C, n, bucket_R, batch_rows=4):
    """#1 diagnostic: six-path NLL at a NON-256 boundary bucket_R, over a fresh
    stream sample. Returns nll[n', K] (n' <= n). Assignment stays at 256; this only
    shows whether the argmin distribution shifts with context length."""
    tokenizer = D.get_tokenizer()
    device = next(model.parameters()).device
    parts, buf, seen = [], [], 0
    it = D.batch_iterator(tokenizer, skip_sequences=0)
    for input_ids, _srcs, _c, _s in it:
        for b in range(input_ids.shape[0]):
            buf.append(input_ids[b]); seen += 1
            if len(buf) >= batch_rows:
                parts.append(_six_path_nll(model, torch.stack(buf).to(device), bucket_R, C))
                buf = []
            if seen >= n:
                break
        if seen >= n:
            break
    if buf:
        parts.append(_six_path_nll(model, torch.stack(buf).to(device), bucket_R, C))
    return torch.cat(parts, 0) if parts else torch.empty(0, C.N_SPECIALISTS)


def _routed_ppl(nll, assign):
    """Mean NLL / PPL of the path each input is ROUTED to, under the frozen LM.
    nll[N,K] are the six-path scores; assign[N] is a CI's runtime routing. This is
    the real 'what does this CI's routing cost in LM PPL' number — same frozen LM,
    only the routing differs."""
    import math
    assign = assign.to(nll.device)
    routed = nll.gather(1, assign.long().unsqueeze(1)).squeeze(1)   # [N]
    mean_nll = float(routed.mean())
    return mean_nll, float(math.exp(min(mean_nll, 20.0)))


PPL_CACHE_FORMAT = 2  # Phase 4: multi-view fit

# Phase 3: alpha sweep values
ALPHA_SWEEP = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)

# Phase 4: PPL-feedback FIT views.
# CALIB/GATE remain exact R=256.
PPL_FIT_PREFIXES = (32, 64, 128, 256)


def _save_ppl_cache(path, payload):
    """Save raw expensive PPL-sweep results for reuse."""
    import os

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    torch.save(payload, path)
    print(f"[cache] saved raw PPL sweep -> {path}")


def _load_ppl_cache(path):
    """Load a previously saved raw PPL-sweep cache."""
    blob = torch.load(path, map_location="cpu", weights_only=False)

    fmt = int(blob.get("cache_format", -1))
    if fmt != PPL_CACHE_FORMAT:
        raise ValueError(
            f"unsupported PPL cache format {fmt}; "
            f"expected {PPL_CACHE_FORMAT}"
        )

    print(f"[cache] loaded raw PPL sweep <- {path}")
    return blob

# ----------------------------------------------------------------------
# quota: raw PPL-best -> final targets with a minimum per-path share
# ----------------------------------------------------------------------
def apply_min_share_quota(nll, raw_best, floor, delta_cap=None):
    """Lift every path below `floor` to the floor using the cheapest ΔNLL moves,
    without pushing any donor path below the floor. Returns (final_labels, report).

    ΔNLL(i,p) = NLL(i,p) - min_q NLL(i,q). Candidates for path p are inputs not
    currently labelled p, sorted ascending by ΔNLL(i,p); we take the cheapest until p
    reaches the floor or we run out of allowable (donor-safe, under-cap) moves. This
    is best-effort: if the floor cannot be reached without violating the donor floor
    or the ΔNLL cap, p is left partially filled (never forced)."""
    N, K = nll.shape
    labels = raw_best.clone()
    best_nll = nll.min(dim=1).values
    target = int(torch.ceil(torch.tensor(floor * N)))
    counts = torch.bincount(labels, minlength=K).tolist()
    moves = []           # (i, from, to, dnll)

    # process starved paths from most-starved first
    order = sorted(range(K), key=lambda k: counts[k])
    for p in order:
        need = target - counts[p]
        if need <= 0:
            continue
        dnll_p = nll[:, p] - best_nll                      # [N] >= 0
        cand = torch.nonzero(labels != p, as_tuple=True)[0]
        cand = cand[torch.argsort(dnll_p[cand])]           # cheapest first
        for i in cand:
            if need <= 0:
                break
            i = int(i)
            donor = int(labels[i])
            if counts[donor] - 1 < target:                 # donor floor protection
                continue
            d = float(dnll_p[i])
            if delta_cap is not None and d > delta_cap:
                break                                      # rest are only costlier
            labels[i] = p
            counts[p] += 1; counts[donor] -= 1; need -= 1
            moves.append((i, donor, p, d))

    final_counts = torch.bincount(labels, minlength=K).float()
    report = {
        "n_moves": len(moves),
        "total_dnll": float(sum(m[3] for m in moves)),
        "mean_dnll": float(sum(m[3] for m in moves) / max(len(moves), 1)),
        "final_shares": (final_counts / N).tolist(),
        "still_below_floor": [k for k in range(K) if final_counts[k] / N < floor],
        "delta_cap": delta_cap,
        "floor": floor,
    }
    return labels, report


# ----------------------------------------------------------------------
# label-mean centroid refit (same BalancedCluster architecture, no head)
# ----------------------------------------------------------------------
def refit_label_mean(Z, labels, K, allow_empty=False):
    """Centroid_k = L2-normalized mean of embeddings whose target label is k.

    An EMPTY label (no inputs assigned to path k) is a real problem — a path with no
    seed content. By default this RAISES so the empty path is never silently papered
    over with a global-mean centroid (which would collapse that path onto the data
    centroid). Pass allow_empty=True to fall back to the global mean for empty labels
    (explicit, logged by the caller). Returns (BalancedCluster, empty_paths)."""
    Zc = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)
    cents, empty = [], []
    for k in range(K):
        members = Zc[labels == k]
        if members.numel() == 0:
            empty.append(k)
            if not allow_empty:
                continue
            cents.append(Zc.mean(dim=0))
        else:
            cents.append(members.mean(dim=0))
    if empty and not allow_empty:
        raise ValueError(
            f"empty target label(s) for path(s) {empty} — cannot form a centroid "
            f"from zero inputs. Enable the min-share quota (--quota) to seed them, "
            f"or pass --allow-empty to fall back to the global-mean centroid "
            f"(explicit, not silent).")
    Cn = F.normalize(torch.stack(cents, 0), p=2, dim=-1, eps=1e-8)
    return BalancedCluster(Cn), empty

def blend_cluster(parent_cluster, ppl_cluster, alpha):
    """Move each parent centroid part-way toward its PPL-target centroid.

    alpha=0.0 -> unchanged parent CI
    alpha=1.0 -> full PPL label-mean refit
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    old = parent_cluster.centroids.float()
    target = ppl_cluster.centroids.float().to(old.device)

    cents = F.normalize(
        (1.0 - alpha) * old + alpha * target,
        p=2, dim=-1, eps=1e-8
    )
    return BalancedCluster(cents)


def _evaluate_alpha_sweep(
        alpha,
        parent,
        encoder,
        ppl_cluster,
        Z,
        nll,
        Z_cal,
        Z_gate,
        nll_gate,
        pref_gate,
        full_gate,
        B):
    """Cheap evaluation of one alpha using already collected/cached evidence."""

    cluster = blend_cluster(parent.cluster, ppl_cluster, alpha)

    # FEEDBACK routing PPL
    fit_assign = cluster.runtime_assign(Z)
    _, fit_ppl = _routed_ppl(nll, fit_assign)

    # Each alpha changes centroid geometry, so recalibrate separately.
    fb = B.calibrate_fallback(cluster, Z_cal)

    thresholds = {
        "T_MARGIN_FALLBACK": fb["T_MARGIN_FALLBACK"],
        "T_SIM_FALLBACK": fb["T_SIM_FALLBACK"],
        "T_SWITCH": parent.t_switch,
    }

    cand_ci = ContentIndex(
        encoder,
        cluster,
        thresholds,
        parent.version_id + f"-alpha{alpha:.3f}-candidate",
        dict(parent.meta),
    )

    passed, gate_report = B.go_no_go(
        cand_ci,
        cluster,
        encoder,
        Z_gate,
        full_gate,
        prefix_texts_gate=pref_gate,
    )

    # Held-out PPL
    gate_assign = cluster.runtime_assign(Z_gate)
    _, gate_ppl = _routed_ppl(nll_gate, gate_assign)

    gate_counts = torch.bincount(
        gate_assign, minlength=parent.K
    ).float()

    gate_shares = gate_counts / gate_counts.sum().clamp_min(1)

    return {
        "alpha": float(alpha),
        "fit_ppl": float(fit_ppl),
        "gate_ppl": float(gate_ppl),
        "gate_min_share": float(gate_shares.min()),
        "gate_max_share": float(gate_shares.max()),
        "passed": bool(passed),
        "gate_report": gate_report,
        "thresholds": thresholds,
    }


# ----------------------------------------------------------------------
# main entry (callable from build_content_index.py dispatch or directly)
# ----------------------------------------------------------------------
def run(argv=None):
    ap = argparse.ArgumentParser(
        description="PPL-feedback Content Index refit (generic; any LM checkpoint).")
    ap.add_argument("--checkpoint", default=None,
                    help="frozen LM weights (.pt) to score paths with (teacher). "
                         "Default: latest.pt pointer in the LM CKPT_DIR.")
    ap.add_argument("--parent", default=None,
                    help="parent CI artifact to refit from / compare against. "
                         "Default: content_index_config.ARTIFACT.")
    ap.add_argument("--n", type=int, default=3000,
                    help="feedback inputs to score (default 3000; up to dataset size).")
    ap.add_argument("--batch-rows", type=int, default=4,
                    help="LM scoring micro-batch (default 4; lower if VRAM-bound).")
    ap.add_argument("--quota", action="store_true",
                    help="apply MIN_PATH_TARGET_SHARE to lift below-floor paths.")
    ap.add_argument("--floor", type=float, default=0.08,
                    help="MIN_PATH_TARGET_SHARE (default 0.08); only used with --quota.")
    ap.add_argument("--delta-cap", type=float, default=0.05,
                help="maximum ΔNLL allowed for quota reassignment "
                     "(default 0.05).")
    ap.add_argument("--alpha", type=float, default=None,
                    help="centroid move toward PPL target: "
                         "0=parent CI, 1=full PPL label-mean refit. "
                         "Saving requires an explicit value.")

    ap.add_argument("--sweep", action="store_true",
                    help="evaluate alpha=0,.10,.25,.50,.75,1.0 on the same "
                         "evidence. Report-only; never saves.")

    ap.add_argument("--allow-empty", action="store_true",
                    help="allow global-mean fallback for empty target labels "
                         "(explicit; default is to FAIL on an empty label).")
    ap.add_argument("--buckets", action="store_true",
                    help="also report raw argmin shares at non-256 prefix buckets "
                         "(diagnostic; assignment stays at 256).")
    ap.add_argument("--calib-n", type=int, default=2000,
                    help="disjoint CALIB inputs to RECALIBRATE thresholds on the new "
                         "centroids (default 2000).")
    ap.add_argument("--gate-n", type=int, default=2000,
                    help="disjoint held-out GATE inputs for GO/NO-GO (default 2000).")
    ap.add_argument("--cache-save", default=None,
                    help="save raw FEEDBACK/CALIB/GATE embeddings and six-path "
                         "NLL results after the expensive LM sweep.")
    ap.add_argument("--cache-load", default=None,
                    help="load a previously saved raw PPL cache and skip the "
                         "expensive HF-stream LM sweep.")
    ap.add_argument("--save", action="store_true",
                    help="save a NEW artifact (bumped version). Off = dry report only.")
    ap.add_argument("--out", default=None,
                    help="output artifact path when --save. Default: parent path with "
                         "the new version id.")
    ap.add_argument("--version-suffix", default=None,
                    help="explicit new version id. Default: parent id + '-pplN'.")
    ap.add_argument("--device", default=None)

    ap.add_argument(
        "--force-save",
        action="store_true",
        help="experimental: save refit even if held-out CI gate/PPL does not pass"
    )

    args = ap.parse_args(argv)

    if args.alpha is not None and not 0.0 <= args.alpha <= 1.0:
        ap.error("--alpha must be between 0.0 and 1.0")

    if args.sweep and args.alpha is not None:
        ap.error("use either --sweep or --alpha X, not both")

    if args.sweep and args.save:
        ap.error("--sweep is report-only; re-run with --alpha X --save")

    if args.save and args.alpha is None:
        ap.error("--save requires explicit --alpha X")

    if args.cache_save and args.cache_load:
        ap.error("use either --cache-save or --cache-load, not both")

    if args.cache_load and args.buckets:
        ap.error(
            "--cache-load cannot be combined with --buckets in Phase 2; "
            "bucket NLL sweeps are not stored in this cache yet"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # LM side (read-only). Imported here so the mode also works standalone.
    import config as C
    import data as D
    import checkpoint as CK
    from model import GeneralistSpecialistModel, build_gpt2_config
    import build_content_index as B   # for calibrate_fallback / go_no_go

    boundary = int(C.SCORE_FROM)                 # assignment boundary = CI deploy view

    print("=" * 70)
    print("PPL-FEEDBACK CONTENT INDEX REFIT (generic mode; LM read-only)")
    print("=" * 70)

    # ---- parent CI (for compare + provenance) ----
    parent_path = args.parent or CIC.ARTIFACT
    parent = ContentIndex.load(parent_path, device=device)
    encoder = parent.encoder                      # reuse the SAME frozen encoder
    K = parent.K
    print(f"[parent] {parent.version_id}  K={K}  from {parent_path}")

    if args.sweep:
        print("[alpha] sweep = " +
              ", ".join(f"{a:g}" for a in ALPHA_SWEEP) +
              "  (report-only)")
    else:
        print(f"[alpha] "
              f"{'1.0 (legacy full-refit dry default)' if args.alpha is None else args.alpha}")

    # PPL feedback must come from data AFTER what this LM checkpoint trained on.
    gap = getattr(CIC, "BUILD_SPLIT_GAP", 10000)

    model = None
    cache = None
    teacher_checkpoint_used = None

    # These are populated now from cache, or later from the live LM sweep.
    Z_cal = None
    Z_gate = None
    nll_gate = None
    pref_gate = None
    full_gate = None
    off_cal = None
    off_gate = None

    if args.cache_load:
        # --------------------------------------------------------------
        # CACHE PATH: no HF-stream LM scoring
        # --------------------------------------------------------------
        cache = _load_ppl_cache(args.cache_load)

        if cache["parent_version_id"] != parent.version_id:
            raise ValueError(
                f"cache parent version {cache['parent_version_id']!r} != "
                f"current parent {parent.version_id!r}"
            )

        if int(cache["K"]) != int(K):
            raise ValueError(
                f"cache K={cache['K']} != parent K={K}"
            )

        if int(cache["boundary"]) != int(boundary):
            raise ValueError(
                f"cache boundary={cache['boundary']} != current boundary={boundary}"
            )

        cached_parent_c = cache.get("parent_centroids")
        if cached_parent_c is not None:
            current_parent_c = parent.cluster.centroids.detach().cpu()
            if not torch.allclose(
                    cached_parent_c.float(),
                    current_parent_c.float(),
                    atol=1e-6,
                    rtol=1e-5):
                raise ValueError(
                    "cache parent centroids do not match the current parent CI"
                )

        if (args.checkpoint is not None
                and cache.get("teacher_checkpoint") is not None
                and str(args.checkpoint) != str(cache["teacher_checkpoint"])):
            raise ValueError(
                "explicit --checkpoint does not match the teacher checkpoint "
                "recorded in --cache-load"
            )

        step = int(cache["teacher_step"])
        consumed = int(cache["consumed_sequences"])
        teacher_checkpoint_used = cache.get(
            "teacher_checkpoint", "unknown"
        )
        feedback_skip = int(cache["feedback_skip"])
        off_cal = int(cache["calib_skip"])
        off_gate = int(cache["gate_skip"])

        # Phase 4: load multi-view data
        cached_prefixes = tuple(
            int(r) for r in cache["fit_prefixes"]
        )

        if cached_prefixes != PPL_FIT_PREFIXES:
            raise ValueError(
                f"cache fit_prefixes={cached_prefixes} != "
                f"current {PPL_FIT_PREFIXES}"
            )

        Z_views = {
            int(r): z for r, z in cache["Z_views"].items()
        }
        nll_views = {
            int(r): x for r, x in cache["nll_views"].items()
        }

        feedback_seq_n = int(cache["feedback_sequence_n"])

        Z = torch.cat(
            [Z_views[r] for r in PPL_FIT_PREFIXES], dim=0
        )
        nll = torch.cat(
            [nll_views[r] for r in PPL_FIT_PREFIXES], dim=0
        )

        sources = cache.get("sources", [])
        pref_txt = cache.get("pref_txt", [])
        full_txt = cache.get("full_txt", [])

        Z_cal = cache["Z_cal"]

        Z_gate = cache["Z_gate"]
        nll_gate = cache["nll_gate"]
        pref_gate = cache["pref_gate"]
        full_gate = cache["full_gate"]

        print(
            f"[teacher] using cached frozen-LM sweep from step={step} "
            f"({teacher_checkpoint_used})"
        )
        print(
            f"[cache] FEEDBACK sequences={feedback_seq_n} "
            f"views/seq={len(PPL_FIT_PREFIXES)} total FIT rows={Z.size(0)}  "
            f"CALIB={Z_cal.size(0)}  GATE={Z_gate.size(0)}"
        )

    else:
        # --------------------------------------------------------------
        # LIVE PATH: run frozen LM and optionally save cache later
        # --------------------------------------------------------------
        model = GeneralistSpecialistModel(build_gpt2_config()).to(device)

        step, consumed = CK.load_checkpoint(
            model,
            path=args.checkpoint,
            map_location="cpu"
        )

        teacher_checkpoint_used = args.checkpoint or "latest.pt"

        print(
            f"[teacher] LM weights step={step} "
            f"from {args.checkpoint or 'latest.pt'} "
            f"(read-only; no CI attached, six paths scored directly)"
        )

        feedback_skip = int(consumed) + gap

        print(
            f"[collect] scoring {args.n} feedback sequences at multiple R={PPL_FIT_PREFIXES} "
            f"(full-vocab, frozen, autocast, batch_rows={args.batch_rows}, "
            f"skip={feedback_skip})"
        )

        Z_views, nll_views, sources, pref_txt, full_txt = \
            _collect_feedback_multiview(
                encoder,
                model,
                args.n,
                PPL_FIT_PREFIXES,
                C,
                D,
                skip=feedback_skip,
                batch_rows=args.batch_rows
            )

        feedback_seq_n = next(
            iter(Z_views.values())
        ).size(0)

        Z = torch.cat(
            [Z_views[r] for r in PPL_FIT_PREFIXES], dim=0
        )

        nll = torch.cat(
            [nll_views[r] for r in PPL_FIT_PREFIXES], dim=0
        )

    N = Z.size(0)

    raw_best, raw_counts, raw_shares = _argmin_shares(nll)
    margins = _winner_margins(nll)

    # ---- per-prefix diagnostics ----
    print("\n[prefix targets] raw PPL-best shares by FIT view:")

    for r in PPL_FIT_PREFIXES:
        _, _, s = _argmin_shares(nll_views[r])
        print(
            f"  R={r:3d}: " +
            "  ".join(
                f"P{k}:{float(s[k])*100:5.1f}%"
                for k in range(K)
            )
        )

    print(
        f"[fit] sequences={feedback_seq_n}  "
        f"views/sequence={len(PPL_FIT_PREFIXES)}  "
        f"total FIT rows={N}"
    )

    # ---- raw PPL target diagnostics ----
    print("\n" + "-" * 70)
    print(
        f"RAW PPL TARGETS  "
        f"(argmin over six paths, multi-view R={PPL_FIT_PREFIXES}, N={N})"
    )
    print("-" * 70)
    for k in range(K):
        below = "  << below floor" if float(raw_shares[k]) < args.floor else ""
        print(f"  P{k}  {int(raw_counts[k]):6d}  {raw_shares[k]*100:5.1f}%  "
              f"{_bar(float(raw_shares[k]))}{below}")
    ms = margins.sort().values
    print(f"  winner margin: median={float(margins.median()):.4f} "
          f"p25={float(ms[int(0.25*N)]):.4f} p10={float(ms[int(0.10*N)]):.4f}")
    if float(margins.median()) < 0.02:
        print("  [!] tiny margins — paths near-tied; raw targets are largely noise "
              "(early/undertrained LM). A refit here bakes noise into centroids.")

    # ---- #4: per-path ΔNLL fill-cost diagnostics for below-floor paths ----
    below = [k for k in range(K) if float(raw_shares[k]) < args.floor]
    print("\n  fill-cost diagnostics (ΔNLL to reach floor):")
    if not below:
        print(f"  no path below {args.floor*100:.0f}% — floor does not bind on this data.")
    else:
        for k in below:
            _fill_cost(nll, raw_best, raw_shares, args.floor, k, delta_cap=args.delta_cap)

    # ---- #1: bucket diagnostic (argmin shares at non-256 boundaries) ----
    if args.buckets:
        print("\n  argmin shares BY PREFIX BUCKET (diagnostic; assignment stays at 256):")
        for lo, hi in C.VARPREFIX_EVAL_BUCKETS:
            R = (lo + hi) // 2
            bnll = _bucket_nll(model, D, C, min(args.n, 1000), R, batch_rows=args.batch_rows)
            if bnll.numel() == 0:
                continue
            _, bc, bs = _argmin_shares(bnll)
            print(f"    [{lo}-{hi} R={R}] " + "  ".join(
                f"P{k}:{bs[k]*100:.0f}%" for k in range(K)))

    # ---- quota -> final targets ----
    if args.quota:
        final_labels, qrep = apply_min_share_quota(
            nll, raw_best, args.floor, delta_cap=args.delta_cap)
        print("\n" + "-" * 70)
        print(f"MIN-SHARE QUOTA  floor={args.floor*100:.0f}%  "
              f"cap={args.delta_cap}")
        print("-" * 70)
        print(f"  moves={qrep['n_moves']}  total ΔNLL={qrep['total_dnll']:.3f}  "
              f"mean ΔNLL/move={qrep['mean_dnll']:.4f}")
        fs = qrep["final_shares"]
        print("  final target shares: " + "  ".join(
            f"P{k}:{fs[k]*100:.1f}%" for k in range(K)))
        if qrep["still_below_floor"]:
            print(f"  [!] still below floor after donor-safe fill: "
                  f"{['P%d'%k for k in qrep['still_below_floor']]} "
                  f"(floor unreachable without forcing — left partial, by design)")
        fit_method = f"label-mean; quota floor={args.floor} cap={args.delta_cap}"
    else:
        final_labels = raw_best
        qrep = {"n_moves": 0, "total_dnll": 0.0, "mean_dnll": 0.0,
                "final_shares": raw_shares.tolist(), "still_below_floor": [],
                "delta_cap": None, "floor": None}
        fit_method = "label-mean; no quota (raw PPL-best targets)"
        print("\n[quota] disabled — final targets = raw PPL-best (use --quota to enable)")

    # ---- build full PPL target centroids, then move toward them by alpha ----
    ppl_cluster, empty_paths = refit_label_mean(
        Z, final_labels, K, allow_empty=args.allow_empty)

    if empty_paths:
        print(f"\n[refit][WARN] empty target label(s) {empty_paths} -> global-mean "
              f"fallback (--allow-empty). These paths have no seed content.")

    # --------------------------------------------------------------
    # Prepare CALIB/GATE evidence (cached or live)
    # --------------------------------------------------------------
    # If cache-load was used, Z_cal, Z_gate, etc. are already loaded.
    if not args.cache_load:
        # Live collection of CALIB/GATE
        off_cal = feedback_skip + feedback_seq_n + gap   # Phase 4: use sequence count, not N
        off_gate = off_cal + args.calib_n + gap

        print(
            "\n[recalib] collecting disjoint CALIB/GATE splits and recalibrating "
            "thresholds on the NEW centroids (parent thresholds are stale after "
            "centroid movement)"
        )

        Z_cal, _, _, _, _ = _collect_feedback(
            encoder,
            model,
            args.calib_n,
            boundary,
            C,
            D,
            skip=off_cal,
            batch_rows=args.batch_rows
        )

        Z_gate, nll_gate, _, pref_gate, full_gate = _collect_feedback(
            encoder,
            model,
            args.gate_n,
            boundary,
            C,
            D,
            skip=off_gate,
            batch_rows=args.batch_rows
        )

        # Save raw sweep data if requested
        if args.cache_save:
            cache_payload = {
                "cache_format": PPL_CACHE_FORMAT,
                "created_at": time.time(),

                "parent_version_id": parent.version_id,
                "parent_artifact": parent_path,
                "parent_centroids": parent.cluster.centroids.detach().cpu(),

                "K": int(K),
                "boundary": int(boundary),

                "teacher_checkpoint": teacher_checkpoint_used,
                "teacher_step": int(step),
                "consumed_sequences": int(consumed),

                "feedback_skip": int(feedback_skip),
                "calib_skip": int(off_cal),
                "gate_skip": int(off_gate),

                # Phase 4: multi-view FEEDBACK
                "fit_prefixes": list(PPL_FIT_PREFIXES),

                "feedback_sequence_n": int(feedback_seq_n),
                "feedback_fit_rows": int(N),
                "calib_n": int(Z_cal.size(0)),
                "gate_n": int(Z_gate.size(0)),

                "Z_views": {
                    r: Z_views[r].detach().cpu()
                    for r in PPL_FIT_PREFIXES
                },

                "nll_views": {
                    r: nll_views[r].detach().cpu()
                    for r in PPL_FIT_PREFIXES
                },

                "sources": sources,
                "pref_txt": pref_txt,
                "full_txt": full_txt,

                "Z_cal": Z_cal.detach().cpu(),

                "Z_gate": Z_gate.detach().cpu(),
                "nll_gate": nll_gate.detach().cpu(),
                "pref_gate": pref_gate,
                "full_gate": full_gate,
            }

            _save_ppl_cache(args.cache_save, cache_payload)

    # --------------------------------------------------------------
    # SWEEP: same targets, same CALIB, same GATE, only alpha changes.
    # --------------------------------------------------------------
    if args.sweep:
        parent_fit_assign = parent.cluster.runtime_assign(Z)
        _, parent_fit_ppl = _routed_ppl(nll, parent_fit_assign)

        parent_gate_assign = parent.cluster.runtime_assign(Z_gate)
        _, parent_gate_ppl = _routed_ppl(nll_gate, parent_gate_assign)

        oracle_gate_assign = nll_gate.argmin(dim=1)
        _, oracle_gate_ppl = _routed_ppl(
            nll_gate, oracle_gate_assign
        )

        rows = []

        print("\n" + "=" * 86)
        print("ALPHA SWEEP — SAME FROZEN LM / SAME FEEDBACK / SAME CALIB / SAME GATE")
        print("=" * 86)

        for a in ALPHA_SWEEP:
            row = _evaluate_alpha_sweep(
                a,
                parent,
                encoder,
                ppl_cluster,
                Z,
                nll,
                Z_cal,
                Z_gate,
                nll_gate,
                pref_gate,
                full_gate,
                B,
            )
            rows.append(row)

        print(
            f"{'alpha':>7}  "
            f"{'FIT PPL':>10}  "
            f"{'GATE PPL':>10}  "
            f"{'Delta':>9}  "
            f"{'min%':>7}  "
            f"{'max%':>7}  "
            f"{'GATE':>6}"
        )
        print("-" * 86)

        for row in rows:
            print(
                f"{row['alpha']:7.2f}  "
                f"{row['fit_ppl']:10.3f}  "
                f"{row['gate_ppl']:10.3f}  "
                f"{row['gate_ppl'] - parent_gate_ppl:+9.3f}  "
                f"{row['gate_min_share']*100:7.1f}  "
                f"{row['gate_max_share']*100:7.1f}  "
                f"{'PASS' if row['passed'] else 'FAIL':>6}"
            )

        print("-" * 86)
        print(f"parent FIT PPL : {parent_fit_ppl:.3f}")
        print(f"parent GATE PPL: {parent_gate_ppl:.3f}")
        print(f"oracle GATE PPL: {oracle_gate_ppl:.3f}")
        print(
            "[sweep] report-only — NO artifact written. "
            "Choose alpha, then re-run with --alpha X --save."
        )
        print("=" * 86)
        return

    # For normal single-alpha mode, continue exactly as before.
    # Saving always requires an explicit --alpha.
    alpha = 1.0 if args.alpha is None else float(args.alpha)

    new_cluster = blend_cluster(
        parent.cluster,
        ppl_cluster,
        alpha
    )

    fit_method = f"{fit_method}; centroid_blend_alpha={alpha}"

    new_runtime = new_cluster.runtime_assign(Z)
    new_counts = torch.bincount(new_runtime, minlength=K).float()

    print("\n" + "-" * 70)
    print(f"REFIT — blended parent -> PPL centroids, alpha={alpha:.3f}")
    print("-" * 70)

    for k in range(K):
        print(f"  P{k}  {int(new_counts[k]):6d}  {new_counts[k]/N*100:5.1f}%  "
              f"{_bar(float(new_counts[k]/N))}")
    reproduce = float((new_runtime == final_labels).float().mean())
    print(f"  target reproduction (runtime==final target): {reproduce*100:.1f}%  "
          f"(the rest is the nearest-centroid generalization gap — expected)")

    # ---- compare parent CI vs new (transition matrices + share change) ----
    old_assign = parent.cluster.runtime_assign(Z)             # parent routing on Z
    print("\n" + "=" * 70)
    print("PARENT CI  vs  REFIT  (on the same feedback embeddings)")
    print("=" * 70)
    _print_matrix("current -> RAW PPL target", _transition_matrix(old_assign, raw_best, K))
    if args.quota:
        _print_matrix("current -> FINAL target (post-quota)",
                      _transition_matrix(old_assign, final_labels, K))
    old_counts = torch.bincount(old_assign, minlength=K).float()
    print("\nper-path share change (parent runtime -> new runtime):")
    for k in range(K):
        o, nn = float(old_counts[k]/N), float(new_counts[k]/N)
        print(f"  P{k}: {o*100:5.1f}%  ->  {nn*100:5.1f}%   (Δ {(nn-o)*100:+.1f} pts)")
    # RESERVED HOOK — drift limiting (NOT enforced). When added it is a PER-PATH
    # SHARE-CHANGE CAP vs the parent CI on THIS feedback set (old_counts/N vs
    # new_counts/N per path), never a per-sample flip cap.

    # ---- #3b: OLD vs NEW routing PPL under the SAME frozen LM ----
    # The decisive number: does the refit's routing actually LOWER LM PPL vs the
    # parent's routing? Both use the same six-path nll[N,K]; only the routing differs.
    old_nll, old_ppl = _routed_ppl(nll, old_assign)
    new_nll, new_ppl = _routed_ppl(nll, new_runtime)
    orc_nll, orc_ppl = _routed_ppl(nll, raw_best)             # oracle (PPL-best) route
    print("\n" + "=" * 70)
    print("ROUTING PPL under the frozen LM (same weights; only routing differs)")
    print("=" * 70)
    print(f"  parent CI routing : PPL {old_ppl:8.3f}  (mean NLL {old_nll:.4f})")
    print(f"  refit  CI routing : PPL {new_ppl:8.3f}  (mean NLL {new_nll:.4f})   "
          f"Δ {new_ppl-old_ppl:+.3f}")
    print(f"  oracle (PPL-best) : PPL {orc_ppl:8.3f}  (lower bound; not routable)")
    improved = new_ppl < old_ppl
    print(f"  -> refit routing {'IMPROVES' if improved else 'does NOT improve'} "
          f"PPL vs parent on the feedback set")

    # ---- recalibrate thresholds for this explicit alpha ----
    fb = B.calibrate_fallback(new_cluster, Z_cal)

    thresholds = {"T_MARGIN_FALLBACK": fb["T_MARGIN_FALLBACK"],
                  "T_SIM_FALLBACK": fb["T_SIM_FALLBACK"],
                  # T_SWITCH: keep parent's unless recalibrated separately; label-mean
                  # recentre doesn't change the switch regime enough to justify a full
                  # two-regime recal here (documented; a later step can recal it).
                  "T_SWITCH": parent.t_switch}
    print(f"[recalib] T_MARGIN {parent.t_margin:.4f}->{thresholds['T_MARGIN_FALLBACK']:.4f}  "
          f"T_SIM {parent.t_sim:.4f}->{thresholds['T_SIM_FALLBACK']:.4f}  "
          f"T_SWITCH {parent.t_switch:.4f} (kept)")

    # build a candidate CI with the NEW centroids + RECALIBRATED thresholds, gate it
    cand_ci = ContentIndex(encoder, new_cluster, thresholds,
                           parent.version_id + "-candidate", dict(parent.meta))
    passed, gate_report = B.go_no_go(
        cand_ci, new_cluster, encoder, Z_gate,
        full_gate, prefix_texts_gate=pref_gate)

    B._print_gate(gate_report)
    print(f"[gate] refit {'PASSED' if passed else 'FAILED'} the held-out GO/NO-GO gate.")

    # ---- HELD-OUT routing PPL: parent vs refit vs oracle ----
    old_gate_assign = parent.cluster.runtime_assign(Z_gate)
    new_gate_assign = new_cluster.runtime_assign(Z_gate)
    oracle_gate_assign = nll_gate.argmin(dim=1)

    old_gate_nll, old_gate_ppl = _routed_ppl(
        nll_gate, old_gate_assign)

    new_gate_nll, new_gate_ppl = _routed_ppl(
        nll_gate, new_gate_assign)

    oracle_gate_nll, oracle_gate_ppl = _routed_ppl(
        nll_gate, oracle_gate_assign)

    print("\n" + "=" * 70)
    print("HELD-OUT ROUTING PPL — DISJOINT GATE SET")
    print("=" * 70)
    print(f"  parent CI : PPL {old_gate_ppl:8.3f}  "
        f"(mean NLL {old_gate_nll:.4f})")
    print(f"  refit CI  : PPL {new_gate_ppl:8.3f}  "
        f"(mean NLL {new_gate_nll:.4f})  "
        f"Delta {new_gate_ppl-old_gate_ppl:+.3f}")
    print(f"  oracle    : PPL {oracle_gate_ppl:8.3f}  "
        f"(mean NLL {oracle_gate_nll:.4f})")

    heldout_improved = new_gate_ppl <= old_gate_ppl

    print(f"  -> held-out routing "
        f"{'IMPROVES/PRESERVES' if heldout_improved else 'WORSENS'} "
        f"PPL vs parent")

    # ---- save (only when requested AND gate passed AND held-out PPL OK) ----
    if not args.save:
        print("\n[dry] --save not given — reported only, NO artifact written.")
        print("=" * 70)
        return

    if not passed and not args.force_save:
        print("\n[save] REFUSED — refit FAILED the held-out CI gate.")
        print("=" * 70)
        return

    if not heldout_improved and not args.force_save:
        print("\n[save] REFUSED — refit worsened held-out routing PPL.")
        print(f"       parent={old_gate_ppl:.3f}  refit={new_gate_ppl:.3f}")
        print("=" * 70)
        return

    gen = int(parent.meta.get("refit_generation", 0)) + 1
    new_version = args.version_suffix or f"{parent.version_id}-ppl{gen}"
    meta = dict(parent.meta)                                   # carry parent meta
    meta.update({
        "refit_generation": gen,
        "parent_version_id": parent.version_id,
        "parent_artifact": parent_path,
        "teacher_checkpoint": teacher_checkpoint_used,
        "teacher_step": int(step),
        "fit_method": fit_method,
        "centroid_blend_alpha": float(alpha),

        "ppl_cache": {
            "loaded_from": args.cache_load,
            "saved_to": args.cache_save,
            "format": PPL_CACHE_FORMAT,
            "feedback_skip": int(feedback_skip),
            "calib_skip": int(off_cal),
            "gate_skip": int(off_gate),
        },

        "assignment_boundary": boundary,
        # Phase 4: multi-view metadata
        "fit_prefixes": list(PPL_FIT_PREFIXES),
        "feedback_sequence_n": int(feedback_seq_n),
        "feedback_fit_rows": int(N),

        "empty_label_paths": empty_paths,
        "quota": {
            "enabled": bool(args.quota),
            "floor": args.floor if args.quota else None,
            "delta_cap": args.delta_cap,
            "moves": qrep["n_moves"],
            "total_dnll": qrep["total_dnll"],
            "mean_dnll": qrep["mean_dnll"],
            "still_below_floor": qrep["still_below_floor"],
        },
        "raw_ppl_shares": raw_shares.tolist(),
        "final_target_shares": qrep["final_shares"],
        "winner_margin_median": float(margins.median()),
        "routing_ppl": {
            "feedback": {
                "parent": old_ppl,
                "refit": new_ppl,
                "oracle": orc_ppl,
            },
            "heldout_gate": {
                "parent": old_gate_ppl,
                "refit": new_gate_ppl,
                "oracle": oracle_gate_ppl,
                "improved": bool(heldout_improved),
            },
        },
        "recalibrated_thresholds": {
            "T_MARGIN_FALLBACK": thresholds["T_MARGIN_FALLBACK"],
            "T_SIM_FALLBACK": thresholds["T_SIM_FALLBACK"],
            "T_SWITCH": thresholds["T_SWITCH"],
            "parent_T_MARGIN": parent.t_margin,
            "parent_T_SIM": parent.t_sim,
            "T_SWITCH_recalibrated": False,   # kept from parent (documented)
        },
        "gate_passed": bool(passed),
        "force_saved": bool(args.force_save),
        "gate_report": gate_report,
        "refit_saved_at": time.time(),
    })
    # Use the RECALIBRATED thresholds (computed on the NEW centroids above) — NOT the
    # parent's stale values, which no longer describe the moved centroids' geometry.
    new_ci = ContentIndex(encoder, new_cluster, thresholds, new_version, meta)
    out = args.out or parent_path.replace(".pt", f".ppl{gen}.pt")
    new_ci.save(out)
    print(f"\n[save] wrote refit artifact -> {out}")
    print(f"       version_id = {new_version}  (generation {gen})")
    print(
        f"       parent = {parent.version_id}  teacher_step = {step}  "
        f"gate = {'PASS' if passed else 'FAIL (FORCED)'}"
    )
    print("       NOTE: bump config CONTENT_INDEX_VERSION + ARTIFACT to this before")
    print("       the LM trains against it (the LM asserts version match on load).")
    print("=" * 70)


if __name__ == "__main__":
    run()