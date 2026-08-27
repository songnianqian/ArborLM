"""
inspect_index.py — read-only inspector for a Content Index artifact.

Shows how the FROZEN index actually partitions data, on whatever dataset you point
it at (defaults to the frozen local slice). Nothing here mutates the artifact.

Two modes, auto-detected from the artifact:

  SINGLE-CENTROID (build_content_index.py output, e.g. content_index.local.pt)
    - one centroid per path (K paths). Reports:
        * cluster occupancy: how many contexts land in each path (runtime top-1).
          v3.1: there is NO GENERAL destination — EVERY context routes to its top-1
          specialist and is counted. Low-confidence (below T_SIM / T_MARGIN) is a
          DIAGNOSTIC COUNT ONLY, not a separate bucket.
        * domain x cluster: which SlimPajama sources dominate each path
    - there are NO prototypes yet; those appear only after the round-level trainer.

  MULTI-PROTOTYPE (content_index_trainer.py output, e.g. content_index.r2.pt)
    - N prototypes per path (design §22). Additionally reports:
        * per-prototype usage: within each path, how its contexts split across its
          prototypes (the "prototype usage" the trainer tracks, §23/§24)

Occupancy is measured on the RUNTIME top-1 partition — the one specialists actually
see — computed on the SAME routing view the deployed CI uses: the first
`route_prefix` GPT-2 tokens (from the artifact's build config), NOT the full
sequence (design §1).

Run from the projectB folder:
    python inspect_index.py
    python inspect_index.py --artifact content_index.local.pt --dataset local_dataset.pt
    python inspect_index.py --artifact content_index.r2.pt --n 2000
"""
import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F


def _fmt_bar(frac, width=24):
    fill = int(round(frac * width))
    return "█" * fill + "·" * (width - fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="content_index.local.pt")
    ap.add_argument("--dataset", default="local_dataset.pt",
                    help="frozen local slice to measure occupancy on")
    ap.add_argument("--n", type=int, default=None,
                    help="cap contexts inspected (default: all in the dataset)")
    ap.add_argument("--route-prefix", type=int, default=None,
                    help="tokens of the runtime routing view to embed (default: the "
                         "prefix the artifact was built on, else config.CI_ROUTE_PREFIX, "
                         "else 256). This must match deployed CI routing.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.artifact):
        print(f"[inspect] artifact not found: {args.artifact}"); sys.exit(2)
    if not os.path.exists(args.dataset):
        print(f"[inspect] dataset not found: {args.dataset}"); sys.exit(2)

    import content_index_config as CIC
    from content_index import ContentIndex, MultiPrototypeCluster

    ci = ContentIndex.load(args.artifact, CIC, args.device)
    K = ci.K
    is_multi = isinstance(ci.cluster, MultiPrototypeCluster)
    M = None
    if is_multi:
        # prototypes per path (assume uniform; report actual per path below)
        counts = torch.bincount(ci.cluster.proto_specialist, minlength=K)
        M = int(counts.max().item())

    print("=" * 64)
    print(f"CONTENT INDEX INSPECTION — {ci.version_id}")
    print("=" * 64)
    print(f"artifact  : {args.artifact}")
    print(f"encoder   : {ci.encoder.name}")
    print(f"K (paths) : {K}")
    print(f"layout    : {'MULTI-PROTOTYPE' if is_multi else 'single-centroid'}"
          + (f"  ({ci.cluster.P} prototypes total, ~{M}/path)" if is_multi else ""))
    thr = f"T_MARGIN={ci.t_margin:.4f} T_SIM={ci.t_sim:.4f} T_SWITCH={ci.t_switch:.4f}"
    print(f"thresholds: {thr}")

    # ---- embed the dataset with the frozen encoder ----
    blob = torch.load(args.dataset, map_location="cpu")
    input_ids = blob["input_ids"]
    sources = blob["sources"]
    N = input_ids.shape[0] if args.n is None else min(args.n, input_ids.shape[0])
    print(f"\n[inspect] embedding {N} contexts from {args.dataset} ...")

    lm_tok = None
    # decode with the dataset's own tokenizer name if available, else the encoder's
    try:
        from transformers import AutoTokenizer
        lm_tok = AutoTokenizer.from_pretrained(blob.get("tokenizer_name", "gpt2"))
    except Exception:
        lm_tok = getattr(ci.encoder, "_tok", None)

    # ---- FIX (2): embed the EXACT runtime routing view, not the whole 1024-token
    # sequence. The deployed CI routes on the first `route_prefix` GPT-2 tokens
    # (design: CI must never see tokens at/after the scored boundary), so occupancy /
    # domain tables here must be computed on tokens[:route_prefix] — otherwise they
    # describe a routing view the LM never uses. Prefer the prefix the artifact was
    # BUILT on (ci.route_prefix_tokens); fall back to the LM's CI_ROUTE_PREFIX, then
    # 256. A CLI --route-prefix overrides for experiments.
    route_prefix = args.route_prefix
    src = "cli --route-prefix"
    if route_prefix is None:
        route_prefix = getattr(ci, "route_prefix_tokens", None)
        src = "artifact route_prefix_tokens"
    if route_prefix is None:
        try:
            import config as _LMC
            route_prefix = int(getattr(_LMC, "CI_ROUTE_PREFIX", 256))
            src = "config.CI_ROUTE_PREFIX"
        except Exception:
            route_prefix = 256; src = "default 256"
    route_prefix = int(route_prefix)
    seq_len = int(input_ids.shape[1])
    print(f"[inspect] routing view: first {route_prefix} tokens "
          f"(source: {src}); sequences are {seq_len} tokens — "
          f"{'CROP applied' if route_prefix < seq_len else 'no crop (prefix >= seq len)'}")

    Z = []
    for i in range(N):
        prefix_ids = input_ids[i][:route_prefix]            # runtime routing view
        txt = lm_tok.decode(prefix_ids.tolist(), skip_special_tokens=True) \
            if lm_tok is not None else ""
        Z.append(ci.encoder.embed_text(txt))
    Z = torch.stack(Z, dim=0)

    # ---- runtime argmax partition (what specialists see) ----
    assign = ci.cluster.runtime_assign(Z)          # [N] path id per context
    sims = ci.cluster.similarities(Z)              # [N, K]

    # fallback: low-confidence flag — top1 below T_SIM, or top1-top2 margin below
    # T_MARGIN. FIX (1): in v3.1 there is NO GENERAL destination, so this is a
    # DIAGNOSTIC COUNT ONLY. Every context still routes to (and is counted under) its
    # top-1 specialist; low-confidence contexts are NOT removed from occupancy.
    top2 = torch.topk(sims, k=min(2, K), dim=1).values
    s1 = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1] if K > 1 else torch.ones_like(s1)
    low_conf = (s1 < ci.t_sim) | (margin < ci.t_margin)
    n_lc = int(low_conf.sum())

    # ---- cluster occupancy: ALL contexts by top-1 specialist (deployed view) ----
    counts = torch.bincount(assign, minlength=K).float()
    shares = (counts / counts.sum().clamp_min(1))
    # low-confidence contexts, broken down by the specialist they still route to
    lc_counts = torch.bincount(assign[low_conf], minlength=K).float() if n_lc else torch.zeros(K)

    print("\n" + "-" * 64)
    print("CLUSTER OCCUPANCY (runtime top-1 specialist, ALL contexts)")
    print("-" * 64)
    for k in range(K):
        lc = int(lc_counts[k])
        lc_s = f"  (low-conf: {lc})" if lc else ""
        print(f"  P{k}  {int(counts[k]):5d}  {shares[k]*100:5.1f}%  "
              f"{_fmt_bar(float(shares[k]))}{lc_s}")
    print(f"  low-confidence    : {n_lc:5d}  {n_lc/N*100:5.1f}%  "
          f"(top1<T_SIM or margin<T_MARGIN — DIAGNOSTIC ONLY; still routed to top-1)")
    print(f"  total routed      : {int(counts.sum())} / {N}  (no GENERAL — all contexts route)")

    # ---- domain x cluster ----
    print("\n" + "-" * 64)
    print("DOMAIN × CLUSTER  (row=source, cell=% of that source routed to path)")
    print("-" * 64)
    # rows: source -> [count per path]. FIX (1): every context counts toward its
    # top-1 specialist (no GENERAL bucket, no fallback removal).
    by_src = defaultdict(lambda: torch.zeros(K))
    src_totals = defaultdict(int)
    for i in range(N):
        by_src[str(sources[i])][assign[i]] += 1
        src_totals[str(sources[i])] += 1
    hdr = "  " + f"{'source':<22}" + "".join(f"  P{k} " for k in range(K)) + "  dom"
    print(hdr)
    for src in sorted(by_src):
        row = by_src[src]
        tot = row.sum().clamp_min(1)
        pct = row / tot * 100
        dom = int(row.argmax())
        cells = "".join(f" {p:4.0f}" for p in pct)
        print(f"  {src:<22}{cells}   P{dom}")
    print("  (each row sums to ~100% across paths; 'dom' = that source's top path)")

    # ---- per-prototype usage (multi-prototype only) ----
    if is_multi:
        print("\n" + "-" * 64)
        print("PROTOTYPE USAGE  (within each path, split of contexts across prototypes)")
        print("-" * 64)
        proto_spec = ci.cluster.proto_specialist
        Zc = F.normalize(Z.float(), p=2, dim=-1, eps=1e-8)
        for k in range(K):
            cols = torch.where(proto_spec == k)[0]
            # ALL contexts routed to path k (v3.1: no fallback exclusion)
            in_k = (assign == k)
            if in_k.sum() == 0:
                print(f"  P{k}: (no routed contexts)")
                continue
            sub = Zc[in_k] @ ci.cluster.prototypes[cols].t()   # [n_k, m_k]
            nearest = sub.argmax(dim=1)
            usage = torch.bincount(nearest, minlength=len(cols)).float()
            ushare = usage / usage.sum().clamp_min(1)
            parts = "  ".join(f"m{m}:{ushare[m]*100:4.1f}%" for m in range(len(cols)))
            dead = int((ushare < 0.02).sum())
            flag = f"  [{dead} near-dead]" if dead else ""
            print(f"  P{k} ({int(in_k.sum())} ctx): {parts}{flag}")
        print("  near-dead = prototype with <2% of its path's mass (rejuvenation "
              "candidate, §24)")

    # ---- quick health read ----
    print("\n" + "-" * 64)
    print("READ")
    print("-" * 64)
    mn, mx = float(shares.min()), float(shares.max())
    print(f"  balance      : min={mn:.3f} max={mx:.3f} imbalance={mx/max(mn,1e-9):.2f}")
    print(f"  low-conf rate: {n_lc/N:.3f}  (diagnostic; all contexts still route to top-1)")
    starved = int((shares < 0.5/K).sum())
    print(f"  starved paths: {starved} (share < half the equal {1/K:.3f})")
    if not is_multi:
        print("  note: single-centroid artifact — no prototypes yet. Per-prototype")
        print("        usage appears after the round-level trainer builds them (§22).")


if __name__ == "__main__":
    main()
