"""
eval.py — Project B v3.1 evaluation (no GENERAL, no fusion).

Runs every EVAL_EVERY steps. Three parts:

  1. PROMPT TEST — greedy per-domain continuations, CI-routed via model.infer:
        [dom ->Pk] 'prompt' -->> 'continuation'
  2. PPL + ACC — full-vocab held-out perplexity AND next-token accuracy. Reported
     numbers are ALWAYS full-vocab (training-time top-k never affects eval).
  3. PER-PATH TABLE — GLOBAL per-path metrics (every specialist Pk scored on EVERY
     doc as its own continuation of h_trunk, so paths are comparable on equal
     footing), plus CI usage:
        PPL(global)   exp(mean specialist NLL over ALL docs)
        ACC(global)   mean next-token accuracy over ALL docs
        usage         how many docs the CI routed to k (top-1)
        %missed        of k's ROUTED docs, fraction where another path has lower NLL
     Rollup: CI-top1 PPL/ACC, CI-top2-best PPL (EXACT), oracle PPL (best of P0..P5).

v3.1 semantics: each specialist is a complete upper continuation of h_trunk; there
is no GENERAL branch and no fusion. Oracle and the table cover P0..P5. All metrics
come FRESH from model.fresh_gains (full-vocab).
"""
import math
import torch

import config as C


def _ppl(mean_nll):
    return float(math.exp(min(float(mean_nll), 20.0)))


def _mutual_info(joint):
    """MI (nats) between source and CI-top1 path from a [S, P] count matrix.
    Diagnostic only — sources are never path targets (design note item 4)."""
    if joint is None:
        return None
    j = joint.float()
    tot = j.sum()
    if tot <= 0:
        return 0.0
    p = j / tot
    ps = p.sum(dim=1, keepdim=True)       # [S,1]
    pp = p.sum(dim=0, keepdim=True)       # [1,P]
    denom = (ps * pp).clamp_min(1e-12)
    mask = p > 0
    return float((p[mask] * (p[mask] / denom[mask]).log()).sum())


def _apply_repetition_penalty(logits, prev_ids, penalty):
    """CTRL-style repetition penalty: positive logits of already-seen tokens are
    divided by `penalty` (>1 discourages repeats), negative logits multiplied.
    logits [V], prev_ids 1-D LongTensor. No-op at penalty=1.0."""
    if penalty is None or penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    uniq = torch.unique(prev_ids)
    sel = logits[uniq]
    logits[uniq] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return logits


def _ban_repeat_ngrams(logits, seq_ids, n):
    """no_repeat_ngram_size: forbid any next token that would complete an n-gram
    already present in seq_ids. logits [V], seq_ids 1-D LongTensor (full sequence)."""
    if not n or n < 1 or seq_ids.numel() < n:
        return logits
    seq = seq_ids.tolist()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else tuple()
    banned = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    for t in banned:
        logits[t] = float("-inf")
    return logits


def _sample_next(logits_last, seq_ids, temperature, top_p, top_k,
                 repetition_penalty, no_repeat_ngram, generator):
    """Pick the next token from last-step logits [1, V]. HF-style order: repetition
    penalty -> no-repeat-ngram ban -> temperature -> top-k -> top-p (nucleus) ->
    multinomial. Reporting-only; does not touch model state or routing."""
    logits = logits_last.squeeze(0).float().clone()  # [V]
    logits = _apply_repetition_penalty(logits, seq_ids, repetition_penalty)
    logits = _ban_repeat_ngrams(logits, seq_ids, no_repeat_ngram)
    if temperature and temperature > 0:
        logits = logits / temperature
    V = logits.size(-1)
    # top-k restriction
    if top_k and 0 < top_k < V:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, logits.new_full((), float("-inf")), logits)
    # top-p (nucleus) restriction
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep the smallest set whose cumulative prob first reaches top_p, INCLUDING
        # the token that crosses the threshold. cum > top_p marks everything strictly
        # past top_p; shifting that mask right by one keeps the crossing token (and
        # always keeps the top token, since position 0 becomes False).
        remove_sorted = cum > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove = sorted_idx[remove_sorted]
        logits[remove] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1, generator=generator)  # [1]
    return nxt.view(1, 1)


@torch.no_grad()
def _generate(model, tokenizer, device, text, n_tokens, mode,
              temperature=None, top_p=None, top_k=None,
              repetition_penalty=None, no_repeat_ngram=None,
              eos_stop=False, generator=None):
    """One CI-routed continuation of `text`. mode='greedy' is RAW argmax (no
    penalties, no stopping) — an intrinsic model diagnostic. mode='sample' applies
    the HF-style controls (temp/top-p/top-k + repetition penalty + no-repeat-ngram +
    optional EOS stop). Routing is identical (model.infer each step) — only next-token
    SELECTION differs. Returns (generated_text, route)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.size(1)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    route = None
    for _ in range(n_tokens):
        logits, top1 = model.infer(ids, tokenizer=tokenizer)
        if route is None:
            route = int(top1[0])
        if mode == "greedy":
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            nxt = _sample_next(logits[:, -1, :], ids[0], temperature, top_p, top_k,
                               repetition_penalty, no_repeat_ngram, generator)
        ids = torch.cat([ids, nxt], dim=1)
        if mode == "sample" and eos_stop and eos_id is not None and int(nxt) == eos_id:
            break
        if ids.size(1) >= C.N_POSITIONS:
            break
    gtxt = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
    return gtxt, route


@torch.no_grad()
def prompt_test(model, tokenizer, device, prompts=None, n_tokens=None):
    """Per-domain continuations, CI-routed via model.infer. Produces BOTH a GREEDY
    and a SAMPLED (temperature/top-p/top-k) continuation for each prompt, and an
    optional fixed REFERENCE from C.DIAG_REFERENCES. Returns a list of per-prompt
    dicts; formatting is done in print_headline. Diagnostics-only — no model,
    routing, optimizer, or data-stream state is modified."""
    prompts = prompts or C.DIAG_PROMPTS
    n_tokens = n_tokens or C.DIAG_GEN_TOKENS
    refs = getattr(C, "DIAG_REFERENCES", {}) or {}
    temp = getattr(C, "DIAG_SAMPLE_TEMPERATURE", 0.8)
    top_p = getattr(C, "DIAG_SAMPLE_TOP_P", 0.9)
    top_k = getattr(C, "DIAG_SAMPLE_TOP_K", 40)
    rep = getattr(C, "DIAG_SAMPLE_REPETITION_PENALTY", 1.15)
    nrng = getattr(C, "DIAG_SAMPLE_NO_REPEAT_NGRAM", 4)
    eos = getattr(C, "DIAG_SAMPLE_EOS_STOP", True)
    seed = getattr(C, "DIAG_SAMPLE_SEED", 1234)

    was_training = model.training
    model.eval()
    results = []
    try:
        for dom, text in prompts.items():
            greedy_txt, route = _generate(model, tokenizer, device, text, n_tokens,
                                          mode="greedy")
            # fresh generator per prompt so SAMPLED is reproducible across checkpoints
            gen = torch.Generator(device=device).manual_seed(seed)
            sampled_txt, _ = _generate(model, tokenizer, device, text, n_tokens,
                                       mode="sample", temperature=temp, top_p=top_p,
                                       top_k=top_k, repetition_penalty=rep,
                                       no_repeat_ngram=nrng, eos_stop=eos, generator=gen)
            results.append({
                "dom": dom, "route": route, "prompt": text,
                "reference": refs.get(dom),
                "greedy": greedy_txt, "sampled": sampled_txt,
            })
    finally:
        if was_training:
            model.train()
    return results


@torch.no_grad()
def headline_eval(model, eval_ids, eval_labels=None, attention_mask=None,
                  base_is_frozen=None, batch_size=8, queue=None, tokenizer=None,
                  prefix_buckets=False):
    """Fresh, full-vocab, queue-off headline eval. Requires a tokenizer (causal
    prefix routing). GLOBAL per-path PPL/ACC + CI usage + exact CI-top2.

    The main headline is always at the exact-256 reference boundary (C.SCORE_FROM).
    When prefix_buckets=True and variable-prefix training is on, ALSO reports CI-top1
    PPL at a representative R for each configured bucket (plus the ref256 point), so
    we can see whether the variable-prefix LM generalizes across context lengths."""
    if queue is not None:
        raise ValueError("headline_eval: v3 has no replay queue — do not pass one.")
    if tokenizer is None:
        raise ValueError("headline_eval requires a tokenizer (causal prefix routing).")
    if C.HEADLINE_REQUIRES_FRESH and base_is_frozen is False:
        print("[eval][WARN] base NOT frozen — treat headline numbers as provisional.")

    device = next(model.parameters()).device
    N = eval_ids.size(0)
    n_spec = C.N_SPECIALISTS

    # GLOBAL accumulators: every path scored on every doc
    g_nll_sum = torch.zeros(n_spec)      # sum specialist NLL over ALL docs, per path
    g_acc_sum = torch.zeros(n_spec)      # sum specialist ACC over ALL docs, per path
    # CI-routed accumulators
    sel_nll_sum = 0.0                    # CI top-1 NLL
    sel_acc_sum = 0.0                    # CI top-1 ACC
    top2_nll_sum = 0.0                   # EXACT CI-top2-best NLL
    oracle_nll_sum = 0.0                 # best of P0..P5 per doc
    usage = torch.zeros(n_spec)          # docs routed (top-1) to each path
    missed = torch.zeros(n_spec)         # of routed docs, another path had lower NLL
    n_seen = 0

    # source -> path joint counts (design note item 4). Diagnostic only — sources
    # are NOT path targets. Built only when eval_labels are supplied.
    src_order = []
    if eval_labels is not None:
        seen = set()
        for s in eval_labels:
            if s not in seen:
                seen.add(s); src_order.append(s)
    src_idx = {s: i for i, s in enumerate(src_order)}
    joint = torch.zeros(len(src_order), n_spec) if src_order else None

    for i in range(0, N, batch_size):
        ids = eval_ids[i:i + batch_size].to(device)
        am = attention_mask[i:i + batch_size].to(device) if attention_mask is not None else None
        out = model.fresh_gains(ids, tokenizer=tokenizer, attention_mask=am)
        B = ids.size(0)
        spec_nll = out["spec_nll"]       # [B, n_spec]
        spec_acc = out["spec_acc"]       # [B, n_spec]
        top1 = out["selected"]           # [B]

        g_nll_sum += spec_nll.sum(dim=0).cpu()
        g_acc_sum += spec_acc.sum(dim=0).cpu()
        sel_nll_sum += float(out["L_selected"].sum())
        top2_nll_sum += float(out["L_top2"].sum())     # EXACT (no approximation)
        oracle_nll_sum += float(out["L_oracle"].sum())
        # CI top-1 accuracy
        sel_acc_sum += float(spec_acc.gather(1, top1.unsqueeze(1)).squeeze(1).sum())

        best_path = spec_nll.argmin(dim=1)
        for b in range(B):
            k = int(top1[b])
            usage[k] += 1
            if int(best_path[b]) != k:
                missed[k] += 1
            if joint is not None:
                joint[src_idx[eval_labels[i + b]], k] += 1
        n_seen += B

    n = max(n_seen, 1)
    global_ppl = [_ppl(g_nll_sum[k] / n) for k in range(n_spec)]
    global_acc = [float(g_acc_sum[k] / n) for k in range(n_spec)]
    miss_pct = [(100.0 * float(missed[k]) / float(usage[k])) if usage[k] > 0 else None
                for k in range(n_spec)]
    best_fixed_path = int(g_nll_sum.argmin())

    # ---- PPL by prefix-length bucket (variable-prefix diagnostic) ----
    # For each bucket we route+score the whole eval set at a representative R (bucket
    # midpoint), plus the exact-256 reference. CI-top1 PPL/ACC only (the headline
    # metric). Skipped unless requested and variable-prefix is enabled.
    bucket_ppl = None
    if prefix_buckets and getattr(C, "VARPREFIX_ENABLED", False):
        bucket_ppl = []
        specs = [(f"{lo}-{hi}", (lo + hi) // 2) for (lo, hi) in C.VARPREFIX_EVAL_BUCKETS]
        specs.append(("ref256", int(C.SCORE_FROM)))
        for label, R in specs:
            bnll, bacc, bn = 0.0, 0.0, 0
            for i in range(0, N, batch_size):
                ids = eval_ids[i:i + batch_size].to(device)
                am = attention_mask[i:i + batch_size].to(device) if attention_mask is not None else None
                out = model.fresh_gains(ids, tokenizer=tokenizer, attention_mask=am,
                                        score_from=R)
                t1 = out["selected"]
                bnll += float(out["L_selected"].sum())
                bacc += float(out["spec_acc"].gather(1, t1.unsqueeze(1)).squeeze(1).sum())
                bn += ids.size(0)
            bn = max(bn, 1)
            bucket_ppl.append({"bucket": label, "R": R,
                               "ppl": _ppl(bnll / bn), "acc": bacc / bn, "n": bn})

    return {
        "n": n_seen,
        "selected_ppl": _ppl(sel_nll_sum / n),
        "selected_acc": sel_acc_sum / n,
        "top2_ppl": _ppl(top2_nll_sum / n),           # EXACT CI-top2-best
        "oracle_ppl": _ppl(oracle_nll_sum / n),       # best of P0..P5 (no GENERAL)
        "best_fixed_ppl": _ppl(float(g_nll_sum.min()) / n),
        "best_fixed_path": best_fixed_path,
        "global_ppl": global_ppl,                     # per-path, over ALL docs
        "global_acc": global_acc,                     # per-path, over ALL docs
        "usage": usage.tolist(),
        "missed_pct": miss_pct,
        "bucket_ppl": bucket_ppl,                     # per prefix-length bucket (or None)
        "src_names": src_order,
        "src_path_joint": (joint.tolist() if joint is not None else None),
        "src_path_mi": (_mutual_info(joint) if joint is not None else None),
    }


def print_headline(m, prompt_lines=None, model=None, stats=None):
    print("=" * 70)
    hdr = f"EVAL (fresh, full-vocab, causal routing)  n={m['n']}"
    if stats and "step" in stats:
        hdr = f"EVAL step {stats['step']}" + (f" / {stats['max_steps']}" if stats.get("max_steps") else "")
    print(hdr)
    print("=" * 70)
    if stats:
        _print_run_stats(stats)
        print("-" * 70)
    if prompt_lines:
        print("prompt test (CI-routed; GREEDY + SAMPLED):")
        for r in prompt_lines:
            print(f"  [{r['dom']} ->P{r['route']}]")
            prompt_tail = " ".join(r["prompt"].split()[-2:])
            if r.get("reference") is not None:
                print(f"    REFERENCE: ... {prompt_tail} =>{r['reference']}")
            print(f"    GREEDY   : ... {prompt_tail} =>{r['greedy']}")
            print(f"    SAMPLED  : ... {prompt_tail} =>{r['sampled']}")
        print("-" * 70)
    print("LM QUALITY")
    print(f"  CI-top1     : PPL {m['selected_ppl']:.3f}   ACC {100*m['selected_acc']:.2f}%")
    print(f"  CI-top2-best: PPL {m['top2_ppl']:.3f}   (exact — better of CI rank-1/2)")
    print(f"  oracle      : PPL {m['oracle_ppl']:.3f}   (best of P0..P5)")
    print(f"  best-fixed  : PPL {m['best_fixed_ppl']:.3f}   (always P{m['best_fixed_path']})")
    if m.get("bucket_ppl"):
        print("-" * 70)
        print("PPL BY PREFIX LENGTH  (CI-top1; ref256 = exact-256 reference)")
        print(f"  {'bucket':10}{'R':>5}{'PPL':>12}{'ACC':>10}{'n':>7}")
        for b in m["bucket_ppl"]:
            print(f"  {b['bucket']:10}{b['R']:>5}{b['ppl']:>12.3f}"
                  f"{100*b['acc']:>9.2f}%{b['n']:>7}")
    print("-" * 70)
    print("PATH QUALITY")
    print(f"  {'path':6}{'PPL(global)':>13}{'ACC(global)':>13}{'usage':>8}{'%missed':>10}")
    dead_paths = []
    for k in range(len(m["global_ppl"])):
        miss = m["missed_pct"][k]
        miss_s = f"{miss:.1f}%" if miss is not None else "  -- "
        if int(m["usage"][k]) == 0:
            dead_paths.append(k)
        print(f"  P{k:<5}{m['global_ppl'][k]:>13.3f}"
              f"{100*m['global_acc'][k]:>12.2f}%{int(m['usage'][k]):>8}{miss_s:>10}")
    print("-" * 70)
    total_missed = sum((m["missed_pct"][k] or 0) * (m["usage"][k] or 0)
                       for k in range(len(m["global_ppl"])))
    total_use = sum(m["usage"]) or 1
    overall_miss = total_missed / total_use
    print(f"  dead paths (0 CI-top1 traffic): "
          + (", ".join(f"P{k}" for k in dead_paths) if dead_paths else "none"))
    print(f"  routing regret: {overall_miss:.1f}% of docs routed to a non-PPL-best path")
    if overall_miss > 25:
        print("  [!] high routing regret — the Content Index may need an EM refit.")
    gap = m["selected_ppl"] - m["oracle_ppl"]
    if gap > 0.5:
        print(f"  [i] CI-top1 - oracle = {gap:.2f} PPL — headroom a refit could capture.")
    # ---- design-note item 4: source -> path heatmap + MI (diagnostic only) ----
    if m.get("src_path_joint") is not None:
        print("-" * 70)
        print_source_path(m)
    # ---- design-note items 5 & 6: MoE + HyperNet diagnostics ----
    if model is not None:
        print("-" * 70)
        print_moe_usage(model)
        print("-" * 70)
        print_hypernet_centroids(model)
    print("=" * 70)


def _print_run_stats(stats):
    """Timing / throughput / VRAM header (design note item 4)."""
    def g(k, d="..."): return stats.get(k, d)
    if "tokens_trained" in stats:
        print(f"  tokens trained : {g('tokens_trained')}")
    if "step_per_min" in stats:
        print(f"  train speed    : {g('step_per_min'):.1f} step/min")
    if "tokens_per_sec" in stats:
        print(f"  token speed    : {g('tokens_per_sec'):.0f} tokens/sec")
    if "elapsed" in stats:
        print(f"  elapsed        : {g('elapsed')}")
    if "eval_time" in stats:
        print(f"  eval time      : {g('eval_time')}")
    peak = _peak_vram_gb()
    if peak is not None:
        print(f"  peak VRAM      : {peak:.2f} GB")


def _peak_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None


def print_source_path(m):
    """Source -> CI-top1 path heatmap (row-normalized) + MI. Diagnostic only —
    sources are NOT path targets (design note item 4)."""
    names = m["src_names"]; joint = m["src_path_joint"]
    n_spec = len(joint[0]) if joint else 0
    print("SOURCE -> PATH  (row-normalized; diagnostic, not a target)")
    header = " " * 18 + "".join(f"  P{p}  " for p in range(n_spec))
    print(header)
    for si, name in enumerate(names):
        row = joint[si]
        tot = sum(row) or 1.0
        short = str(name).replace("RedPajama", "")[:16].ljust(16)
        cells = "".join(f" {row[p]/tot:.2f} " for p in range(n_spec))
        print(f"  {short}{cells}")
    mi = m.get("src_path_mi")
    if mi is not None:
        print(f"  source/path MI = {mi:.4f} nats")


def print_moe_usage(model):
    """Slot-level soft-MoE usage (design note item 6): E0..E4 kept separate, gate
    mass + top-1 share + gate entropy + dead experts."""
    if not hasattr(model, "get_moe_slot_usage"):
        return
    u = model.get_moe_slot_usage()
    print("Soft-MoE usage — training")
    print(f"  {'':4}{'expert':10}{'gate mass':>12}{'top1 share':>12}")
    for e in u["experts"]:
        print(f"  E{e['slot']}  {e['kind']:10}{e['gate_mass_pct']:>11.1f}%{e['top1_pct']:>11.1f}%")
    dead = u["dead_experts"]
    print(f"  gate entropy = {u['gate_entropy']:.3f}   "
          f"dead experts = {('E'+',E'.join(map(str, dead))) if dead else 'none'}")


def print_hypernet_centroids(model):
    """Per-HyperNet centroid usage (design note item 5), from train_usage_lifetime."""
    if not hasattr(model, "get_hypernet_centroid_diag"):
        return
    rep = model.get_hypernet_centroid_diag()
    print("HyperNet centroid usage — training lifetime")
    for name, r in rep.items():
        top = "  ".join(f"C{ci}={pct:.1f}%" for ci, pct in r["top5"][:5])
        print(f"  {name:8} {r['used']}/{r['total']} used  dead={r['dead']}  "
              f"eff={r['eff']:.1f}  H={r['entropy']:.2f}  gain={r['gain']:.3f}")
        if top:
            print(f"           top: {top}")
