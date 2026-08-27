"""
train.py — Project B v3.1 training loop: FROZEN Content Index routing, no GENERAL.

Per optimizer step (GRAD_ACCUM micro-batches of MICRO_BATCH samples each):
  1. PER-SAMPLE ROUTE: for EACH sample, decode its causal prefix (first
     CI_ROUTE_PREFIX tokens) and ask the frozen Content Index for its TOP-N ranked
     SPECIALIST ids (P0..P5). There is no GENERAL destination. Every sample gets
     its OWN route.
  2. GROUP-BY-DESTINATION FORWARD: the trunk runs ONCE over the batch; then each
     active specialist runs ONCE over the rows routed to it. Each specialist output
     is scored directly as the LM continuation (top-k sampled CE if active) — the
     the CI top-2 specialists are scored SEPARATELY, not fused; rank-1 is the
     primary LM path and rank-2 is the conditional rescue candidate.
  3. LOSS: CI rank-1 is the primary LM path. CI rank-2 is always forwarded/scored
   but receives LM-loss weight only as a conditional specialist rescue when its
   recent top-1 usage is below its dynamic floor and its NLL is competitive with
   rank-1. For rescued samples:
       L_sample = (L_rank1 + PATH_RESCUE_WEIGHT * L_rank2)
                  / (1 + PATH_RESCUE_WEIGHT)
   otherwise:
       L_sample = L_rank1
   If PATH_RESCUE_ENABLED=False, the legacy mean(L_rank1, L_rank2) objective is
   used. HyperNet/MoE balance losses remain separate. No L_general, no lambda_G,
   no fusion alpha.
  4. ASSERTION: destination assignments == effective_batch * CI_TRAIN_TOPN
     (e.g. MICRO_BATCH 8 * GRAD_ACCUM 8 * top-2 = 128), catching the per-sample
     routing-bug class.
  5. REJUVENATION: early/mid only, HyperNet centroids.

No learned selector, no replay queue. Eval (every EVAL_EVERY): prompt test +
full-vocab PPL/ACC + per-path table. HEADLINE routing/PPL are full-vocab even when
training uses top-k.
"""
import signal
import math
import torch

import config as C
import data as D
import checkpoint as CK
from model import GeneralistSpecialistModel, build_gpt2_config
from content_index import ContentIndex


class _SinglePathRouter:
    """Trivial constant router used when C.SINGLE_PATH is on: there is exactly one
    specialist (P0), so every input 'routes' to it and NO Content Index artifact is
    loaded. Implements just the surface the LM/eval touch:
        .K, .version_id, .assert_frozen(), .attach_to_optimizer_guard(opt)
        .encoder.embed_text(text) -> z ;  .score_z(z) -> (_c,_s1,_s2,_m, sims)
    sims is length-K (=1) constant, so every argsort ranks P0 first. It holds NO
    parameters, so the optimizer guard is trivially satisfied.
    """
    class _Encoder:
        def embed_text(self, text):
            return None                       # z is unused; score_z ignores it

    def __init__(self, k):
        self.K = int(k)
        self.version_id = "SINGLE_PATH-no-index"
        self.encoder = self._Encoder()

    def assert_frozen(self):
        return True

    def attach_to_optimizer_guard(self, optimizer):
        return True                            # no CI tensors exist to leak

    def score_z(self, z):
        sims = torch.zeros(self.K)             # constant -> argsort ranks P0..P{K-1}
        return None, None, None, None, sims

_STOP = {"requested": False}


def _install_sigint():
    def handler(signum, frame):
        _STOP["requested"] = True
        print("\n[interrupt] will stop-and-save at next completed optimizer step.")
    signal.signal(signal.SIGINT, handler)


def _fmt_hms(secs):
    """Format seconds as Hh Mm Ss (or Mm Ss / Ss) for readable timing lines."""
    secs = max(0.0, float(secs))
    h = int(secs // 3600); m = int((secs % 3600) // 60); s = int(secs % 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def _prefix_coverage_line(hist):
    """One-line coverage report of sampled prefix lengths R since the last eval:
    mean/min/max plus counts per configured bucket (design: verify coverage)."""
    if not hist:
        return "[prefix] (variable prefix off — fixed at SCORE_FROM)"
    import statistics as _st
    counts = {}
    for R in hist:
        b = C.prefix_bucket(R)
        counts[b] = counts.get(b, 0) + 1
    order = [f"{lo}-{hi}" for (lo, hi) in C.VARPREFIX_EVAL_BUCKETS] + ["ref256", "other"]
    parts = [f"{b}:{counts[b]}" for b in order if counts.get(b)]
    return (f"[prefix] n={len(hist)} mean={_st.mean(hist):.0f} "
            f"min={min(hist)} max={max(hist)} | " + "  ".join(parts))


def _dtype():
    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}[C.DTYPE]


def _autocast(device):
    """Mixed-precision context for the LM forwards. Honors DTYPE: bf16/fp16 enable
    CUDA autocast; fp32 (or CPU) is a no-op. BF16 needs no GradScaler; FP16 would —
    we use BF16 by default (DTYPE='bf16'), so no scaler is required."""
    import contextlib
    dt = _dtype()
    if device == "cuda" and dt in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return contextlib.nullcontext()


def _build_optimizer(model):
    # TRUNK_LR_MULT == 1.0 -> original 2-group optimizer (decay / no_decay). This
    # keeps checkpoints saved before the trunk-LR feature loadable: same group count.
    # TRUNK_LR_MULT != 1.0 -> 4 groups ({base, specialist} x {decay, no_decay}) so
    # the base (trunk) can train at a different LR. Switching the mult changes the
    # group count, so the FIRST resume across that switch rebuilds optimizer state
    # fresh (checkpoint.load_checkpoint handles the mismatch gracefully).
    if C.TRUNK_LR_MULT == 1.0:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": C.WEIGHT_DECAY},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=C.LEARNING_RATE, betas=(0.9, 0.95))

    # Split base (trunk) from specialists. "Base" is scoped to base_reference_params()
    # — the SAME set the freeze mechanism covers — so reducing and freezing the trunk
    # act on identical parameters. Base groups carry lr = LEARNING_RATE * TRUNK_LR_MULT.
    base_ids = {id(p) for _n, p in model.base_reference_params()}
    base_decay, base_no_decay, spec_decay, spec_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_base = id(p) in base_ids
        if p.ndim < 2:
            (base_no_decay if is_base else spec_no_decay).append(p)
        else:
            (base_decay if is_base else spec_decay).append(p)
    base_lr = C.LEARNING_RATE * C.TRUNK_LR_MULT
    groups = [
        {"params": spec_decay,    "weight_decay": C.WEIGHT_DECAY, "lr": C.LEARNING_RATE},
        {"params": spec_no_decay, "weight_decay": 0.0,            "lr": C.LEARNING_RATE},
        {"params": base_decay,    "weight_decay": C.WEIGHT_DECAY, "lr": base_lr},
        {"params": base_no_decay, "weight_decay": 0.0,            "lr": base_lr},
    ]
    # Drop empty groups (e.g. after a hard freeze the base groups are empty) so the
    # scheduler's per-group lr list stays aligned with non-empty groups only.
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=C.LEARNING_RATE, betas=(0.9, 0.95))


def _build_scheduler(opt):
    def lr_lambda(step):
        if step < C.WARMUP_STEPS:
            return step / max(1, C.WARMUP_STEPS)
        prog = (step - C.WARMUP_STEPS) / max(
            1, C.LR_SCHEDULE_STEPS - C.WARMUP_STEPS
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def kd_lambda(step):
    """KD weight schedule for Run 2: linear decay TEACHER_KD_WEIGHT_START ->
    TEACHER_KD_WEIGHT_END across [0, MAX_STEPS]. Returns 0.0 when the teacher is
    disabled so Run 1 is pure CE by construction."""
    if not getattr(C, "TEACHER_ENABLED", False):
        return 0.0
    prog = min(max(step / max(1, C.MAX_STEPS), 0.0), 1.0)
    a, b = C.TEACHER_KD_WEIGHT_START, C.TEACHER_KD_WEIGHT_END
    return a + (b - a) * prog


def _decode_route_prefix(tokenizer, ids_row, R=None):
    """Decode ONLY the causal routing prefix (first R tokens; R defaults to the fixed
    CI_ROUTE_PREFIX) for the router. CRITICAL (causal correctness): the LM scores
    targets from position R onward, so the router must NOT see any token at or beyond
    R — otherwise the routing decision leaks the future of the very tokens the chosen
    path then predicts. Routing on ids[:R] keeps the router strictly inside the
    un-scored prefix. R is the per-micro-batch variable boundary in variable-prefix
    training (config.sample_prefix); with variable prefix off it is CI_ROUTE_PREFIX."""
    r = int(C.CI_ROUTE_PREFIX if R is None else R)
    pref = ids_row[:r]
    return tokenizer.decode(pref.tolist(),
                            skip_special_tokens=C.CI_DECODE_SKIP_SPECIAL)


def _route_topn(model, z_or_text):
    """Frozen CI -> up to CI_TRAIN_TOPN DISTINCT SPECIALIST destination ids (P0..P5).

    v3.1: there is no GENERAL destination. The CI ranks the six specialists by
    descending cosine similarity and this returns the top-N specialist ids. The two
    returned paths are later scored SEPARATELY and their losses averaged (no fusion).

    Built from the CI's own primitives (score_z -> sims[K]). Deterministic; reads
    nothing from LM loss (design §6/§10/§12).

    `z_or_text` may be a precomputed z or the (already prefix-truncated) routing text."""
    ci = model.content_index
    topn = C.CI_TRAIN_TOPN
    if isinstance(z_or_text, torch.Tensor):
        z = z_or_text
    else:
        z = ci.encoder.embed_text(z_or_text)
    _cand, _s1, _s2, _margin, sims = ci.score_z(z)
    order = torch.argsort(sims, descending=True).tolist()      # specialist ids best->worst
    return [int(d) for d in order[:topn]]


# ---- rescue helpers ----
def _new_path_rescue_state():
    """Training-only EMA state.

    Not checkpointed. After a process restart/resume the rescue
    mechanism re-warms for PATH_RESCUE_WARMUP_STEPS.
    """
    return {
        "steps": 0,
        "initialized": False,

        # Actual CI top-1 routing share.
        "usage_ema": torch.full(
            (C.N_SPECIALISTS,),
            1.0 / C.N_SPECIALISTS
        ),

        # For each path:
        # among samples where it was CI rank-2,
        # how often was its NLL competitive with rank-1?
        "competitive_ema": torch.zeros(
            C.N_SPECIALISTS
        ),

        "competitive_seen": torch.zeros(
            C.N_SPECIALISTS,
            dtype=torch.bool
        ),
    }


def _path_rescue_floors(state):
    """Dynamic floor: 3% ... 5% according to competitiveness."""
    q = state["competitive_ema"].clamp(0.0, 1.0)
    lo = float(C.PATH_RESCUE_FLOOR_MIN)
    hi = float(C.PATH_RESCUE_FLOOR_MAX)
    return lo + (hi - lo) * q


def train_one_step(
        model, opt, sched, tokenizer,
        micro_batches, step, device,
        rescue_state):
    topk = C.active_vocab_topk(step)          # 0 (full) until TOPK_START_STEP

    opt.zero_grad(set_to_none=True)

    agg = {
        "lm_final": 0.0,
        "kd": 0.0,
        "kd_lambda": 0.0,
        "lm_blend": 0.0,
        "balance": 0.0,
        "spec_balance": 0.0,
        "moe_balance": 0.0,
        "n_mb": 0,
    }
    dest_counts = torch.zeros(C.N_SPECIALISTS)
    total_route_calls = 0        # must equal effective batch (one route per sample)
    total_assignments = 0        # must equal effective_batch * CI_TRAIN_TOPN
    prefix_samples = []          # sampled R per micro-batch (coverage logging)

    # Rescue usage is TOP-1 usage because inference uses top-1.
    # dest_counts below still records total top-2 training exposure.
    rank1_counts = torch.zeros(C.N_SPECIALISTS)

    # How often each path occurs as rank-2, and how often it is competitive.
    runnerup_opp = torch.zeros(C.N_SPECIALISTS)
    runnerup_comp = torch.zeros(C.N_SPECIALISTS)

    rescue_count = 0

    for ids, srcs in micro_batches:
        ids = ids.to(device)
        am = None
        B = ids.size(0)

        # ---- VARIABLE CAUSAL PREFIX: one R per micro-batch. The CI routes on
        # tokens[:R] (never sees token R or later); the LM scores targets from
        # position R to the end. Routing is chosen ONCE here and held fixed for the
        # whole sequence (no rerouting during the continuation). ----
        R = C.sample_prefix()
        prefix_samples.append(R)

        # ---- PER-SAMPLE routing: each of the B rows gets its OWN CI route from its
        # OWN tokens[:R] prefix (not row 0's). ----
        per_sample_dests = []
        for b in range(B):
            text = _decode_route_prefix(tokenizer, ids[b], R=R)
            dests = _route_topn(model, text)      # specialist ids only (no GENERAL)
            if len(dests) != C.CI_TRAIN_TOPN:
                raise RuntimeError(
                    f"routing expects exactly CI top-{C.CI_TRAIN_TOPN}, "
                    f"got {len(dests)} destinations"
                )

            per_sample_dests.append(dests)

            # Only rank-1 counts as actual routing usage.
            rank1_counts[int(dests[0])] += 1

            total_route_calls += 1
            total_assignments += len(dests)
            for d in dests:
                dest_counts[int(d)] += 1

        # ---- group-by-destination forward: each specialist scored on its own
        # output (top-k CE if active); top-N specialists scored separately.
        # score_from=R makes the LM loss start at the same R the router saw. ----
        with _autocast(device):
            out = model.training_forward(
                ids, am, per_sample_dests=per_sample_dests, topk=topk, score_from=R)
            if not C.PATH_RESCUE_ENABLED:
                # Exact old behavior:
                # mean(rank1 loss, rank2 loss).
                L_final = out["L_final"]
            else:
                rank_nll = out["rank_nll"]

                if (rank_nll.ndim != 2
                        or rank_nll.size(0) != B
                        or rank_nll.size(1) != 2):
                    raise RuntimeError(
                        f"rank_nll must be [B,2], "
                        f"got {tuple(rank_nll.shape)}"
                    )

                # CI rank-1 is ALWAYS the main path.
                L1 = rank_nll[:, 0]

                # CI rank-2 is ONLY a rescue candidate.
                L2 = rank_nll[:, 1]

                # No gradient through the rescue decision itself.
                dnll = (
                    L2.detach() - L1.detach()
                ).float().cpu()

                weights = torch.zeros_like(L1)

                floors = _path_rescue_floors(rescue_state)
                usage = rescue_state["usage_ema"]

                # We need a little history before trusting the EMA.
                # Also require full-vocab training loss for an apples-to-apples
                # rank1/rank2 comparison.
                rescue_active = (
                    rescue_state["initialized"]
                    and rescue_state["steps"]
                        >= C.PATH_RESCUE_WARMUP_STEPS
                    and topk == 0
                )

                for b in range(B):
                    p2 = int(per_sample_dests[b][1])

                    runnerup_opp[p2] += 1

                    # Negative dNLL is allowed:
                    # if rank-2 is actually BETTER than rank-1,
                    # it is certainly competitive.
                    competitive = (
                        float(dnll[b])
                        <= float(C.PATH_RESCUE_DNLL_MAX)
                    )

                    if competitive:
                        runnerup_comp[p2] += 1

                    if (
                        rescue_active
                        and competitive
                        and float(usage[p2]) < float(floors[p2])
                    ):
                        weights[b] = float(
                            C.PATH_RESCUE_WEIGHT
                        )
                        rescue_count += 1

                # Rank-1 owns the sample.
                # Rank-2 gets only a small normalized auxiliary share.
                #
                # w=0.10 gives approximately:
                #     rank1 90.9%
                #     rank2  9.1%
                #
                # w=0 gives pure rank-1 training.
                L_final = (
                    (L1 + weights * L2)
                    / (1.0 + weights)
                ).mean()

            hn_balance = out["trunk_balance"] + out["spec_balance"]   # v3.1: no general
            moe_balance = out["moe_balance"]

            # ---- KD blending (Run 2) ----
            if getattr(C, "TEACHER_ENABLED", False):
                kd = out["kd_loss"]
                lam = kd_lambda(step)
                L_lm = (1.0 - lam) * L_final + lam * kd
            else:
                kd = L_final.new_zeros(())
                lam = 0.0
                L_lm = L_final

            loss = (
                L_lm
                + C.HYPERNET_BALANCE_COEF * hn_balance
                + C.MOE_BALANCE_COEF * moe_balance
            )

        (loss / len(micro_batches)).backward()

        agg["lm_final"] += float(L_final.detach())
        agg["kd"] += float(kd.detach())
        agg["kd_lambda"] += float(lam)
        agg["lm_blend"] += float(L_lm.detach())
        agg["balance"] += float(hn_balance if isinstance(hn_balance, float) else hn_balance.detach())
        agg["spec_balance"] += float(out["spec_balance"] if isinstance(out["spec_balance"], float)
                                     else out["spec_balance"].detach())
        agg["moe_balance"] += float(moe_balance if isinstance(moe_balance, float)
                                    else moe_balance.detach())
        agg["n_mb"] += 1

    # ---- routing-correctness assertion (catches the #1 bug class) ----
    # Effective batch = MICRO_BATCH * GRAD_ACCUM. One route per sample, CI_TRAIN_TOPN
    # assignments each. With MB=8, GA=8, top-2: 64 route calls, 128 assignments.
    eff_batch = C.MICRO_BATCH * C.GRAD_ACCUM
    expected_assignments = eff_batch * C.CI_TRAIN_TOPN
    assert total_route_calls == eff_batch, (
        f"[routing bug] {total_route_calls} route calls != effective batch "
        f"{eff_batch} — every sample must get its OWN CI route, not row 0's.")
    assert total_assignments == expected_assignments, (
        f"[routing bug] {total_assignments} destination assignments != "
        f"{expected_assignments} (eff_batch {eff_batch} x top-{C.CI_TRAIN_TOPN}).")

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], C.GRAD_CLIP)
    opt.step()
    sched.step()

    # ---- update rescue EMA once per optimizer step ----
    if C.PATH_RESCUE_ENABLED:
        decay = float(C.PATH_RESCUE_EMA_DECAY)

        # Rank-1 routing distribution over the whole effective batch.
        step_share = (
            rank1_counts
            / max(float(total_route_calls), 1.0)
        )

        if not rescue_state["initialized"]:
            rescue_state["usage_ema"].copy_(step_share)
            rescue_state["initialized"] = True
        else:
            rescue_state["usage_ema"].mul_(decay).add_(
                step_share,
                alpha=1.0 - decay
            )

        # Update competitiveness only when the path actually
        # appeared as CI rank-2 during this optimizer step.
        for p in range(C.N_SPECIALISTS):
            if runnerup_opp[p] > 0:
                rate = (
                    runnerup_comp[p]
                    / runnerup_opp[p]
                )

                if not bool(rescue_state["competitive_seen"][p]):
                    rescue_state["competitive_ema"][p] = rate
                    rescue_state["competitive_seen"][p] = True
                else:
                    rescue_state["competitive_ema"][p] = (
                        decay
                        * rescue_state["competitive_ema"][p]
                        + (1.0 - decay) * rate
                    )

        rescue_state["steps"] += 1

    rejuv_report = {}
    if (C.REJUV_ENABLED and step > C.REJUV_START_STEP
            and step <= C.REJUV_STOP_STEP and step % C.REJUV_EVERY == 0):
        rejuv_report = model.rejuvenate(step, optimizer=opt)
        if rejuv_report:
            print(f"[rejuv step {step}] revived {rejuv_report}")

    n = max(agg["n_mb"], 1)
    metrics = {
        "lm_final": agg["lm_final"] / n,
        "kd": agg["kd"] / n,
        "kd_lambda": agg["kd_lambda"] / n,
        "lm_blend": agg["lm_blend"] / n,
        "balance": agg["balance"] / n,
        "spec_balance": agg["spec_balance"] / n,
        "moe_balance": agg["moe_balance"] / n,
        "topk": topk,
        "dest_counts": dest_counts.tolist(),      # length N_SPECIALISTS (no GENERAL)
        "assignments": total_assignments,
        "rejuv": rejuv_report,
        "prefix_samples": prefix_samples,         # sampled R this step (coverage log)
        "rank1_counts": rank1_counts.tolist(),
        "rescue_count": int(rescue_count),
        "rescue_usage_ema":
            rescue_state["usage_ema"].tolist(),
        "rescue_competitive_ema":
            rescue_state["competitive_ema"].tolist(),
        "rescue_floor":
            _path_rescue_floors(
                rescue_state
            ).tolist(),
    }
    return metrics


def run_headline(model, tokenizer, step, force=False):
    """Full eval: prompt test + full-vocab PPL + per-path table. Base should be
    frozen for a clean headline; warns (not blocks) if not."""
    frozen = C.base_frozen(step)
    if frozen and not force:
        model.assert_reference_frozen()
    import eval as EB
    eval_ids, eval_labels, _audit = D.build_eval_set(tokenizer)
    m = EB.headline_eval(model, eval_ids, eval_labels=eval_labels,
                         base_is_frozen=frozen, tokenizer=tokenizer)
    plines = EB.prompt_test(model, tokenizer, next(model.parameters()).device)
    EB.print_headline(m, prompt_lines=plines, model=model, stats={"step": step})
    return m


def main():
    C.validate()

    # ---- KD startup guards ----
    if getattr(C, "TEACHER_ENABLED", False):
        assert C.MULTIPATH_ENABLED, "KD requires MULTIPATH_ENABLED"
        assert C.FREEZE_TRUNK, "KD requires FREEZE_TRUNK"
        assert C.FREEZE_EMBEDDINGS, "KD requires FREEZE_EMBEDDINGS"
        assert C.FREEZE_FINAL_LN, "KD requires FREEZE_FINAL_LN"
        assert C.FREEZE_LM_HEAD, "KD requires FREEZE_LM_HEAD"
        assert C.CI_TRAIN_TOPN == 1, "KD requires CI_TRAIN_TOPN == 1"
        assert not C.PATH_RESCUE_ENABLED, "KD cannot be used with PATH_RESCUE"
        assert C.VOCAB_TOPK == 0, "KD requires VOCAB_TOPK == 0 (full vocabulary)"

    C.print_resolved()
    _install_sigint()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = D.get_tokenizer()

    if getattr(C, "MULTIPATH_ENABLED", False):
        # Multi-path reduced-width conversion: frozen mature C-Rich shared base +
        # N narrow (PATH_D_MODEL) paths + a frozen 768 teacher specialist. Initial
        # weights come from the C-Rich checkpoint via load_c_rich (NOT the resume
        # path); a fresh multipath run then has no rolling checkpoint yet, so
        # CK.load_checkpoint below returns (0,0) and training starts from here.
        from multipath_model import build_multipath_model
        crich_ckpt = getattr(C, "CRICH_INIT_CKPT", None)
        model, load_rep = build_multipath_model(build_gpt2_config(),
                                                ckpt_path=crich_ckpt, apply_freeze=True,
                                                device=device)
        if load_rep is not None:
            print(f"[multipath] loaded C-Rich base+teacher: {load_rep['loaded']} tensors "
                  f"(missing={len(load_rep['missing_dest'])}, "
                  f"unexpected={len(load_rep['unexpected_source'])})")
        else:
            print("[multipath][WARN] no CRICH_INIT_CKPT set — narrow paths AND shared "
                  "base are randomly initialized (only valid for a shape smoke test).")
        # explicit param-count table (spec item 9)
        print("[multipath] parameter counts:")
        for k, v in model.param_report().items():
            print(f"           {k:34}: {v:,}")
    else:
        model = GeneralistSpecialistModel(build_gpt2_config()).to(device)

    # ---- load + attach the FROZEN Content Index (design §12/§26/§30) ----
    if getattr(C, "SINGLE_PATH", False):
        # No routing, no Content Index artifact: attach a constant P0 router.
        ci = _SinglePathRouter(C.N_SPECIALISTS)
        model.attach_content_index(ci)
        print(f"[single-path] no Content Index — constant router (K={ci.K}); "
              f"every input runs the single path P0 "
              f"(MOE_MODE={C.MOE_MODE}, DISABLE_HYPERNET={C.DISABLE_HYPERNET})")
    else:
        if not C.CONTENT_INDEX_ENABLED:
            raise RuntimeError("CONTENT_INDEX_ENABLED is False — v3 requires the "
                               "frozen Content Index router.")
        ci = ContentIndex.load(C.CONTENT_INDEX_ARTIFACT, device=device)
        if ci.version_id != C.CONTENT_INDEX_VERSION:
            raise RuntimeError(
                f"Content Index version mismatch: artifact '{ci.version_id}' != config "
                f"'{C.CONTENT_INDEX_VERSION}'. The LM must train against the expected "
                f"index (design §26). Rebuild or fix CONTENT_INDEX_VERSION.")
        if ci.K != C.N_SPECIALISTS:
            raise RuntimeError(
                f"Content Index K ({ci.K}) != N_SPECIALISTS ({C.N_SPECIALISTS}). "
                f"A multipath run needs a K={C.N_SPECIALISTS} index; reuse the existing "
                f"K={C.N_SPECIALISTS} artifact or rebuild one before this run.")
        model.attach_content_index(ci)
        print(f"[CI] attached frozen Content Index {ci.version_id} (K={ci.K}); "
              f"no GENERAL/fusion — routes are specialists P0..P{C.N_SPECIALISTS-1} "
              f"(train top-{C.CI_TRAIN_TOPN}, infer top-1)")

    opt = _build_optimizer(model)
    sched = _build_scheduler(opt)
    # STARTUP GUARD: no Content Index tensor may be in the LM optimizer (design §30).
    model.assert_router_out_of_optimizer(opt)
    print("[CI] optimizer guard OK — no Content Index tensor in the LM optimizer.")

    step, consumed = CK.load_checkpoint(model, opt, sched)

    # Training-policy state only.
    # It intentionally re-warms after process restart/resume.
    rescue_state = _new_path_rescue_state()

    if C.PATH_RESCUE_ENABLED:
        print(
            f"[rescue] ON: CI top-2 only; "
            f"rank1 main, rank2 conditional "
            f"w={C.PATH_RESCUE_WEIGHT}; "
            f"dNLL<={C.PATH_RESCUE_DNLL_MAX}; "
            f"dynamic floor="
            f"{100*C.PATH_RESCUE_FLOOR_MIN:.0f}-"
            f"{100*C.PATH_RESCUE_FLOOR_MAX:.0f}%; "
            f"EMA warmup="
            f"{C.PATH_RESCUE_WARMUP_STEPS} steps"
        )

    if C.base_frozen(step):
        model.set_base_frozen(True)
        print(f"[freeze] base frozen on resume (step {step} >= "
              f"{C.FREEZE_BASE_AT_STEP}).")

    data_iter = D.batch_iterator(tokenizer, skip_sequences=consumed)
    print(f"[train] starting at step {step}, consumed {consumed}")

    import time
    t_run_start = time.time()          # wall clock for the whole run
    t_window = t_run_start             # start of the current 10-step print window
    t_eval_window = t_run_start        # start of the current eval window
    step_at_window = step
    step_at_eval = step

    just_froze = False
    prefix_hist = []                    # sampled R values since last eval (coverage)
    while step < C.MAX_STEPS and not _STOP["requested"]:
        if C.base_frozen(step) and not just_froze and any(
                p.requires_grad for p in model.trunk.parameters()):
            model.set_base_frozen(True)
            just_froze = True
            print(f"[freeze] base (trunk) frozen at step {step}. "
                  f"Specialists continue training.")

        micro = []
        last_consumed = consumed
        for _ in range(C.GRAD_ACCUM):
            try:
                ids, srcs, last_consumed, _stream = next(data_iter)
            except StopIteration:
                data_iter = D.batch_iterator(tokenizer, skip_sequences=last_consumed)
                ids, srcs, last_consumed, _stream = next(data_iter)
            micro.append((ids, srcs))

        metrics = train_one_step(
            model,
            opt,
            sched,
            tokenizer,
            micro,
            step,
            device,
            rescue_state,
        )
        consumed = last_consumed
        step += 1
        prefix_hist.extend(metrics.get("prefix_samples", []))

        if step % 10 == 0:
            now = time.time()
            win_steps = step - step_at_window
            win_sps = win_steps / max(now - t_window, 1e-9)     # steps/sec this window
            elapsed = now - t_run_start
            remaining = (C.MAX_STEPS - step) / max(win_sps, 1e-9)
            t_window = now; step_at_window = step
            dc = "/".join(str(int(x)) for x in metrics["dest_counts"])
            pct = 100.0 * step / max(1, C.MAX_STEPS)
            lf_tag = f"Lfinal(topk{metrics['topk']})" if metrics["topk"] else "Lfinal"
            print(f"[step {step}/{C.MAX_STEPS} ({pct:.1f}%)] "
                  f"CE={metrics['lm_final']:.4f} "
                  f"KD={metrics['kd']:.4f} "
                  f"kdλ={metrics['kd_lambda']:.3f} "
                  f"LM={metrics['lm_blend']:.4f} "
                  f"specBal={metrics['spec_balance']:.4f} "
                  f"moeBal={metrics['moe_balance']:.4f} "
                  f"dest=[{dc}] (P0..P5)  assign={metrics['assignments']}  "
                  f"| {60*win_sps:.1f} step/min  "
                  f"ETA {_fmt_hms(remaining)}")

            if C.PATH_RESCUE_ENABLED:
                u = "/".join(
                    f"{100*x:.1f}"
                    for x in metrics["rescue_usage_ema"]
                )

                f = "/".join(
                    f"{100*x:.1f}"
                    for x in metrics["rescue_floor"]
                )

                q = "/".join(
                    f"{100*x:.0f}"
                    for x in metrics[
                        "rescue_competitive_ema"
                    ]
                )

                seen = int(sum(metrics["rank1_counts"]))

                print(
                    f"[rescue] used="
                    f"{metrics['rescue_count']}/{seen} "
                    f"top1EMA%=[{u}] "
                    f"floor%=[{f}] "
                    f"competitive%=[{q}]"
                )

        if step % C.EVAL_EVERY == 0:
            import eval as EB
            now = time.time()
            train_secs = now - t_eval_window          # training time since last eval
            eval_steps = step - step_at_eval
            spm = eval_steps / max(train_secs, 1e-9) * 60
            tokens_trained = consumed * C.MAX_LENGTH
            tok_per_sec = (eval_steps * C.MICRO_BATCH * C.GRAD_ACCUM * C.MAX_LENGTH) \
                / max(train_secs, 1e-9)
            print(f"[timing] {eval_steps} train steps in {_fmt_hms(train_secs)} "
                  f"({spm:.1f} step/min) | total elapsed {_fmt_hms(now - t_run_start)}")
            print(_prefix_coverage_line(prefix_hist))
            prefix_hist = []                          # reset coverage window after eval
            t_eval_start = time.time()
            eval_ids, eval_labels, _audit = D.build_eval_set(tokenizer)
            frozen = C.base_frozen(step)
            m = EB.headline_eval(model, eval_ids, eval_labels=eval_labels,
                                 base_is_frozen=frozen, tokenizer=tokenizer,
                                 prefix_buckets=True)
            plines = EB.prompt_test(model, tokenizer, device)
            stats = {"step": step, "max_steps": C.MAX_STEPS,
                     "tokens_trained": tokens_trained, "step_per_min": spm,
                     "tokens_per_sec": tok_per_sec,
                     "elapsed": _fmt_hms(now - t_run_start),
                     "eval_time": _fmt_hms(time.time() - t_eval_start)}
            EB.print_headline(m, prompt_lines=plines, model=model, stats=stats)
            print(f"[timing] eval took {_fmt_hms(time.time() - t_eval_start)}")
            # reset training-window MoE gate/usage accumulators so the NEXT eval's
            # MoE report reflects only the upcoming window (centroid lifetime and
            # HyperNet train_usage_window are managed by rejuvenation separately).
            if hasattr(model, "reset_expert_usage"):
                model.reset_expert_usage()
            t_eval_window = time.time()               # reset AFTER eval so eval time
            step_at_eval = step                       # isn't counted as train time

        is_ms = step in C.MILESTONE_STEPS
        if step % C.SAVE_EVERY == 0 or is_ms:
            CK.save_checkpoint(model, opt, sched, step, consumed, is_milestone=is_ms)

    CK.save_checkpoint(model, opt, sched, step, consumed, is_milestone=True)
    print("[train] stopped. Running final headline eval...")
    run_headline(model, tokenizer, step, force=not C.base_frozen(step))


if __name__ == "__main__":
    main()