"""
content_trainer_config.py — Round-level Content Index TRAINER configuration.

Kept SEPARATE from content_index_config.py on purpose. That file configures the
FROZEN runtime artifact (design §3, §26, §30). THIS file configures the offline
trainer that reads measured PPL and produces the NEXT artifact (design §4 loop,
§13). The runtime never imports this; only content_index_trainer.py does.

The trainer implements the co-training UPDATE half of the loop (design §4):

    prior artifact + ppl_round1.json (measured per-specialist NLLs, NO GENERAL)
      -> soft PPL responsibilities q_d(x)               (§13)
      -> contested-only relabeling vs carried targets    (§18)
      -> responsibility-weighted centroid M-step         (§13, §22)
      -> multi-prototype specialists                     (§22, §23)
      -> dead-prototype detection + PPL-aware rejuvenation (§24, §25, §26)
      -> anti-oscillation damping                        (§13 "hard labels oscillate")
      -> freeze + save the NEXT artifact                 (§26, §30)

Everything is a deliberate, documented knob. No hidden heuristics.
"""

# ======================================================================
# SOFT PPL RESPONSIBILITIES (design §13)   q_k(x) ∝ exp(-L_k(x) / TAU)
# ======================================================================
# NO GENERAL: the softmax is over the K specialists only (6 paths). TAU controls
# how peaked the responsibilities are: small TAU -> near-hard best-path (oscillates),
# large TAU -> near-uniform (no learning signal). TAU is calibrated to reproduce a
# moderately peaked target from the measured NLL spread.
TAU                 = None       # None -> auto-calibrate from measured NLL std (below)
TAU_AUTO_TARGET_TOP1 = 0.55      # pick TAU so the MEDIAN context's top-1 q ≈ this
TAU_MIN             = 0.02       # calibration clamps
TAU_MAX             = 1.00
# RESPONSIBILITY_MODE — DEAD KNOB (kept as a tombstone so imports don't break).
# With GENERAL removed there is no gain baseline; responsibilities are always the
# raw-NLL softmax q_k ∝ exp(-L_k/τ). The trainer no longer reads this value.
RESPONSIBILITY_MODE = "raw_nll"  # inert; raw is the only mode now

# ======================================================================
# CONTESTED-ONLY RELABELING (design §18)
# ======================================================================
# Contexts whose routing is NOT contested keep their carried soft targets for this
# round; only contested ones are re-derived from fresh PPL. A context is contested
# if ANY of these fire (design §18 bullet list):
CONTEST_MARGIN_MAX      = 0.05   # low Content Index top1/top2 margin (CI geometry)
CONTEST_GENERAL_TIE_MAX = 0.05   # DEAD KNOB (no GENERAL) — trainer no longer reads it
# Ratified D3 — replaces the old delta_cost/near_dcost band. A context is contested
# when the LM is nearly INDIFFERENT between its two best specialists, i.e. its PPL
# top1/top2 margin (L_2nd_best − L_best, in nats) is small. The threshold is
# CALIBRATED per round: the 15th percentile of this round's PPL margins, CAPPED at
# 0.05 nat. Adaptive so it tracks the LM sharpening across rounds; capped so we
# never call a context "indifferent" when its two best paths actually differ by
# more than 0.05 nat (above that, routing genuinely matters — don't relabel it).
CONTEST_PPL_MARGIN_QUANTILE = 0.15   # contest the smallest-margin 15% of contexts...
CONTEST_PPL_MARGIN_CAP      = 0.05   # ...but never above this absolute margin (nats)
CONTEST_ROUTE_CHANGED   = True   # route differs from prior artifact's route
CONTEST_TEACHER_DISAGREE = 0.15  # |q_new_top1 - q_prev_top1| above this = disagreement
# If there is no prior target file (round 1 -> round 2 bootstrap), EVERY context is
# treated as contested (nothing to carry). After that, stable contexts are skipped.
FIRST_ROUND_ALL_CONTESTED = True

# ======================================================================
# CENTROID / PROTOTYPE M-STEP (design §13, §22, §23)
# ======================================================================
# Responsibility-weighted refit: for specialist k, each context x contributes with
# weight q_k(x) to k's prototypes. Within a specialist, a context is routed to its
# NEAREST prototype (max cos), and that prototype's new position is the
# responsibility-weighted mean of the contexts nearest to it (soft k-means M-step,
# multi-prototype). NO GENERAL: responsibilities sum to 1 over the K specialists,
# so all mass forms specialist prototypes (no fallback destination to dilute pull).
PROTOTYPES_PER_SPECIALIST = 4    # §22 example: 6 specialists × 4 prototypes = 24
MSTEP_ITERS               = 10   # inner Lloyd iterations of the weighted M-step
MSTEP_MIN_WEIGHT          = 1e-6 # ignore contexts with negligible responsibility

# ======================================================================
# ANTI-OSCILLATION DAMPING (design §13 "hard assignment can oscillate")
# ======================================================================
# New centroids are blended with the prior artifact's centroids so a single noisy
# PPL round can't yank the partition. c_new = (1-DAMP)*c_fit + DAMP*c_prior.
# DAMP=0 -> full trust in this round; DAMP~0.3 -> conservative. Rejuvenated
# prototypes (below) are EXEMPT from damping — they must move to their new region.
CENTROID_DAMP = 0.30

# ======================================================================
# DEAD-PROTOTYPE DETECTION + REJUVENATION (design §23, §24, §25, §26)
# ======================================================================
# A prototype is a DEAD CANDIDATE if its usage share (fraction of its specialist's
# assigned mass that routes to it) stays below DEAD_USAGE_FRAC over the round's
# assignment window (§23 "99.7/0.2/0.1/0" is death; §24 "usage below threshold").
DEAD_USAGE_FRAC   = 0.02         # < 2% of the specialist's mass = dead candidate
# PPL-AWARE rejuvenation target (§25): pick the context maximizing
#   R(x) = Δ_k(x) * (1 - max_m cos(z(x), c_{k,m}))
# i.e. specialist k demonstrably helps there (positive gain) AND its prototypes
# cover it poorly. We only consider contexts with Δ_k(x) > REJUV_MIN_GAIN.
REJUV_ENABLED     = True
REJUV_MIN_GAIN    = 0.02         # context must show at least this specialist gain
REJUV_MAX_PER_SPECIALIST = 1     # rejuvenate at most this many dead protos/round/spec
# GRACE PERIOD (§26): a freshly rejuvenated prototype cannot be re-killed for this
# many rounds. Tracked in the artifact meta as a per-prototype age counter.
REJUV_GRACE_ROUNDS = 1

# ======================================================================
# THRESHOLD RE-CALIBRATION AFTER UPDATE (design §14, §15)
# ======================================================================
# After centroids move, the old fallback/switch thresholds no longer match the new
# geometry. Re-derive them from the SAME contexts using the build script's method
# (quantiles of margin / top1). We reuse content_index_config's quantiles so the
# calibration is identical to the initial build.
RECALIBRATE_THRESHOLDS = True

# ======================================================================
# ROUND BOOKKEEPING / ARTIFACT (design §26, §30)
# ======================================================================
# The trainer bumps the round counter and writes a new versioned artifact so LM
# checkpoints can pin exactly which Content Index they trained against (§26).
NEXT_VERSION_SUFFIX = None       # None -> auto "-rN" appended to the base version id
# where the new artifact goes; None -> sibling of the input artifact named
# content_index.rN.pt
OUTPUT_ARTIFACT     = None
# also emit the derived soft targets so the NEXT round can do contested-only carry
SOFT_TARGETS_OUT    = None       # None -> sibling "soft_targets.rN.json"


def resolve():
    return {k: globals()[k] for k in globals()
            if k.isupper() and not k.startswith("_")}
