"""
config.py — Project B v3.1: Shared Trunk + Frozen Content Index + Six Specialists.

ARCHITECTURE (design note v3.1, FIXED):
    input -> 8-layer shared TRUNK (HyperNet A[0:4] off / B[4:8] on) -> h_trunk
          -> Frozen Content Index -> one or more ranked specialist ids (P0..P5)
          -> selected specialist (4 layers + its own HyperNet) -> h_p
          -> LN_f -> shared LM head -> logits

There is NO GENERAL path, NO fusion, and NO alpha. The selected specialist is a
COMPLETE upper continuation of h_trunk (design note §2/§3).

Key v3.1 deltas vs v3 (GENERAL-always-fused):
  * GENERAL branch, general HyperNet, fusion alpha, _fuse, L_general — all removed.
  * The trunk (which every input passes through) carries the common/general
    capability; specialists carry context-specific capability (design note §3).
  * Routing is a FROZEN Content Index (no LM gradient): train top-CI_TRAIN_TOPN
    specialists scored SEPARATELY, per-sample loss = mean(L_Pa, L_Pb); infer top-1.
  * MoE expert pool is the 5-view set: perc2, perc4, perc4, reglu1d, film (§7).
  * Specialist HyperNets ON (one 32-centroid HyperNet per path, §8.2).

Memory regimes (design note §4):
    * INFERENCE : trunk(8) + ONE specialist(4) resident = 12 executed blocks.
    * TRAINING  : trunk + top-2 specialists per sample = up to 16 block executions;
                  the per-path diagnostic sweep measures all six (no grad).

NOTE: a number of v2 selector knobs below (SELECTOR_*, QBAL_*, QUEUE_*, TEACHER_*,
RESP_SCHEDULE/lambda_G, ROUTING_HYPERNET_*) are DEAD in v3.1 — routing is frozen
and external. They are retained only so ancillary modules that still import them
resolve; nothing in the v3.1 train/eval path reads them. They can be deleted once
lgr.py and the old diagnostics are removed.
"""

import os

# ======================================================================
# GLOBALLY-FIXED CONSTANTS — choose once. Keep checkpoints weight-compatible.
# ======================================================================
PROJECT        = "B"
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE     = 50257
D_MODEL        = 768
N_HEADS        = 12
N_POSITIONS    = 1024

# ---- fixed architecture (design note v3.1 §2 / §11 / §19) ----
TRUNK_LAYERS       = 8          # shared trunk
TRUNK_SPLIT_AT     = 4          # HyperNet A = layers [0,4), HyperNet B = layers [4,8)
GENERAL_LAYERS     = 0          # v3.1: GENERAL branch removed entirely
SPECIALIST_LAYERS  = 4          # each specialist path
N_SPECIALISTS      = 6          # P0..P5 (six specialists)
# Depth per token (representation) = TRUNK + one specialist path = 12.
# Inference FLOPs: trunk(8) + one specialist(4) = 12 block executions (no GENERAL).
N_LAYERS_EFFECTIVE = TRUNK_LAYERS + SPECIALIST_LAYERS                        # 12
# block executions when ONE specialist runs (inference, FLOPs-honest)
N_BLOCKS_EXECUTED  = TRUNK_LAYERS + SPECIALIST_LAYERS                        # 12

# ---- v3.1: no fusion ----
# The selected specialist output IS the LM continuation: logits = head(LN_f(h_p)).
# There is no h_general, no alpha, no fusion. (design note §2/§3/§19)

# ---- clustered-hypernet expert sizing (reused mechanism from Project A) ----
# Trunk carries TWO hypernets (A/B); B is ON with the trunk centroid budget.
HYPERNET_CENTROIDS_TRUNK_A = 128
HYPERNET_CENTROIDS_TRUNK_B = 128    # design note §8.1 / §19
HYPERNET_CENTROIDS_SPEC    = 64     # each specialist is one "domain" -> narrower (§8.2)
HYPERNET_TEMPERATURE       = 5.0
HYPERNET_BALANCE_MODE      = "switch"
HYPERNET_BALANCE_COEF      = 0.01
# token-level expert pool inside every HyperNetBlock (design note §7 / §19):
#   E0 perc2 | E1 perc4 | E2 perc4 | E3 reglu1d | E4 film   (5 experts)
OURS_EXPERT_KINDS = ["perc2", "perc4", "perc4", "reglu1d", "film"]

# ---- soft MoE (NeedFix item 1) ----------------------------------------------
# Soft mixture over the experts kept per token. None -> use ALL experts (full
# differentiable softmax mixture over E0..E4, the intended Project B setting).
# Set to an int < len(OURS_EXPERT_KINDS) only to experiment with sparse soft-MoE;
# do NOT set it to 1 (that reproduces the top-1 masking bug this run fixes).
MOE_SOFT_TOPK = len(OURS_EXPERT_KINDS)      # = 5 -> full soft mixture

# ---- MoE gate load-balance (NeedFix item 2) ---------------------------------
# Separate, SMALL coefficient on the switch-style aggregate-soft-mass balance loss.
# Prevents expert collapse/dead experts WITHOUT forcing per-token uniform gates, so
# specialization is still free to emerge. Total objective:
#   L_total = L_LM + HYPERNET_BALANCE_COEF*L_HN_balance + MOE_BALANCE_COEF*L_MoE_balance
MOE_BALANCE_COEF = 0.01

# ---- ABLATION / VARIANT TOGGLES ---------------------------------------------
# SINGLE_PATH: strip routing entirely. One specialist path, no Content Index at
#   all — every input flows trunk -> the single path -> head. When True, train.py
#   attaches a trivial constant router (always P0) instead of loading the frozen
#   CI artifact, so no content_index.py artifact / version match is needed. This
#   is the setting for the three FFN A/B/C tests: they differ ONLY in the FFN
#   (MOE_MODE / DISABLE_HYPERNET), with the path held fixed. Forces
#   N_SPECIALISTS=1 and CI_TRAIN_TOPN=1 below. Keep its own CKPT_DIR.
SINGLE_PATH = False   # Changed to False to allow six specialists
#
# DISABLE_HYPERNET: skip the clustered-hypernet reasoning vector AND its dense
#   reason_proj entirely (block becomes attention + FFN). A TRUE no-hypernet path
#   (no centroid compute, no wide projection). Trainable variant; keep its own
#   CKPT_DIR (not checkpoint-compatible with hypernet-on).
DISABLE_HYPERNET = False
#
# MOE_MODE: which FFN each block uses in place of / as the mixer. This is a
#   first-class ARCHITECTURE choice (trainable, checkpointed) — a run's checkpoints
#   are only compatible with the same MOE_MODE. Use a distinct CKPT_DIR per mode.
#     "multi_expert" : the per-token multi-expert MoE (perc2/perc4/reglu1d/film)
#                      with gating dispatch (the full model).
#     "lowrank"      : a SINGLE uniform low-rank FFN (d -> r -> d), no expert
#                      dispatch. Difference vs multi_expert isolates the per-token
#                      dispatch cost (gating + gather/scatter + many small matmuls).
#     "dense"        : the TRUE dense GPT-2 FFN (d -> 4d -> d, MLP4E), no expert
#                      dispatch and no gating. The real dense baseline (Model A).
MOE_MODE = "multi_expert"          # "multi_expert" | "lowrank" | "dense"
MOE_LOWRANK_R = 256                # bottleneck rank for MOE_MODE="lowrank"
# (legacy DISABLE_MOE removed — use MOE_MODE="lowrank" for the no-dispatch baseline)

# ---- HyperNet placement (design note v3.1 §8 / §19 / §26) -------------------
# Simplified additive-correction HyperNet. Placement controls WHICH blocks get a
# hypernet; blocks without one run MoE-only (no hypernet cost). v3.1 target:
#   trunk 0..SPLIT-1 : none                  (HYPERNET_TRUNK_A = False)
#   trunk SPLIT..end : ONE shared 64-centroid hypernet (HYPERNET_TRUNK_B = True)
#   specialist paths : one independent 32-centroid hypernet PER PATH (HYPERNET_SPEC = True)
HYPERNET_TRUNK_A = False           # trunk layers [0, SPLIT): no hypernet
HYPERNET_TRUNK_B = True            # trunk layers [SPLIT, end): shared hypernet
HYPERNET_SPEC    = True            # each specialist: own 32-centroid hypernet (§8.2)
# GENERAL branch removed in v3.1 — HYPERNET_GENERAL retained (False) only so any
# stale reference resolves; it drives nothing.
HYPERNET_GENERAL = False


GATE_HIDDEN       = 256

# ======================================================================
# SELECTOR (design note §6, v2.0) — ROUTING-HYPERNET prototype classifier
# over the GENERAL representation.
# ======================================================================
N_PROTOTYPES_PER_SPEC = 4       # v2.0: 6 specialists x 4 = 24 prototypes
SELECTOR_QUERIES      = 4        # learned cross-attention queries -> doc index
SELECTOR_PREFIX       = 256      # first <=256 GENERAL hidden states for indexing
# routing HyperNet: produces a strong input-context ROUTING VECTOR before the
# prototype match (reuses ClusteredHypernetwork). Behind a flag so the pilot can
# bisect to plain nearest-prototype routing (index straight to prototypes).
ROUTING_HYPERNET_ENABLED   = True
ROUTING_HYPERNET_CENTROIDS = 128
# v1.1 retained: PURE SUPERVISED selector. Label-free geometry losses OFF.
SELECTOR_LABEL_FREE_LOSSES = False   # keep False — do NOT re-enable clustering/etc.
SELECTOR_CE_COEF      = 1.0     # weight on soft CE / KL(q || selector) — FRESH (live)
# [FIX #9] replay CE is a SEPARATE, explicit coefficient. The queue replays stored
# (index, q) — training routing HyperNet + prototypes + score params, but NOT the
# learned queries / cross-attention context reader / index projection (those only
# see gradient from the live CE, by design — the queue stores the already-computed
# index). Start replay WEAKER than fresh supervision; tune from diagnostics.
REPLAY_CE_COEF        = 0.5
# selector observer-only until specialists differentiate (design note §8)
SELECTOR_START_STEP   = 300
# routing temperature schedule (sample-hard early -> deterministic top-1 late)
SELECTOR_TEMP_START        = 2.0
SELECTOR_TEMP_END          = 0.5
SELECTOR_SAMPLE_UNTIL_STEP = 4000

# ======================================================================
# K3 QUANTILE BALANCING (design note §6, v2.0) — SEPARATE from teacher CE.
# NOT a loss: it is a NO-GRAD exposure correction. accumulate_usage() maintains
# proto/path routing-mass EMAs; refresh_exposure_bias() (once per optimizer step)
# caches a per-specialist additive bias that is NEGATIVE for above-quantile hogs;
# and it is applied ONLY at TRAINING-TIME specialist selection (train._select_subset
# slot 1, once the selector is active) — never in the teacher CE gradient and never
# at inference/headline routing (those use raw learned logits). Ranks routing mass
# and caps the hogs toward the quantile; does NOT force final path usage to 1/6.
# ======================================================================
QBAL_ENABLED          = True
# per-prototype (24) and per-path (6) balance coefficients; proto > path.
# TUNING (review): the bias is added to CLIP-scaled selector logits, whose spread
# grows as logit_scale sharpens, so a small coefficient can be negligible. Watch
# the printed  ratio = max|exposure_bias| / selector_logit_std  every EVAL_EVERY:
# if ratio << 1 the bias does nothing and these must be raised. Start here, tune
# up from the pilot until the bias measurably shifts route shares without swamping
# the learned selector (aim ratio ~0.1-0.3 early, decaying late).
QBAL_PROTO_COEF_START = 0.10
QBAL_PROTO_COEF_END   = 0.02
QBAL_PATH_COEF_START  = 0.03
QBAL_PATH_COEF_END    = 0.005
QBAL_DECAY_STEPS      = 6000
# quantile target: penalize routing mass above this quantile of the usage
# distribution toward it (K3-style: don't equalize, just cap the hogs).
QBAL_QUANTILE         = 0.75
# EMA of routing mass that the quantile ranking is computed over (cross-microbatch
# memory so balance works at micro-batch 1, same rationale as slot_ema).
QBAL_EMA_DECAY        = 0.99

# ======================================================================
# TEACHER (design note §6) — soft specialist-gain target, replay-smoothed.
# v2.0: gain measured through the FUSED head. All specialists read DETACHED
# h_trunk; hG is detached and reused for fusion + the L_general reference.
# ======================================================================
#   hP[p]  = Pp(detach(h_trunk))
#   L_p    = NLL( head( detach(hG) + alpha * hP[p] ) )      (p = 0..5)
#   gain_p = L_general - L_p
#   q_p    = softmax(gain_p / tau) = softmax(-(L_p - min_j L_j)/tau)  (detached)
TEACHER_TAU        = 0.05     # raw-nats temperature; calibrate from gain gaps
# measure-6 / update-subset (design note §8): all six measured for the target,
# only this many specialists receive LM gradients per step (memory bound).
SPEC_GRAD_PATHS    = 1        # specialists updated with gradient per input
# subset selection policy: "round_robin" early (keep all alive), optionally
# "gain_biased" late (after specialization emerges). Team schedule decision.
SUBSET_POLICY      = "round_robin"     # "round_robin" | "gain_biased"
SUBSET_GAIN_BIAS_FROM_STEP = 20000     # switch to gain_biased after this (if enabled)

# ---- replay queue: TRAINING-ONLY teacher smoothing (design note §6, §8, §9) ----
# Absorbs measure-6 / update-subset staleness so the selector is less reactive to
# one-step fluctuations. NEVER used for headline metrics (see EVAL section).
QUEUE_ENABLED   = True
QUEUE_CAP       = 128
QUEUE_TTL_STEPS = 32          # base TTL; scaled down late (see ttl_for_step)
QUEUE_SAMPLE    = 32          # selector replay batch per step
QUEUE_TTL_MIN   = 8           # floor when the base is nearly frozen

# ======================================================================
# TRAINING OBJECTIVE (design note §4, v2.0 — NO gamma)
#   L_total = L_final + lambda_G * L_general + lambda_sel * CE(q, selector)
#             + HYPERNET_BALANCE_COEF * (trunk+general+spec hypernet balance)
#             + K3 quantile balance (selector, separate mechanism)
#   L_general = head(hG) ; L_final = head(hG + alpha * hP_selected)
# There is NO grad-scale hook: nothing feeds GENERAL->specialist. L_final still
# updates the shared TRUNK (both hG and hP depend on h_trunk) — ordinary
# shared-trunk training, not cross-branch contamination.
# ======================================================================
# responsibility annealing (design note §5, v2.0): ONLY lambda_G is scheduled.
# The gamma column is GONE. alpha is learnable, not scheduled.
RESP_SCHEDULE = [
    # (step,   lambda_G)
    (0,        0.50),   # early: build competent shared foundation
    (15000,    0.20),   # middle: shift responsibility to specialists
    (40000,    0.08),   # late: specialists do most correction
]
# late freeze (design note §5/§9): freeze trunk+general (+ln_f, head) after this
# step so the headline L_general reference is fixed. 0 = never auto-freeze.
FREEZE_BASE_AT_STEP = 0               # freeze from step 0 (full base freeze)
# one base LR (design note §4): do NOT use a lower general LR as the control.
LEARNING_RATE = 3e-4
# Trunk (base) LR multiplier, applied on top of LEARNING_RATE and the LR schedule.
#   1.0 = trunk trains at the same LR as specialists (default; existing behavior)
#   <1  = trunk trains slower than specialists (e.g. 0.25)
#   0.0 = trunk contributes no LR — but for a TRUE freeze prefer FREEZE_BASE_AT_STEP
#         (requires_grad=False), which also removes trunk params from the optimizer
#         and its momentum/decay state. TRUNK_LR_MULT only scales the step size.
# Scope = base_reference_params() (trunk blocks + trunk HyperNets + wte/wpe/ln_f/
# lm_head) — exactly what the freeze mechanism covers, so reduce and freeze agree.
# Env override: TRUNK_LR_MULT=0.25 python train.py
TRUNK_LR_MULT = float(os.environ.get("TRUNK_LR_MULT", "1.0"))
WEIGHT_DECAY  = 0.1
GRAD_CLIP     = 1.0
WARMUP_STEPS  = 500
LR_SCHEDULE_STEPS = 100000   # preserve original schedule
MAX_STEPS     = 10000            # KD smoke run (short)
DTYPE         = "bf16"

# ---- Explicit freeze flags (new) ----
# These allow fine‑grained freezing of specific base components.
# When True, the respective parameters are set to requires_grad=False.
# The existing FREEZE_BASE_AT_STEP=0 already freezes trunk, ln_f, and head;
# the flags below are added for clarity and future use.
MULTIPATH_ENABLED = True            # master toggle for multi‑path routing
PATH_D_MODEL = 576                  # per‑specialist hidden dimension (future use)
PATH_N_HEADS = 9                    # per‑specialist number of heads (future use)
FREEZE_TRUNK = True                 # freeze all trunk layers
FREEZE_EMBEDDINGS = True            # freeze token and position embeddings
FREEZE_FINAL_LN = True              # freeze final layer norm
FREEZE_LM_HEAD = True               # freeze LM head

# ---- KD / Teacher (Run 2) settings ----
TEACHER_ENABLED = True              # enable knowledge distillation from frozen 768 teacher
TEACHER_KD_TEMP = 2.0               # temperature for softmax in KD
TEACHER_KD_WEIGHT_START = 0.5       # initial KD weight
TEACHER_KD_WEIGHT_END = 0.0         # final KD weight (linear decay)

# ======================================================================
# CHECKPOINTING — reuse Project A's checkpoint.py verbatim (atomic Drive writes).
# ======================================================================
# New smoke‑run directory (separate from the single‑path run)
DRIVE_DIR = "/content/drive/MyDrive/checkpoints_multipath_576_6path_kd_long"
CKPT_DIR  = DRIVE_DIR + "/checkpoints"

# Initialisation checkpoint from the C‑Rich step‑8200 run (must point to the
# actual step‑8200 checkpoint of the single‑path trunk model)
CRICH_INIT_CKPT = "/content/drive/MyDrive/checkpoints_single_multi_expert_hypernet_rich/checkpoints/ckpt_milestone_step10000.pt"
# ^^^ Replace this with the exact path to the step8200 checkpoint you intend to use.

SAVE_EVERY = 200
KEEP_LAST  = 3
#MILESTONE_STEPS = (100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000)
MILESTONE_STEPS = [
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]


# ======================================================================
# REJUVENATION (design note §6, v2.0) — revive persistently dead routing
# prototypes / HyperNet centroids. Early/mid only; stops late. Never auto-revive
# a whole dead PATH.
# ======================================================================
REJUV_ENABLED       = True
REJUV_START_STEP    = 2000      # [FIX #5] no rejuvenation before a real observation
                                #  window — the first check happens strictly after
                                #  this step (step 0 is never eligible).
REJUV_STOP_STEP     = 40000     # rejuvenation runs early/mid only
REJUV_EVERY         = 200       # cadence (optimizer steps)
REJUV_MIN_EXPOSURE  = 2000      # min summed usage before a hypernet is eligible
REJUV_NOISE_STD     = 0.01
REJUV_DONOR_FRAC    = 2.0 / 3.0 # 2/3 donor-cloned, 1/3 random restart
# routing-prototype rejuvenation: revive from a real routing vector of an input
# where the dead prototype's specialist has HIGH teacher gain (+ small noise).
REJUV_PROTO_ENABLED = True
REJUV_PROTO_MIN_DEAD_STEPS = 2000   # a prototype must be dead this long first

# ======================================================================
# CAUSAL SCORING (design note §7) — selector reads first 256 GENERAL states, so
# LM loss for routing comparison is scored ONLY on target positions >= SCORE_FROM
# to avoid prefix leakage. Same convention/plumbing as Project A's SCORE_FROM.
# ======================================================================
SCORE_FROM = SELECTOR_PREFIX   # first scored TARGET position (0-indexed); also the
                               # exact-256 REFERENCE boundary used by eval/inference
                               # and as the fallback when variable prefix is off.

# ---- VARIABLE CAUSAL ROUTING PREFIX (training only) ------------------------
# Instead of a fixed 256-token prefix, each TRAINING micro-batch samples a boundary
#   R ~ Uniform{VARPREFIX_MIN .. VARPREFIX_MAX}   (inclusive)
# and for that micro-batch:
#   * the CI routes on tokens[:R]  (causal: never sees token R or later)
#   * the LM scores next-token targets from position R through the end (pos 1023)
#   * routing is chosen ONCE from tokens[:R] and held fixed for the whole sequence.
# This trains the LM for variable context lengths; exact 256 (SCORE_FROM) stays the
# clean reference/runtime diagnostic. Inference is UNCHANGED here (still top-1 at the
# fixed prefix) — the inference rule is a separate later decision.
# R is per-MICRO-BATCH (all samples in one micro-batch share R); with GRAD_ACCUM
# micro-batches per step, each optimizer step still sees GRAD_ACCUM distinct R values.
VARPREFIX_ENABLED = True
VARPREFIX_MIN     = 32          # inclusive
VARPREFIX_MAX     = 255         # inclusive (256 reserved as the exact reference)
# eval PPL is additionally bucketed by prefix length so we can see whether the model
# generalizes across context lengths (design: report PPL per bucket).
VARPREFIX_EVAL_BUCKETS = [(32, 63), (64, 95), (96, 127), (128, 191), (192, 255)]
# the exact-256 reference is always evaluated too (labeled "ref256").

# ======================================================================
# DATA — same document-level streamed segmentation as Project A's data.py.
# ======================================================================
DATA_SOURCE   = "hf"
DATASET_NAME  = "DKYoon/SlimPajama-6B"
DATASET_CONFIG = None
DATASET_SPLIT = "train"
TRAIN_FILE    = "/content/drive/MyDrive/Project1/wikitext-103/wiki.train.txt"
VALID_FILE    = "/content/drive/MyDrive/Project1/wikitext-103/wiki.valid.txt"
SINGLE_DOMAIN_LABEL = "wikitext"
# LOCAL_FORMAT selects the local reader when DATA_SOURCE=="local":
#   "wikitext" -> _local_wikitext_docs (heading-delimited .txt, single domain)
#   "jsonl"    -> _local_jsonl_docs   (SlimPajama-style JSONL/dir of shards; keeps
#                 meta.redpajama_set_name so per-domain routing diagnostics work)
# For a local SlimPajama smoke run set DATA_SOURCE="local", LOCAL_FORMAT="jsonl",
# and point TRAIN_FILE/VALID_FILE at the shard file or directory.
LOCAL_FORMAT  = "wikitext"
MAX_DOC_CHARS = 200000
MAX_LENGTH    = 1024
WINDOW_SIZE   = 1
SHUFFLE_SEED  = 1234
SHUFFLE_BUFFER = 10000
EVAL_SAMPLES  = 64

# [FIX #10] Project-A hand-domain balanced-path RESERVATION is REMOVED. Project B
# specialists have NO hand-assigned domains — specialization emerges from measured
# specialist gain — so there is no "correct path per domain" and no reason to skip
# the first 250k sequences of the shuffled stream. Fresh B training starts at the
# stream head (after the eval guard only). We keep a SOURCE-LABELED diagnostic set
# (heatmap / MI / oracle-path shares by source) but never call a path the correct
# target for a domain. The DOMAIN_TO_PATH map and BALANCED_EVAL_* reserve are gone.
N_PATHS = N_SPECIALISTS         # alias: diagnostics count over specialists as "paths"
# how many source-labeled sequences the diagnostic set collects (no reservation —
# it is streamed from a modest skip past the eval guard, not a 50k carve-out).
SELECTOR_DIAG_SAMPLES = 192
SELECTOR_DIAG_SKIP    = 50_000  # skip past held-out eval region; NOT a reservation

# ======================================================================
# BATCHING
# ======================================================================
MICRO_BATCH = 2
GRAD_ACCUM  = 32

# ======================================================================
# EVAL (design note §9) — HEADLINE metrics use FRESH losses from the FROZEN base,
# queue OFF. This section governs the final measurement stage.
# ======================================================================
EVAL_EVERY = 200
HEADLINE_REQUIRES_FRESH = True

# diagnostics prompts (one per hidden domain) — optional, used by diagnostics.py
DIAG_PROMPTS = {
    "arxiv": "From the standpoint of the Killing-Cartan classification of simple Lie algebras and their highest-weight representation theory, it remains genuinely difficult to explain the striking uniformity that appears across different Lie groups, which suggests",
    "code":  "I am trying to solve a differential equation with the ode45 solver in MATLAB, but I want the parameter vectors C1 through C4 to advance by one entry on every iteration, so that each successive call uses",
    "web":   "The Ohio Supreme Court recently considered whether a high school student who leaves a backpack behind on the bus can reasonably expect it to remain unopened, and whether a warrantless search in that situation would",
    "c4":    "After growing up with poor nutritional habits that cost me years of confusion and frustrating yo-yo dieting, I finally decided to build a simple step-by-step health program, and the first lesson I learned was",
    "wiki":  "Loren Gold is an American keyboardist, vocalist, and songwriter best known as the touring keyboardist and backup vocalist for The Who, and over his career he has also performed and recorded with",
}
DIAG_GEN_TOKENS = 60

# Sampled-generation controls for the SAMPLED diagnostic pass (HF-style). The GREEDY
# pass is unchanged — it stays a raw intrinsic model diagnostic (pure argmax, no
# penalties, no stopping) so we can still see collapse/repetition directly. These
# apply ONLY to the sampled pass and affect diagnostics only, never training or PPL.
DIAG_SAMPLE_TEMPERATURE      = 0.8
DIAG_SAMPLE_TOP_P            = 0.9
DIAG_SAMPLE_TOP_K            = 40
DIAG_SAMPLE_REPETITION_PENALTY = 1.15   # CTRL-style: divide logits of already-seen tokens
DIAG_SAMPLE_NO_REPEAT_NGRAM  = 4        # ban any 4-gram from repeating
DIAG_SAMPLE_EOS_STOP         = True     # stop a sample early at the EOS token
DIAG_SAMPLE_SEED             = 1234     # fixed so SAMPLED is reproducible across checkpoints

# Optional fixed REFERENCE continuation per prompt (comparison only; generation need
# not match). Leave empty to print PROMPT/GREEDY/SAMPLED without a REFERENCE line.
DIAG_REFERENCES = {}



# ======================================================================
# CONTENT INDEX ROUTING (v3) — replaces the learned SelectorB entirely.
# The frozen Content Index (content_index.py, built offline by
# build_content_index.py) decides the executed path. NO LM-loss feedback reaches
# the router (design note §1/§12/§30). The learned selector, its replay queue CE,
# and K3 quantile balancing are REMOVED — routing is frozen: cluster k -> Pk.
# This is a FRESH run from step 0 (v2.0 selector checkpoints do NOT load).
# ======================================================================
CONTENT_INDEX_ENABLED = True
# artifact produced by build_content_index.py; version_id is recorded in every LM
# checkpoint so the PPL harness can confirm it measured the same index (design §26).
# CONTENT_INDEX_ARTIFACT = (DRIVE_DIR + "/content_index/content_index.pt")
# CONTENT_INDEX_ARTIFACT = "/content/drive/MyDrive/Project1/MultiplePathHypernetPro/content_index.local.pt"
CONTENT_INDEX_ARTIFACT = (
    "/content/drive/MyDrive/Project1/MultiplePathHypernetPro/"
    "content_index.step3500.alpha025.pt"
)
# CONTENT_INDEX_VERSION  = "CI-v3-minilm-meanpool-bkmeans"   # expected; asserted on load
# -p256-mv-src: rebuilt artifact (multi-view / source-aware). Must match the
# artifact's version_id exactly; train.py asserts ci.version_id == this on load.
# CONTENT_INDEX_VERSION = "CI-v3-minilm-meanpool-bkmeans-p256-mv-src"
CONTENT_INDEX_VERSION = (
    "CI-v3-minilm-meanpool-bkmeans-p256-mv-src-ppl3500-a025"
)
# v3.1: destinations are P0..P5 ONLY. There is no GENERAL destination (the trunk
# is the shared/general computation; design note §6). The CI ranks the six
# specialists by cosine similarity; its K equals N_SPECIALISTS.

# TRAINING routes the top-N ranked CI SPECIALISTS (memory-bounded). The two paths
# are scored SEPARATELY and the per-sample loss is mean(L_Pa, L_Pb) — they are NOT
# fused (design note §6/§10). INFERENCE executes CI top-1 only (the memory win).
CI_TRAIN_TOPN = 1                      # forward/score CI rank-1 only (top-2 disabled)

# ---- soft self-balancing specialist rescue (training only) -------------------
# Rank-1 is always the main LM path.
# Rank-2 is always forwarded/scored, but gets LM gradient only when:
#   1) its recent top-1 usage is below its dynamic floor, and
#   2) its NLL is competitive with rank-1.
# No all-six training sweep. Inference remains CI top-1.
PATH_RESCUE_ENABLED      = False
PATH_RESCUE_WEIGHT       = 0.10
PATH_RESCUE_DNLL_MAX     = 0.05
PATH_RESCUE_EMA_DECAY    = 0.95

# Dynamic usage floor:
# weakly competitive path -> near 3%
# strongly competitive path -> toward 5%
PATH_RESCUE_FLOOR_MIN    = 0.03
PATH_RESCUE_FLOOR_MAX    = 0.05

# After process start/resume, collect statistics before rescue begins.
PATH_RESCUE_WARMUP_STEPS = 20

# ---- top-k sampled cross-entropy (TRAINING-ONLY speedup) ----------------------
# VOCAB_TOPK restricts the CE softmax denominator to the top-k highest-logit vocab
# entries UNION the true target (target always included). It APPROXIMATES the loss
# (different gradient than full-vocab), so it is a training-time speedup only:
#   * 0        -> exact full-vocab CE (default; use this until the model is stable)
#   * 512/1024 -> sampled top-k CE
# ALL REPORTED PPL/ACC ARE FULL-VOCAB (eval / fresh_gains always pass topk=0), so
# headline numbers stay comparable regardless of this knob.
# Top-k is gated: it activates ONLY when step >= TOPK_START_STEP AND VOCAB_TOPK>0.
# Rationale: early training needs the exact gradient; enable top-k after the model
# has settled (e.g. around/after the first EM refit). The TRAINING-LOSS print will
# step-discontinuously drop when top-k turns on — that is expected, not a training
# event; the loss line is labeled with the active regime.
VOCAB_TOPK = 0                         # 0 (off/full) | 512 | 1024
TOPK_START_STEP = 20000                # top-k inactive before this step (set near
                                       # your first EM refit; harmless if VOCAB_TOPK=0)
# When VOCAB_TOPK>0, training_forward uses a per-sample candidate cache: the FIRST CI
# path scored for a sample computes the full shared LM head and stores its top-K vocab
# ids; the SECOND path projects ONLY those K rows (∪ true target) — a genuine LM-head
# compute saving (a [T,K] gather+dot instead of a [T,V] matmul), not just a cheaper
# CE. The cache is a per-micro-batch local (reset every step, never checkpointed). The
# second path's candidate set is the first path's top-K, a deliberate training-only
# approximation (like top-k CE); all REPORTED PPL/ACC remain full-vocab (topk=0).
# VOCAB_TOPK_CHUNK bounds the cached-path gather W[cand] ([n,chunk,K+1,d]) so peak
# memory stays small at long T / large K; math is identical to un-chunked. Lower it
# if the cached path is memory-bound; raise it (up to MAX_LENGTH) for fewer kernels.
VOCAB_TOPK_CHUNK = 128
# VOCAB_TOPK_RECOMPUTE: chunking bounds the FORWARD gather, but autograd would still
# retain every chunk's gathered W[cand] for backward — so retained training memory can
# approach the un-chunked ~1.1 GiB. With recompute on, each chunk is wrapped in
# gradient checkpointing: its Wc/candidate-logits are dropped after forward and
# recomputed in backward, so retained memory is ~one chunk. Costs one extra gather+
# einsum per chunk in backward. Only active when grad is on (no eval-time cost).
VOCAB_TOPK_RECOMPUTE = True

# Decode ids->text for the router: the CI encoder routes on raw text, so the train
# loop decodes each segment once per micro-batch. Cache is keyed by step to bound
# memory (routing is deterministic given the frozen index, so this is pure compute).
CI_DECODE_SKIP_SPECIAL = True
# CAUSAL ROUTING PREFIX (correctness-critical): the router may read ONLY the first
# CI_ROUTE_PREFIX tokens. The LM scores targets from SCORE_FROM onward, so the
# router must stay strictly inside the un-scored prefix or it leaks the future of
# the tokens the routed path predicts. Must equal SELECTOR_PREFIX (the trunk-hidden
# prefix the old selector read) and be <= SCORE_FROM. Asserted in validate().
CI_ROUTE_PREFIX = SELECTOR_PREFIX


# ----------------------------------------------------------------------
# Derived helpers / schedules
# ----------------------------------------------------------------------
def expert_kinds():
    return list(OURS_EXPERT_KINDS)


def num_mlps():
    return len(OURS_EXPERT_KINDS)


def uses_path_split():
    # data.py gates its balanced-eval reservation on this; Project B does split.
    return True


def _interp(step, schedule, idx):
    """Piecewise-linear interpolation of schedule[*][idx] at `step`."""
    pts = schedule
    if step <= pts[0][0]:
        return pts[0][idx]
    if step >= pts[-1][0]:
        return pts[-1][idx]
    for (s0, *_), (s1, *_) in zip(pts, pts[1:]):
        if s0 <= step <= s1:
            row0 = next(r for r in pts if r[0] == s0)
            row1 = next(r for r in pts if r[0] == s1)
            frac = (step - s0) / max(1, (s1 - s0))
            return row0[idx] + frac * (row1[idx] - row0[idx])
    return pts[-1][idx]


def lambda_G(step):
    """General LM auxiliary-loss weight at `step` (design note §4/§5)."""
    return float(_interp(step, RESP_SCHEDULE, 1))


def base_frozen(step):
    """
    Whether the base components (trunk, embeddings, final LN, LM head) are frozen.
    Multipath‑aware: returns True if any of the explicit freeze flags is True,
    OR if the step‑based freeze (FREEZE_BASE_AT_STEP) is active.
    For the smoke run, all flags are True, so this always returns True.
    """
    if MULTIPATH_ENABLED:
        if FREEZE_TRUNK or FREEZE_EMBEDDINGS or FREEZE_FINAL_LN or FREEZE_LM_HEAD:
            return True
    return bool(FREEZE_BASE_AT_STEP) and step >= FREEZE_BASE_AT_STEP


def active_vocab_topk(step):
    """The top-k value to use for the TRAINING CE at this step, honoring the phase
    gate: returns VOCAB_TOPK only when VOCAB_TOPK>0 AND step>=TOPK_START_STEP, else
    0 (full-vocab). Eval/reporting never calls this — it always uses topk=0."""
    if int(VOCAB_TOPK) > 0 and step >= int(TOPK_START_STEP):
        return int(VOCAB_TOPK)
    return 0


def sample_prefix(rng=None):
    """Sample the per-micro-batch causal routing boundary R (training only).
    Returns SCORE_FROM (the fixed reference) when variable prefix is disabled, else
    a uniform integer in [VARPREFIX_MIN, VARPREFIX_MAX] inclusive. `rng` may be a
    python random.Random for reproducibility; otherwise uses the module `random`."""
    if not VARPREFIX_ENABLED:
        return int(SCORE_FROM)
    import random as _r
    lo, hi = int(VARPREFIX_MIN), int(VARPREFIX_MAX)
    return (rng or _r).randint(lo, hi)


def prefix_bucket(R):
    """Label R by its eval bucket range (or 'ref256' for the exact reference)."""
    R = int(R)
    if R >= int(SCORE_FROM):
        return "ref256"
    for lo, hi in VARPREFIX_EVAL_BUCKETS:
        if lo <= R <= hi:
            return f"{lo}-{hi}"
    return "other"


def _lin(step, v_start, v_end, decay_steps):
    t = min(max(step / max(1, decay_steps), 0.0), 1.0)
    return v_start + t * (v_end - v_start)


def selector_temperature(step):
    return _lin(step, SELECTOR_TEMP_START, SELECTOR_TEMP_END, SELECTOR_SAMPLE_UNTIL_STEP)


def selector_hard_sample(step):
    return step < SELECTOR_SAMPLE_UNTIL_STEP


def qbal_proto_coef(step):
    return _lin(step, QBAL_PROTO_COEF_START, QBAL_PROTO_COEF_END, QBAL_DECAY_STEPS)


def qbal_path_coef(step):
    return _lin(step, QBAL_PATH_COEF_START, QBAL_PATH_COEF_END, QBAL_DECAY_STEPS)


def selector_active(step):
    """Selector trains only after specialists differentiate (design note §8)."""
    return step >= SELECTOR_START_STEP


def ttl_for_step(step):
    """Replay-queue TTL shrinks as the base approaches the late freeze: early
    (base still moving) a longer TTL is fine; near/after the freeze a shorter TTL
    keeps the target fresh. v2.0 has no gamma, so we key the shrink on progress
    toward FREEZE_BASE_AT_STEP instead of gamma. Linear between QUEUE_TTL_STEPS
    (early) and QUEUE_TTL_MIN (at/after freeze)."""
    if not FREEZE_BASE_AT_STEP:
        return QUEUE_TTL_STEPS
    frac = min(max(step / max(FREEZE_BASE_AT_STEP, 1), 0.0), 1.0)   # 0 early -> 1 late
    ttl = QUEUE_TTL_STEPS + frac * (QUEUE_TTL_MIN - QUEUE_TTL_STEPS)
    return int(round(ttl))


def subset_gain_biased(step):
    """Whether the gradient-subset is chosen by gain (late) vs round-robin."""
    return (SUBSET_POLICY == "gain_biased") and (step >= SUBSET_GAIN_BIAS_FROM_STEP)


def run_tag():
    hn = f"HN[tB={'on' if HYPERNET_TRUNK_B else 'off'},spec={'on' if HYPERNET_SPEC else 'off'}]"
    return (f"PROJECT=B/v3.1 trunk{TRUNK_LAYERS}(A[0:{TRUNK_SPLIT_AT}]/B) "
            f"|| spec{SPECIALIST_LAYERS}x{N_SPECIALISTS} (no GENERAL, no fusion) "
            f"d={D_MODEL} {hn} experts={'+'.join(OURS_EXPERT_KINDS)} "
            f"CI={CONTENT_INDEX_VERSION} topN={CI_TRAIN_TOPN} "
            f"seq={MAX_LENGTH} mb={MICRO_BATCH}x{GRAD_ACCUM}")


def _git_hash():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def resolve():
    d = {k: globals()[k] for k in globals()
         if k.isupper() and not k.startswith("_")}
    d.update({
        "_resolved_run_tag": run_tag(),
        "_resolved_expert_kinds": expert_kinds(),
        "_resolved_effective_depth": N_LAYERS_EFFECTIVE,
        "_resolved_blocks_executed": N_BLOCKS_EXECUTED,
        "_git_hash": _git_hash(),
    })
    return d


def validate():
    errs = []
    if MOE_MODE not in ("multi_expert", "lowrank", "dense"):
        errs.append(f"MOE_MODE ({MOE_MODE!r}) must be 'multi_expert', 'lowrank', or 'dense'")
    if MOE_MODE == "lowrank" and not (1 <= MOE_LOWRANK_R <= D_MODEL):
        errs.append(f"MOE_LOWRANK_R ({MOE_LOWRANK_R}) must be 1..D_MODEL ({D_MODEL})")
    if MAX_LENGTH > N_POSITIONS:
        errs.append(f"MAX_LENGTH ({MAX_LENGTH}) > N_POSITIONS ({N_POSITIONS})")
    if GRAD_ACCUM < 1 or MICRO_BATCH < 1:
        errs.append("MICRO_BATCH and GRAD_ACCUM must be >= 1")
    if N_SPECIALISTS < (1 if SINGLE_PATH else 2):
        errs.append("N_SPECIALISTS must be >= 2 (or exactly 1 when SINGLE_PATH)")
    if not (0.0 <= TRUNK_LR_MULT <= 1.0):
        errs.append(f"TRUNK_LR_MULT ({TRUNK_LR_MULT}) must be in 0.0..1.0 "
                    f"(0=no trunk step; for a true freeze use FREEZE_BASE_AT_STEP)")
    if VARPREFIX_ENABLED:
        if not (1 <= VARPREFIX_MIN <= VARPREFIX_MAX):
            errs.append(f"VARPREFIX_MIN ({VARPREFIX_MIN}) must be 1..VARPREFIX_MAX "
                        f"({VARPREFIX_MAX})")
        if VARPREFIX_MAX >= MAX_LENGTH:
            errs.append(f"VARPREFIX_MAX ({VARPREFIX_MAX}) must be < MAX_LENGTH "
                        f"({MAX_LENGTH}) — need scored targets after R")
        if VARPREFIX_MAX >= SCORE_FROM:
            errs.append(f"VARPREFIX_MAX ({VARPREFIX_MAX}) must be < SCORE_FROM "
                        f"({SCORE_FROM}) so exact-{SCORE_FROM} stays the clean reference")
    if not (0 < TRUNK_SPLIT_AT < TRUNK_LAYERS):
        errs.append(f"TRUNK_SPLIT_AT ({TRUNK_SPLIT_AT}) must be in 1..{TRUNK_LAYERS-1}")
    if SPEC_GRAD_PATHS < 1 or SPEC_GRAD_PATHS > N_SPECIALISTS:
        errs.append(f"SPEC_GRAD_PATHS ({SPEC_GRAD_PATHS}) must be 1..{N_SPECIALISTS}")
    if SELECTOR_PREFIX >= MAX_LENGTH:
        errs.append("SELECTOR_PREFIX must be < MAX_LENGTH (need scored targets after prefix)")
    if SCORE_FROM < SELECTOR_PREFIX:
        errs.append("SCORE_FROM must be >= SELECTOR_PREFIX (avoid prefix leakage)")
    # Content Index routing (v3): the learned selector is gone; routing is frozen.
    if CONTENT_INDEX_ENABLED:
        if VOCAB_TOPK not in (0, 512, 1024):
            errs.append(f"VOCAB_TOPK ({VOCAB_TOPK}) must be 0, 512, or 1024")
        if VOCAB_TOPK and TOPK_START_STEP < 0:
            errs.append("TOPK_START_STEP must be >= 0")
        if not (1 <= CI_TRAIN_TOPN <= N_SPECIALISTS):
            errs.append(f"CI_TRAIN_TOPN ({CI_TRAIN_TOPN}) must be 1..N_SPECIALISTS "
                        f"({N_SPECIALISTS}) — v3.1 routes specialists only, no GENERAL")
        # rescue validation
        if PATH_RESCUE_ENABLED and CI_TRAIN_TOPN != 2:
            errs.append("PATH_RESCUE_ENABLED requires CI_TRAIN_TOPN == 2")
        if not (0.0 < PATH_RESCUE_WEIGHT <= 1.0):
            errs.append("PATH_RESCUE_WEIGHT must be in (0,1]")
        if PATH_RESCUE_DNLL_MAX < 0.0:
            errs.append("PATH_RESCUE_DNLL_MAX must be >= 0")
        if not (0.0 <= PATH_RESCUE_EMA_DECAY < 1.0):
            errs.append("PATH_RESCUE_EMA_DECAY must be in [0,1)")
        if not (0.0 < PATH_RESCUE_FLOOR_MIN <= PATH_RESCUE_FLOOR_MAX < 1.0):
            errs.append(
                "PATH_RESCUE_FLOOR_MIN/MAX must satisfy 0 < MIN <= MAX < 1"
            )
        if PATH_RESCUE_WARMUP_STEPS < 0:
            errs.append("PATH_RESCUE_WARMUP_STEPS must be >= 0")
        # causal routing boundary: the router must not see scored tokens.
        if CI_ROUTE_PREFIX > SCORE_FROM:
            errs.append(f"CI_ROUTE_PREFIX ({CI_ROUTE_PREFIX}) > SCORE_FROM "
                        f"({SCORE_FROM}) — router would leak future (scored) tokens")
        if CI_ROUTE_PREFIX < 1:
            errs.append("CI_ROUTE_PREFIX must be >= 1 (router needs some context)")
    steps = [r[0] for r in RESP_SCHEDULE]
    if steps != sorted(steps):
        errs.append("RESP_SCHEDULE steps must be ascending")
    if any(len(r) != 2 for r in RESP_SCHEDULE):
        errs.append("v2.0 RESP_SCHEDULE rows are (step, lambda_G) — no gamma column")
    if errs:
        raise ValueError("Config B validation failed:\n  - " + "\n  - ".join(errs))
    return True


def print_resolved():
    r = resolve()
    print("=" * 64)
    print("RESOLVED CONFIG — PROJECT B v2.0 (parallel siblings)")
    print("=" * 64)
    print(f"run_tag   : {r['_resolved_run_tag']}")
    print(f"git       : {r['_git_hash']}")
    print(f"arch      : trunk {TRUNK_LAYERS} (A[0:{TRUNK_SPLIT_AT}] / B[{TRUNK_SPLIT_AT}:{TRUNK_LAYERS}]) "
          f"|| specialist {SPECIALIST_LAYERS} x {N_SPECIALISTS}  (NO GENERAL, NO fusion)")
    print(f"depth/tok : {N_LAYERS_EFFECTIVE} layers (trunk + one {SPECIALIST_LAYERS}-layer specialist)")
    print(f"blocks/exec: {N_BLOCKS_EXECUTED} (trunk + 1 specialist; inference FLOPs-honest)")
    print(f"output    : logits = head(LN_f(specialist(h_trunk)))  — specialist IS the continuation")
    print(f"experts   : {' | '.join(f'E{i} {k}' for i, k in enumerate(OURS_EXPERT_KINDS))}")
    print(f"hypernet  : trunk_A={'on' if HYPERNET_TRUNK_A else 'off'} "
          f"trunk_B={'on' if HYPERNET_TRUNK_B else 'off'}({HYPERNET_CENTROIDS_TRUNK_B}) "
          f"spec={'on' if HYPERNET_SPEC else 'off'}({HYPERNET_CENTROIDS_SPEC}/path)")
    print(f"routing   : FROZEN Content Index {CONTENT_INDEX_VERSION}; "
          f"train top-{CI_TRAIN_TOPN} forwarded, infer top-1")
    print(f"rescue    : {'ON' if PATH_RESCUE_ENABLED else 'OFF'}; "
          f"rank1 main + conditional rank2; "
          f"w={PATH_RESCUE_WEIGHT} dNLL<={PATH_RESCUE_DNLL_MAX} "
          f"floor={100*PATH_RESCUE_FLOOR_MIN:.0f}-"
          f"{100*PATH_RESCUE_FLOOR_MAX:.0f}%")
    print(f"rejuv     : {'ON' if REJUV_ENABLED else 'OFF'} "
          f"(<= step {REJUV_STOP_STEP}, every {REJUV_EVERY})")
    print(f"freeze base: step {FREEZE_BASE_AT_STEP} (trunk only; specialists keep training)")
    print(f"trunk LR   : x{TRUNK_LR_MULT:g} of base LR {LEARNING_RATE:g}"
          + ("  (same as specialists)" if TRUNK_LR_MULT == 1.0
             else "  (trunk trains slower)" if TRUNK_LR_MULT > 0.0
             else "  (trunk LR=0; note: not a true freeze — see FREEZE_BASE_AT_STEP)"))
    print(f"score_from : target pos >= {SCORE_FROM} (causal, no prefix leak)")
    print(f"seq/pos    : MAX_LENGTH={MAX_LENGTH} (cap {N_POSITIONS})")
    print(f"batch      : {MICRO_BATCH} x {GRAD_ACCUM} = {MICRO_BATCH*GRAD_ACCUM}")
    if DISABLE_HYPERNET or MOE_MODE != "multi_expert":
        print(f"VARIANT    : DISABLE_HYPERNET={DISABLE_HYPERNET}  MOE_MODE={MOE_MODE}"
              + (f" (r={MOE_LOWRANK_R})" if MOE_MODE == "lowrank" else "")
              + "  <-- NOT the full model; use a distinct CKPT_DIR")
    print("=" * 64)