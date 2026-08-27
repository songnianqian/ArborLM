"""
content_index_config.py — Content Index (v3) configuration.

Kept SEPARATE from the LM config (config.py) on purpose: the Content Index is an
independent, frozen artifact (design note §3, §26, §29). The LM never reaches into
these values; it only calls ContentIndex.route(...) and reads the saved version ID
that the LM checkpoint records (§26, invariant §30.3).

REVIEW FIXES folded in (teammate review, 2026-08):
  #3  three-way data split FIT -> CALIBRATION -> GATE (calibration != gate set).
  #4  switch behaviour split into same-state STABILITY and TOPIC-SHIFT responsiveness;
      T_SWITCH is calibrated between the two, not a bare 1.5*margin heuristic.
  #5  chunking is TOKENIZER-EXACT (<=CHUNK_TOKENS encoder tokens, true overlap).
  #6  the whole embedding pipeline (backend, revision, caps, norm) is pinned so the
      frozen artifact reproduces byte-for-byte across machines.
  +OOD calibration statistics are collected and saved (design §26).

Thresholds that the design says to CALIBRATE, not hardcode (§14, §15), are left as
None here and FILLED IN by build_content_index.py from held-out geometry, then baked
into the saved artifact. Values below are only defaults / sweep ranges for the build.
"""

# ======================================================================
# FROZEN SEMANTIC ENCODER (design §6, §28 step 1)
# ======================================================================
# Cheapest-first: a frozen pretrained sentence encoder. Correctness must NOT depend
# on precomputed IDs — the same pipeline runs live on unseen text (design §5).
ENCODER_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION   = None        # pin to a specific commit hash for reproducibility;
                                 #  None = default branch. build script records the
                                 #  resolved revision into the artifact regardless.
ENCODER_BACKEND    = "auto"      # "auto" | "st" | "hf". [#6] auto picks
                                 #  sentence-transformers if importable else raw HF.
                                 #  The RESOLVED backend is pinned into the artifact
                                 #  so two machines can't silently embed differently.
ENCODER_DIM        = 384         # all-MiniLM-L6-v2 width; auto-checked/overwritten on load
ENCODER_DEVICE     = "cuda"      # "cuda" | "cpu"; build falls back to cpu
ENCODER_BATCH      = 64          # chunk-encode micro-batch
ENCODER_MAX_TOKENS = 256         # encoder truncation cap per chunk (== CHUNK_TOKENS)

# ======================================================================
# CHUNKING (design §7) — represent the WHOLE context, not the first 256 tokens.
# [#5] TOKENIZER-EXACT: chunks are sliced on the ENCODER's own tokenizer so every
# chunk is provably <= CHUNK_TOKENS encoder tokens and the overlap really is 25%.
# ======================================================================
CHUNK_TOKENS   = 256   # requested content cap; runtime reserves special-token slots
CHUNK_OVERLAP  = 0.25            # 25% token overlap (design §7)
MAX_CHUNKS     = 64              # hard cap so a pathological doc can't explode cost
# short input (<= one chunk) -> a single embedding, no padding. long input ->
# overlapping token windows -> mean pool. Both handled by the same code.

# ======================================================================
# ROUTING PREFIX (ci_prefix_patch) — MUST equal the LM's CI_ROUTE_PREFIX.
# ======================================================================
# The LM routes on the FIRST ROUTE_PREFIX_TOKENS tokens of each context (its causal
# routing prefix, == config.CI_ROUTE_PREFIX == SELECTOR_PREFIX). The Content Index
# must FIT and CALIBRATE on that SAME view — build z from the first
# ROUTE_PREFIX_TOKENS tokens, not the full segment — or occupancy, fallback rate,
# and T_SIM/T_MARGIN are measured on a distribution the LM never queries.
ROUTE_PREFIX_TOKENS = 256          # MUST equal the LM's CI_ROUTE_PREFIX

# ----------------------------------------------------------------------
# MULTI-VIEW PREFIX SWEEP — TWO roles (D1=B ratified).
# ----------------------------------------------------------------------
# The runtime routing view stays ROUTE_PREFIX_TOKENS (256): the LM queries the first
# 256 tokens VERBATIM at inference. The sweep below is used two ways:
#   (1) FIT AUGMENTATION (D1=B): the FIT split embeds one view per natural-break
#       prefix (32/64/128 snapped, 256 exact) so the CENTROIDS see short prefixes and
#       short-prefix routing is robust. This DOES shape the frozen partition.
#   (2) SELF-CONSISTENCY DIAGNOSTIC: after the gate, a sample is routed at each view
#       and compared to its 256-view route (geometry consistency). Diagnostic only.
# CALIB, GATE, thresholds, and the headline GO/NO-GO deliberately stay on the pure
# 256 deployment view (calibrate on what's deployed), NOT the augmented cloud. True
# PPL accuracy/regret vs LM P0..P5 NLL is measured later by the harness, not here.
# 256 must stay in the sweep (it's the runtime view and the diagnostic label).
# The 32 target is snapped FORWARD (32..48) since a backward band at the smallest
# length would require a boundary exactly at 32; its recorded token_len is actual.
ROUTE_PREFIX_TOKENS_SWEEP = [32, 64, 128, 256]
MULTIVIEW_EVAL_ENABLED = True       # run the self-consistency diagnostic after the gate
MULTIVIEW_EVAL_SAMPLES = 2000       # sequences to sample for the diagnostic pass

# ----------------------------------------------------------------------
# SOURCE-DIVERSE SEEDING (INITIAL build only — design §9).
# ----------------------------------------------------------------------
# Sources are collected but must NEVER be a routing target (§9). This biases the
# k-means++ SEED so each new seed prefers a source not yet chosen, spreading initial
# centroids across sources. Unlike the earlier rarity weighting it WORKS on a
# source-balanced dataset. SEED-ONLY and honest about it: on well-separated data the
# Lloyd iterations + restarts can converge regardless of init, so the effect may wash
# out — it's a nudge, not a guarantee, and is kept seed-only so source never enters
# the objective (§9). 0.0 disables it exactly (byte-identical seeding). Subsequent CI
# rounds are PPL-label driven and pass NO source pressure.
SOURCE_PRESSURE_WEIGHT = 0.30       # 0.0 = off; ~0.3 = weak; 1.0 = strong seed spread

# ======================================================================
# POOLING / CONTENT VECTOR z (design §6 step 2)
# ======================================================================
POOL           = "mean"          # mean pool of chunk embeddings, then L2-normalize
NORMALIZE_Z    = True            # z unit-norm so cosine == dot (design §13)

# ======================================================================
# CLUSTERING (design §9, §10) — balanced / capacity-constrained FIT ONLY.
# [#1] The capacity constraint trains the CENTROIDS. Evaluation (balance, silhouette,
# margin, stability) ALWAYS uses the runtime nearest-centroid argmax partition — the
# partition specialists actually see — never the constrained fit assignment.
# ======================================================================
K_SWEEP        = (3, 4, 5, 6, 7, 8, 9, 10)
K_DEFAULT      = 6               # fallback if a sweep isn't run (== N_SPECIALISTS)
CAPACITY_FACTOR = 1.30           # each cluster <= CAPACITY_FACTOR * (N/K) during FIT
KMEANS_ITERS    = 50
KMEANS_RESTARTS = 4
KMEANS_SEED     = 1234
# balance gate on the RUNTIME partition (design §10, §24.1/§24.2): a cluster below
# this fraction of the equal share is "starved" and fails the gate.
MIN_CLUSTER_SHARE = 0.5          # each runtime cluster >= 0.5 * (1/K) of the mass

# [#2] HARD STOP when the measured K disagrees with the LM specialist count. The
# design says this must be a deliberate architecture decision (design §9), never a
# silent proceed. If True and chosen_K != N_SPECIALISTS, the build refuses to save a
# GO artifact and asks you to resolve it (change path count, or force K explicitly).
REQUIRE_K_EQUALS_SPECIALISTS = True
FORCE_K = None                   # set to an int to OVERRIDE the sweep after an explicit
                                 #  decision (design §9 option B: keep 6, tolerate
                                 #  redundant clusters). None = use the sweep winner.

# ======================================================================
# RUNTIME CONFIDENCE + HYSTERESIS (design §13-§17) — CALIBRATED, not hardcoded.
# Filled by build_content_index.py from the CALIBRATION split, saved into the
# artifact; the frozen runtime reads them from the artifact, never from here.
# ======================================================================
T_MARGIN_FALLBACK = None         # s1-s2 below this -> uncertain -> GENERAL (§16,§17)
T_SIM_FALLBACK    = None         # s1 below this (far from all centroids) -> GENERAL (§14)
T_SWITCH          = None         # hysteresis margin for incumbent takeover (§15)
# fallback calibration: quantiles of the CALIBRATION-split margin / top1 distributions.
CAL_MARGIN_QUANTILE = 0.10       # T_MARGIN_FALLBACK := 10th pct of calib margins
CAL_SIM_QUANTILE    = 0.05       # T_SIM_FALLBACK    := 5th pct of calib top1 sims
# [#4] T_SWITCH is calibrated to sit between two measured regimes:
#   - same-state margin jitter (should NOT trigger a switch)
#   - genuine topic-shift advantage (SHOULD trigger a switch)
# We pick T_SWITCH at a high quantile of same-state score-gaps (so normal jitter
# stays below it) and verify it sits below a low quantile of topic-shift gaps.
# CAL_SWITCH_OVER_MARGIN is only the FALLBACK heuristic if the two-regime calibration
# can't be run (e.g. too few conversational simulations).
CAL_SWITCH_SAMESTATE_Q = 0.90    # T_SWITCH >= 90th pct of same-state incumbent gaps
CAL_SWITCH_OVER_MARGIN = 1.5     # fallback: T_SWITCH := 1.5 * T_MARGIN_FALLBACK

# ======================================================================
# OOD CALIBRATION (design §26 "OOD calibration statistics") — [+review]
# The build measures the ID top1-similarity distribution and records summary stats
# so GENERAL fallback is grounded in geometry, and (optionally) checks separation
# against a shuffled-token OOD proxy. Saved into the artifact.
# ======================================================================
OOD_ENABLED       = True
OOD_PROXY_SAMPLES = 1000         # shuffled-token / random-text OOD proxies to score

# ======================================================================
# STABILITY vs TOPIC-SHIFT DIAGNOSTICS (design §18, §19, §24) — [#4]
# ======================================================================
# same-state stability (design §19, §24.4): different VIEWS of the same state
# (overlap change / history-length change / paraphrase) must route the same.
STABILITY_TRIALS      = 4
STABILITY_MIN_SAMERATE = 0.80    # >= 80% same-cluster across views (§24.4)
# short/long consistency (design §11, §19, §24.5): SAME current state with full vs
# shortened HISTORY (NOT first-third of a document) must agree.
SHORTLONG_MIN_AGREE   = 0.75
# topic-shift responsiveness (design §18): when the content genuinely changes to a
# strongly different cluster, the router SHOULD switch. Too-sticky is now a failure.
TOPICSHIFT_MIN_SWITCHRATE = 0.60 # >= 60% of strong genuine shifts should switch
# same-state oscillation ceiling (design §18): repeated same-state views should
# almost never switch.
SAMESTATE_MAX_SWITCHRATE  = 0.10 # <= 10% switching under same-state perturbation
MIN_SPECIALIST_COVERAGE   = 0.70

# perturbation strength expressed as a target COSINE change, not a raw noise std
# ([smaller-review]: sqrt(384)*0.02 was a ~21° perturbation, not "small").
PERTURB_COSINE = 0.98            # embedding-noise views aim for cos ~= 0.98 (~11°)

# ======================================================================
# BUILD DATA — three disjoint splits [#3]. Reuses the LM's streamed corpus
# (config.DATA_SOURCE / DATASET_*) via data.py so the partition is fit on the SAME
# distribution the LM will see (design §10).
#   FIT       : fit centroids + run the K sweep
#   CALIB     : calibrate thresholds (margin / sim / switch / OOD)
#   GATE      : untouched GO/NO-GO evaluation (never seen by calibration)
# ======================================================================
FIT_SAMPLES    = 20000
CALIB_SAMPLES  = 2000            # [#3] threshold calibration only
GATE_SAMPLES   = 2000            # [#3] held-out GO gate only (disjoint from calib)
BUILD_SKIP     = 0               # skip N sequences before collecting (avoid LM eval region)
BUILD_SEED     = 2025

# ======================================================================
# ARTIFACT (design §26, §27) — everything the frozen runtime needs, versioned.
# ======================================================================
# NOTE: version suffixes reflect the geometry that produced this artifact.
#   -p256 : runtime routing view is the first 256 tokens (unchanged, deployment).
#   -mv   : centroids were fit on the 32..256 MULTI-VIEW cloud (D1=B augmentation);
#           CALIB/GATE/thresholds still describe the 256 deployment view.
#   -src  : initial partition used source-diverse k-means++ seeding.
# Drop a suffix if you disable that feature (SOURCE_PRESSURE_WEIGHT=0 -> drop -src;
# single-view fit -> drop -mv) so the version id stays honest. The LM checkpoint
# pins this exact string (§26).
VERSION_ID   = "CI-v3-minilm-meanpool-bkmeans-p256-mv-src"
ARTIFACT     = ("/content/drive/MyDrive/projectB_generalist_specialist"
                "/content_index/content_index.pt")


def resolve():
    return {k: globals()[k] for k in globals()
            if k.isupper() and not k.startswith("_")}
