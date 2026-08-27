"""
multipath_model.py — mature C-Rich shared trunk (768) + N narrow specialist paths.

Converts the single full-width d=768 specialist into N independently trainable
paths at PATH_D_MODEL (default 576), each:

    h_trunk(768) -> down_proj 768->Dp -> 4x HyperNetBlock@Dp (+ spec HyperNet@Dp)
                 -> up_proj Dp->768 -> (shared frozen LN_f + LM head)

The shared trunk / embeddings / trunk-HyperNet / LN / head come from the mature
C-Rich checkpoint and are frozen (per config). Only the paths + their down/up
projections train in Run 1.

Design decisions baked in (see conversion spec):
  * per-path down AND up projections (each path its own view into the frozen trunk)
  * paths are FRESH init (no 768->576 weight copy); KD (Run 2) transfers behavior
  * the mature 768 teacher-specialist is ALWAYS constructed + loaded (decision A),
    forwarded ONLY when C.TEACHER_ENABLED. While FREEZE_TRUNK it reuses the same
    h_trunk as the student (no duplicate teacher trunk).

MultiPathModel subclasses GeneralistSpecialistModel so ALL routing/scoring/infer/
fresh_gains logic is inherited unchanged — every specialist access there goes
through run_specialist(h_trunk, p, ...), which we override to insert the
projections. The narrow paths keep the attribute names `specialists` /
`spec_hypernets` so param-group building and hypernet diagnostics still find them.
"""
import copy
import torch
import torch.nn as nn

import config as C
from model import GeneralistSpecialistModel, build_gpt2_config
from clustered_hypernet import ClusteredHypernetwork, HyperNetBlock


def _narrow_cfg(base_cfg, d_model, n_heads):
    """A GPT2Config clone at reduced width, so HyperNetBlock / ClusteredHypernetwork
    (which read n_embd / n_head) build at PATH_D_MODEL without any change to them."""
    cfg = copy.deepcopy(base_cfg)
    cfg.n_embd = int(d_model)
    cfg.n_head = int(n_heads)
    cfg.hidden_size = int(d_model)   # some HF paths read hidden_size
    return cfg


class MultiPathModel(GeneralistSpecialistModel):
    """Shared 768 trunk (inherited) + N narrow paths + optional frozen 768 teacher."""

    def __init__(self, gpt2_config=None):
        # Build the parent normally FIRST. This constructs the 768 trunk, embeddings,
        # LN, head, trunk-hypernets, AND a set of 768 specialists/spec_hypernets sized
        # by C.N_SPECIALISTS. We then REPLACE the specialist stack with narrow paths.
        super().__init__(gpt2_config)
        base_cfg = self.config
        d_full = C.D_MODEL
        d_path = C.PATH_D_MODEL
        n_paths = C.N_SPECIALISTS

        # ---- keep a frozen 768 TEACHER specialist (decision A: always built) ----
        # One 768 specialist stack + its spec-hypernet, matching the OLD single-path
        # C-Rich specialist (P0). Loaded from checkpoint by load_c_rich(); frozen;
        # forwarded only when C.TEACHER_ENABLED. Named distinctly so it is NOT part
        # of the student specialist list and does NOT collide with new-path keys.
        # ITEM 7: it is constructed here (so load_c_rich has a home for the old-spec
        # tensors and Run 2 is a flag flip) but build_multipath_model keeps it on CPU
        # while TEACHER_ENABLED is False, so it consumes no GPU memory in Run 1.
        teacher_cfg = base_cfg
        if C.HYPERNET_SPEC:
            self.teacher_spec_hypernet = ClusteredHypernetwork(
                teacher_cfg, num_centroids=C.HYPERNET_CENTROIDS_SPEC,
                temperature=C.HYPERNET_TEMPERATURE, balance_mode=C.HYPERNET_BALANCE_MODE)
        else:
            self.teacher_spec_hypernet = None
        base_idx = C.TRUNK_LAYERS
        self.teacher_specialist = nn.ModuleList([
            HyperNetBlock(teacher_cfg, base_idx + j, self.teacher_spec_hypernet)
            for j in range(C.SPECIALIST_LAYERS)])

        # ---- REPLACE the parent's 768 specialists with NARROW paths ----
        path_cfg = _narrow_cfg(base_cfg, d_path, C.PATH_N_HEADS)
        self.path_cfg = path_cfg

        if C.HYPERNET_SPEC:
            self.spec_hypernets = nn.ModuleList([
                ClusteredHypernetwork(path_cfg, num_centroids=C.HYPERNET_CENTROIDS_SPEC,
                                      temperature=C.HYPERNET_TEMPERATURE,
                                      balance_mode=C.HYPERNET_BALANCE_MODE)
                for _ in range(n_paths)])
        else:
            self.spec_hypernets = [None] * n_paths
        self.specialists = nn.ModuleList([
            nn.ModuleList([HyperNetBlock(path_cfg, base_idx + j, self.spec_hypernets[p])
                           for j in range(C.SPECIALIST_LAYERS)])
            for p in range(n_paths)])

        # per-path down/up projections (768<->Dp)
        self.down_proj = nn.ModuleList([nn.Linear(d_full, d_path) for _ in range(n_paths)])
        self.up_proj   = nn.ModuleList([nn.Linear(d_path, d_full) for _ in range(n_paths)])

        # Fresh generic init for the NEW blocks + projections ONLY. Crucially we do
        # NOT touch the spec HyperNets: their constructor sets hyper_proj near-zero
        # (std=1e-3) so each HN starts as a ~no-op additive correction
        # (identity-on-MoE). The generic std=0.02 initializer would CLOBBER that.
        # NOTE: nn.Module.apply() recurses into a module AND all its descendants, so
        # skipping a ClusteredHypernetwork module is NOT enough — apply would still
        # hit the HN's inner nn.Linear children. Instead we collect the id() of every
        # module inside any spec HyperNet and skip modules that belong to one.
        #
        # reason_proj / reset_reason_proj: the HyperNetBlock in THIS codebase uses the
        # simplified additive-correction hypernet and has NO reason_proj (block
        # children are ln_1/attn/ln_2/moe/mlp_dropout only), so there is no
        # reset_reason_proj() to call — identity-on-MoE is carried entirely by the
        # near-zero hyper_proj init we protect above. If the block is ever restored to
        # the older wide-reason_proj design, call block.reset_reason_proj() on each new
        # narrow block HERE, right after this generic init (see reset_reason_projs()).
        hypernet_module_ids = set()
        if C.HYPERNET_SPEC:
            for hn in self.spec_hypernets:
                if hn is not None:
                    for sub in hn.modules():
                        hypernet_module_ids.add(id(sub))
        self._hypernet_module_ids = hypernet_module_ids

        def _init_skip_hn(m):
            if id(m) in hypernet_module_ids:
                return  # inside a spec HyperNet — leave its careful init alone
            self._init_weights(m)

        for mod in (self.specialists, self.down_proj, self.up_proj):
            mod.apply(_init_skip_hn)

        # Forward-compatible hook: if the block design regains reason_proj, this
        # restores identity init on every new narrow block. No-op on the current
        # simplified block (method absent), so it's always safe to call.
        self.reset_reason_projs()

        # keep head tied to embeddings (parent set this; re-assert after any apply)
        self.lm_head.weight = self.wte.weight

    def reset_reason_projs(self):
        """Call reset_reason_proj() on every new narrow specialist block IF the block
        design provides it. On the current simplified additive-correction HyperNet the
        blocks have no reason_proj, so this is a safe no-op; identity-on-MoE is instead
        preserved by the near-zero hyper_proj init. On an older wide-reason_proj block
        design this restores the identity init after generic re-init. Returns the count
        of blocks reset (0 on the simplified design)."""
        n = 0
        for path in self.specialists:
            for blk in path:
                fn = getattr(blk, "reset_reason_proj", None)
                if callable(fn):
                    fn(); n += 1
        return n

    # ---- narrow-path continuation: project 768 -> Dp -> path -> Dp -> 768 ----
    def run_specialist(self, h_trunk, p, attention_mask):
        h = self.down_proj[p](h_trunk)                       # [B,T,768] -> [B,T,Dp]
        h, bal, moe_bal = self._run_stack(self.specialists[p], h, attention_mask)
        h = self.up_proj[p](h)                               # [B,T,Dp] -> [B,T,768]
        return h, bal, moe_bal

    # ---- teacher continuation (768), reuses the shared h_trunk ----
    @torch.no_grad()
    def teacher_logits(self, h_trunk, attention_mask=None):
        """Teacher = frozen 768 C-Rich specialist on the SAME h_trunk, then the shared
        frozen LN_f + LM head. Valid because FREEZE_TRUNK makes student trunk ==
        teacher trunk. Returns full-vocab logits [B,T,V]. Only call when TEACHER_ENABLED."""
        h = h_trunk
        for blk in self.teacher_specialist:
            h = blk(h, attention_mask=attention_mask)[0]
        h = self.ln_f(h)
        return self.lm_head(h)

    # ---- teacher device management (ITEM 7) ----
    def teacher_to(self, device):
        """Move the teacher specialist + its HN to `device`. Called only when the
        teacher is actually needed (Run 2), so Run 1 keeps it off-GPU."""
        self.teacher_specialist.to(device)
        if self.teacher_spec_hypernet is not None:
            self.teacher_spec_hypernet.to(device)
        return self

    def train(self, mode: bool = True):
        """Override so model.train() NEVER puts the teacher into train mode. The
        teacher must stay eval() (frozen, no dropout, no train-time HN stats) even
        when the student is training. All other submodules follow `mode` as usual."""
        super().train(mode)
        self.teacher_specialist.eval()
        if self.teacher_spec_hypernet is not None:
            self.teacher_spec_hypernet.eval()
        return self

    # ---- KD token scoring: MIRRORS GeneralistSpecialistModel._token_ce exactly ----
    def _token_kd(self, student_hidden, teacher_hidden, input_ids, attention_mask,
                  temperature, score_from=None):
        """Token-aligned KD KL(p_teacher || p_student) over EXACTLY the CE-scored
        positions. Uses the identical crop as _token_ce: predictions at [start:-1],
        start = max(sf-1, 0), sf = score_from (variable-prefix R). This guarantees KD
        never supervises tokens before the causal routing boundary that CE ignores.

        student_hidden / teacher_hidden: [B,T,768] post-path / post-teacher-specialist
        hidden (BEFORE ln_f/lm_head — we apply the shared frozen head here, same as CE).
        Returns (kd_per_token [B,Ts], mask [B,Ts] or None)."""
        sf = int(C.SCORE_FROM if score_from is None else score_from)
        start = max(sf - 1, 0)
        T = temperature

        # shared frozen head on both (same LN_f + lm_head CE uses)
        s_logits = self.lm_head(self.ln_f(student_hidden))[:, start:-1, :]      # [B,Ts,V]
        with torch.no_grad():
            t_logits = self.lm_head(self.ln_f(teacher_hidden))[:, start:-1, :]  # [B,Ts,V]

        # KL(p_t || p_s) * T^2, temperature-scaled (standard KD)
        log_p_s = torch.log_softmax(s_logits / T, dim=-1)
        p_t     = torch.softmax(t_logits / T, dim=-1)
        # per-token KL = sum_v p_t * (log p_t - log p_s); use kl_div for stability
        kd = torch.nn.functional.kl_div(
            log_p_s, p_t, reduction="none").sum(dim=-1) * (T * T)              # [B,Ts]

        m = None
        if attention_mask is not None:
            m = attention_mask[:, start + 1:].float()                          # [B,Ts]
        return kd, m

    # ---- KD auxiliary methods (override the base class stubs) ----
    def _training_aux_prepare(
        self, h_trunk, input_ids, attention_mask, score_from
    ):
        """Pre‑compute teacher hidden once per micro‑batch."""
        if not C.TEACHER_ENABLED:
            return None

        # KD requires exactly one routed path per sample (CI_TRAIN_TOPN == 1)
        assert C.CI_TRAIN_TOPN == 1

        with torch.no_grad():
            h = h_trunk
            for blk in self.teacher_specialist:
                h = blk(h, attention_mask=attention_mask)[0]
        return h

    def _training_aux_group(
        self, teacher_hidden, hp, idx, ids_p, am_p, path_id, score_from
    ):
        """Compute KD loss for a group of rows that routed to the same specialist."""
        if teacher_hidden is None:
            return None

        th = teacher_hidden.index_select(0, idx)

        kd, mask = self._token_kd(
            hp, th, ids_p, am_p,
            float(C.TEACHER_KD_TEMP),
            score_from=score_from,
        )

        if mask is not None:
            return (kd * mask).sum(), mask.sum()

        return kd.sum(), kd.new_tensor(float(kd.numel()))

    # ---- rejuvenation guard (ITEM 5) ----
    @torch.no_grad()
    def rejuvenate(self, step, optimizer=None):
        """Rejuvenation mutates HyperNet centroids IN-PLACE (a manual .copy_(), not a
        gradient step), so requires_grad=False does NOT protect a frozen trunk HN.
        Here we skip any hypernet whose parameters are frozen — so with FREEZE_TRUNK
        the trunk A/B centroids are never rewritten, while the trainable per-path
        spec HNs still rejuvenate normally. Mirrors the parent loop otherwise."""
        if not C.REJUV_ENABLED or step > C.REJUV_STOP_STEP:
            return {}
        report = {}
        for name, hyper in self.named_hypernets():
            if hyper is None:
                continue
            # frozen HN (e.g. trunk under FREEZE_TRUNK) -> never mutate its centroids
            if not hyper.centroids.requires_grad:
                continue
            if int(hyper.train_usage_lifetime.sum()) < C.REJUV_MIN_EXPOSURE:
                continue
            revived = self._rejuvenate_one_hypernet(
                hyper, optimizer=optimizer,
                noise_std=C.REJUV_NOISE_STD, donor_frac=C.REJUV_DONOR_FRAC)
            if revived:
                report[name] = len(revived)
            hyper.train_usage_window.zero_()
        return report

    # ---- freezing ----
    def apply_freezes(self):
        """Set requires_grad per config. Frozen params are also excluded from the
        optimizer by trainable_parameters(). Idempotent."""
        def _set(mod, flag):
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = flag

        # teacher is ALWAYS frozen
        _set(self.teacher_specialist, False)
        _set(self.teacher_spec_hypernet, False)

        if C.FREEZE_EMBEDDINGS:
            _set(self.wte, False); _set(self.wpe, False)
        if C.FREEZE_TRUNK:
            _set(self.trunk, False)
            _set(self.trunk_hypernet_a, False)
            _set(self.trunk_hypernet_b, False)
        if C.FREEZE_FINAL_LN:
            _set(self.ln_f, False)
        if C.FREEZE_LM_HEAD:
            _set(self.lm_head, False)
            # lm_head.weight is tied to wte.weight; freezing one freezes both. If
            # embeddings must stay trainable while head is frozen (or vice versa),
            # the tie has to be broken first — not needed for Run 1 (both frozen).

        # student paths + projections ALWAYS trainable
        _set(self.specialists, True)
        _set(self.down_proj, True)
        _set(self.up_proj, True)
        if C.HYPERNET_SPEC:
            _set(self.spec_hypernets, True)

    def trainable_parameters(self):
        """Params with requires_grad=True — feed THIS to the optimizer so frozen
        trunk/teacher never receive updates or optimizer state."""
        return [p for p in self.parameters() if p.requires_grad]

    # ---- param-count report (spec item: make the memory advantage explicit) ----
    def param_report(self):
        # ITEM 8: count UNIQUE parameters by id(). Within a path, all 4 blocks share
        # ONE spec-hypernet instance (block.hypernet), so a naive sum over
        # specialists + spec_hypernets double-counts each shared HN. Deduping by id
        # fixes it; also correctly handles the wte/lm_head weight tie.
        def _n(*mods):
            seen, tot = set(), 0
            stack = []
            for m in mods:
                if m is None:
                    continue
                if isinstance(m, (list, tuple, nn.ModuleList)):
                    stack.extend(m)
                else:
                    stack.append(m)
            for m in stack:
                if m is None:
                    continue
                for p in m.parameters():
                    if id(p) not in seen:
                        seen.add(id(p)); tot += p.numel()
            return tot

        # shared base: dedupe so the tied lm_head.weight (== wte.weight) counts once
        shared = _n(self.wte, self.wpe, self.ln_f, self.lm_head,
                    self.trunk, self.trunk_hypernet_a, self.trunk_hypernet_b)
        old_768_spec = _n(self.teacher_specialist, self.teacher_spec_hypernet)
        # one path = its blocks (incl. its shared HN, counted once) + its projections
        one_path = _n(self.specialists[0], self.down_proj[0], self.up_proj[0],
                      (self.spec_hypernets[0] if C.HYPERNET_SPEC else None))
        all_paths = _n(self.specialists, self.down_proj, self.up_proj,
                       (self.spec_hypernets if C.HYPERNET_SPEC else None))
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        active_infer = shared + one_path   # trunk + ONE narrow path at inference
        return {
            "shared_frozen_base": shared,
            "old_768_specialist(teacher)": old_768_spec,
            "one_narrow_path": one_path,
            "all_narrow_paths": all_paths,
            "trainable": trainable,
            "frozen": frozen,
            "active_inference(trunk+1path)": active_infer,
        }


# ======================================================================
# Checkpoint load: mature C-Rich -> shared base (student) + teacher specialist.
# ======================================================================
# Key layout in a C-Rich single-path checkpoint's model.state_dict():
#   SHARED (load into student, then freeze):
#     wte.* wpe.* ln_f.* lm_head.*  (lm_head tied to wte, usually absent as own key)
#     trunk.*  trunk_hypernet_a.*  trunk_hypernet_b.*
#   OLD 768 SPECIALIST (load into teacher#     into teacher_specialist / teacher_spec_hypernet; NEVER into the narrow paths):
#     specialists.0.*      -> teacher_specialist.*
#     spec_hypernets.0.*   -> teacher_spec_hypernet.*
#
# The narrow student paths (specialists.*/spec_hypernets.* on THIS model, at
# PATH_D_MODEL=576) are intentionally left at their fresh init — Run 2 KD, not a
# 768->576 weight copy, is what transfers behaviour into them.

# Destination prefixes that are the SHARED frozen base: copied verbatim from the
# C-Rich checkpoint (same names, same 768 width) into this model.
_SHARED_PREFIXES = (
    "wte.", "wpe.", "ln_f.", "lm_head.",
    "trunk.", "trunk_hypernet_a.", "trunk_hypernet_b.",
)


def load_c_rich(model, ckpt_path, map_location="cpu", verbose=True):
    """Load a mature single-path C-Rich checkpoint into `model` (a MultiPathModel):

      * SHARED base (wte/wpe/ln_f/lm_head/trunk/trunk_hypernet_a/b) -> same-named
        params on the student (768 width, verbatim).
      * OLD 768 specialist P0 -> the frozen teacher:
            specialists.0.*    -> teacher_specialist.*
            spec_hypernets.0.* -> teacher_spec_hypernet.*
      * The narrow student paths at PATH_D_MODEL are NOT touched (fresh init).
      * lm_head.weight is re-tied to wte.weight after load.

    Returns a report dict:
        {loaded, missing_dest, unexpected_source, teacher_loaded, shared_loaded}
    """
    ckpt = torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,
    )
    # Accept either a raw state_dict or a wrapped checkpoint {'model': sd, ...}.
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        src = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        src = ckpt["state_dict"]
    else:
        src = ckpt
    # Strip a DDP 'module.' prefix if present.
    if any(k.startswith("module.") for k in src):
        src = {k[len("module."):] if k.startswith("module.") else k: v
               for k, v in src.items()}

    dest = model.state_dict()
    new_sd = dict(dest)                    # start from current (keeps fresh narrow paths)
    loaded, shared_loaded, teacher_loaded = 0, 0, 0
    used_src = set()

    def _assign(dst_key, tensor):
        nonlocal loaded
        if dst_key in dest and dest[dst_key].shape == tensor.shape:
            new_sd[dst_key] = tensor.clone()
            loaded += 1
            return True
        return False

    for k, v in src.items():
        # ---- shared frozen base: verbatim same-named copy ----
        if any(k.startswith(p) for p in _SHARED_PREFIXES):
            if _assign(k, v):
                shared_loaded += 1
                used_src.add(k)
            continue
        # ---- old 768 specialist P0 -> teacher ----
        if k.startswith("specialists.0."):
            dst = "teacher_specialist." + k[len("specialists.0."):]
            if _assign(dst, v):
                teacher_loaded += 1
                used_src.add(k)
            continue
        if k.startswith("spec_hypernets.0."):
            dst = "teacher_spec_hypernet." + k[len("spec_hypernets.0."):]
            if _assign(dst, v):
                teacher_loaded += 1
                used_src.add(k)
            continue
        # Everything else (old narrow-less paths P1.., extra 768 specialists,
        # optimizer/step bookkeeping) is deliberately ignored — the new narrow
        # student paths must NOT receive old 768 weights.

    # NEVER copy old 768 specialist weights into the 576 student paths: those keys
    # (specialists.1+.* and the student specialists.*) are simply never assigned.
    missing_dest = [k for k in dest
                    if k not in used_src
                    and (any(k.startswith(p) for p in _SHARED_PREFIXES)
                         or k.startswith("teacher_specialist.")
                         or k.startswith("teacher_spec_hypernet."))
                    and k != "lm_head.weight"          # tied, re-tied below
                    and new_sd[k] is dest[k]]           # untouched dest tensor
    unexpected_source = [k for k in src if k not in used_src]

    model.load_state_dict(new_sd, strict=False)
    # Re-tie the head to the token embedding (the checkpoint's lm_head.weight, if
    # present, was loaded into wte via the tie; make the alias explicit again).
    model.lm_head.weight = model.wte.weight

    if verbose:
        print(f"[multipath] load_c_rich: shared={shared_loaded} teacher={teacher_loaded} "
              f"total={loaded} (missing={len(missing_dest)}, "
              f"unexpected={len(unexpected_source)})")

    return {
        "loaded": loaded,
        "shared_loaded": shared_loaded,
        "teacher_loaded": teacher_loaded,
        "missing_dest": missing_dest,
        "unexpected_source": unexpected_source,
    }


def build_multipath_model(gpt2_config=None, ckpt_path=None, apply_freeze=True,
                          device="cpu"):
    """Construct a MultiPathModel, optionally init it from a C-Rich checkpoint,
    apply freezes, and place it on `device`.

      * The student (shared base + narrow paths) goes to `device`.
      * The frozen 768 teacher goes to `device` ONLY when C.TEACHER_ENABLED
        (ITEM 7: it stays on CPU in Run 1 so it costs no GPU memory).

    Returns (model, load_report) where load_report is None if no ckpt was given.
    """
    model = MultiPathModel(gpt2_config or build_gpt2_config())

    load_rep = None
    if ckpt_path is not None:
        load_rep = load_c_rich(model, ckpt_path, map_location="cpu")

    if apply_freeze:
        model.apply_freezes()

    model.to(device)

    # Teacher placement: only pull it onto the training device when it will be used.
    if getattr(C, "TEACHER_ENABLED", False):
        if hasattr(model, "teacher_to"):
            model.teacher_to(device)
        else:
            model.teacher_specialist.to(device)
            if model.teacher_spec_hypernet is not None:
                model.teacher_spec_hypernet.to(device)
    else:
        model.teacher_specialist.to("cpu")
        if model.teacher_spec_hypernet is not None:
            model.teacher_spec_hypernet.to("cpu")

    return model, load_rep
