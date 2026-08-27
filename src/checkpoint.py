"""
checkpoint.py — full-state save/load with atomic Drive writes.

A checkpoint that only saves model weights is a TRAP: on resume the Adam moments
reset and convergence quietly breaks, and the data stream restarts from the top.
So the bundle contains EVERYTHING needed to continue the exact curve:

    model, optimizer, scheduler, consumed_sequences (data position),
    step, RNG states (torch/cuda/numpy/python), and the run config snapshot.

Atomic write: save to a .tmp file, then os.replace() (atomic) onto the final
name. The previous good checkpoint is only pruned AFTER the new one is fully
written, so a Colab death mid-save never leaves you with zero valid checkpoints.

Retention: KEEP_LAST rolling checkpoints (oldest pruned) PLUS every milestone
checkpoint kept forever (the paper comparison points).
"""
import os
import glob
import random
import numpy as np
import torch

import config as C


def _ensure_dir():
    os.makedirs(C.CKPT_DIR, exist_ok=True)


def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _config_snapshot():
    # fully-resolved config (raw constants + derived values + git hash) so every
    # checkpoint traces to an unambiguous configuration.
    return C.resolve()


def save_checkpoint(model, optimizer, scheduler, step, consumed_sequences,
                    is_milestone=False):
    _ensure_dir()
    tag = "milestone" if is_milestone else "rolling"
    final = os.path.join(C.CKPT_DIR, f"ckpt_{tag}_step{step}.pt")
    tmp = final + ".tmp"

    bundle = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "consumed_sequences": consumed_sequences,
        "rng": _rng_state(),
        "config": _config_snapshot(),
        "run_tag": C.run_tag(),
    }

    # atomic: write tmp, fsync, replace
    torch.save(bundle, tmp)
    os.replace(tmp, final)   # atomic on the same filesystem

    # update "latest" pointer (also atomic)
    latest = os.path.join(C.CKPT_DIR, "latest.pt")
    latest_tmp = latest + ".tmp"
    torch.save({"path": final, "step": step}, latest_tmp)
    os.replace(latest_tmp, latest)

    if not is_milestone:
        _prune_rolling()
    print(f"[checkpoint] saved {os.path.basename(final)} "
          f"(step {step}, consumed {consumed_sequences}, {tag})")
    return final


def _prune_rolling():
    """Keep only the newest KEEP_LAST rolling checkpoints. Milestones untouched."""
    rolling = sorted(
        glob.glob(os.path.join(C.CKPT_DIR, "ckpt_rolling_step*.pt")),
        key=lambda p: int(p.split("step")[-1].split(".pt")[0]),
    )
    for old in rolling[:-C.KEEP_LAST]:
        try:
            os.remove(old)
            print(f"[checkpoint] pruned old rolling {os.path.basename(old)}")
        except OSError:
            pass


def find_latest():
    latest = os.path.join(C.CKPT_DIR, "latest.pt")
    if not os.path.exists(latest):
        return None
    ptr = torch.load(latest, map_location="cpu", weights_only=False)
    return ptr["path"] if os.path.exists(ptr["path"]) else None


def load_checkpoint(model, optimizer=None, scheduler=None, path=None, map_location="cpu"):
    """Restore full state. Returns (step, consumed_sequences). If no checkpoint
    exists, returns (0, 0) so training starts fresh."""
    if path is None:
        path = find_latest()
    if path is None:
        print("[checkpoint] none found — starting fresh (step 0).")
        return 0, 0

    bundle = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(bundle["model"])

    # Optimizer state is tied to the param-group layout at save time. If the layout
    # changed since (e.g. TRUNK_LR_MULT switched the optimizer between 2 and 4
    # groups), the saved state no longer matches. Rather than crash, skip the stale
    # optimizer state and continue with a FRESH optimizer: model weights and step
    # are preserved; only Adam moments reset (they re-warm, like any restart). The
    # NEXT checkpoint saves in the new layout and resumes cleanly thereafter.
    opt_loaded = True
    if optimizer is not None and bundle.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(bundle["optimizer"])
        except ValueError as e:
            if "different number of parameter groups" in str(e):
                opt_loaded = False
                print("[checkpoint][WARN] optimizer param-group layout changed since "
                      "this checkpoint was saved (likely a TRUNK_LR_MULT change). "
                      "Resuming with FRESH optimizer state — Adam moments re-init and "
                      "the LR re-warms. Model weights + step are preserved.")
            else:
                raise
    # The scheduler's per-group LR bookkeeping is only consistent with a restored
    # optimizer. If we skipped the optimizer state, skip the scheduler too and let
    # it re-derive from the fresh optimizer (it already re-warms on resume).
    if scheduler is not None and bundle.get("scheduler") is not None and opt_loaded:
        scheduler.load_state_dict(bundle["scheduler"])
    _set_rng_state(bundle["rng"])

    step = bundle["step"]
    consumed = bundle["consumed_sequences"]
    print(f"[checkpoint] resumed {os.path.basename(path)} "
          f"@ step {step}, consumed {consumed}  ({bundle.get('run_tag','')})")
    return step, consumed
