"""
clustered_hypernet.py — C-Rich ClusteredHypernetwork + HyperNetBlock.

Restores selected parts of the richer Dyck HyperNet mechanism INSIDE the current
Project-B/C architecture (8-layer trunk + 4-layer specialist path, heterogeneous
multi-expert FFN, trunk/path HN separation, normal HN LR, gain ~0.1). What is
restored from the Dyck HyperNet:

  1. RICH 3*M reasoning vector R_HN = concat[router_out, dsin_probs, dcos_probs],
     each channel M-dim, per-channel normalized by ONE shared LayerNorm(M).
       trunk M=128 -> R_HN=384 ;  specialist M=64 -> R_HN=192.
  2. CONCATENATION + learned projection instead of the additive correction:
       ffn_out = Linear([ moe_out | R_HN ])      ([d + 3M] -> d), owned by the block.
  3. SPLIT of the two roles of the hidden state h:
       - centroid routing / state selection: on RAW h (current-C behavior),
         normalize(h) -> cosine(h, centroids) -> hard top-1 STE. NOT on RoPE Y.
       - rich feature generation only: build_Y_from_X(h) -> (Y, sX, cX); Y ignored;
         dsin=cX, dcos=-sX; normalize, cosine vs the SAME centroid bank, softmax.

Deliberately NOT restored (Dyck logic side channel / other old bits):
  - logic_map, ReasoningLogicInjector, the 4-D [num/op/logic/delim] token-category
    vector and its 4->d_model injection into token embeddings
  - RL centroid-selection machinery ; old layer-history/state averaging
  - old dense GPT-2 d->4d->d FFN ; 10x HN LR ; old high (~0.743) initial gain
  - Y-based centroid routing (routing stays on raw h)

Switch-style load balance retained. One Hypernet instance is SHARED across the
layers of its cluster (trunk, or one path).
"""
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config


class ClusteredHypernetwork(nn.Module):
    def __init__(self, config: GPT2Config, num_centroids: int = 128,
                 normalize: bool = True, temperature: float = 5.0,
                 top_k_train: int = 1, top_k_eval: int = 1,
                 balance_mode: str = "switch", balance_eps: float = 1e-9,
                 min_gain: float = 0.05):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_centroids = num_centroids
        self.normalize = normalize
        self.temperature = temperature
        self.top_k_train = int(max(1, top_k_train))
        self.top_k_eval = int(max(1, top_k_eval))
        self.balance_mode = str(balance_mode or "switch").lower()
        self.balance_eps = float(balance_eps)

        self.centroids = nn.Parameter(torch.randn(num_centroids, self.hidden_size))
        nn.init.xavier_uniform_(self.centroids)

        self.router_perceptron = nn.Sequential(
            nn.Linear(num_centroids, num_centroids), nn.ReLU(),
        )
        # C-Rich: the HyperNet emits a RICH 3*M reasoning vector
        #   R_HN = [ router_out , dsin_probs , dcos_probs ]      (dim = 3*M)
        # to be CONCATENATED with the MoE output and learned-projected back to
        # d_model *by the block* (HyperNetBlock owns the [d + 3M] -> d Linear). The
        # old additive M->d_model hyper_proj is REMOVED — combination is now concat,
        # not addition.
        #
        # Per-channel normalization before concat, matching the old Dyck design: ONE
        # shared LayerNorm(M) is reused for all three M-dim channels (router_out,
        # dsin_probs, dcos_probs) — not three separate norms.
        self.reason_dim = 3 * num_centroids
        self.channel_norm = nn.LayerNorm(num_centroids)

        # ---- centroid usage counters (design-note review fixes #2/#3/#4) ----
        # centroid_usage_counts : LIFETIME over ALL forwards (train + sweep + eval).
        #   Kept for diagnostics/back-compat (get_centroid_usage reads this). It is
        #   POLLUTED by measurement passes by design and must NOT gate rejuvenation.
        # train_usage_window    : gradient-bearing TRAINING passes only, RESETTABLE.
        #   This is the dead-detection signal for rejuvenation — reset every window
        #   so "dead" means "persistently unused in the RECENT window", not "never
        #   used since the last reset" (#3/#4).
        # train_usage_lifetime  : gradient-bearing TRAINING passes only, cumulative.
        #   The exposure/eligibility gate (REJUV_MIN_EXPOSURE) reads this so a
        #   hypernet that simply hasn't trained yet is never "revived".
        # count_mode selects which counters a forward updates:
        #   "train"  -> lifetime(all) + train_window + train_lifetime
        #   "sweep"  -> lifetime(all) only  (teacher/eval measurement; never train_*)
        #   "off"    -> nothing
        self.register_buffer('centroid_usage_counts',
                             torch.zeros(num_centroids, dtype=torch.long))
        self.register_buffer('train_usage_window',
                             torch.zeros(num_centroids, dtype=torch.long))
        self.register_buffer('train_usage_lifetime',
                             torch.zeros(num_centroids, dtype=torch.long))
        self.count_mode = "train"     # caller sets to "sweep" during teacher/eval

        self.min_gain = min_gain
        # Init so the gain STARTS SMALL (~0.1), not 0.05+softplus(0)=0.743.
        # We want min_gain + softplus(raw) ~= target_gain.
        import math as _m
        target_gain = 0.1
        sp = max(target_gain - min_gain, 1e-4)           # desired softplus output
        raw0 = _m.log(_m.expm1(sp))                       # inverse softplus
        self.hyper_gain_raw = nn.Parameter(torch.tensor(float(raw0)))
        self.disable_hypernet = False

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def create_rope_position_encoding(self, seq_len, device, dtype):
        half_dim = self.hidden_size // 2
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=dtype)
            * (-math.log(10000.0) / half_dim))
        angles = positions * inv_freq.unsqueeze(0)
        sin, cos = torch.sin(angles), torch.cos(angles)
        sin_pos = torch.zeros(seq_len, self.hidden_size, device=device, dtype=dtype)
        cos_pos = torch.zeros(seq_len, self.hidden_size, device=device, dtype=dtype)
        sin_pos[:, ::2] = sin_pos[:, 1::2] = sin
        cos_pos[:, ::2] = cos_pos[:, 1::2] = cos
        return sin_pos, cos_pos

    def build_Y_from_X(self, X: torch.Tensor):
        b, s, e = X.shape
        sin_pos, cos_pos = self.create_rope_position_encoding(s, X.device, X.dtype)
        X_rot = X * cos_pos.unsqueeze(0) + self.rotate_half(X) * sin_pos.unsqueeze(0)
        sX, cX = torch.sin(X_rot), torch.cos(X_rot)
        Y = torch.empty_like(X_rot)
        Y[..., ::2], Y[..., 1::2] = sX[..., ::2], cX[..., ::2]
        return Y, sX, cX

    def get_centroid_distribution(self, Y_norm: torch.Tensor, training: bool = True):
        b, s, _ = Y_norm.shape
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        similarity = torch.matmul(Y_norm, centroids_norm.t())
        logits = similarity / (self.temperature if training else 1.0)
        soft_probs = F.softmax(logits, dim=-1)

        k = self.top_k_train if training else self.top_k_eval
        k = int(max(1, min(k, self.num_centroids)))
        if k == 1:
            selected = torch.argmax(soft_probs, dim=-1)
            sparse = F.one_hot(selected, self.num_centroids).to(soft_probs.dtype)
        else:
            topk_vals, topk_idx = torch.topk(soft_probs, k=k, dim=-1)
            topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.balance_eps)
            sparse = torch.zeros_like(soft_probs)
            sparse.scatter_(dim=-1, index=topk_idx, src=topk_w)
            selected = topk_idx[..., 0]

        routing_probs = sparse + soft_probs - soft_probs.detach()   # straight-through

        if self.balance_mode in ("off", "none", "0"):
            balance_loss = logits.new_zeros(())
        else:
            importance = soft_probs.mean(dim=(0, 1))
            load = sparse.detach().mean(dim=(0, 1))
            switch_loss = float(self.num_centroids) * torch.sum(importance * load)
            if self.balance_mode == "l2_uniform":
                target = 1.0 / float(self.num_centroids)
                balance_loss = ((load - target) ** 2).mean()
            elif self.balance_mode == "both":
                target = 1.0 / float(self.num_centroids)
                balance_loss = switch_loss + ((load - target) ** 2).mean()
            else:
                balance_loss = switch_loss
        return soft_probs, routing_probs, selected, balance_loss

    def forward(self, layer_inputs: List[torch.Tensor], training: bool = True):
        """C-Rich HyperNet: restores the rich 3*M reasoning vector from the Dyck
        HyperNet, wired into the current Project-B/C architecture.

        The two roles of the hidden state h are SPLIT (they were fused in the old
        Dyck code, which routed on the RoPE-transformed Y):

          (a) centroid routing / state selection  — UNCHANGED current-C behavior:
                normalize(h) -> cosine(h, centroids) -> hard top-1 STE.
              We do NOT route on the RoPE/sin/cos Y.

          (b) rich feature generation ONLY — from the SAME h:
                build_Y_from_X(h) -> (Y, sX, cX)          [Y is ignored here]
                dsin = cX ,  dcos = -sX
                normalize each, cosine vs the SAME centroid bank, softmax
                  -> dsin_probs, dcos_probs                     each [B,S,M]
                router_out = router_perceptron(STE_routing)     [B,S,M]

        The three M-dim channels are each passed through ONE shared LayerNorm(M)
        (per-channel normalization, shared module — old Dyck design), scaled by the
        learnable gain (~0.1 start), then concatenated:
                R_HN = [router_out, dsin_probs, dcos_probs]     [B,S,3M]

        Returns R_HN (NOT a d_model correction). The block concatenates R_HN with
        the MoE output and learn-projects [d + 3M] -> d.
        """
        if not layer_inputs:
            return None, {}
        X = layer_inputs[0]
        b, s, _ = X.shape

        centroids_norm = F.normalize(self.centroids, p=2, dim=-1, eps=1e-8)

        # ---- (a) centroid routing on RAW h — unchanged current-C behavior --------
        h_norm = F.normalize(X, p=2, dim=-1, eps=1e-8)
        similarity = torch.matmul(h_norm, centroids_norm.t())          # [B,S,M]
        soft = F.softmax(similarity / self.temperature, dim=-1)         # [B,S,M]
        selected = torch.argmax(soft, dim=-1)                          # [B,S]
        hard = F.one_hot(selected, self.num_centroids).to(soft.dtype)  # [B,S,M]
        # straight-through: hard one-hot forward, soft gradient backward
        routing = hard + soft - soft.detach()                          # [B,S,M]

        # balance loss (load balancing over centroids), same switch-style objective
        balance_loss = self._balance_loss(soft, hard)

        # GPU-only diagnostic counters — NO .item()/.cpu() in forward (sync fix).
        # Routing (selection) is unchanged, so organizational-health diagnostics
        # (usage counters, dead detection, rejuvenation) keep working as before.
        with torch.no_grad():
            mode = getattr(self, "count_mode", "train")
            if mode != "off":
                counts = torch.bincount(selected.reshape(-1),
                                        minlength=self.num_centroids
                                        ).to(self.centroid_usage_counts.device)
                self.centroid_usage_counts += counts
                if mode == "train":
                    self.train_usage_window += counts
                    self.train_usage_lifetime += counts

        # ---- (b) rich feature channels from the SAME h ---------------------------
        # router_out: selected-centroid signal (STE one-hot through M->M perceptron)
        router_out = self.router_perceptron(
            routing.reshape(b * s, self.num_centroids)
        ).reshape(b, s, self.num_centroids)                            # [B,S,M]

        # sin/cos-derived features. build_Y_from_X returns (Y, sX, cX); Y (the
        # centroid-routing feature of the OLD design) is intentionally discarded.
        _Y, sX, cX = self.build_Y_from_X(X)
        dsin = cX                                                       # d/dx sin = cos
        dcos = -sX                                                      # d/dx cos = -sin

        # compare derivative features against the SAME centroid bank -> M-dim probs.
        # NOTE: no temperature scaling here — the old Dyck rich-feature channels
        # softmaxed their cosine similarities DIRECTLY. Temperature=5 applies ONLY
        # to the raw-h centroid routing softmax above, not to these feature channels.
        dsin_norm = F.normalize(dsin, p=2, dim=-1, eps=1e-8)
        dcos_norm = F.normalize(dcos, p=2, dim=-1, eps=1e-8)
        dsin_probs = F.softmax(
            torch.matmul(dsin_norm, centroids_norm.t()), dim=-1)
        dcos_probs = F.softmax(
            torch.matmul(dcos_norm, centroids_norm.t()), dim=-1)

        # per-channel LayerNorm (ONE shared LayerNorm(M) reused for all three),
        # then gain-scale (small ~0.1 start), then concat into the 3M vector.
        gain = self.min_gain + F.softplus(self.hyper_gain_raw)
        router_out = self.channel_norm(router_out)
        dsin_probs = self.channel_norm(dsin_probs)
        dcos_probs = self.channel_norm(dcos_probs)
        reason = torch.cat([router_out, dsin_probs, dcos_probs], dim=-1)  # [B,S,3M]
        reason = reason * gain

        info = {'soft_probs': soft, 'routing_probs': routing,
                'selected': selected, 'balance_loss': balance_loss}
        return reason, info

    def _balance_loss(self, soft, hard):
        """Switch-style load balance over centroids (importance * load)."""
        importance = soft.mean(dim=(0, 1))                 # [M]
        load = hard.mean(dim=(0, 1))                        # [M]
        if self.balance_mode == "switch":
            return float(self.num_centroids) * torch.sum(importance * load)
        target = 1.0 / float(self.num_centroids)
        return torch.sum((importance - target) ** 2)


class _LowRankFFN(nn.Module):
    """Uniform low-rank FFN (d -> r -> d), no per-token expert dispatch. A single
    trainable mixer used when MOE_MODE='lowrank'. Same input/output shape as the
    multi-expert MoE's primary output, so it drops in as self.moe. Comparing this
    to MOE_MODE='multi_expert' isolates the cost of the per-token gating + expert
    dispatch (gather/scatter, many small matmuls) from the FFN compute itself."""
    def __init__(self, d, r, dropout_p=0.1):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.drop(self.up(self.act(self.down(x))))


class HyperNetBlock(nn.Module):
    """Transformer block with clustered-hypernet augmentation AND per-token
    low-rank expert MoE. The MoE (MultiMLPLayer: perc2/perc4/reglu1d/film)
    replaces the plain FFN; the clustered-hypernet reasoning vector is injected
    onto the MoE output via a small projection (concat -> down-project). Shares
    one ClusteredHypernetwork instance (passed in)."""
    def __init__(self, config, layer_idx, hypernet):
        super().__init__()
        self.layer_idx = layer_idx
        self.hypernet = hypernet
        self.hidden_size = config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head,
            dropout=config.attn_pdrop, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # per-token expert MoE (replaces the plain FFN)
        import config as _C
        from context_readers_model import MultiMLPLayer, MLP4E
        _moe_mode = getattr(_C, "MOE_MODE", "multi_expert")
        if _moe_mode == "lowrank":
            # single uniform low-rank FFN (d -> r -> d), NO per-token expert dispatch.
            # Trainable + checkpointed; isolates the dispatch cost vs multi_expert.
            r = int(getattr(_C, "MOE_LOWRANK_R", 256))
            self.moe = _LowRankFFN(config.n_embd, r, dropout_p=config.resid_pdrop)
        elif _moe_mode == "dense":
            # TRUE dense GPT-2 FFN (d -> 4d -> d), NO per-token expert dispatch and
            # NO gating — the real dense baseline (Model A). Same call shape as
            # _LowRankFFN, so it drops in as self.moe. Trainable + checkpointed;
            # keep its own CKPT_DIR (not weight-compatible with other MOE_MODEs).
            self.moe = MLP4E(config.n_embd, dropout_p=config.resid_pdrop)
        else:
            self.moe = MultiMLPLayer(
                layer_idx=layer_idx, hidden_size=config.n_embd,
                num_mlps=len(_C.OURS_EXPERT_KINDS),
                gate_hidden=getattr(config, "gate_hidden", 256),
                expert_kinds=list(_C.OURS_EXPERT_KINDS))

        # C-Rich combination: the hypernet (when present) returns a RICH 3*M
        # reasoning vector R_HN. It is CONCATENATED with the MoE output and
        # learn-projected back to d_model:
        #       ffn_out = reason_proj( [ moe_out | R_HN ] )     ([d + 3M] -> d)
        # A block with hypernet=None has NO hypernet (per PoC placement) and no
        # reason_proj — MoE-only, zero hypernet cost (the honest A/B "OFF" path).
        if self.hypernet is not None:
            reason_dim = self.hypernet.reason_dim          # 3 * num_centroids
            self.reason_proj = nn.Linear(self.hidden_size + reason_dim,
                                         self.hidden_size)
            self.reset_reason_proj()
        else:
            self.reason_proj = None
        self.mlp_dropout = nn.Dropout(config.resid_pdrop)

    def reset_reason_proj(self):
        """Initialize reason_proj so C-Rich STARTS ~= plain MoE: identity on the
        moe_out block, near-zero (std 1e-3) on the R_HN block, so the hypernet earns
        influence via training (mirrors the old near-zero additive-correction init).

        IMPORTANT: this MUST be re-called at the MODEL level AFTER the model-wide
        self.apply(_init_weights), because that global init recurses into every
        nn.Linear — including reason_proj — and would otherwise clobber this init
        with the generic normal_(0, 0.02). No-op for hypernet-free blocks."""
        if self.reason_proj is None:
            return
        with torch.no_grad():
            self.reason_proj.weight.zero_()
            self.reason_proj.weight[:, :self.hidden_size] = torch.eye(
                self.hidden_size)
            nn.init.normal_(self.reason_proj.weight[:, self.hidden_size:],
                            mean=0.0, std=1e-3)
            nn.init.zeros_(self.reason_proj.bias)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        import config as _C
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        B, S, _ = hidden_states.shape
        causal = torch.triu(torch.ones(S, S, device=hidden_states.device,
                                       dtype=torch.bool), diagonal=1)
        key_padding = None
        if attention_mask is not None and attention_mask.dim() == 2:
            key_padding = ~attention_mask.bool()
        attn_out, attn_w = self.attn(hidden_states, hidden_states, hidden_states,
                                     attn_mask=causal, key_padding_mask=key_padding,
                                     need_weights=output_attentions)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        # FFN mixer (multi-expert MoE, or uniform low-rank / dense per MOE_MODE)
        moe_balance = None
        if getattr(_C, "MOE_MODE", "multi_expert") in ("lowrank", "dense"):
            # plain FFN call (no gating, no soft mixture, no balance loss)
            moe_out = self.moe(hidden_states)
        else:
            # design note item 1: FULL soft mixture over all E0..E4 (top_k = M), not
            # top-1 masking. MOE_SOFT_TOPK defaults to the expert count.
            n_exp = len(getattr(self.moe, "expert_kinds", [])) or self.moe.num_mlps
            soft_topk = int(getattr(_C, "MOE_SOFT_TOPK", n_exp) or n_exp)
            moe_ret = self.moe(hidden_states, soft_routing=True, top_k=soft_topk)
            if isinstance(moe_ret, tuple):
                moe_out = moe_ret[0]
                if len(moe_ret) >= 3:
                    moe_balance = moe_ret[2]         # aggregate soft-mass balance loss
            else:
                moe_out = moe_ret

        # HyperNet by CONCATENATION + learned projection (C-Rich design):
        #   ffn_out = reason_proj( [ moe_out | R_HN ] )
        # No hypernet on this block (placement) OR DISABLE_HYPERNET -> MoE only.
        # This is also the honest "no HyperNet" A/B path: zero hypernet cost.
        if self.hypernet is None or getattr(_C, "DISABLE_HYPERNET", False):
            ffn_out = self.mlp_dropout(moe_out)
            info = {}
        else:
            reason, info = self.hypernet([hidden_states], training=self.training)
            if reason is None:
                ffn_out = self.mlp_dropout(moe_out)
            else:
                fused = torch.cat([moe_out, reason], dim=-1)   # [B,S,d+3M]
                ffn_out = self.mlp_dropout(self.reason_proj(fused))
        if moe_balance is not None:
            info = dict(info); info["moe_balance_loss"] = moe_balance
        hidden_states = residual + ffn_out
        out = (hidden_states, info)
        if output_attentions:
            out = out + (attn_w,)
        return out
