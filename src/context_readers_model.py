# MIT License
#
# Copyright (c) 2025 Songnian Qian
# Multi-MLP Model: Multiple MLP blocks with Gating on All Layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
import math
from transformers import GPT2LMHeadModel, GPT2Config

import functools
import time
import re
from typing import Any, Dict

expert_kinds = ["mlp2e", "mlpe", "perc2", "perc4", "reglu1d", "film"]

class Perc4(nn.Module):
    """
    Rank-4 perceptron (very low-cost, O(E)):
      y = biasvec + Σ_{i=1..4} alpha_i * GELU(<w_i, x> + b_i) * v_i
    """
    def __init__(self, E, **kw):
        super().__init__()                    # ← FIXED
        self.w = nn.Parameter(torch.empty(4, E))
        self.v = nn.Parameter(torch.empty(4, E))
        self.alpha = nn.Parameter(torch.ones(4))
        self.b = nn.Parameter(torch.zeros(4))         # scalar biases per rank
        self.biasvec = nn.Parameter(torch.zeros(E))   # residual bias
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, x):
        # works for x shape [..., E]
        proj = x.matmul(self.w.t()) + self.b           # [..., 4]
        act  = F.gelu(proj) * self.alpha               # [..., 4]
        y    = act.matmul(self.v) + self.biasvec       # [..., E]
        return y


class ReGLU1D(nn.Module):
    """
    ReGLU pair (rank-1 + gate). Feature-wise gain after a squashed projection:
      s = <u, x> + b ; g = σ(s)
      y = g * (a ⊙ x) + biasvec
    """
    def __init__(self, E, **kw):
        super().__init__()                    # ← FIXED
        self.u = nn.Parameter(torch.empty(E))  # projection
        self.a = nn.Parameter(torch.ones(E))   # gate scale (feature-wise)
        self.b = nn.Parameter(torch.zeros(1))  # scalar bias
        self.biasvec = nn.Parameter(torch.zeros(E))
        nn.init.normal_(self.u, std=0.02)

    def forward(self, x):
        # x shape [..., E]
        s = x.matmul(self.u) + self.b          # [... ]
        g = torch.sigmoid(s).unsqueeze(-1)     # [..., 1]
        y = g * (x * self.a) + self.biasvec    # [..., E]
        return y


class FiLM(nn.Module):
    """
    FiLM expert: per-token modulation γ,β from a tiny bottleneck MLP:
      t = GELU(Down(x)); [γ,β] = Up(t) ; y = γ ⊙ x + β
    """
    def __init__(self, E, **kw):
        super().__init__()                    # ← FIXED
        r = max(16, E // 64)                  # tiny bottleneck
        self.down = nn.Linear(E, r, bias=True)
        self.up   = nn.Linear(r, 2*E, bias=True)

    def forward(self, x):
        t  = F.gelu(self.down(x))             # [B,S,r]
        gb = self.up(t)                       # [B,S,2E]
        gamma, beta = gb.chunk(2, dim=-1)
        y = gamma * x + beta
        return y

class MLP4E(nn.Module):
    """
    TRUE dense GPT-2 FFN: E -> 4E -> E with GELU; GPT-2 style init.
    This is the real dense baseline (Model A) — NOT a low-rank expert.
    """
    def __init__(self, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.c_fc   = nn.Linear(hidden_size, 4 * hidden_size)
        self.act    = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        nn.init.normal_(self.c_fc.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)      # [B,S,4E]
        x = self.act(x)
        x = self.c_proj(x)    # [B,S,E]
        x = self.dropout(x)
        return x


class DoubleMLP(nn.Module):
    """
    E -> 2E -> E with GELU; GPT-2 style init.
    """
    def __init__(self, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.c_fc   = nn.Linear(hidden_size, 2 * hidden_size)
        self.act    = nn.GELU()
        self.c_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        nn.init.normal_(self.c_fc.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)      # [B,S,2E]
        x = self.act(x)
        x = self.c_proj(x)    # [B,S,E]
        x = self.dropout(x)
        return x

class ExpertC_RankN(nn.Module):
    """
    Rank-N cosine-style perceptron:
      y = biasvec + Σ_{i=1..N} alpha_i * GELU(<w_i, x> + b_i) * v_i
    Shapes:
      w:[N,E], v:[N,E], alpha:[N], b:[N], biasvec:[E]
    """
    def __init__(self, E: int, rank: int = 2, **kw):
        super().__init__()
        assert rank >= 1, "rank must be >= 1"
        self.rank = int(rank)
        self.w = nn.Parameter(torch.empty(self.rank, E))
        self.v = nn.Parameter(torch.empty(self.rank, E))
        self.alpha = nn.Parameter(torch.ones(self.rank))
        self.b = nn.Parameter(torch.zeros(self.rank))       # scalar bias per rank
        self.biasvec = nn.Parameter(torch.zeros(E))         # residual bias
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [..., E]
        s = x.matmul(self.w.t()) + self.b                  # [..., N]
        g = F.gelu(s)                                      # [..., N]
        # broadcast alpha over leading dims, then contract with v: [...,N] @ [N,E] -> [...,E]
        a = self.alpha.view(*([1] * (g.dim() - 1)), self.rank)
        y = (a * g).matmul(self.v) + self.biasvec          # [..., E]
        return y

class ExpertC_Rank2(ExpertC_RankN):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size, rank=2)
 
class SingleMLP(nn.Module):
    """
    Single MLP block matching GPT-2 architecture but with E->E dimensions
    (instead of the standard E->4E->E)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Standard GPT-2 uses 4*hidden_size, but we use hidden_size for compactness
        self.c_fc = nn.Linear(hidden_size, hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()  # GPT-2 uses GELU (actually "gelu_new" but GELU is close)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights similar to GPT-2
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        returns: [B, S, E]
        """
        x = self.c_fc(x)      # [B, S, E]
        x = self.act(x)       # [B, S, E]
        x = self.c_proj(x)    # [B, S, E]
        x = self.dropout(x)   # [B, S, E]
        return x

class GatingMLP(nn.Module):
    """Gate for selecting among multiple MLP blocks."""
    def __init__(self, hidden_size: int, num_mlps: int, gate_hidden: int = 256):
        super().__init__()
        self.num_mlps = num_mlps
        self.force_uniform = False  # can be toggled externally
        self.net = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_mlps)
        )

    def forward(self, x, *, force_uniform: bool | None = None):  # x: [B,S,E]
        """
        Returns logits over MLPs: [B,S,M].
        If force_uniform=True (or self.force_uniform), return all-zero logits
        => softmax is uniform across M, which is what we want for Phase A.
        """
        if force_uniform is None:
            force_uniform = self.force_uniform

        if force_uniform:
            B, S, _ = x.shape
            # zeros -> uniform after softmax; robust to any temperature scaling
            return x.new_zeros(B, S, self.num_mlps)

        return self.net(x)

class MultiMLPLayer(nn.Module):
    """Multiple MLP blocks for a single layer with gating"""
    def __init__(self, layer_idx: int, hidden_size: int,
                 num_mlps: int = 4, gate_hidden: int = 256,
                 expert_kinds: list[str] | None = None,  # <--- NEW
                 dropout_p: float = 0.1):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Build experts
        kinds = expert_kinds if (expert_kinds and len(expert_kinds) > 0) else None
        if kinds is None:
            # default: keep previous behavior (homogeneous SingleMLP)
            kinds = ["mlpe"] * num_mlps
        self.num_mlps = len(kinds)
        self.expert_kinds = [str(k).lower() for k in kinds]
        self.expert_bias = nn.Parameter(torch.zeros(self.num_mlps))

        experts = []
        for k in kinds:
            k = k.lower()

            if k in ("mlp4e", "dense", "gpt2ffn"):
                experts.append(MLP4E(hidden_size, dropout_p))
            elif k in ("mlp2e", "double", "wide2e"):
                experts.append(DoubleMLP(hidden_size, dropout_p))
            elif k in ("mlpe", "single", "e2e"):
                experts.append(SingleMLP(hidden_size))
            elif k in ("reglu1d", "reglu"):
                experts.append(ReGLU1D(hidden_size))
            elif k in ("film", "filme"):
                experts.append(FiLM(hidden_size))
            else:
                # percN / rankN / gelupercN
                m = re.fullmatch(r"(?:gelu)?(?:perc|rank)(\d+)", k)
                if m:
                    r = int(m.group(1))
                    experts.append(ExpertC_RankN(hidden_size, rank=r))
                else:
                    raise ValueError(f"Unknown expert kind '{k}'")

        self.mlps = nn.ModuleList(experts)

        # Post-mix scale to keep residual addition well-behaved
        self.post_mix_alpha = nn.Parameter(torch.tensor(0.5))

        # Gate for selecting among M experts
        self.gate = GatingMLP(hidden_size, self.num_mlps, gate_hidden)

        # Stats
        self.register_buffer("gate_entropy", torch.zeros(1))
        self.register_buffer("usage_counts", torch.zeros(self.num_mlps, dtype=torch.float32))
        self.register_buffer("selection_counts", torch.zeros(self.num_mlps, dtype=torch.float32))
        # GPU accumulators for gate std/entropy (design note item 7): NO .item() in
        # the forward. These sum over forwards; divide by fwd_count at report time.
        self.register_buffer("gate_std_sum", torch.zeros(1))
        self.register_buffer("gate_entropy_sum", torch.zeros(1))
        self.register_buffer("fwd_count", torch.zeros(1))

        self._last_gate_std = None  # legacy; populated only by explicit report calls
        self._last_gate_entropy = None

    @torch.no_grad()
    def gate_diag(self):
        """Report-time (eval) gate diagnostics — reads the GPU accumulators and
        converts to Python ONCE here, never in the training forward. Returns mean
        gate std and mean gate entropy over accumulated training forwards."""
        n = float(self.fwd_count.item()) or 1.0
        std = float(self.gate_std_sum.item()) / n
        ent = float(self.gate_entropy_sum.item()) / n
        self._last_gate_std, self._last_gate_entropy = std, ent
        return {"gate_std": std, "gate_entropy": ent, "n_forwards": int(n)}

    @torch.no_grad()
    def reset_gate_diag(self):
        self.gate_std_sum.zero_(); self.gate_entropy_sum.zero_(); self.fwd_count.zero_()

    def _moe_balance_loss(self, pi_soft):
        """Switch-style load balance over experts on AGGREGATE soft gate mass
        (design note item 2). importance = mean soft prob per expert; load = same
        aggregate mass (soft-MoE has no hard dispatch), so this is M * sum(imp*load),
        minimized when mass is spread across experts — WITHOUT pushing any individual
        token toward a uniform [1/M,...] gate. Grad-bearing (pi_soft carries grad)."""
        M = pi_soft.size(-1)
        if M <= 1:
            return pi_soft.new_zeros(())
        importance = pi_soft.mean(dim=(0, 1))              # [M]
        return float(M) * torch.sum(importance * importance)
        """Get statistics for monitoring"""
        total_usage = self.usage_counts.sum().item()
        total_selections = self.selection_counts.sum().item()
        
        if total_usage > 0:
            usage_percentages = (self.usage_counts / total_usage * 100).tolist()
        else:
            usage_percentages = [0.0] * self.num_mlps
            
        if total_selections > 0:
            selection_percentages = (self.selection_counts / total_selections * 100).tolist()
        else:
            selection_percentages = [0.0] * self.num_mlps
        
        return {
            'layer_idx': self.layer_idx,
            'usage_counts': self.usage_counts.tolist(),
            'usage_percentages': usage_percentages,
            'selection_counts': self.selection_counts.tolist(),
            'selection_percentages': selection_percentages,
            'total_usage': total_usage,
            'total_selections': total_selections,
            'num_mlps': self.num_mlps,
        }
    
    def forward(
        self,
        x,
        *,
        soft_routing: bool = True,
        tau: float = 1.0,
        top_k: int = 1,
        compute_only_selected: bool = True,
        routing_cfg: dict | None = None,
    ):
        """
        x: [B, S, E]
        soft_routing: True → soft mixture over experts
        tau: temperature (also used for Gumbel in hard path)
        top_k: number of experts to keep in sparse/hard path
        compute_only_selected: compute only selected experts when routing is sparse
        routing_cfg: optional dict to override the above and to set 'force_uniform_gate'
        """

        # ---- resolve config (routing_cfg overrides kwargs) ----
        if routing_cfg:
            soft_routing = bool(routing_cfg.get("soft_routing", soft_routing))
            tau = float(routing_cfg.get("tau", tau))
            top_k = int(routing_cfg.get("top_k", top_k))
            compute_only_selected = bool(routing_cfg.get("compute_only_selected", compute_only_selected))
            force_uniform = bool(routing_cfg.get("force_uniform_gate", False))
        else:
            force_uniform = False

        B, S, E = x.shape
        device = x.device
        M = self.num_mlps

        # ---- gating (one call) ----
        gate_logits = self.gate(x, force_uniform=force_uniform)
        if not force_uniform:
            gate_logits = gate_logits + self.expert_bias.to(gate_logits.dtype)
        pi_soft = F.softmax(gate_logits / max(tau, 1e-6), dim=-1)           # [B, S, M]

        # gate diagnostics accumulate on-GPU (design note item 7 — NO .item() in the
        # training forward). Converted to Python only at report time via gate_diag().
        if self.training:
            with torch.no_grad():
                if pi_soft.size(-1) > 1:
                    self.gate_std_sum += pi_soft.float().std(dim=-1).mean()
                p = pi_soft.clamp_min(1e-8)
                self.gate_entropy_sum += (-(p * p.log()).sum(dim=-1)).mean()
                self.fwd_count += 1

        if soft_routing:
            # ===== Soft mixture over experts (design note item 1: full E0..E4 soft
            # mixture when top_k >= M — which is the intended Project B setting) =====
            if top_k < M:
                topv, topi = torch.topk(pi_soft, k=top_k, dim=-1)           # [B, S, k]
                mask = torch.zeros_like(pi_soft).scatter_(-1, topi, 1.0)
                weights = pi_soft * mask
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
            else:
                weights = pi_soft                                           # [B, S, M]

            outputs = [mlp(x) for mlp in self.mlps]                         # M × [B,S,E]
            all_out = torch.stack(outputs, dim=-1)                          # [B, S, E, M]
            # match dtypes to avoid AMP promotion issues
            weights = weights.to(all_out.dtype)
            mixed = (all_out * weights.unsqueeze(-2)).sum(dim=-1)           # [B, S, E]
            mixed = self.post_mix_alpha * mixed

            # MoE load-balance loss (design note item 2): switch-style over the
            # AGGREGATE soft gate mass across tokens — prevents collapse/dead experts
            # WITHOUT forcing per-token uniformity. Grad-bearing (uses pi_soft).
            moe_balance = self._moe_balance_loss(pi_soft)

            # --- usage stats (no grad, GPU) ---
            if self.training:
                with torch.no_grad():
                    self.usage_counts.add_(pi_soft.sum(dim=(0, 1)).to(self.usage_counts.dtype))
                    sel = F.one_hot(pi_soft.argmax(dim=-1), num_classes=M).sum(dim=(0, 1))
                    self.selection_counts.add_(sel.to(self.selection_counts.dtype))
                    ent = (-(pi_soft.clamp_min(1e-9).log() * pi_soft).sum(dim=-1)).mean()
                    self.gate_entropy.add_(ent.to(self.gate_entropy.device))

            return mixed, pi_soft, moe_balance

        # ===== Hard / Sparse routing =====
        top_k = int(min(max(top_k, 1), M))

        if top_k == 1 and compute_only_selected:
            # ---- hard top-1 with Gumbel ----
            pi_hard = F.gumbel_softmax(gate_logits, tau=tau, hard=True, dim=-1)  # [B, S, M]
            selected_idx = pi_hard.argmax(dim=-1)                                # [B, S]

            out = torch.zeros_like(x)  # matches dtype & device
            for mlp_idx in torch.unique(selected_idx):
                mask = (selected_idx == mlp_idx)                                 # [B, S]
                if mask.any():
                    pos = torch.where(mask)
                    x_sel = x[pos]                                               # [N, E]
                    y_sel = self.mlps[int(mlp_idx)](x_sel)                       # [N, E]
                    y_sel = (self.post_mix_alpha * y_sel).to(out.dtype)
                    out[pos] = y_sel

            # --- stats (no grad) ---
            if self.training:
                with torch.no_grad():
                    self.usage_counts.add_(pi_soft.sum(dim=(0, 1)).to(self.usage_counts.dtype))
                    sel_counts = torch.zeros(M, device=device, dtype=self.selection_counts.dtype)
                    uniq, cnt = torch.unique(selected_idx, return_counts=True)
                    sel_counts[uniq] = cnt.to(sel_counts.dtype)
                    self.selection_counts.add_(sel_counts)
                    ent = (-(pi_soft.clamp_min(1e-9).log() * pi_soft).sum(dim=-1)).mean()
                    self.gate_entropy.add_(ent.to(self.gate_entropy.device))

            return out, pi_hard, out.new_zeros(())

        # ---- general top-k sparse mixture (compute selected experts only) ----
        topv, topi = gate_logits.topk(k=top_k, dim=-1)                           # [B, S, k]
        pi_k = F.softmax(topv / max(tau, 1e-6), dim=-1)                          # [B, S, k]

        out = torch.zeros_like(x)
        for k in range(top_k):
            mlp_idx_map = topi[..., k]                                           # [B, S]
            w_k = pi_k[..., k]                                                   # [B, S]
            for mlp_idx in torch.unique(mlp_idx_map):
                mask = (mlp_idx_map == mlp_idx)
                if mask.any():
                    pos = torch.where(mask)
                    x_sel = x[pos]                                               # [N, E]
                    y_sel = self.mlps[int(mlp_idx)](x_sel).to(out.dtype)         # [N, E]
                    w_sel = w_k[pos].unsqueeze(-1).to(out.dtype)                 # [N, 1]
                    # accumulate (sum over selected experts)
                    out[pos] += self.post_mix_alpha * y_sel * w_sel

        # --- stats (no grad) ---
        if self.training:
            with torch.no_grad():
                self.usage_counts.add_(pi_soft.sum(dim=(0, 1)).to(self.usage_counts.dtype))
                sel_counts = torch.zeros(M, device=device, dtype=self.selection_counts.dtype)
                for k in range(top_k):
                    idx = topi[..., k].reshape(-1)
                    uniq, cnt = torch.unique(idx, return_counts=True)
                    sel_counts[uniq] = cnt.to(sel_counts.dtype)
                self.selection_counts.add_(sel_counts)
                ent = (-(pi_soft.clamp_min(1e-9).log() * pi_soft).sum(dim=-1)).mean()
                self.gate_entropy.add_(ent.to(self.gate_entropy.device))

        pi_sparse = torch.zeros_like(gate_logits)
        pi_sparse.scatter_(-1, topi, pi_k)                                       # [B, S, M] sparse weights
        return out, pi_sparse, out.new_zeros(())

# Simple attention mechanism (GPT-2 compatible)
class GPT2Attention(nn.Module):
    """GPT-2 compatible multi-head attention"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection like GPT-2
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(getattr(config, 'attn_pdrop', 0.1))
        self.resid_dropout = nn.Dropout(getattr(config, 'resid_pdrop', 0.1))
        
        # Register causal mask as BOOLEAN
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False
        )
        
    def _attn(self, q, k, v, attention_mask=None):
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask - now properly boolean
        seq_len = q.size(-2)
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(~causal_mask, -1e4)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, 1, S]
            mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, S, E = hidden_states.shape
        
        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(E, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self._attn(q, k, v, attention_mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return (attn_output,)

class CustomGPT2Block(nn.Module):
    """GPT-2 block with Multi-MLP layer replacing standard MLP"""
    def __init__(self, config, layer_idx: int, num_mlps: int = 4,
                 expert_kinds: list[str] | None = None):   # <--- NEW
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        self.mlp_dropout = nn.Dropout(getattr(config, 'resid_pdrop', 0.1))

        self.mlp = MultiMLPLayer(
            layer_idx=layer_idx,
            hidden_size=config.n_embd,
            num_mlps=num_mlps,
            gate_hidden=getattr(config, "gate_hidden", 256),
            expert_kinds=expert_kinds,                          # <--- pass through
            dropout_p=getattr(config, "resid_pdrop", 0.1),
        )

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        # routing knobs (defaults kept for BC)
        soft_routing: bool = True,
        tau: float = 1.0,
        top_k: int = 1,
        compute_only_selected: bool = True,
        # NEW: per-call routing config dict
        routing_cfg: dict | None = None,
        # NEW: surface per-token gate weights for diagnostics (BC: off by default)
        return_gates: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask)
        attn_out = attn_outputs[0]
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        # ---- merge config: routing_cfg overrides the kwargs above ----
        if routing_cfg is not None:
            soft_routing = bool(routing_cfg.get("soft_routing", soft_routing))
            tau = float(routing_cfg.get("tau", tau))
            top_k = int(routing_cfg.get("top_k", top_k))
            compute_only_selected = bool(routing_cfg.get("compute_only_selected", compute_only_selected))
            force_uniform_gate = bool(routing_cfg.get("force_uniform_gate", False))
        else:
            force_uniform_gate = False

        # Build a single config dict to pass to the MLP (per-call, no state leak)
        mlp_cfg = {
            "soft_routing": soft_routing,
            "tau": tau,
            "top_k": top_k,
            "compute_only_selected": compute_only_selected,
            "force_uniform_gate": force_uniform_gate,  # Phase A: set True from the trainer
        }

        # Call the MLP with routing_cfg (MultiMLPLayer.forward returns
        # (out, gate_weights, moe_balance) — baseline ignores the balance term).
        _ret = self.mlp(
            hidden_states,
            routing_cfg=mlp_cfg,
        )
        mlp_out, gate_weights = _ret[0], _ret[1]

        mlp_out = self.mlp_dropout(mlp_out)
        hidden_states = mlp_out + residual

        outputs = (hidden_states,)
        if use_cache:
            outputs += (None,)
        if output_attentions:
            outputs += (None,)
        if return_gates:
            # gate_weights: [B, S, M] per-token expert weights for this layer
            outputs += (gate_weights,)
        return outputs

class MultiMLPGPT2Model(nn.Module):
    def __init__(self, config, use_pretrained=False, pretrained_model=None,
                 num_mlps=2, expert_kinds: list[str] | None = None):   # <--- NEW
        super().__init__()
        self.config = config
        self.num_mlps = num_mlps
        self.expert_kinds = expert_kinds

        print(f"Creating Multi-MLP GPT2 model")

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.h = nn.ModuleList([
            CustomGPT2Block(config, layer_idx=i,
                            num_mlps=num_mlps,
                            expert_kinds=expert_kinds)          # <--- pass through
            for i in range(config.n_layer)
        ])
       
        # Initialize weights BEFORE tying
        self.apply(self._init_weights)
        if use_pretrained and pretrained_model:
            self._load_pretrained_backbone(pretrained_model)
        self.lm_head.weight = self.wte.weight
        
        # Tie LM head weights to token embeddings (after initialization)
        self.lm_head.weight = self.wte.weight
        print("Tied LM head weights to token embeddings")
        
        print(f"Initialized {config.n_layer} layers with {num_mlps} perceptrons each")

    @staticmethod
    def _init_perc2_from_hf(expert: ExpertC_Rank2, hf_mlp):
        W1 = hf_mlp.c_fc.weight.data    # [H,E]
        b1 = hf_mlp.c_fc.bias.data      # [H]
        W2 = hf_mlp.c_proj.weight.data  # [E,H]
        b2 = hf_mlp.c_proj.bias.data    # [E]
        init_rank2_from_hf(expert, W1, b1, W2, b2)

    def _load_pretrained_backbone(self, name: str):
        """
        Load GPT-2 backbone (embeddings, layer norms, attention) from HF weights
        and copy them into this model. Skips our Multi-MLP blocks.
        """
        print(f"Loading GPT-2 backbone from '{name}' (skipping Multi-MLP weights)")
        hf = GPT2LMHeadModel.from_pretrained(name)  # <-- HF model, not our class
        sd = hf.state_dict()
        own = self.state_dict()

        def copy_if_exists(dst_key: str, src_key: str):
            if src_key in sd and dst_key in own:
                if own[dst_key].shape == sd[src_key].shape:
                    with torch.no_grad():
                        own[dst_key].copy_(sd[src_key])

        # --- embeddings & final layer norm ---
        copy_if_exists("wte.weight", "transformer.wte.weight")
        copy_if_exists("wpe.weight", "transformer.wpe.weight")
        copy_if_exists("ln_f.weight", "transformer.ln_f.weight")
        copy_if_exists("ln_f.bias",   "transformer.ln_f.bias")

        # --- per-layer attention + LNs (skip our MLPs entirely) ---
        n = min(self.config.n_layer, hf.transformer.config.n_layer)
        for i in range(n):
            # layer norms
            copy_if_exists(f"h.{i}.ln_1.weight", f"transformer.h.{i}.ln_1.weight")
            copy_if_exists(f"h.{i}.ln_1.bias",   f"transformer.h.{i}.ln_1.bias")
            copy_if_exists(f"h.{i}.ln_2.weight", f"transformer.h.{i}.ln_2.weight")
            copy_if_exists(f"h.{i}.ln_2.bias",   f"transformer.h.{i}.ln_2.bias")

            # attention (QKV + out proj)
            copy_if_exists(f"h.{i}.attn.c_attn.weight", f"transformer.h.{i}.attn.c_attn.weight")
            copy_if_exists(f"h.{i}.attn.c_attn.bias",   f"transformer.h.{i}.attn.c_attn.bias")
            copy_if_exists(f"h.{i}.attn.c_proj.weight", f"transformer.h.{i}.attn.c_proj.weight")
            copy_if_exists(f"h.{i}.attn.c_proj.bias",   f"transformer.h.{i}.attn.c_proj.bias")

        # --- tie LM head to token embeddings (GPT-2 style) ---
        self.lm_head.weight = self.wte.weight
        print("Backbone preload complete; LM head tied to embeddings.")

    def iter_backbone_modules(self):
        for i, blk in enumerate(self.h):
            yield f"h.{i}.ln_1", blk.ln_1
            yield f"h.{i}.attn", blk.attn
            yield f"h.{i}.ln_2", blk.ln_2

    def iter_gate_modules(self):
        for i, blk in enumerate(self.h):
            yield f"h.{i}.mlp.gate", blk.mlp.gate

    def iter_mlp_modules(self):
        for i, blk in enumerate(self.h):
            for j, m in enumerate(blk.mlp.mlps):
                yield f"h.{i}.mlp.mlps.{j}", m

    def set_requires_grad(self, *, backbone=None, gates=None, mlps=None, lm_head=None, embeddings=None):
        def _set(mods, flag):
            if flag is None: return
            for _, m in mods:
                for p in m.parameters(): p.requires_grad = bool(flag)

        if embeddings is not None:
            for p in list(self.wte.parameters()) + list(self.wpe.parameters()):
                p.requires_grad = bool(embeddings)
        if lm_head is not None:
            for p in self.lm_head.parameters():
                p.requires_grad = bool(lm_head)

        _set(self.iter_backbone_modules(), backbone)
        _set(self.iter_gate_modules(), gates)
        _set(self.iter_mlp_modules(), mlps)

    def parameter_groups(self, lr_backbone=0.0, lr_gates=0.0, lr_mlps=0.0, lr_lm=0.0, weight_decay=0.01):
        groups = []
        def _add(mods, lr):
            params = []
            for _, m in mods:
                params += list(m.parameters())
            if params:
                groups.append({"params": [p for p in params if p.requires_grad], "lr": lr, "weight_decay": weight_decay})
        # embeddings & lm
        emb_params = list(self.wte.parameters()) + list(self.wpe.parameters())
        if any(p.requires_grad for p in emb_params):
            groups.append({"params": [p for p in emb_params if p.requires_grad], "lr": lr_backbone, "weight_decay": weight_decay})
        if any(p.requires_grad for p in self.lm_head.parameters()):
            groups.append({"params": [p for p in self.lm_head.parameters() if p.requires_grad], "lr": lr_lm, "weight_decay": weight_decay})
        # rest
        _add(self.iter_backbone_modules(), lr_backbone)
        _add(self.iter_gate_modules(), lr_gates)
        _add(self.iter_mlp_modules(), lr_mlps)
        return [g for g in groups if len(g["params"]) > 0]

    def _prepare_attention_mask(self, attention_mask, input_ids, dtype):
        if attention_mask is None:
            return None
        # Simple binary mask for our simplified attention
        return attention_mask.to(input_ids.device)

    def _init_weights(self, module):
        """Initialize weights for all components"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_layer_stats(self):
        """Get statistics for all layers"""
        stats = {}
        for i, layer in enumerate(self.h):
            stats[f'layer_{i}'] = layer.mlp.get_mlp_stats()
        return stats

    def set_routing_mode(self, layer_configs):
        """
        Set routing mode for each layer
        layer_configs: dict mapping layer_idx to config dict with keys:
            - soft_routing: bool
            - tau: float
            - top_k: int
            - compute_only_selected: bool
        """
        self.layer_configs = layer_configs

    # @trace_forward("MultiMLPGPT2Model")
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        labels=None,
        global_routing_config=None,
        return_gates=False,
    ):
        # ---- defaults / config ----
        gate_collect = [] if return_gates else None
        if return_dict is None:
            return_dict = getattr(self.config, "use_return_dict", True)
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)

        # Never cache during supervised/KD training
        if labels is not None:
            use_cache = False

        # Friendly default routing; Phase A can pass {"force_uniform_gate": True}
        if global_routing_config is None:
            global_routing_config = {
                "soft_routing": True,
                "tau": 1.2,
                "top_k": 1,
                "compute_only_selected": False,
                "force_uniform_gate": False,
            }

        # ---- shapes & ids ----
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, dtype=torch.long, device=input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        # ---- embeddings ----
        inputs_embeds   = self.wte(input_ids)        # [B,S,E]
        position_embeds = self.wpe(position_ids)     # [B,S,E]
        hidden_states   = inputs_embeds + position_embeds

        # ---- extended attention mask ----
        extended_mask = self._prepare_attention_mask(attention_mask, input_ids, hidden_states.dtype)

        # ---- collectors ----
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions    = () if output_attentions else None

        # ---- transformer blocks ----
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Merge per-layer override onto the global config
            layer_cfg = dict(global_routing_config)
            per_layer_overrides = getattr(self, "layer_configs", {}).get(i, None)
            if per_layer_overrides:
                layer_cfg.update(per_layer_overrides)

            outputs = block(
                hidden_states,
                layer_past=(past_key_values[i] if past_key_values is not None else None),
                attention_mask=extended_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                routing_cfg=layer_cfg,  # <-- single dict passed down
                return_gates=return_gates,
            )

            # expected: (hidden_states, present?, attn_weights?, gates?)
            hidden_states = outputs[0]
            if return_gates:
                gate_collect.append(outputs[-1])  # [B,S,M] for this layer

            if use_cache:
                presents = presents + ((outputs[1] if len(outputs) > 1 else None),)

            if output_attentions:
                attn_idx = 2 if use_cache else 1
                all_attentions = all_attentions + ((outputs[attn_idx] if len(outputs) > attn_idx else None),)

        # ---- final norm & LM head ----
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)  # [B,S,V]

        # ---- loss (shifted, CAUSAL CROP) ----
        # Uniform scoring window with the Path Model (Option 2): score only target
        # positions >= score_from. The dense baseline has no selector and is already
        # causal, but uses the SAME window so its PPL is directly comparable. After
        # the shift, shifted index i scores target i+1, so slice at score_from-1.
        loss = None
        n_scored = None
        if labels is not None:
            _sf = int(getattr(self.config, "score_from", 0))
            start = max(_sf - 1, 0)
            shift_logits = logits[..., start:-1, :].contiguous()
            shift_labels = labels[..., start + 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            n_scored = int((shift_labels != -100).sum().item())

        # If we "cached" only Nones, return None for past_key_values
        if use_cache and presents is not None:
            past_out = presents if any(p is not None for p in presents) else None
        else:
            past_out = None

        # ---- outputs ----
        if not return_dict:
            out = (logits,)
            if past_out is not None:
                out = out + (past_out,)
            if output_hidden_states:
                out = out + (all_hidden_states,)
            if output_attentions:
                out = out + (all_attentions,)
            if loss is not None:
                out = (loss,) + out
            return out

        from transformers.modeling_outputs import CausalLMOutputWithPast
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_out,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        if return_gates:
            # list of [B,S,M] per layer; consumed by diagnostics for per-sequence MI
            result.gate_weights = gate_collect
        result.n_scored = n_scored
        return result

    @staticmethod
    def create_multi_lp_model(config_name="gpt2", num_mlps=2, expert_kinds: list[str] | None = None):
        """Factory function to create Multi-MLP model"""
        config = GPT2Config.from_pretrained(config_name)
        config.num_mlps = num_mlps

        model = MultiMLPGPT2Model(
            config=config,
            use_pretrained=False,
            pretrained_model=config_name,
            num_mlps=num_mlps,
            expert_kinds=expert_kinds,    # <--- pass through
        )
        return model, config

class MultiMLPTrainer:
    """Simplified trainer class for Multi-MLP models"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.global_step = 0
        
    def train_step(self, batch, optimizer):
        """Simple training step - just forward/backward, no phase logic"""
        self.model.train()
        device = self.device

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
            output_attentions=False,
        )
        
        loss = outputs.loss
        return {"total_loss": float(loss.detach().item())}

    def get_all_layer_stats(self):
        """Get routing statistics for all layers"""
        return self.model.get_layer_stats()

@torch.no_grad()
def init_rank2_from_hf(expert: ExpertC_Rank2, W1, b1, W2, b2):
    """
    Project the HF MLP (W2,GELU,W1,biases) onto 2 rank-1 components via SVD of M=W2@W1.
    """
    E = W2.shape[0]
    M = W2 @ W1                              # [E,E]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Top-2 components (u_i, s_i, v_i)
    for i in range(2):
        u_i = U[:, i]                        # [E]
        s_i = S[i].item()
        v_i = Vh[i, :]                       # [E]
        expert.w[i].copy_(v_i)               # input projection
        expert.v[i].copy_(u_i)               # output shaping
        # scale to roughly match gain; 2.0 offsets GELU’s ~0.5 around 0
        expert.alpha[i].fill_( 2.0 * s_i / (v_i.norm() + 1e-8) )

    u_bias = W2 @ b1                         # [E]
    expert.b.copy_( torch.tensor([u_bias.mean().item(), u_bias.mean().item()]) )
    expert.biasvec.copy_( b2 + 0.5 * u_bias / E )

@torch.no_grad()
def init_expertC_from_hf_rank2(expert: ExpertC_Rank2, W1, b1, W2, b2):
    M = W2 @ W1
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    for i in range(2):
        u = U[:, i]
        v = Vh[i, :]
        s = S[i].item()
        expert.w[i].copy_(v)
        expert.v[i].copy_(u)
        expert.alpha[i].fill_( 2.0 * s / (v.norm() + 1e-8) )
    u_bias = W2 @ b1
    expert.b.copy_( torch.tensor([u_bias.mean().item(), u_bias.mean().item()]) )
    expert.biasvec.copy_( b2 + 0.5 * u_bias / u_bias.shape[0] )

@torch.no_grad()
def init_expertC_from_hf_rankN(expert: ExpertC_RankN, W1, b1, W2, b2):
    """
    Project HF MLP (W2, GELU, W1) onto N rank-1 components via top-N SVD of M = W2 @ W1.
    Fills expert.w[v_i], expert.v[u_i], expert.alpha[gain], expert.b, expert.biasvec.
    """
    E = W2.shape[0]
    N = expert.rank
    M = W2 @ W1  # [E, E]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    K = min(N, U.size(1))
    for i in range(K):
        u_i = U[:, i]          # [E]
        s_i = S[i].item()
        v_i = Vh[i, :]         # [E]
        expert.w[i].copy_(v_i)
        expert.v[i].copy_(u_i)
        expert.alpha[i].fill_(2.0 * s_i / (v_i.norm() + 1e-8))  # “2.0” offsets GELU’s ~0.5 linear region

    # Bias projection (same idea as your rank-2 init)
    u_bias = W2 @ b1
    mean_b = u_bias.mean().item()
    expert.b[:K].fill_(mean_b)
    expert.biasvec.copy_(b2 + 0.5 * u_bias / E)

@torch.no_grad()
def init_perc_from_mlp_svd(perc_module, W1, b1, W2, b2):
    E = W2.shape[0]
    M = W2 @ W1               # [E, E]

    # Top-1 SVD of M
    # torch.linalg.svd: M = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    u = U[:, 0]               # [E]
    s1 = S[0].item()          # scalar
    v = Vh[0, :]              # [E]

    # Choose w along v; choose alpha to reflect overall gain
    w = v.clone()
    alpha = 2.0 * s1 / (w.norm(p=2).item() + 1e-8)  # "2.0" offsets the GELU ~0.5 linear factor

    # Bias handling
    u_bias = (W2 @ b1)        # [E]
    b_scalar = u_bias.mean().item()
    biasvec = b2 + 0.5 * u_bias / E

    perc_module.w.copy_(w)
    perc_module.b.fill_(b_scalar)
    perc_module.scale.fill_(alpha)
    perc_module.biasvec.copy_(biasvec)

@torch.no_grad()
def init_perc_from_mlp_linearize(perc_module,  # your GELUPerceptron(E)
                                 W1, b1, W2, b2):
    """
    W1: [H, E], b1: [H]
    W2: [E, H], b2: [E]
    Initializes perc_module.{w,b,scale,biasvec}.
    """
    E = W2.shape[0]         # hidden size
    H = W1.shape[0]         # inner width (2E or 4E)

    # 1) Contract the effective linear term: M = W2 @ W1 -> [E, E]
    M = W2 @ W1             # [E, E]
    w = M.mean(dim=0)       # [E]  (row-average collapse)

    # 2) Contract the bias-through-MLP term: u = W2 @ b1 -> [E]
    u = W2 @ b1             # [E]
    b_scalar = u.mean().item()  # scalar

    # 3) alpha & biasvec
    # Start with alpha ≈ 1.0 (GELU linear region ~ 0.5 is already considered implicitly)
    alpha = 1.0

    # You can absorb part of u into biasvec for a closer match:
    biasvec = b2 + 0.5 * u / E

    # 4) Optional normalization to keep magnitudes tame:
    # Scale w so that alpha * 0.5 * ||w|| has similar average magnitude to rows of M
    target = M.abs().mean().item()
    denom  = (w.abs().mean().item() + 1e-8)
    scale_fix = target / denom
    w = w * scale_fix
    # Adjust alpha inversely to keep product similar
    alpha = alpha / max(scale_fix, 1e-8)

    # 5) Load into perceptron
    perc_module.w.copy_(w)
    perc_module.b.fill_(b_scalar)
    perc_module.scale.fill_(alpha)
    perc_module.biasvec.copy_(biasvec)
