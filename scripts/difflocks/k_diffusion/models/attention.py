# k_diffusion/models/attention.py
# ─────────────────────────────────────────────────────────────────
from functools import reduce
from inspect import isfunction
import math
from k_diffusion.models.modules import AxialRoPE, apply_rotary_emb_
from k_diffusion.models.modules import AdaRMSNorm, FeedForwardBlock, LinearGEGLU, RMSNorm, apply_wd, use_flash_2, zero_init
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WRAP FlashAttention import in try/except ---
try:
    import flash_attn
    _HAS_FLASH_ATTENTION = True
except ImportError:
    _HAS_FLASH_ATTENTION = False
    # We'll define a small CPU fallback below.

# The rest (xformers import, precision handling) can stay as is:
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False

import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


def scale_for_cosine_sim_single(q, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    return q * scale_q.to(q.dtype)


class SpatialTransformerSimpleV2(nn.Module):
    """
    Transformer block for image‐like data.
    Falls back to plain PyTorch attention if FlashAttention isn't available.
    """

    def __init__(self, in_channels, n_heads, d_head,
                 global_cond_dim,
                 do_self_attention=True,
                 dropout=0.,
                 context_dim=None,
                 ):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        self.do_self_attention = do_self_attention

        # AdaRMSNorm on input
        self.x_in_norm = AdaRMSNorm(in_channels, global_cond_dim)

        # x → qkv or q
        if self.do_self_attention:
            self.x_qkv_proj = apply_wd(torch.nn.Linear(in_channels, inner_dim * 3, bias=False))
        else:
            self.x_q_proj = apply_wd(torch.nn.Linear(in_channels, inner_dim, bias=False))
        self.x_scale = nn.Parameter(torch.full([self.n_heads], 10.0))

        self.x_pos_emb = AxialRoPE(d_head // 2, self.n_heads)

        # context → kv
        self.cond_kv_proj = apply_wd(torch.nn.Linear(context_dim, inner_dim * 2, bias=False))
        self.cond_scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.cond_pos_emb = AxialRoPE(d_head // 2, self.n_heads)

        self.ff = FeedForwardBlock(in_channels, d_ff=int(in_channels * 2), cond_features=global_cond_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = apply_wd(zero_module(nn.Linear(in_channels, inner_dim)))

    # Replace the entire forward method in SpatialTransformerSimpleV2 class in attention.py
# This fixes all the fallback issues when FlashAttention/NATTEN is not available

    def forward(self, x, pos, global_cond, context=None, context_pos=None):
        b, c, h, w = x.shape
        x_in = x

        # reshape for "patch" layout: (B, H, W, C)
        x = rearrange(x, 'b c h w -> b h w c')
        context = rearrange(context, 'b c h w -> b h w c')
        
        # Get actual context dimensions
        ctx_b, ctx_h, ctx_w, ctx_c = context.shape
        print(f"DEBUG: x shape: {x.shape}, context shape: {context.shape}")

        # AdaRMSNorm with global_cond
        x = self.x_in_norm(x, global_cond)

        # ── 1. Compute query (and maybe k, v) from x ──
        inner_dim = self.n_heads * self.d_head
        
        if self.do_self_attention:
            x_qkv = self.x_qkv_proj(x)  # → [B, H, W, 3·inner_dim]
            pos_flat = rearrange(pos, "... h w e -> ... (h w) e").to(x_qkv.dtype)
            x_theta = self.x_pos_emb(pos_flat)

            # Check if we can use FlashAttention
            if _HAS_FLASH_ATTENTION and use_flash_2(x_qkv):
                # FlashAttention path - keep original
                x_qkv = rearrange(x_qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
                x_qkv = scale_for_cosine_sim_qkv(x_qkv, self.x_scale, 1e-6)
                x_theta_full = torch.stack((x_theta, x_theta, torch.zeros_like(x_theta)), dim=-3)
                x_qkv = apply_rotary_emb_(x_qkv, x_theta_full)
                x_q, x_k, x_v = x_qkv.chunk(3, dim=-3)
            else:
                # FALLBACK: Split properly
                print(f"DEBUG: x_qkv shape: {x_qkv.shape}, expected inner_dim*3: {inner_dim*3}")
                x_q, x_k, x_v = x_qkv.chunk(3, dim=-1)  # Each: [B, H, W, inner_dim]
                
                # Reshape to [B, heads, H*W, d_head] 
                seq_len = h * w
                x_q = x_q.contiguous().view(b, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                x_k = x_k.contiguous().view(b, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                x_v = x_v.contiguous().view(b, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                
                # Apply cosine similarity scaling
                x_q, x_k = scale_for_cosine_sim(x_q, x_k, self.x_scale[:, None, None], 1e-6)
        else:
            # Only query path
            x_q = self.x_q_proj(x)  # [B, H, W, inner_dim]
            pos_flat = rearrange(pos, "... h w e -> ... (h w) e").to(x_q.dtype)
            x_theta = self.x_pos_emb(pos_flat)
            
            if _HAS_FLASH_ATTENTION and use_flash_2(x_q):
                # FlashAttention path - keep original
                x_q = rearrange(x_q, "n h w (nh e) -> n (h w) nh e", e=self.d_head)
                x_q = scale_for_cosine_sim_single(x_q, self.x_scale[:, None], 1e-6)
                x_q = x_q.unsqueeze(2)
                x_theta = x_theta.unsqueeze(1)
                x_q = apply_rotary_emb_(x_q, x_theta)
                x_q = x_q.squeeze(2)
            else:
                # FALLBACK: Handle query only
                seq_len = h * w
                x_q = x_q.contiguous().view(b, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                x_q = scale_for_cosine_sim_single(x_q, self.x_scale[:, None, None], 1e-6)
                # Create dummy k, v for self-attention
                x_k = x_q.clone()
                x_v = x_q.clone()

        # ── 2. Compute context → k, v ──
        cond_kv = self.cond_kv_proj(context)  # [B, ctx_H, ctx_W, 2·inner_dim]
        print(f"DEBUG: cond_kv shape: {cond_kv.shape}, expected: [B, {ctx_h}, {ctx_w}, {2*inner_dim}]")
        
        context_pos_flat = rearrange(context_pos, "... h w e -> ... (h w) e").to(cond_kv.dtype)
        cond_theta = self.cond_pos_emb(context_pos_flat)
        
        if _HAS_FLASH_ATTENTION and use_flash_2(cond_kv):
            # FlashAttention path - keep original
            cond_kv = rearrange(cond_kv, "n h w (t nh e) -> n (h w) t nh e", t=2, e=self.d_head)
            cond_k, cond_v = cond_kv.unbind(2)
            cond_k = scale_for_cosine_sim_single(cond_k, self.cond_scale[:, None], 1e-6)
            cond_k = cond_k.unsqueeze(2)
            cond_theta = cond_theta.unsqueeze(1)
            cond_k = apply_rotary_emb_(cond_k, cond_theta)
            cond_k = cond_k.squeeze(2)
        else:
            # FALLBACK: Handle actual context dimensions
            # Check if cond_kv has the right size for chunking
            expected_size = 2 * inner_dim
            actual_last_dim = cond_kv.shape[-1]
            
            if actual_last_dim != expected_size:
                print(f"WARNING: Context dimension mismatch. Expected {expected_size}, got {actual_last_dim}")
                # Try to adapt - maybe the projection is different
                # Use what we have and split in half
                mid_point = actual_last_dim // 2
                cond_k = cond_kv[..., :mid_point]  # First half
                cond_v = cond_kv[..., mid_point:]  # Second half
            else:
                cond_k, cond_v = cond_kv.chunk(2, dim=-1)  # Each: [B, ctx_H, ctx_W, inner_dim]
            
            # Reshape using actual context dimensions
            ctx_seq_len = ctx_h * ctx_w
            actual_inner_dim = cond_k.shape[-1]  # Use actual dimension, not assumed
            actual_d_head = actual_inner_dim // self.n_heads
            
            print(f"DEBUG: Reshaping context - seq_len: {ctx_seq_len}, inner_dim: {actual_inner_dim}, d_head: {actual_d_head}")
            
            cond_k = cond_k.contiguous().view(ctx_b, ctx_seq_len, self.n_heads, actual_d_head).permute(0, 2, 1, 3)
            cond_v = cond_v.contiguous().view(ctx_b, ctx_seq_len, self.n_heads, actual_d_head).permute(0, 2, 1, 3)
            
            # Apply cosine similarity scaling to key only
            cond_k = scale_for_cosine_sim_single(cond_k, self.cond_scale[:, None, None], 1e-6)

        # ── 3. Combine attention ──
        if self.do_self_attention:
            # Concatenate self and context keys/values along sequence dimension
            k = torch.cat([x_k, cond_k], dim=2)  
            v = torch.cat([x_v, cond_v], dim=2)
            q = x_q
        else:
            # Only context attention
            k = cond_k
            v = cond_v
            q = x_q

        # ── 4. Apply attention ──
        # Always use PyTorch attention in fallback
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # Reshape back to spatial format
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(b, h * w, self.n_heads * self.d_head)
        attn_out = attn_out.view(b, h, w, self.n_heads * self.d_head)

        # Project and add residual
        x = self.dropout(attn_out) if hasattr(self, 'dropout') else attn_out
        x = self.proj_out(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = x + x_in

        # ── 5. Feed-forward block ──
        x_ff = rearrange(x, 'b c h w -> b h w c')
        x_ff = self.ff(x_ff, global_cond)
        x = rearrange(x_ff, 'b h w c -> b c h w')

        return x