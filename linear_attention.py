import os, tarfile, urllib.request, json, math, copy, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode

class ReLULinearSelfAttention(nn.Module):
    def __init__(self, src_self_attn: nn.Module, eps: float = 1e-6, row_normalize: bool = True):
        super().__init__()
        self.query   = src_self_attn.query
        self.key     = src_self_attn.key
        self.value   = src_self_attn.value
        self.dropout = getattr(src_self_attn, "dropout", nn.Dropout(0.0))
        self.num_heads = src_self_attn.num_attention_heads
        self.head_dim  = src_self_attn.attention_head_size
        self.hidden_size = self.num_heads * self.head_dim
        self.eps = eps
        self.row_normalize = row_normalize

    def _shape(self, x, b):
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, head_mask=None, output_attentions: bool=False):
        b, L, _ = hidden_states.shape
        q = self._shape(self.query(hidden_states), b)  # (b,h,L,d)
        k = self._shape(self.key(hidden_states),   b)
        v = self._shape(self.value(hidden_states), b)

        # ReLU feature map + 1/sqrt(d) scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        q_phi = F.relu(q) * scale
        k_phi = F.relu(k) * scale

        # KV pre-aggregation
        S = torch.matmul(k_phi.transpose(-2, -1), v)        # (b,h,d,d_v==d)
        z = k_phi.sum(dim=-2)                               # (b,h,d)

        num = torch.matmul(q_phi, S)                        # (b,h,L,d)
        den = torch.sum(q_phi * z.unsqueeze(-2), dim=-1)    # (b,h,L)

        context = num / (den.unsqueeze(-1) + 1e-6)          # (b,h,L,d)
        if head_mask is not None:
            context = context * head_mask[:, :, None, None]
        context = context.transpose(1, 2).reshape(b, L, self.hidden_size)
        context = self.dropout(context)
        return (context, None)
    
class ELU1LinearSelfAttention(nn.Module):
    def __init__(self, src_self_attn: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.query   = src_self_attn.query
        self.key     = src_self_attn.key
        self.value   = src_self_attn.value
        self.dropout = getattr(src_self_attn, "dropout", nn.Dropout(0.0))
        self.num_heads = src_self_attn.num_attention_heads
        self.head_dim  = src_self_attn.attention_head_size
        self.hidden_size = self.num_heads * self.head_dim
        self.eps = eps

    def _shape(self, x, b):
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, head_mask=None, output_attentions: bool=False):
        b, L, _ = hidden_states.shape
        out_dtype = hidden_states.dtype

        # do attention math in fp32 for stability
        hs = hidden_states.float()
        q = self._shape(self.query(hs), b)   # (b,h,L,d)
        k = self._shape(self.key(hs),   b)
        v = self._shape(self.value(hs), b)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Positive, smooth map; avoids dead zeros
        q_phi = (F.elu(q, 1.0) + 1.0) * scale
        k_phi = (F.elu(k, 1.0) + 1.0) * scale

        # Pre-aggregation (linear time)
        S = torch.matmul(k_phi.transpose(-2, -1), v)      # (b,h,d,d)
        z = k_phi.sum(dim=-2)                             # (b,h,d)

        num = torch.matmul(q_phi, S)                      # (b,h,L,d)
        den = torch.sum(q_phi * z.unsqueeze(-2), dim=-1)  # (b,h,L)
        den = den.clamp_min(self.eps)

        context = num / den.unsqueeze(-1)                 # (b,h,L,d)

        if head_mask is not None:
            context = context * head_mask[:, :, None, None]

        context = context.transpose(1, 2).reshape(b, L, self.hidden_size)
        context = self.dropout(context)
        return (context.to(out_dtype), None)

class FAVORLinearSelfAttention(nn.Module):
    """
    Linear-time softmax approximation with positive random features.
    m = rf_dim controls the feature budget; 64–128 works well for ViT-B.
    """
    def __init__(self, src_self_attn: nn.Module, rf_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.query   = src_self_attn.query
        self.key     = src_self_attn.key
        self.value   = src_self_attn.value
        self.dropout = getattr(src_self_attn, "dropout", nn.Dropout(0.0))
        self.num_heads = src_self_attn.num_attention_heads
        self.head_dim  = src_self_attn.attention_head_size
        self.hidden_size = self.num_heads * self.head_dim
        self.rf_dim = rf_dim
        self.eps = eps
        # Per-head Gaussian matrix for random features (fixed, not learned)
        W = torch.randn(self.num_heads, self.head_dim, rf_dim)
        self.register_buffer("rf_W", W, persistent=False)

    def _shape(self, x, b):
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _phi(self, x):  # x: (b,h,L,d) -> (b,h,L,m) positive features
        # Project then stabilize with max-subtraction; include exp(-||x||^2/2)
        proj = torch.einsum("bhld,hdm->bhlm", x, self.rf_W)       # (b,h,L,m)
        proj = proj - proj.max(dim=-1, keepdim=True).values       # stabilize exp
        l2   = 0.5 * (x.pow(2).sum(dim=-1, keepdim=True))         # (b,h,L,1)
        return torch.exp(proj - l2)                                # positive

    def forward(self, hidden_states, head_mask=None, output_attentions: bool=False):
        b, L, _ = hidden_states.shape
        out_dtype = hidden_states.dtype

        hs = hidden_states.float()
        q = self._shape(self.query(hs), b)
        k = self._shape(self.key(hs),   b)
        v = self._shape(self.value(hs), b)

        # mild scaling helps numerics
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q * scale
        k = k * scale

        q_phi = self._phi(q)                         # (b,h,L,m)
        k_phi = self._phi(k)                         # (b,h,L,m)

        # Pre-aggregation in feature space (linear time in L)
        # S: (b,h,m,d), z: (b,h,m)
        S = torch.einsum("bhlm,bhld->bhmd", k_phi, v)
        z = k_phi.sum(dim=-2)

        num = torch.einsum("bhlm,bhmd->bhld", q_phi, S)           # (b,h,L,d)
        den = torch.einsum("bhlm,bhm->bhl",   q_phi, z).clamp_min(self.eps)  # (b,h,L)

        context = num / den.unsqueeze(-1)

        if head_mask is not None:
            context = context * head_mask[:, :, None, None]

        context = context.transpose(1, 2).reshape(b, L, self.hidden_size)
        context = self.dropout(context)
        return (context.to(out_dtype), None)
    
class HSwishLinearSelfAttention(nn.Module):
    def __init__(self, src_self_attn: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.query   = src_self_attn.query
        self.key     = src_self_attn.key
        self.value   = src_self_attn.value
        self.dropout = getattr(src_self_attn, "dropout", nn.Dropout(0.0))
        self.num_heads   = src_self_attn.num_attention_heads
        self.head_dim    = src_self_attn.attention_head_size
        self.hidden_size = self.num_heads * self.head_dim
        self.eps = eps
        self.pre_scale = self.head_dim ** (-0.25)  # temperature match

    def _shape(self, x, b):
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _hswish_plus_one(x):
        # φ(x) = 1 + x * relu6(x+3) / 6  (always > 0)
        return 1.0 + x * F.relu6(x + 3.0) * (1.0 / 6.0)

    def forward(self, hidden_states, head_mask=None, output_attentions: bool=False):
        b, L, _ = hidden_states.shape
        out_dtype = hidden_states.dtype

        # do projections in fp32 under AMP for stability
        hs = hidden_states.float()
        q = self._shape(self.query(hs), b)
        k = self._shape(self.key(hs),   b)
        v = self._shape(self.value(hs), b)

        # scale both q and k
        q = q * self.pre_scale
        k = k * self.pre_scale

        # φ in fp32; clamp to guarantee strictly-positive
        q_phi = torch.clamp(self._hswish_plus_one(q), min=1e-6)
        k_phi = torch.clamp(self._hswish_plus_one(k), min=1e-6)

        # Pre-aggregation (fp32)
        S = torch.matmul(k_phi.transpose(-2, -1), v)      # (b,h,d,d_v)
        z = k_phi.sum(dim=-2)                             # (b,h,d)

        num = torch.matmul(q_phi, S)                      # (b,h,L,d_v)
        den = torch.sum(q_phi * z.unsqueeze(-2), dim=-1)  # (b,h,L)
        den = den.clamp_min(self.eps)

        context = num / den.unsqueeze(-1)                 # (b,h,L,d_v)
        if head_mask is not None:
            context = context * head_mask[:, :, None, None]

        context = context.transpose(1, 2).reshape(b, L, self.hidden_size)
        context = self.dropout(context)
        return (context.to(out_dtype), None)


# def replace_vit_self_attention_with_linear(module: nn.Module):
#     for name, child in list(module.named_children()):
#         if child.__class__.__name__.endswith("SelfAttention") and hasattr(child, "query"):
#             setattr(module, name, ReLULinearSelfAttention(child))
#         else:
#             replace_vit_self_attention_with_linear(child)

def replace_vit_self_attention_with_linear(module: nn.Module, impl="elu1", **kwargs):
    impl_map = {
        "relu": ReLULinearSelfAttention,
        "elu1": ELU1LinearSelfAttention,        
        "favor": FAVORLinearSelfAttention,
        "hswish":HSwishLinearSelfAttention,
    }
    cls = impl_map[impl]
    for name, child in list(module.named_children()):
        if child.__class__.__name__.endswith("SelfAttention") and hasattr(child, "query"):
            setattr(module, name, cls(child, **kwargs))
        else:
            replace_vit_self_attention_with_linear(child, impl=impl, **kwargs)