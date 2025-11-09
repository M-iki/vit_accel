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

def replace_vit_self_attention_with_linear(module: nn.Module):
    for name, child in list(module.named_children()):
        if child.__class__.__name__.endswith("SelfAttention") and hasattr(child, "query"):
            setattr(module, name, ReLULinearSelfAttention(child))
        else:
            replace_vit_self_attention_with_linear(child)