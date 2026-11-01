"""Shared representation network and corpus classification layers."""

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class SharedHeads(nn.Module):
    def __init__(self, d_in, class_counts, hidden=512, dropout=0.3, n_spk=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden // 2, c) for c in class_counts])
        self.spk = nn.Linear(hidden // 2, max(n_spk, 1))

    def encode(self, x, ds):
        return self.trunk(x)

    def forward(self, x, ds, target_head=None):
        h = self.encode(x, ds)
        if target_head is not None:
            return self.heads[target_head](h)
        out = torch.zeros(
            x.size(0), max(hh.out_features for hh in self.heads), device=x.device
        )
        for d in torch.unique(ds):
            mask = ds == d
            out[mask, : self.heads[int(d)].out_features] = self.heads[int(d)](h[mask])
        return out
