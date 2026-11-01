"""Normalization, random seeds, and parallel-sentence assignment."""

import os
import numpy as np


def set_seed(seed=42):
    import random, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    os.environ["PYTHONHASHSEED"] = str(seed)


def speaker_center(blocks, groups, use_std=False):
    out = [b.copy() for b in blocks]
    for g in np.unique(groups):
        idx = groups == g
        for bi in range(len(out)):
            mu = out[bi][idx].mean(0, keepdims=True)
            out[bi][idx] -= mu
            if use_std:
                sd = out[bi][idx].std(0, keepdims=True) + 1e-06
                out[bi][idx] /= sd
    return out


def group_assignment_probs(cost, sent_te, spk_te, n_cls):
    """Hungarian assignment within each (speaker, sentence) group from a cost matrix."""
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    p = np.zeros((len(cost), n_cls), np.float32)
    groups = defaultdict(list)
    for i, (s, sp) in enumerate(zip(sent_te, spk_te)):
        groups[sp, s].append(i)
    for (sp, s), idx in groups.items():
        sub = cost[idx]
        if len(idx) == n_cls:
            ri, ci = linear_sum_assignment(-sub)
            p[np.array(idx)[ri], ci] = 1.0
        else:
            q = np.exp(sub - sub.max(1, keepdims=True))
            q /= q.sum(1, keepdims=True)
            p[idx] = q
    return p
