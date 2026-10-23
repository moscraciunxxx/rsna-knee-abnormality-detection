"""Frozen DenseNet121 head fit. Gold 58 and the mill-870 holdout stay out of the loss."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rsna_knee.metrics import macro_roc_auc
from rsna_knee.v4_split import llm_soft_holdout_indices, make_steven_v4_split, should_abort_gold_overfit

HEAD_EPOCHS = 25
HEAD_LR = 1e-2
HEAD_WEIGHT_DECAY = 0.05
HEAD_SEED = 2026


def densenet_id_split(train, v4) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return fit, mill-870 holdout, and gold ids. The three sets are disjoint."""
    split = make_steven_v4_split(train, v4)
    fit_idx, hold_idx = llm_soft_holdout_indices(split.n_train, frac=0.2, seed=2026)
    fit_ids = tuple(split.train_ids[int(i)] for i in fit_idx.tolist())
    hold_ids = tuple(split.train_ids[int(i)] for i in hold_idx.tolist())
    gold_ids = tuple(split.val_ids)
    if set(fit_ids) & set(hold_ids):
        raise ValueError("holdout leaked into the DenseNet fit")
    if set(fit_ids) & set(gold_ids) or set(hold_ids) & set(gold_ids):
        raise ValueError("gold leaked into the DenseNet fit or holdout")
    return fit_ids, hold_ids, gold_ids


def fit_linear_head(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    x_eval: np.ndarray,
    *,
    epochs: int = HEAD_EPOCHS,
    lr: float = HEAD_LR,
    weight_decay: float = HEAD_WEIGHT_DECAY,
    seed: int = HEAD_SEED,
) -> np.ndarray:
    """BCE-with-logits on soft targets. Returns probabilities for x_eval."""
    if x_fit.shape[0] != y_fit.shape[0] or y_fit.shape != w_fit.shape:
        raise ValueError("fit rows must match targets and weights")
    if x_fit.shape[1] != x_eval.shape[1]:
        raise ValueError("eval width must match fit width")
    generator = torch.Generator().manual_seed(int(seed))
    layer = nn.Linear(int(x_fit.shape[1]), int(y_fit.shape[1]))
    nn.init.xavier_uniform_(layer.weight, generator=generator)
    nn.init.zeros_(layer.bias)
    opt = torch.optim.AdamW(layer.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    x = torch.as_tensor(x_fit, dtype=torch.float32)
    y = torch.as_tensor(np.clip(y_fit, 0.0, 1.0), dtype=torch.float32)
    w = torch.as_tensor(w_fit, dtype=torch.float32)
    n = int(x.shape[0])
    layer.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, 256):
            idx = perm[start : start + 256]
            opt.zero_grad()
            loss = (loss_fn(layer(x[idx]), y[idx]) * w[idx]).mean()
            loss.backward()
            opt.step()
    layer.eval()
    with torch.no_grad():
        logits = layer(torch.as_tensor(x_eval, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
    return np.asarray(probs, dtype=np.float64)


def gold_gate(y_gold: np.ndarray, scores: np.ndarray) -> tuple[float, bool]:
    """Macro AUC on expert gold. Abort when it is strictly above 0.94."""
    macro = macro_roc_auc(y_gold, scores)
    return macro, should_abort_gold_overfit(macro)
