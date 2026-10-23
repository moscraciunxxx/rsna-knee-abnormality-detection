"""Optimizer, scaler, and collapse check for the no-OAI CoAtNet.

Fold-0 v2 used a flat 2e-5 step with no GradScaler and batch 1. Its OOF
standard deviation was about 2.5e-5 and the loss stayed near 0.63.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 0.05
ACCUM = 4
COLLAPSE_STD = 1e-3


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per * weight).sum() / weight.sum().clamp_min(1e-6)


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {"params": backbone, "lr": BACKBONE_LR},
            {"params": head, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine down to 0. ``step`` is 1-based after an update."""
    if total < 1:
        return 1.0
    if step <= warmup:
        return float(step) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lrs(opt: torch.optim.Optimizer, step: int, warmup: int, total: int) -> None:
    scale = lr_multiplier(step, warmup, total)
    for group in opt.param_groups:
        group["lr"] = float(group["initial_lr"]) * scale


@dataclass
class Stepper:
    opt: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    accum: int = ACCUM
    micro: int = 0
    updates: int = 0

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss / self.accum).backward()
        self.micro += 1

    def ready(self) -> bool:
        return self.micro > 0 and self.micro % self.accum == 0

    def update(self, warmup: int, total: int) -> None:
        self.scaler.unscale_(self.opt)
        params = [p for group in self.opt.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.updates += 1
        apply_lrs(self.opt, self.updates, warmup, total)


def probability_std(probs: torch.Tensor) -> float:
    return float(probs.detach().float().std(unbiased=False))


def collapsed(probs: torch.Tensor, floor: float = COLLAPSE_STD) -> bool:
    if probs.numel() < 2:
        return False
    return probability_std(probs) < floor
