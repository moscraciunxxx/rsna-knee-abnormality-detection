"""Local 3060 smoke: r336 ViT-B dummy 6-slot 336 forward. No train DICOMs required."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / ".cache" / "dinov2b" / "r336_best.pt"
N_SLOT = 6
N_OUT = 12
POOL_PARTS = 2
N_STUDY_BENCH = 8


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(0.0)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim * POOL_PARTS, N_SLOT, N_OUT)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=336):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        feat = torch.cat([out[:, 0], patch.mean(1)], dim=1).reshape(B, S, -1)
        return self.head(feat, mask)


def vram(msg: str) -> None:
    if not torch.cuda.is_available():
        print(msg, "cpu")
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{msg} alloc={alloc:.0f}MiB reserved={reserved:.0f}MiB peak={peak:.0f}MiB")


def main() -> None:
    assert PT.is_file(), PT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    ck = torch.load(PT, map_location="cpu", weights_only=False)
    cfg = Dinov2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        patch_size=14,
        image_size=518,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        use_swiglu_ffn=False,
    )
    bb = Dinov2Model(cfg)
    model = Model(bb, 768)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print("missing", len(missing), "unexpected", len(unexpected))
    model.eval()
    model.to(device)
    vram("loaded")
    imgs = torch.randint(0, 256, (1, N_SLOT, 3, 336, 336), dtype=torch.uint8, device=device)
    mask = torch.ones(1, N_SLOT, device=device)
    use_amp = device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs, mask)
        else:
            logits = model(imgs, mask)
    print("logits", tuple(logits.shape), float(logits.min()), float(logits.max()))
    vram("after_first")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_STUDY_BENCH):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(imgs, mask)
            else:
                _ = model(imgs, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per = dt / N_STUDY_BENCH
    print(f"sec_per_study={per:.3f} n={N_STUDY_BENCH}")
    print(f"est_870_forward_only_min={870 * per / 60:.1f}")
    print(f"est_870_with_dicom_decode_x3_min={870 * per * 3 / 60:.1f}")
    vram("after_bench")
    print("LOCAL_GPU_SMOKE_OK")


if __name__ == "__main__":
    main()
