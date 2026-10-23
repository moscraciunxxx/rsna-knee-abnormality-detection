# DINOv3 ViT-S/16+ replace of MCL + Baker's on the V10 mill. Not mill-RadImageNet,
# not public native384, not CNN-tri 0.50-all-12. Fail closed to mill.
from pathlib import Path as _D3Path
import sys as _d3_sys
import glob as _d3_glob
import math as _d3_math
import pandas as _d3_pd
import numpy as _d3_np
import gc as _d3_gc

_FUSION_PY = r'''___FUSION_PY___'''
_d3_fusion_path = _D3Path('/kaggle/working') / 'fusion.py'
_d3_fusion_path.write_text(_FUSION_PY, encoding='utf-8')
_d3_sys.path.insert(0, str(_d3_fusion_path.parent))
from fusion import (
    DINO3S_REPLACE_FINDINGS,
    DINO3S_REPLACE_WEIGHT,
    overlay_weights_from_findings,
    per_finding_rank_blend,
)

_d3_torch = torch
_d3_nn = nn
_d3_F = F

_D3_HIDDEN = 384
_D3_HEADS = 6
_D3_HEAD_DIM = _D3_HIDDEN // _D3_HEADS
_D3_INTER = 1536
_D3_PATCH = 16
_D3_REGS = 4
_D3_LAYERS = 12
_D3_MODEL = 256
_D3_FINDING = 12
_D3_SLOT = 6
_D3_SLICES = 12
_D3_WINDOW = 3
_D3_NWIN = _D3_SLICES - _D3_WINDOW + 1
_D3_IMG = 336
_D3_CROP = 130.0
_D3_FOCAL = 0.125
_D3_MEAN = _d3_torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_D3_STD = _d3_torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_D3_SLOT_BANDS = (
    (0.1, 0.9),
    (0.15, 0.85),
    (0.1, 0.9),
    (0.1, 0.9),
    (0.15, 0.85),
    (0.1, 0.9),
)


def _d3_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return _d3_torch.cat((-x2, x1), dim=-1)


class _D3Rope(_d3_nn.Module):
    def __init__(self, theta=100.0):
        super().__init__()
        inv_freq = 1 / theta ** _d3_torch.arange(0, 1, 4 / _D3_HEAD_DIM, dtype=_d3_torch.float32)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        nh, nw = height // _D3_PATCH, width // _D3_PATCH
        device = pixel_values.device
        coords_h = _d3_torch.arange(0.5, nh, dtype=_d3_torch.float32, device=device) / nh
        coords_w = _d3_torch.arange(0.5, nw, dtype=_d3_torch.float32, device=device) / nw
        coords = _d3_torch.stack(_d3_torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * _d3_math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return _d3_torch.cos(angles), _d3_torch.sin(angles)


class _D3Embeddings(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = _d3_nn.Parameter(_d3_torch.zeros(1, 1, _D3_HIDDEN))
        self.mask_token = _d3_nn.Parameter(_d3_torch.zeros(1, 1, _D3_HIDDEN))
        self.register_tokens = _d3_nn.Parameter(_d3_torch.zeros(1, _D3_REGS, _D3_HIDDEN))
        self.patch_embeddings = _d3_nn.Conv2d(3, _D3_HIDDEN, kernel_size=_D3_PATCH, stride=_D3_PATCH)

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        b = x.shape[0]
        return _d3_torch.cat(
            [
                self.cls_token.expand(b, -1, -1),
                self.register_tokens.expand(b, -1, -1),
                x,
            ],
            dim=1,
        )


class _D3Attention(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_HIDDEN, bias=False)
        self.v_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_HIDDEN, bias=True)
        self.q_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_HIDDEN, bias=True)
        self.o_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_HIDDEN, bias=True)
        self.scaling = _D3_HEAD_DIM ** -0.5

    def forward(self, hidden, cos, sin):
        b, t, _ = hidden.shape
        q = self.q_proj(hidden).view(b, t, _D3_HEADS, _D3_HEAD_DIM).transpose(1, 2)
        k = self.k_proj(hidden).view(b, t, _D3_HEADS, _D3_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(hidden).view(b, t, _D3_HEADS, _D3_HEAD_DIM).transpose(1, 2)
        n_patches = sin.shape[-2]
        n_prefix = t - n_patches
        q_p, q_h = q.split((n_prefix, n_patches), dim=-2)
        k_p, k_h = k.split((n_prefix, n_patches), dim=-2)
        cos_b = cos[None, None, :, :]
        sin_b = sin[None, None, :, :]
        q_h = q_h * cos_b + _d3_rotate_half(q_h) * sin_b
        k_h = k_h * cos_b + _d3_rotate_half(k_h) * sin_b
        q = _d3_torch.cat((q_p, q_h), dim=-2)
        k = _d3_torch.cat((k_p, k_h), dim=-2)
        attn = _d3_torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = attn.softmax(dim=-1)
        out = _d3_torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(b, t, _D3_HIDDEN)
        return self.o_proj(out)


class _D3LayerScale(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = _d3_nn.Parameter(_d3_torch.ones(_D3_HIDDEN))

    def forward(self, x):
        return x * self.lambda1


class _D3GatedMLP(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_INTER, bias=True)
        self.up_proj = _d3_nn.Linear(_D3_HIDDEN, _D3_INTER, bias=True)
        self.down_proj = _d3_nn.Linear(_D3_INTER, _D3_HIDDEN, bias=True)

    def forward(self, x):
        return self.down_proj(_d3_F.silu(self.gate_proj(x)) * self.up_proj(x))


class _D3Layer(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = _d3_nn.LayerNorm(_D3_HIDDEN, eps=1e-5)
        self.attention = _D3Attention()
        self.layer_scale1 = _D3LayerScale()
        self.norm2 = _d3_nn.LayerNorm(_D3_HIDDEN, eps=1e-5)
        self.mlp = _D3GatedMLP()
        self.layer_scale2 = _D3LayerScale()

    def forward(self, x, cos, sin):
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class _D3Backbone(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = _D3Embeddings()
        self.layer = _d3_nn.ModuleList([_D3Layer() for _ in range(_D3_LAYERS)])
        self.norm = _d3_nn.LayerNorm(_D3_HIDDEN, eps=1e-5)
        self.rope = _D3Rope()

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        cos, sin = self.rope(pixel_values)
        for blk in self.layer:
            x = blk(x, cos, sin)
        x = self.norm(x)
        return x[:, 0]


class _D3Head(_d3_nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_queries = _d3_nn.Parameter(_d3_torch.zeros(_D3_FINDING, _D3_MODEL))
        self.attn_gate = _d3_nn.Parameter(_d3_torch.zeros(_D3_FINDING))
        self.slot_prior = _d3_nn.Parameter(_d3_torch.zeros(_D3_FINDING, _D3_SLOT))
        self.classifier_w = _d3_nn.Parameter(_d3_torch.zeros(_D3_FINDING, _D3_MODEL))
        self.classifier_b = _d3_nn.Parameter(_d3_torch.zeros(_D3_FINDING))
        self.feature_proj = _d3_nn.Linear(1152, _D3_MODEL)
        self.slot_embed = _d3_nn.Embedding(_D3_SLOT, _D3_MODEL)
        self.slot_norm = _d3_nn.LayerNorm(_D3_MODEL)
        self.study_norm = _d3_nn.LayerNorm(_D3_MODEL)

    def pool_study(self, cls_slots, slot_mask):
        feats = []
        slot_ids = []
        masks = []
        for s in range(_D3_SLOT):
            for i in range(_D3_NWIN):
                cat = _d3_torch.cat([cls_slots[s, i + j] for j in range(_D3_WINDOW)], dim=-1)
                feat = self.feature_proj(cat)
                feat = self.slot_norm(feat + self.slot_embed.weight[s])
                feats.append(feat)
                slot_ids.append(s)
                masks.append(slot_mask[s])
        feat = _d3_torch.stack(feats, dim=0)
        mask = _d3_torch.stack(masks, dim=0)
        slot_ids = _d3_torch.tensor(slot_ids, device=feat.device, dtype=_d3_torch.long)
        scale = _D3_MODEL ** 0.5
        scores = (self.slot_queries @ feat.T) / scale
        scores = scores + self.slot_prior[:, slot_ids]
        if float(mask.sum()) < 0.5:
            return _d3_torch.zeros(_D3_FINDING, device=feat.device, dtype=feat.dtype)
        invalid = mask < 0.5
        scores = scores.masked_fill(invalid.unsqueeze(0), -1.0e4)
        k = max(1, int(round(int(mask.sum().item()) * _D3_FOCAL)))
        valid_scores = scores.masked_fill(invalid.unsqueeze(0), -1.0e9)
        thresh = _d3_torch.topk(valid_scores, k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(valid_scores < thresh, -1.0e4)
        attn = _d3_torch.softmax(scores, dim=-1)
        pooled_attn = attn @ feat
        denom = mask.sum().clamp_min(1.0)
        mean = (feat * mask.unsqueeze(-1)).sum(0) / denom
        gate = _d3_torch.sigmoid(self.attn_gate).unsqueeze(-1)
        pooled = gate * pooled_attn + (1.0 - gate) * mean
        pooled = _d3_torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = self.study_norm(pooled)
        logits = (pooled * self.classifier_w).sum(-1) + self.classifier_b
        return _d3_torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


def _d3_find_weight_dir():
    extra = (
        '/kaggle/input/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model',
        '/kaggle/input/datasets/surasan092/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model',
    )
    found = _boot_find_dir(['focal_fold0_final.pt'], extra=extra)
    if found is None:
        raise FileNotFoundError('DINOv3-S+ v13.1 weights not mounted')
    return found


def _d3_load_member(path, device):
    blob = _d3_torch.load(path, map_location='cpu', weights_only=False)
    ms = blob['model_state']
    bb = _D3Backbone()
    hd = _D3Head()
    bb_sd = {k[len('backbone.') :]: v for k, v in ms.items() if k.startswith('backbone.')}
    missing, unexpected = bb.load_state_dict(bb_sd, strict=False)
    if missing:
        raise RuntimeError(f'dino3s backbone missing {missing}')
    hd.load_state_dict({k: v for k, v in ms.items() if not k.startswith('backbone.')}, strict=True)
    bb.eval().to(device)
    hd.eval().to(device)
    for p in list(bb.parameters()) + list(hd.parameters()):
        p.requires_grad_(False)
    return bb, hd


def _d3_read_slot(rec, band):
    old = SLICE_BAND
    try:
        globals()['SLICE_BAND'] = band
        return read_slot(rec, n_slice=_D3_SLICES, out_size=_D3_IMG)
    finally:
        globals()['SLICE_BAND'] = old


_d3_primary = _D3Path('/kaggle/working') / 'submission.csv'
try:
    if not _d3_primary.is_file():
        raise RuntimeError('dino3s: mill submission.csv missing')
    mill = _d3_pd.read_csv(_d3_primary, dtype={'StudyInstanceUID': str})
    test = _d3_pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    expected_ids = test.StudyInstanceUID.astype(str).tolist()
    test_series = _d3_pd.read_csv(
        ROOT / 'test_series.csv',
        dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str},
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    weight_dir = _d3_find_weight_dir()
    fold_paths = sorted(_d3_glob.glob(str(_D3Path(weight_dir) / 'focal_fold*_final.pt')))
    if not fold_paths:
        raise FileNotFoundError(f'no focal_fold*_final.pt under {weight_dir}')
    _d3_gc.collect()
    if _d3_torch.cuda.is_available():
        _d3_torch.cuda.empty_cache()
    globals().update(
        SLOTS=list(SLOTS_RECOVERED),
        N_SLOT=len(SLOTS_RECOVERED),
        IMG=_D3_IMG,
        CACHE_IMG=_D3_IMG,
        CROP_MM=float(_D3_CROP),
        RULES=dict(RULES_NATIVE),
    )
    headers = annotate(walk('test_series'))
    slot_map = pick_slots(headers, plane)
    lat_map = lat_of(headers, 'dino3s ')
    device = _d3_torch.device('cuda:0' if _d3_torch.cuda.is_available() else 'cpu')
    mean = _D3_MEAN.to(device)
    std = _D3_STD.to(device)
    fold_logits = []
    for fp in fold_paths:
        bb, hd = _d3_load_member(fp, device)
        rows = []
        for uid in expected_ids:
            chosen = slot_map.get(uid, {})
            lat = lat_map.get(uid)
            cls_slots = _d3_torch.zeros(_D3_SLOT, _D3_SLICES, _D3_HIDDEN, device=device)
            slot_mask = _d3_torch.zeros(_D3_SLOT, device=device)
            images = []
            owners = []
            for s, (name, plane_name, _fluid, _fs) in enumerate(SLOTS_RECOVERED):
                rec = chosen.get(name)
                if rec is None:
                    continue
                vol = _d3_read_slot(rec, _D3_SLOT_BANDS[s])
                if vol is None:
                    continue
                vol = normalise_laterality(vol, plane_name, lat)
                if vol.shape[0] < _D3_SLICES:
                    pad = vol[-1:].repeat(_D3_SLICES - vol.shape[0], 1, 1)
                    vol = _d3_torch.cat([vol, pad], dim=0)
                vol = vol[:_D3_SLICES]
                images.append(vol)
                owners.append(s)
                slot_mask[s] = 1.0
            if images:
                stacked = _d3_torch.cat(images, dim=0).to(device)
                x = stacked.float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
                x = (x - mean) / std
                with _d3_torch.no_grad():
                    cls = bb(x).float()
                offset = 0
                for s, vol in zip(owners, images):
                    n = vol.shape[0]
                    cls_slots[s, :n] = cls[offset : offset + n]
                    offset += n
            with _d3_torch.no_grad():
                logits = hd.pool_study(cls_slots, slot_mask)
            rows.append(logits.detach().float().cpu().numpy())
            del images
        fold_logits.append(_d3_np.stack(rows, axis=0))
        del bb, hd
        _d3_gc.collect()
        if _d3_torch.cuda.is_available():
            _d3_torch.cuda.empty_cache()
    mean_logits = _d3_np.mean(_d3_np.stack(fold_logits, axis=0), axis=0)
    mean_logits = _d3_np.nan_to_num(mean_logits, nan=0.0, posinf=0.0, neginf=0.0)
    overlay = mill.copy()
    ranked = _d3_pd.DataFrame(mean_logits, columns=list(TARGETS))
    overlay[list(TARGETS)] = ranked.rank(method='average', pct=True).to_numpy(_d3_np.float64)
    overlay[list(TARGETS)] = overlay[list(TARGETS)].clip(0.0, 1.0)
    weights = overlay_weights_from_findings(DINO3S_REPLACE_FINDINGS, DINO3S_REPLACE_WEIGHT)
    out = per_finding_rank_blend(mill, overlay, weights)
    out.to_csv(_d3_primary, index=False)
    print(
        '[dino3s] DINOv3-S+ replace '
        + ', '.join(DINO3S_REPLACE_FINDINGS)
        + f' | w={DINO3S_REPLACE_WEIGHT:.2f} | folds={len(fold_paths)} '
        + '| encoder=dinov3-vits16plus | not mill-RadImageNet | not native384 '
        + f'| shape={out.shape}',
        flush=True,
    )
except Exception as _d3_err:
    import traceback as _d3_tb
    print(
        f'[dino3s] failed; retaining V10 mill: {type(_d3_err).__name__}: {_d3_err}',
        flush=True,
    )
    _d3_tb.print_exc()
