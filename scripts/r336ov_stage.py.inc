# r336ov: mill V10 + nursrijan DINOv2-B r336 overlay on PF OA + Synovitis @ 0.20.
# Private 58-row fold (goldfold): mill macro 0.9575, r336 0.9695.
# PF OA +0.0502 (21 pos); Synovitis +0.0370 (27 pos). Not 1.0 replace. Fail-closed.
from pathlib import Path as _OVPath
import sys as _ov_sys
import gc as _ov_gc
import pandas as _ov_pd
import numpy as _ov_np

_FUSION_PY = r'''___FUSION_PY___'''
_ov_fusion_path = _OVPath('/kaggle/working') / 'fusion.py'
_ov_fusion_path.write_text(_FUSION_PY, encoding='utf-8')
_ov_sys.path.insert(0, str(_ov_fusion_path.parent))
from fusion import (
    R336OV_OVERLAY_FINDINGS,
    R336OV_OVERLAY_WEIGHT,
    overlay_weights_from_findings,
    per_finding_rank_blend,
)

_ov_work = _OVPath('/kaggle/working')
_ov_primary = _ov_work / 'submission.csv'


def _ov_find_r336():
    base = _OVPath('/kaggle/input')
    if not base.is_dir():
        return None
    skip = {'train_series', 'test_series', 'test_images'}
    stack = [base]
    depth = 0
    while stack and depth < 80:
        cur = stack.pop(0)
        depth += 1
        try:
            kids = list(cur.iterdir())
        except OSError:
            continue
        names = {k.name for k in kids}
        if 'r336_best.pt' in names:
            return cur / 'r336_best.pt'
        for k in kids:
            if k.is_dir() and k.name not in skip and not k.name.startswith('1.2.'):
                stack.append(k)
    return None


try:
    if not _ov_primary.is_file():
        raise RuntimeError('r336ov: mill submission.csv missing')
    mill = _ov_pd.read_csv(_ov_primary, dtype={'StudyInstanceUID': str})
    ckpt = _ov_find_r336()
    if ckpt is None:
        raise RuntimeError('r336_best.pt not mounted')
    print(f'[r336ov] loading {ckpt}', flush=True)
    from transformers import Dinov2Config as _OVDinov2Config, Dinov2Model as _OVDinov2Model
    _ov_ck = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    _ov_bb = _OVDinov2Model(_OVDinov2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        patch_size=14,
        image_size=518,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        use_swiglu_ffn=False,
    ))
    globals().update(
        SLOTS=list(SLOTS_RECOVERED),
        N_SLOT=len(SLOTS_RECOVERED),
        CACHE_IMG=336,
        IMG=336,
        GROUP=3,
        CACHE_SLICES=3,
        CROP_MM=130.0,
        RULES=dict(RULES_NATIVE),
    )
    _ov_dev = DEVS[0] if DEVS else torch.device('cpu')
    _ov_model = Model(_ov_bb, 768, pool='cls_mean', prior=False).to(_ov_dev)
    missing, unexpected = _ov_model.load_state_dict(_ov_ck['model'], strict=False)
    print(f'[r336ov] load missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}', flush=True)
    if missing:
        raise RuntimeError(f'r336 load missing keys: {missing[:8]}')
    _ov_model.eval()
    _ov_test = _ov_pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    _ov_series = _ov_pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    _ov_plane = dict(zip(_ov_series['SeriesInstanceUID'], _ov_series['Anatomical_Plane']))
    _ov_hte = annotate(walk('test_series'))
    _ov_slots = pick_slots(_ov_hte, _ov_plane)
    _ov_st, _ov_C, _ov_M = build_cache(_ov_slots, _ov_plane, lat_of(_ov_hte, 'r336ov '), 'r336ov')
    _ov_img = int(_ov_ck.get('cfg', {}).get('img', 336))
    _ov_idx = _ov_np.arange(len(_ov_st))
    _ov_pred = predict_member(
        _ov_model, _ov_C, _ov_M, _ov_idx, _ov_dev, _ov_img, group=3, starts=[0]
    )
    r336 = _ov_pd.DataFrame(_ov_pred, columns=list(TARGETS))
    r336.insert(0, 'StudyInstanceUID', [str(x) for x in _ov_st])
    r336 = r336.set_index('StudyInstanceUID').reindex(mill['StudyInstanceUID'].astype(str)).reset_index()
    if r336[list(TARGETS)].isna().any().any():
        raise RuntimeError('r336ov: r336 rows missing after reindex to mill')
    r336.to_csv(_ov_work / 'r336_test.csv', index=False)
    weights = overlay_weights_from_findings(R336OV_OVERLAY_FINDINGS, R336OV_OVERLAY_WEIGHT)
    out = per_finding_rank_blend(mill, r336, weights)
    out.to_csv(_ov_primary, index=False)
    print(
        '[r336ov] PF OA, Synovitis | w='
        + f'{R336OV_OVERLAY_WEIGHT:.2f} | dinov2-vitb14-r336 | goldfold PF OA +0.050 Synovitis +0.037 '
        + '| not 1.0 replace | not mill-RadImageNet | not native384 | not GOLD_ONLY '
        + f'| shape={out.shape}',
        flush=True,
    )
    del _ov_model, _ov_bb, _ov_ck, _ov_C, _ov_M
    _ov_gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as _ov_err:
    import traceback as _ov_tb
    print(f'[r336ov] failed; retaining mill csv: {type(_ov_err).__name__}: {_ov_err}', flush=True)
    _ov_tb.print_exc()
