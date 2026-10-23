# goldfold: mill V10 + nursrijan r336 on the 58 expert TRAIN studies.
# Save Version only. DO NOT Submit to Competition (IDs are train, not hidden test).
from pathlib import Path as _GFPath
import sys as _gf_sys
import os as _gf_os
import gc as _gf_gc
import pandas as _gf_pd
import numpy as _gf_np

_FUSION_PY = r'''___FUSION_PY___'''
_gf_fusion_path = _GFPath('/kaggle/working') / 'fusion.py'
_gf_fusion_path.write_text(_FUSION_PY, encoding='utf-8')
_gf_sys.path.insert(0, str(_gf_fusion_path.parent))
from fusion import FINDINGS as _GF_FINDINGS, ID_COLUMN as _GF_ID

from sklearn.metrics import roc_auc_score as _gf_auc

_gf_work = _GFPath('/kaggle/working')
_gf_primary = _gf_work / 'submission.csv'
print('[goldfold] mill encode of 58 gold studies finished; computing AUCs. DO NOT SUBMIT.', flush=True)


def _gf_macro(y, s):
    per = []
    for j, name in enumerate(_GF_FINDINGS):
        yt = y[:, j]
        if len(_gf_np.unique(yt)) < 2:
            per.append(float('nan'))
            continue
        try:
            per.append(float(_gf_auc(yt, s[:, j])))
        except Exception:
            per.append(float('nan'))
    vals = [v for v in per if v == v]
    return per, (float(_gf_np.mean(vals)) if vals else float('nan'))


def _gf_find_r336():
    base = _GFPath('/kaggle/input')
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
    if not _gf_primary.is_file():
        raise RuntimeError('goldfold: mill submission.csv missing')
    mill = _gf_pd.read_csv(_gf_primary, dtype={_GF_ID: str})
    mill.to_csv(_gf_work / 'mill_v10_gold58.csv', index=False)
    train = _gf_pd.read_csv(ROOT / 'train.csv', dtype={_GF_ID: str})
    gold = train.loc[train[list(_GF_FINDINGS)].notna().all(axis=1), [_GF_ID, *_GF_FINDINGS]].copy()
    gold[_GF_ID] = gold[_GF_ID].astype(str)
    merged = gold.merge(mill, on=_GF_ID, suffixes=('_y', '_s'), how='inner')
    y = merged[[f'{c}_y' for c in _GF_FINDINGS]].to_numpy(dtype=float)
    mill_s = merged[[f'{c}_s' for c in _GF_FINDINGS]].to_numpy(dtype=float)
    mill_per, mill_macro = _gf_macro(y, mill_s)
    print(f'[goldfold] mill vs gold n={len(merged)} macro={mill_macro:.4f}', flush=True)
    for name, auc in zip(_GF_FINDINGS, mill_per):
        n_pos = int((gold[name] == 1).sum())
        n_neg = int((gold[name] == 0).sum())
        print(f'[goldfold] mill {name:22s} {auc:.4f} pos={n_pos} neg={n_neg}', flush=True)

    ckpt = _gf_find_r336()
    r336_macro = float('nan')
    r336_per = [float('nan')] * len(_GF_FINDINGS)
    if ckpt is None:
        print('[goldfold] r336_best.pt not mounted; mill-only gold table written', flush=True)
    else:
        print(f'[goldfold] loading {ckpt}', flush=True)
        from transformers import Dinov2Config as _GFDinov2Config, Dinov2Model as _GFDinov2Model
        _gf_ck = torch.load(str(ckpt), map_location='cpu', weights_only=False)
        _gf_bb_cfg = _GFDinov2Config(
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
        )
        _gf_bb = _GFDinov2Model(_gf_bb_cfg)
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
        _gf_dev = DEVS[0] if DEVS else torch.device('cpu')
        _gf_model = Model(_gf_bb, 768, pool='cls_mean', prior=False).to(_gf_dev)
        missing, unexpected = _gf_model.load_state_dict(_gf_ck['model'], strict=False)
        print(f'[goldfold] load missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}', flush=True)
        _gf_model.eval()
        _gf_series = _goldfold_test_series()
        _gf_plane = dict(zip(_gf_series['SeriesInstanceUID'], _gf_series['Anatomical_Plane']))
        _gf_hte = annotate(walk('test_series'))
        _gf_slots = pick_slots(_gf_hte, _gf_plane)
        _gf_st, _gf_C, _gf_M = build_cache(_gf_slots, _gf_plane, lat_of(_gf_hte, 'gold_r336 '), 'gold_r336')
        _gf_img = int(_gf_ck.get('cfg', {}).get('img', 336))
        _gf_idx = _gf_np.arange(len(_gf_st))
        _gf_pred = predict_member(
            _gf_model, _gf_C, _gf_M, _gf_idx, _gf_dev, _gf_img, group=3, starts=[0]
        )
        r336 = _gf_pd.DataFrame(_gf_pred, columns=list(_GF_FINDINGS))
        r336.insert(0, _GF_ID, list(_gf_st))
        r336.to_csv(_gf_work / 'r336_gold58.csv', index=False)
        m2 = gold.merge(r336, on=_GF_ID, suffixes=('_y', '_s'), how='inner')
        r336_s = m2[[f'{c}_s' for c in _GF_FINDINGS]].to_numpy(dtype=float)
        y2 = m2[[f'{c}_y' for c in _GF_FINDINGS]].to_numpy(dtype=float)
        r336_per, r336_macro = _gf_macro(y2, r336_s)
        print(f'[goldfold] r336 vs gold n={len(m2)} macro={r336_macro:.4f} stored_auc={_gf_ck.get("auc")}', flush=True)
        print(f'{"finding":22} {"mill":7} {"r336":7} {"delta":7} n_pos n_neg', flush=True)
        for name, a, b in zip(_GF_FINDINGS, mill_per, r336_per):
            n_pos = int((gold[name] == 1).sum())
            n_neg = int((gold[name] == 0).sum())
            dlt = (b - a) if (a == a and b == b) else float('nan')
            print(f'{name:22} {a:7.4f} {b:7.4f} {dlt:7.4f} {n_pos:5d} {n_neg:5d}', flush=True)
            if dlt == dlt and dlt > 0:
                print(f'[goldfold] WATCH {name} r336 beats mill by {dlt:.4f} — overlay only after this Save Version, w<<0.50', flush=True)
        del _gf_model, _gf_bb, _gf_ck, _gf_C, _gf_M
        _gf_gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f'[goldfold] mill_macro={mill_macro:.4f} r336_macro={r336_macro:.4f} '
        f'| not overlay | not submit | shape={mill.shape}',
        flush=True,
    )
except Exception as _gf_err:
    import traceback as _gf_tb
    print(f'[goldfold] failed: {type(_gf_err).__name__}: {_gf_err}', flush=True)
    _gf_tb.print_exc()
