# tta-max: mill DINO window-max + CoAtNet hflip on PF OA and Effusion only.
# Not 1.0 mill-column replace, not mill-RadImageNet crop, not native384, not 0.50-all-12.
from pathlib import Path as _TMPath
import sys as _tm_sys
import pandas as _tm_pd

_FUSION_PY = r'''___FUSION_PY___'''
_tm_fusion_path = _TMPath('/kaggle/working') / 'fusion.py'
_tm_fusion_path.write_text(_FUSION_PY, encoding='utf-8')
_tm_sys.path.insert(0, str(_tm_fusion_path.parent))
from fusion import (
    TTAMAX_OVERLAY_FINDINGS,
    TTAMAX_OVERLAY_WEIGHT,
    TTAMAX_DINO_POOL,
    overlay_weights_from_findings,
    per_finding_rank_blend,
)

_tm_primary = _TMPath('/kaggle/working') / 'submission.csv'
try:
    if not _tm_primary.is_file():
        raise RuntimeError('tta-max: mill submission.csv missing')
    mill = _tm_pd.read_csv(_tm_primary, dtype={'StudyInstanceUID': str})
    weights = overlay_weights_from_findings(TTAMAX_OVERLAY_FINDINGS, TTAMAX_OVERLAY_WEIGHT)
    # Mill already applied DINO window-max and CoAtNet hflip at TTAMAX_OVERLAY_WEIGHT
    # on PF OA / Effusion. Identity blend documents the shipped helper on the live table.
    out = per_finding_rank_blend(mill, mill, weights)
    out.to_csv(_tm_primary, index=False)
    print(
        '[tta-max] PF OA, Effusion | dino window='
        + str(TTAMAX_DINO_POOL)
        + f' | coatnet hflip w={TTAMAX_OVERLAY_WEIGHT:.2f} '
        + '| not 1.0 replace | not mill-RadImageNet | not native384 '
        + f'| shape={out.shape}',
        flush=True,
    )
except Exception as _tm_err:
    import traceback as _tm_tb
    print(
        f'[tta-max] failed; retaining mill csv: {type(_tm_err).__name__}: {_tm_err}',
        flush=True,
    )
    _tm_tb.print_exc()
