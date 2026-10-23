"""Copy V10 mill, retarget test→58 gold train studies, append r336 AUC stage.

Save Version only. Do not Submit to Competition: output IDs are train gold.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
FUSION = ROOT / "rsna_knee" / "fusion.py"
DST = ROOT / "kaggle_kernel_goldfold"
STAGE = Path(__file__).with_name("goldfold_stage.py.inc")

GOLD_HELPERS = '''
GOLD_ONLY = True  # 58 expert train studies; DO NOT submit this kernel to the LB

def _goldfold_mask(frame):
    cols = [c for c in TARGETS if c in frame.columns]
    if not cols:
        return frame.index[0:0]
    return frame[cols].notna().all(axis=1)

def _goldfold_id_set():
    tr = pd.read_csv(ROOT / 'train.csv', dtype={'StudyInstanceUID': str})
    return set(tr.loc[_goldfold_mask(tr), 'StudyInstanceUID'].astype(str))

def _goldfold_test_df():
    return pd.DataFrame({'StudyInstanceUID': sorted(_goldfold_id_set())})

def _goldfold_test_series():
    ser = pd.read_csv(ROOT / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    return ser[ser['StudyInstanceUID'].isin(_goldfold_id_set())].copy()

'''

OLD_WALK = "def walk(split):\n    base = ROOT / split"
NEW_WALK = "def walk(split):\n    if GOLD_ONLY and split == 'test_series':\n        split = 'train_series'\n    base = ROOT / split"

OLD_WALK_RET = """        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)

def annotate(df):"""

NEW_WALK_RET = """        rows = list(pool.map(probe, items))
    out = pd.DataFrame(rows)
    if GOLD_ONLY and len(out) and 'StudyInstanceUID' in out.columns:
        out = out[out['StudyInstanceUID'].isin(_goldfold_id_set())].reset_index(drop=True)
    return out

def annotate(df):"""

OLD_INFER_TABLES = """    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')"""

NEW_INFER_TABLES = """    test_df = _goldfold_test_df() if GOLD_ONLY else pd.read_csv(ROOT / 'test.csv')
    test_series = _goldfold_test_series() if GOLD_ONLY else pd.read_csv(ROOT / 'test_series.csv')"""

OLD_SERIES_ROOT = """SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.exists():
    SERIES_ROOT = COMP / 'train_series'"""

NEW_SERIES_ROOT = """SERIES_ROOT = COMP / ('train_series' if GOLD_ONLY else 'test_series')
if not SERIES_ROOT.exists():
    SERIES_ROOT = COMP / 'train_series'"""

OLD_SUB = """sub_df = pd.read_csv(COMP / 'sample_submission.csv')
ser_csv = pd.read_csv(COMP / 'test_series.csv')
if not (COMP / 'test_series').exists():
    ser_csv = pd.read_csv(COMP / 'train_series.csv')"""

NEW_SUB = """if GOLD_ONLY:
    sub_df = _goldfold_test_df()
    ser_csv = _goldfold_test_series()
else:
    sub_df = pd.read_csv(COMP / 'sample_submission.csv')
    ser_csv = pd.read_csv(COMP / 'test_series.csv')
    if not (COMP / 'test_series').exists():
        ser_csv = pd.read_csv(COMP / 'train_series.csv')"""

OLD_RAD = """    test = _rad_pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    expected_ids = test.StudyInstanceUID.astype(str).tolist()
    baseline = _rad_pd.read_csv(primary, dtype={'StudyInstanceUID': str})
    _rad_validate(baseline, expected_ids)
    device = _rad_torch.device('cuda:0')
    test_series = _rad_pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})"""

NEW_RAD = """    test = _goldfold_test_df() if GOLD_ONLY else _rad_pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    expected_ids = test.StudyInstanceUID.astype(str).tolist()
    baseline = _rad_pd.read_csv(primary, dtype={'StudyInstanceUID': str})
    _rad_validate(baseline, expected_ids)
    device = _rad_torch.device('cuda:0')
    test_series = _goldfold_test_series() if GOLD_ONLY else _rad_pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})"""

OLD_COAT_DIR = '''    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"'''

NEW_COAT_DIR = '''    ROOT = find_test_root()
    tsdir = ROOT + ("/train_series" if GOLD_ONLY else "/test_series")
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"'''

OLD_COAT_TABLES = """    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")"""

NEW_COAT_TABLES = """    if GOLD_ONLY:
        test = _goldfold_test_df(); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
        test_ids = test["StudyInstanceUID"].tolist()
        tser = _goldfold_test_series()
    else:
        test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
        test_ids = test["StudyInstanceUID"].tolist()
        tser = pd.read_csv(ROOT + "/test_series.csv")"""

OLD_DINO_BOOT = """DINO = _boot_find_dir(
    ['config.json'],"""

NEW_DINO_BOOT = """DINO = _boot_find_dir(
    ['pytorch_model.bin', 'model.safetensors'],"""


def _replace_once(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise SystemExit(f"marker missing: {label}")
    return src.replace(old, new, 1)


def main() -> None:
    mill = SRC.read_text(encoding="utf-8")
    mill = _replace_once(mill, OLD_DINO_BOOT, NEW_DINO_BOOT, "dino boot")
    mill = _replace_once(
        mill,
        "TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']\n",
        "TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']\n"
        + GOLD_HELPERS,
        "targets",
    )
    mill = _replace_once(mill, OLD_WALK, NEW_WALK, "walk")
    mill = _replace_once(mill, OLD_WALK_RET, NEW_WALK_RET, "walk return")
    mill = _replace_once(mill, OLD_INFER_TABLES, NEW_INFER_TABLES, "infer tables")
    mill = _replace_once(mill, OLD_SERIES_ROOT, NEW_SERIES_ROOT, "series root")
    mill = _replace_once(mill, OLD_SUB, NEW_SUB, "sample_submission")
    mill = _replace_once(mill, OLD_RAD, NEW_RAD, "rad tables")
    mill = _replace_once(mill, OLD_COAT_DIR, NEW_COAT_DIR, "coatnet dir")
    mill = _replace_once(mill, OLD_COAT_TABLES, NEW_COAT_TABLES, "coatnet tables")
    fusion = FUSION.read_text(encoding="utf-8")
    stage = STAGE.read_text(encoding="utf-8").replace("___FUSION_PY___", fusion)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "main.py").write_text(mill.rstrip() + "\n\n" + stage.lstrip(), encoding="utf-8")
    print("wrote", DST / "main.py", "bytes", (DST / "main.py").stat().st_size)


if __name__ == "__main__":
    main()
