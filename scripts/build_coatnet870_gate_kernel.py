"""Build the mill-870 private gate around ctogaurav's public CoAtNet/d4 notebook.

Save Version only. Do not Submit. Do not push while another GPU kernel is running.

Push (only when the GPU lane is free, session cap 4 h):

    kaggle kernels push -p kaggle_kernel_coatnet870_gate -t 14400
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.coatnet_gate import SHIM_SOURCE

NB_PATH = ROOT / "refs" / "ctogaurav__rsna-knee-pure-coatnet-0946" / "rsna-knee-pure-coatnet-0946.ipynb"
CLAIM_META = ROOT / "refs" / "ctogaurav__rsna-knee-pure-coatnet-0946" / "kernel-metadata.json"
DST = ROOT / "kaggle_kernel_coatnet870_gate"
MILL_CSV = ROOT / "data" / "mill_v10_holdout870.csv"

FIND_ROOT_OLD = """def find_root():
    return _one_direct('competition', [
        os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        '/kaggle/input/rsna-knee-abnormality-detection',
    ], ['test.csv', 'test_series.csv', 'test_series'])
"""

FIND_ROOT_NEW = """def find_root():
    forced = os.environ.get('RSNA_COMP_ROOT', '')
    if forced:
        root = Path(forced)
        need = ['test.csv', 'test_series.csv', 'test_series']
        missing = [item for item in need if not (root / item).exists()]
        if missing:
            raise FileNotFoundError('RSNA_COMP_ROOT %s missing %s' % (root, missing))
        print('[coatnet870] find_root forced RSNA_COMP_ROOT', root, flush=True)
        return root
    return _one_direct('competition', [
        '/kaggle/input/competitions/rsna-knee-abnormality-detection',
        '/kaggle/input/rsna-knee-abnormality-detection',
    ], ['test.csv', 'test_series.csv', 'test_series'])
"""

BOOT = r'''#!/usr/bin/env python3
"""Public CoAtNet/d4 blend on the seed-2026 LLM-soft holdout (n=870).

Save Version only. DO NOT SUBMIT. These IDs are train holdout, not the hidden test.
The author claim 0.943 is not this account's publicScore. Gate vs mill V10 before any TEST Submit.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _gpu_is_t4_or_newer():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        print("nvidia-smi:", out.strip(), flush=True)
        bad = ("P100" in out) or ("6.0" in out)
        good = any(x in out for x in ("T4", "V100", "A100", "A10", "L4", "7.5", "8.0", "8.6", "8.9", "9.0"))
        return (not bad) or good
    except Exception as exc:
        print("nvidia-smi probe failed:", exc, flush=True)
        return True


if not _gpu_is_t4_or_newer():
    print("Hiding pre-Volta GPU before importing torch", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

print(
    "[coatnet870] PRIVATE GATE on seed-2026 holdout. DO NOT SUBMIT. "
    "Session cap 14400s. Author 0.943 is not our CLI.",
    flush=True,
)
'''

RUNNER = r'''
def _gate_find_real_comp():
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series.csv").is_file() and (path / "train_series").is_dir():
            return path
    raise FileNotFoundError("competition train_series not mounted")


def _gate_find_v4():
    skip = {"train_series", "test_series", "test_images", "train_images"}
    base = Path("/kaggle/input")
    hits = []
    if base.is_dir():
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
            if "llm_labels_v4_blend.csv" in names:
                hits.append(cur / "llm_labels_v4_blend.csv")
            for kid in kids:
                if kid.is_dir() and kid.name not in skip and not kid.name.startswith("1.2."):
                    stack.append(kid)
    if not hits:
        raise FileNotFoundError("steven v4 llm_labels_v4_blend.csv not mounted")
    print("[coatnet870] v4", hits[0], flush=True)
    return hits[0]


def _gate_run():
    real = _gate_find_real_comp()
    v4 = _gate_find_v4()
    ids = _gate_holdout_ids(real / "train.csv", v4)
    dest = Path("/kaggle/working/_holdout_comp")
    _gate_stage_comp(real, ids, dest)
    os.environ["RSNA_COMP_ROOT"] = str(dest)
    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": list(ids)}).to_csv(work / "coatnet870_ids.csv", index=False)
    print("[coatnet870] RSNA_COMP_ROOT", dest, "n", len(ids), "DO NOT SUBMIT", flush=True)


_gate_run()
'''

TAIL = r'''
# <<<COATNET870_TAIL
# Pipeline finished. Keep preds for the 870 only. Remove submission.csv so this
# kernel cannot be submitted. Authoritative overlay policy is local:
#   python scripts/saturday_after_coatnet.py --coatnet <coatnet_holdout870.csv>
# That calls rsna_knee.mill_holdout.decide_private_overlay. Do not Submit.
print("[coatnet870] pipeline returned. Validating holdout CSV. DO NOT SUBMIT.", flush=True)
_gate_pred_path = Path("/kaggle/working/submission.csv")
_gate_ids_path = Path("/kaggle/working/coatnet870_ids.csv")
_gate_out_path = Path("/kaggle/working/coatnet_holdout870.csv")
_gate_note_path = Path("/kaggle/working/DO_NOT_SUBMIT_HOLDOUT.txt")


def _gate_auc_table(y, scores):
    from sklearn.metrics import roc_auc_score

    per = []
    for j in range(y.shape[1]):
        yt = y[:, j]
        if len(np.unique(yt)) < 2:
            per.append(float("nan"))
            continue
        try:
            per.append(float(roc_auc_score(yt, scores[:, j])))
        except Exception:
            per.append(float("nan"))
    vals = [v for v in per if v == v]
    macro = float(np.mean(vals)) if vals else float("nan")
    return per, macro


try:
    if not _gate_pred_path.is_file():
        raise RuntimeError("coatnet gate: submission.csv missing after pipeline")
    if not _gate_ids_path.is_file():
        raise RuntimeError("coatnet gate: coatnet870_ids.csv missing")
    _gate_ids = pd.read_csv(_gate_ids_path, dtype={"StudyInstanceUID": str})["StudyInstanceUID"].astype(str).tolist()
    _gate_pred = pd.read_csv(_gate_pred_path, dtype={"StudyInstanceUID": str})
    if list(_gate_pred.columns) != SUBMISSION_HEADER:
        raise RuntimeError("coatnet gate header %s" % (list(_gate_pred.columns),))
    if "Baker's" not in _gate_pred.columns:
        raise RuntimeError("Baker's column missing")
    _gate_pred["StudyInstanceUID"] = _gate_pred["StudyInstanceUID"].astype(str)
    _gate_pred = _gate_pred.drop_duplicates("StudyInstanceUID")
    if set(_gate_pred["StudyInstanceUID"]) != set(_gate_ids) or len(_gate_pred) != len(_gate_ids):
        raise RuntimeError(
            "coatnet gate ID set mismatch pred=%s holdout=%s" % (len(_gate_pred), len(_gate_ids))
        )
    _gate_pred = _gate_pred.set_index("StudyInstanceUID").loc[_gate_ids].reset_index()
    _gate_pred = _gate_pred[SUBMISSION_HEADER]
    _gate_values = _gate_pred[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(_gate_values).all():
        raise RuntimeError("coatnet gate non-finite predictions")
    _gate_pred.to_csv(_gate_out_path, index=False)
    print("[coatnet870] wrote", _gate_out_path, "shape", tuple(_gate_pred.shape), flush=True)
except Exception:
    if _gate_pred_path.is_file():
        _gate_pred_path.unlink()
        print("[coatnet870] removed submission.csv after failed validation. DO NOT SUBMIT.", flush=True)
    raise
else:
    if _gate_pred_path.is_file():
        _gate_pred_path.unlink()
        print("[coatnet870] removed submission.csv. DO NOT SUBMIT this holdout kernel.", flush=True)

_gate_note_path.write_text(
    "DO NOT SUBMIT vitaliecervinschi/rsna-knee-coatnet870-gate.\n"
    "coatnet_holdout870.csv is the seed-2026 train holdout, not the hidden test.\n"
    "Keep 55779066 until scripts/saturday_after_coatnet.py licenses a separate TEST overlay.\n",
    encoding="utf-8",
)

try:
    _gate_mill_path = Path(__file__).resolve().parent / "mill_v10_holdout870.csv"
    if not _gate_mill_path.is_file():
        print("[coatnet870] mill_v10_holdout870.csv not beside the script; join locally", flush=True)
    else:
        _gate_mill = pd.read_csv(_gate_mill_path, dtype={"StudyInstanceUID": str})
        _gate_mill["StudyInstanceUID"] = _gate_mill["StudyInstanceUID"].astype(str)
        _gate_mill = _gate_mill.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        _gate_missing_mill = [i for i in _gate_ids if i not in _gate_mill.index]
        if _gate_missing_mill:
            print("[coatnet870] mill CSV missing %s holdout UIDs" % (len(_gate_missing_mill),), flush=True)
        else:
            _gate_v4 = pd.read_csv(_gate_find_v4(), dtype={"StudyInstanceUID": str})
            _gate_v4["StudyInstanceUID"] = _gate_v4["StudyInstanceUID"].astype(str)
            _gate_v4 = _gate_v4.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
            _gate_y = (_gate_v4.loc[_gate_ids, TARGETS].to_numpy(dtype=float) >= 0.5).astype(float)
            _gate_new = _gate_pred.set_index("StudyInstanceUID").loc[_gate_ids, TARGETS].to_numpy(dtype=float)
            _gate_old = _gate_mill.loc[_gate_ids, TARGETS].to_numpy(dtype=float)
            _gate_new_per, _gate_new_macro = _gate_auc_table(_gate_y, _gate_new)
            _gate_old_per, _gate_old_macro = _gate_auc_table(_gate_y, _gate_old)
            print(
                "[coatnet870] mill_macro=%.4f coat_macro=%.4f n=%s DO NOT SUBMIT"
                % (_gate_old_macro, _gate_new_macro, len(_gate_ids)),
                flush=True,
            )
            print("%-22s %7s %7s %7s" % ("finding", "mill", "coat", "delta"), flush=True)
            for _name, _m, _n in zip(TARGETS, _gate_old_per, _gate_new_per):
                _dlt = (_n - _m) if (_m == _m and _n == _n) else float("nan")
                print("%-22s %7.4f %7.4f %7.4f" % (_name, _m, _n, _dlt), flush=True)
            print(
                "[coatnet870] authoritative decision: python scripts/saturday_after_coatnet.py "
                "(decide_private_overlay). Do not write overlay_spec from this kernel. DO NOT SUBMIT.",
                flush=True,
            )
except Exception as _gate_cmp_err:
    print("[coatnet870] mill compare skipped: %s: %s" % (type(_gate_cmp_err).__name__, _gate_cmp_err), flush=True)
    print("[coatnet870] CSV is on disk. Join locally. DO NOT SUBMIT.", flush=True)
# >>>COATNET870_TAIL
'''


def _notebook_body(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines = []
        for line in src.splitlines():
            if line.strip() == "from __future__ import annotations":
                continue
            lines.append(line)
        chunks.append("\n".join(lines).strip("\n"))
    body = "\n\n# ===== notebook cell =====\n\n".join(chunks)
    if body.count(FIND_ROOT_OLD) != 1:
        raise SystemExit("find_root block missing or duplicated in the CoAtNet notebook")
    return body.replace(FIND_ROOT_OLD, FIND_ROOT_NEW, 1)


def _metadata() -> dict:
    claim = json.loads(CLAIM_META.read_text(encoding="utf-8"))
    datasets = list(claim["dataset_sources"])
    extra = "stevenleehans/rsna-knee-llm-report-labels"
    if extra not in datasets:
        datasets.append(extra)
    return {
        "id": "vitaliecervinschi/rsna-knee-coatnet870-gate",
        "title": "RSNA Knee coatnet870 gate",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets,
        "kernel_sources": list(claim["kernel_sources"]),
        "competition_sources": list(claim["competition_sources"]),
        "model_sources": list(claim["model_sources"]),
        "docker_image": claim["docker_image"],
    }


def main() -> None:
    if not NB_PATH.is_file():
        raise SystemExit(f"missing notebook {NB_PATH}")
    if not MILL_CSV.is_file():
        raise SystemExit(f"missing mill holdout csv {MILL_CSV}")
    DST.mkdir(exist_ok=True)
    body = _notebook_body(NB_PATH)
    script = (
        BOOT
        + "\n# <<<COATNET870_SHIM\n"
        + SHIM_SOURCE.strip()
        + "\n# >>>COATNET870_SHIM\n"
        + RUNNER
        + "\n# ===== notebook cell =====\n\n"
        + body
        + "\n"
        + TAIL
    )
    if script.count("from __future__ import annotations") != 1:
        raise SystemExit("future import must appear exactly once")
    compile(script, str(DST / "main.py"), "exec")
    (DST / "main.py").write_text(script, encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(
        json.dumps(_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(MILL_CSV, DST / "mill_v10_holdout870.csv")
    print(f"wrote {DST / 'main.py'} bytes={len(script.encode('utf-8'))}")


if __name__ == "__main__":
    main()
