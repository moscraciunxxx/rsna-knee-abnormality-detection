"""Build kaggle_kernel_top_public_dissect_gate (PRIVATE GATE, DO NOT SUBMIT).

Runs the public nartaa 0.949 CoAtNet checkpoint (exact velciraptor/anatomical-mirror
inference code, which scored 0.949 on BOTH 37c64f and 2757e0) on gold58 + seed-2026
holdout870, and the goodpjw2008 ConvNeXt reader on gold58. Pinned to anchor image 37c64f.
Does not push. Does not Submit.
"""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "external_notebooks" / "matterhorn3838__rsna-knee-v2-velciraptor-dinosaur-speed" / "rsna-knee-v2-velciraptor-dinosaur-speed.ipynb"
DST = ROOT / "kaggle_kernel_top_public_dissect_gate"
ANCHOR_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

TEMPLATE = r'''"""PRIVATE GATE: top-public dissect. DO NOT SUBMIT. Keep 55779066 selected.

Arm A: nartaa raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt (public 0.949) with the exact
       scored velciraptor inference code, on gold58 + seed-2026 holdout870 (928 train studies).
       NOTE: checkpoint is 'alldata' -> it trained on holdout870 studies; only gold58 is clean.
Arm B: goodpjw2008 ConvNeXt 2.5D reader (3 folds) on gold58 only.
Outputs: nartaa384_eval928.csv, convnext_gold58.csv. Train-study predictions must never hit the LB.
"""
import base64, glob, os, shutil, subprocess, sys, threading, time, zlib
from pathlib import Path
import numpy as np
import pandas as pd

T0 = time.perf_counter()
BUDGET = 5400
def _wd():
    time.sleep(BUDGET); print(f"[dissect] WATCHDOG hard kill {BUDGET}s", flush=True); os._exit(99)
threading.Thread(target=_wd, daemon=True).start()
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"
W = Path("/kaggle/working")
NARTAA_CODE_B64 = "@@CODE@@"

def log(m): print(f"[dissect] {m} t={time.perf_counter()-T0:.0f}s", flush=True)

def find_comp():
    for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"), Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (p / "train.csv").is_file() and (p / "train_series").is_dir():
            return p
    for p in Path("/kaggle/input").rglob("train_series.csv"):
        if (p.parent / "train_series").is_dir():
            return p.parent
    raise FileNotFoundError("comp root")

def find_file(name):
    skip = {"train_series", "test_series", "test_images", "train_images"}
    stack = [Path("/kaggle/input")]; n = 0
    while stack and n < 200:
        cur = stack.pop(0); n += 1
        try: kids = list(cur.iterdir())
        except OSError: continue
        for k in kids:
            if k.name == name: log(f"found {name} at {k}"); return k
        stack += [k for k in kids if k.is_dir() and k.name not in skip and not k.name.startswith("1.2.")]
    raise FileNotFoundError(name)

def gold_ids(train):
    m = train[LABELS].notna().all(axis=1)
    g = train.loc[m, ID].astype(str).tolist()
    assert len(g) == 58, len(g)
    return g

def holdout_ids(train, v4):
    gold = set(gold_ids(train))
    lab = v4.drop_duplicates(ID).copy(); lab[ID] = lab[ID].astype(str)
    tr = [u for u in lab[ID].tolist() if u not in gold]
    rng = np.random.default_rng(2026); perm = rng.permutation(len(tr))
    n_ho = max(1, min(len(tr) - 1, int(round(len(tr) * 0.2))))
    ids = [tr[int(i)] for i in np.sort(perm[:n_ho]).tolist()]
    assert len(ids) == 870 and not (set(ids) & gold)
    return ids

def stage(ids, real, dest):
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(parents=True)
    ser = pd.read_csv(real / "train_series.csv", dtype={ID: str, "SeriesInstanceUID": str})
    ser = ser[ser[ID].isin(set(ids))]
    pd.DataFrame({ID: ids}).to_csv(dest / "test.csv", index=False)
    ser.to_csv(dest / "test_series.csv", index=False)
    pd.DataFrame([{ID: u, **{f: 0.5 for f in LABELS}} for u in ids]).to_csv(dest / "sample_submission.csv", index=False)
    (dest / "test_series").mkdir()
    for u in ids:
        if (real / "train_series" / u).is_dir():
            os.symlink(real / "train_series" / u, dest / "test_series" / u, target_is_directory=True)
    log(f"staged {dest.name} studies={len(ids)} series={len(ser)}")
    return dest

def main():
    log("PRIVATE GATE top-public dissect. DO NOT SUBMIT. Keep 55779066.")
    import torch, timm
    log(f"python {sys.version.split()[0]} torch {torch.__version__} timm {timm.__version__} gpus {torch.cuda.device_count()}")
    real = find_comp()
    train = pd.read_csv(real / "train.csv", dtype={ID: str})
    v4 = pd.read_csv(find_file("llm_labels_v4_blend.csv"), dtype={ID: str})
    gold = gold_ids(train); hold = holdout_ids(train, v4)
    pd.DataFrame({ID: gold}).to_csv(W / "gold58_ids.csv", index=False)
    pd.DataFrame({ID: hold}).to_csv(W / "holdout870_ids.csv", index=False)
    # ---- Arm B first (cheap): ConvNeXt reader on gold58 ----
    try:
        g58 = stage(gold, real, W / "_gold_comp")
        ck0 = find_file("cnxt_v0_fold0.pt"); assets = ck0.parent
        ckpts = sorted(glob.glob(str(assets / "cnxt_v0_fold*.pt")))
        own = W / "own_src"; own.mkdir(exist_ok=True)
        for nm in ("infer.py", "knee.py", "preprocess.py"):
            shutil.copy2(assets / nm, own / nm)
        tgt = W / "_own_env"; tgt.mkdir(exist_ok=True)
        whl = [w.name for w in assets.glob("*.whl") if "cp311" not in w.name]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(tgt), "--find-links", str(assets)] + whl, check=False)
        env = dict(os.environ); env["PYTHONPATH"] = str(own) + os.pathsep + str(tgt) + os.pathsep + env.get("PYTHONPATH", "")
        out = W / "_own_gold.csv"
        rc = subprocess.call([sys.executable, str(own / "infer.py"), str(g58), "test", str(out), *ckpts], env=env, timeout=1500)
        if rc == 0 and out.is_file():
            p = pd.read_csv(out, dtype={ID: str}).set_index(ID).reindex(gold).reset_index()[[ID, *LABELS]]
            p.to_csv(W / "convnext_gold58.csv", index=False); out.unlink()
            log(f"convnext_gold58.csv rows={len(p)} nan={int(p[LABELS].isna().sum().sum())}")
        else:
            log(f"ConvNeXt reader failed rc={rc} (non-fatal)")
    except Exception as e:
        log(f"ConvNeXt arm error {type(e).__name__}: {e} (non-fatal)")
    # ---- Arm A: nartaa 0.949 CoAtNet on gold58 + holdout870 ----
    ids = gold + hold
    ev = stage(ids, real, W / "_eval_comp")
    code = zlib.decompress(base64.b64decode(NARTAA_CODE_B64)).decode("utf-8")
    assert code.count('ROOT = find_test_root()') == 1 and code.count('out = "/kaggle/working/submission.csv"') == 1
    code = code.replace('ROOT = find_test_root()', 'ROOT = "/kaggle/working/_eval_comp"')
    code = code.replace('out = "/kaggle/working/submission.csv"', 'out = "/kaggle/working/nartaa384_eval928.csv"')
    (W / "nartaa_infer.py").write_text(code, encoding="utf-8")
    left = int(BUDGET - (time.perf_counter() - T0) - 120)
    log(f"running nartaa CoAtNet on {len(ids)} studies, timeout {left}s")
    rc = subprocess.call([sys.executable, str(W / "nartaa_infer.py")], timeout=left)
    log(f"nartaa rc={rc}")
    for p in [W / "submission.csv"]:
        if p.exists(): p.unlink(); log("removed submission.csv. DO NOT SUBMIT.")
    for d in ["_eval_comp", "_gold_comp", "_own_env", "own_src"]:
        shutil.rmtree(W / d, ignore_errors=True)
    (W / "DO_NOT_SUBMIT_HOLDOUT.txt").write_text("Train-study predictions (gold58 + holdout870). Never submit. Keep 55779066.\n")
    log("done. DO NOT SUBMIT.")

if __name__ == "__main__":
    main()
'''


def main() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 2
    hash_cell, infer_cell = code_cells
    assert "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef" in hash_cell
    infer_cell = infer_cell.replace("# %% [code]\n", "", 1)
    full = hash_cell + "\n\n" + infer_cell
    b64 = base64.b64encode(zlib.compress(full.encode("utf-8"), 9)).decode("ascii")
    DST.mkdir(exist_ok=True)
    (DST / "main.py").write_text(TEMPLATE.replace("@@CODE@@", b64), encoding="utf-8")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-top-public-dissect-gate",
        "title": "RSNA Knee top public dissect gate",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "stevenleehans/rsna-knee-llm-report-labels",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": ANCHOR_IMAGE,
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {DST} (main.py {len((DST / 'main.py').read_text(encoding='utf-8'))} chars). Do NOT Submit.")


if __name__ == "__main__":
    main()
