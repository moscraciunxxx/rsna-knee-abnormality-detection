"""Copy mill-holdout (packing + holdout IDs), skip mill ensemble, encode r336 ViT-B on the 870.

dreaddevelopment/rsna-knee-model is EfficientNet-B0 ~17 MB — head-sized, not ViT-B.
Scored encode is nursrijan r336_best.pt (loadable ViT-B 12-head). Not DINOsaur V5.
Save Version only. Do not Submit.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_mill_holdout"
DST = ROOT / "kaggle_kernel_vitb870"

VITB_BLOCK = r'''
VITB_HOLD_ONLY = True  # r336 ViT-B 12-head on seed-2026 LLM-soft holdout. Not mill ensemble. Not DINOsaur V5.
if VITB_HOLD_ONLY:
    from pathlib import Path as _VBPath
    import pandas as _vb_pd
    import numpy as _vb_np
    from sklearn.metrics import roc_auc_score as _vb_auc

    print("[vitb870] VITB_HOLD_ONLY: skip mill V10 ensemble; encode r336 ViT-B on LLM-soft holdout. DO NOT SUBMIT.", flush=True)
    _vb_work = _VBPath("/kaggle/working")
    _vb_work.mkdir(exist_ok=True)

    def _vb_find_r336():
        base = _VBPath("/kaggle/input")
        if not base.is_dir():
            return None
        skip = {"train_series", "test_series", "test_images"}
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
            if "r336_best.pt" in names:
                return cur / "r336_best.pt"
            for k in kids:
                if k.is_dir() and k.name not in skip and not k.name.startswith("1.2."):
                    stack.append(k)
        return None

    def _vb_macro(y, s):
        per = []
        for j in range(y.shape[1]):
            yt = y[:, j]
            if len(_vb_np.unique(yt)) < 2:
                per.append(float("nan"))
                continue
            try:
                per.append(float(_vb_auc(yt, s[:, j])))
            except Exception:
                per.append(float("nan"))
        vals = [v for v in per if v == v]
        return per, (float(_vb_np.mean(vals)) if vals else float("nan"))

    ckpt = _vb_find_r336()
    if ckpt is None:
        raise FileNotFoundError("r336_best.pt not mounted; cannot encode ViT-B holdout")
    print(f"[vitb870] loading {ckpt}", flush=True)
    from transformers import Dinov2Config as _VBDinov2Config, Dinov2Model as _VBDinov2Model
    _vb_ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _vb_bb_cfg = _VBDinov2Config(
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
    _vb_bb = _VBDinov2Model(_vb_bb_cfg)
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
    _vb_dev = DEVS[0] if DEVS else torch.device("cpu")
    _vb_model = Model(_vb_bb, 768, pool="cls_mean", prior=False).to(_vb_dev)
    missing, unexpected = _vb_model.load_state_dict(_vb_ck["model"], strict=False)
    print(f"[vitb870] load missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}", flush=True)
    _vb_model.eval()
    ho_ids = list(_holdout_id_tuple())
    gold = set(
        pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
        .loc[lambda t: t[list(TARGETS)].notna().all(axis=1), "StudyInstanceUID"]
        .astype(str)
    )
    leaked = set(ho_ids) & gold
    print(f"[vitb870] llm-soft holdout n={len(ho_ids)} leaked_vs_gold58={len(leaked)} seed=2026", flush=True)
    if leaked:
        raise RuntimeError("gold 58 leaked into vitb holdout IDs")
    _vb_series = _goldfold_test_series()
    _vb_plane = dict(zip(_vb_series["SeriesInstanceUID"], _vb_series["Anatomical_Plane"]))
    _vb_hte = annotate(walk("test_series"))
    _vb_slots = pick_slots(_vb_hte, _vb_plane)
    _vb_st, _vb_C, _vb_M = build_cache(_vb_slots, _vb_plane, lat_of(_vb_hte, "vitb870 "), "vitb870")
    _vb_img = int(_vb_ck.get("cfg", {}).get("img", 336))
    _vb_idx = _vb_np.arange(len(_vb_st))
    _vb_pred = predict_member(
        _vb_model, _vb_C, _vb_M, _vb_idx, _vb_dev, _vb_img, group=3, starts=[0]
    )
    vitb = _vb_pd.DataFrame(_vb_pred, columns=list(TARGETS))
    vitb.insert(0, "StudyInstanceUID", list(_vb_st))
    vitb.to_csv(_vb_work / "vitb_holdout870.csv", index=False)
    # valid 12-col file so the kernel does not crash; IDs are train holdout — DO NOT SUBMIT
    vitb.to_csv(_vb_work / "submission.csv", index=False)
    v4 = _vb_pd.read_csv(_find_v4_csv(), dtype={"StudyInstanceUID": str})
    v4["StudyInstanceUID"] = v4["StudyInstanceUID"].astype(str)
    v4 = v4.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    missing_v4 = [i for i in ho_ids if i not in v4.index]
    if missing_v4:
        raise RuntimeError(f"v4 labels missing for {len(missing_v4)} holdout UIDs")
    y = (v4.loc[ho_ids, list(TARGETS)].to_numpy(dtype=float) >= 0.5).astype(float)
    vitb_i = vitb.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    missing_vb = [i for i in ho_ids if i not in vitb_i.index]
    if missing_vb:
        print(
            f"[vitb870] VITB PREDS: MISSING {len(missing_vb)}/{len(ho_ids)} holdout UIDs",
            flush=True,
        )
        raise RuntimeError("VITB PREDS: MISSING")
    vb_s = vitb_i.loc[ho_ids, list(TARGETS)].to_numpy(dtype=float)
    vb_per, vb_macro = _vb_macro(y, vb_s)
    print(f"[vitb870] vitb vs llm-soft holdout n={len(ho_ids)} macro={vb_macro:.4f}", flush=True)
    for name, auc in zip(TARGETS, vb_per):
        print(f"[vitb870] vitb {name:22s} {auc:.4f}", flush=True)
    mill_csv = Path(__file__).resolve().parent / "mill_v10_holdout870.csv"
    if mill_csv.is_file():
        mill = _vb_pd.read_csv(mill_csv, dtype={"StudyInstanceUID": str})
        mill["StudyInstanceUID"] = mill["StudyInstanceUID"].astype(str)
        mill = mill.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        missing_m = [i for i in ho_ids if i not in mill.index]
        if missing_m:
            print(f"[vitb870] mill-870 CSV missing {len(missing_m)} UIDs", flush=True)
        else:
            mill_s = mill.loc[ho_ids, list(TARGETS)].to_numpy(dtype=float)
            mill_per, mill_macro = _vb_macro(y, mill_s)
            print(f"[vitb870] mill vs llm-soft holdout n={len(ho_ids)} macro={mill_macro:.4f}", flush=True)
            print(f"{'finding':22} {'mill':7} {'vitb':7} {'delta':7}", flush=True)
            n_lose = 0
            for name, m, n in zip(TARGETS, mill_per, vb_per):
                dlt = (n - m) if (m == m and n == n) else float("nan")
                print(f"{name:22} {m:7.4f} {n:7.4f} {dlt:7.4f}", flush=True)
                if dlt == dlt and n < m:
                    n_lose += 1
            if n_lose == len(TARGETS):
                print("[vitb870] STOP_PUBLIC_TRAINS: vitb < mill on every finding; keep 55779066; DO NOT SUBMIT", flush=True)
            else:
                print("[vitb870] not all-lose; overlay only if mill-870 join licenses subset ≪0.50; DO NOT SUBMIT from this kernel", flush=True)
    else:
        print("[vitb870] mill_v10_holdout870.csv not shipped in kernel folder; join locally after download", flush=True)
    print(
        f"[vitb870] RESEARCH_METHOD=nursrijan r336 ViT-B/14 SlotHead on mill-870 UIDs "
        f"(not dreaddevelopment EfficientNet-B0, not DINOsaur V5, not mill ensemble) "
        f"| vitb_holdout_macro={vb_macro:.4f} n={len(ho_ids)} leaked={len(leaked)} "
        f"| Baker's ok | DO NOT SUBMIT (train holdout IDs)",
        flush=True,
    )

'''


def main() -> None:
    DST.mkdir(exist_ok=True)
    import shutil

    text = (SRC / "main.py").read_text(encoding="utf-8")
    needle = "run_dinov2()\n"
    idx = text.find(needle)
    if idx < 0:
        raise SystemExit("run_dinov2() not found")
    # Truncate mill ensemble: Saturday T4 must not fall through to run_dinov2().
    text = text[:idx] + VITB_BLOCK
    if "\nrun_dinov2()\n" in text:
        raise SystemExit("run_dinov2() call still present after truncate")
    DST.mkdir(exist_ok=True)
    (DST / "main.py").write_text(text, encoding="utf-8")
    mill_src = ROOT / "data" / "mill_v10_holdout870.csv"
    if mill_src.is_file():
        shutil.copy2(mill_src, DST / "mill_v10_holdout870.csv")
    meta = """{
  "id": "vitaliecervinschi/rsna-knee-vitb870",
  "title": "RSNA Knee vitb870",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "stevenleehans/rsna-knee-llm-report-labels",
    "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
    "nursrijan/rsna-knee-dinov2-base-llm-models"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "rsna-knee-abnormality-detection"
  ],
  "model_sources": [
    "metaresearch/dinov2/PyTorch/small/1"
  ],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
}
"""
    (DST / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    print(f"wrote {DST / 'main.py'} bytes={len(text)}")


if __name__ == "__main__":
    main()
