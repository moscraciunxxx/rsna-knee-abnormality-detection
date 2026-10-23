"""Frozen public DINOv2-B linear head. Gold 58 and the mill-870 holdout stay out of the loss.

Not a Submit kernel. Not r336's trained SlotHead. Not an unfreeze.
Abort the overlay path if gold macro exceeds 0.94.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_mill_holdout"
DST = ROOT / "kaggle_kernel_probe_b"

PROBE_BLOCK = r'''
PROBE_B_ONLY = True  # frozen public DINOv2-B, new linear head. Not r336 SlotHead. Not mill ensemble. DO NOT SUBMIT.
if PROBE_B_ONLY:
    from pathlib import Path as _PBPath
    import json as _pb_json
    import pandas as _pb_pd
    import numpy as _pb_np
    import torch.nn as _pb_nn
    from sklearn.metrics import roc_auc_score as _pb_auc

    print("[probe-b] PROBE_B_ONLY: frozen DINOv2-B linear head. Gold 58 and mill-870 holdout are not in the loss. DO NOT SUBMIT.", flush=True)
    _pb_work = _PBPath("/kaggle/working")
    _pb_work.mkdir(exist_ok=True)
    _PB_ABORT = 0.94

    def _pb_find_base():
        skip = {"train_series", "test_series", "test_images"}
        base = _PBPath("/kaggle/input")
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
            if "config.json" in names and ("model.safetensors" in names or "pytorch_model.bin" in names):
                cfg = _pb_json.loads((cur / "config.json").read_text(encoding="utf-8"))
                if int(cfg.get("hidden_size", 0)) == 768 and int(cfg.get("num_hidden_layers", 0)) == 12:
                    return cur
            for k in kids:
                if k.is_dir() and k.name not in skip and not k.name.startswith("1.2."):
                    stack.append(k)
        return None

    def _pb_macro(y, s):
        per = []
        for j in range(y.shape[1]):
            yt = y[:, j]
            if len(_pb_np.unique(yt)) < 2:
                per.append(float("nan"))
                continue
            try:
                per.append(float(_pb_auc(yt, s[:, j])))
            except Exception:
                per.append(float("nan"))
        vals = [v for v in per if v == v]
        return per, (float(_pb_np.mean(vals)) if vals else float("nan"))

    _pb_dir = _pb_find_base()
    if _pb_dir is None:
        raise FileNotFoundError("public DINOv2-B (hidden 768, 12 layers) not mounted")
    print(f"[probe-b] backbone {_pb_dir}", flush=True)
    from transformers import AutoModel as _PBAuto
    _pb_bb = _PBAuto.from_pretrained(str(_pb_dir), local_files_only=True)
    _pb_trainable = 0
    for _prm in _pb_bb.parameters():
        _prm.requires_grad = False
        _pb_trainable += int(_prm.requires_grad)
    _pb_bb.eval()
    _pb_dim = int(_pb_bb.config.hidden_size)
    if _pb_dim != 768 or _pb_trainable != 0:
        raise RuntimeError(f"backbone not frozen base: dim={_pb_dim} trainable_flags={_pb_trainable}")
    print(f"[probe-b] frozen dim={_pb_dim} trainable_param_flags={_pb_trainable}", flush=True)

    _pb_train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    _pb_all = set(_pb_train["StudyInstanceUID"].astype(str))
    globals()["_goldfold_id_set"] = lambda: _pb_all
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
    _pb_dev = DEVS[0] if DEVS else torch.device("cpu")
    _pb_bb = _pb_bb.to(_pb_dev)
    _pb_mean = torch.tensor([0.485, 0.456, 0.406], device=_pb_dev).view(1, 3, 1, 1)
    _pb_std = torch.tensor([0.229, 0.224, 0.225], device=_pb_dev).view(1, 3, 1, 1)
    ho_ids = list(_holdout_id_tuple())
    gold = set(
        _pb_train.loc[_pb_train[list(TARGETS)].notna().all(axis=1), "StudyInstanceUID"].astype(str)
    )
    if set(ho_ids) & gold:
        raise RuntimeError("gold 58 leaked into probe holdout IDs")
    v4 = _pb_pd.read_csv(_find_v4_csv(), dtype={"StudyInstanceUID": str})
    v4["StudyInstanceUID"] = v4["StudyInstanceUID"].astype(str)
    v4 = v4.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    fit_ids = [i for i in v4.index.astype(str) if i not in gold and i not in set(ho_ids)]
    if set(fit_ids) & gold or set(fit_ids) & set(ho_ids):
        raise RuntimeError("gold or holdout leaked into the probe train loss")
    print(f"[probe-b] fit n={len(fit_ids)} holdout n={len(ho_ids)} gold n={len(gold)} leaked_fit={len(set(fit_ids)&gold)}", flush=True)

    _pb_t0 = time.time()
    _pb_series = _goldfold_test_series()
    _pb_plane = dict(zip(_pb_series["SeriesInstanceUID"], _pb_series["Anatomical_Plane"]))
    _pb_hte = annotate(walk("test_series"))
    _pb_slots = pick_slots(_pb_hte, _pb_plane)
    _pb_st, _pb_C, _pb_M = build_cache(_pb_slots, _pb_plane, lat_of(_pb_hte, "probe-b "), "probe-b")
    _pb_cache_s = time.time() - _pb_t0
    print(f"[probe-b] cache_s={_pb_cache_s:.1f} n_cached={len(_pb_st)}", flush=True)
    _pb_row = {str(s): i for i, s in enumerate(_pb_st)}

    def _pb_featurize(ids):
        idx = [_pb_row[i] for i in ids]
        out = []
        with torch.inference_mode():
            for b in range(0, len(idx), 4):
                sel = idx[b:b + 4]
                m = torch.from_numpy(_pb_M[sel]).to(_pb_dev)
                rows = torch.from_numpy(_pb_np.ascontiguousarray(_pb_C[sel])).to(_pb_dev).float().div_(255.0)
                bb, ss = rows.shape[:2]
                flat = rows.reshape(bb * ss, *rows.shape[2:])
                flat = (flat - _pb_mean) / _pb_std
                hid = _pb_bb(pixel_values=flat).last_hidden_state
                feat = torch.cat([hid[:, 0], hid[:, 1:].mean(1)], dim=1).reshape(bb, ss, -1)
                w = m.unsqueeze(-1)
                pooled = (feat * w).sum(1) / w.sum(1).clamp_min(1e-6)
                out.append(pooled.float().cpu())
        return torch.cat(out, 0)

    _pb_need = [i for i in list(fit_ids) + list(ho_ids) + list(gold) if i not in _pb_row]
    if _pb_need:
        raise RuntimeError(f"probe cache missing {len(_pb_need)} required studies")
    if _pb_cache_s > 4.0 * 3600:
        _pb_log = _pb_work / "protocol_log.txt"
        _pb_log.write_text(
            f"abort=time cache_s={_pb_cache_s:.1f} budget_h=4 gold_in_train=0 unfreeze=0\n",
            encoding="utf-8",
        )
        print("[probe-b] ABORT time budget. No overlay. DO NOT SUBMIT.", flush=True)
    else:
        _probe8 = time.time()
        _ = _pb_featurize(list(fit_ids)[:8])
        _dt8 = time.time() - _probe8
        _proj_h = (_dt8 / 8.0) * (len(fit_ids) + len(ho_ids) + len(gold)) / 3600.0
        print(f"[probe-b] 8-study feature { _dt8:.2f}s projected_feature_h={_proj_h:.2f}", flush=True)
        Xtr = _pb_featurize(fit_ids)
        ytr = v4.loc[fit_ids, list(TARGETS)].to_numpy(dtype=_pb_np.float32)
        ytr = _pb_np.nan_to_num(ytr, nan=0.5)
        ytr = _pb_np.clip(ytr, 0.0, 1.0)
        mu = Xtr.mean(0)
        sd = Xtr.std(0).clamp_min(1e-6)
        Xtr_n = (Xtr - mu) / sd
        _pb_head = _pb_nn.Linear(Xtr_n.shape[1], len(TARGETS)).to(_pb_dev)
        opt = torch.optim.AdamW(_pb_head.parameters(), lr=1e-3, weight_decay=0.1)
        loss_fn = _pb_nn.BCEWithLogitsLoss()
        Y = torch.from_numpy(ytr).to(_pb_dev)
        X = Xtr_n.to(_pb_dev)
        _pb_head.train()
        for _ep in range(40):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(_pb_head(X), Y)
            loss.backward()
            opt.step()
        _pb_head.eval()
        print(f"[probe-b] head_fit loss={float(loss.detach().cpu()):.4f} epochs=40 weight_decay=0.1", flush=True)

        def _pb_predict(ids):
            feat = (( _pb_featurize(ids) - mu) / sd).to(_pb_dev)
            with torch.inference_mode():
                return torch.sigmoid(_pb_head(feat)).float().cpu().numpy()

        g_ids = [i for i in gold if i in v4.index or i in set(_pb_train["StudyInstanceUID"].astype(str))]
        g_ids = sorted(gold)
        y_gold = (
            _pb_train.drop_duplicates("StudyInstanceUID")
            .set_index("StudyInstanceUID")
            .loc[g_ids, list(TARGETS)]
            .to_numpy(dtype=float)
        )
        s_gold = _pb_predict(g_ids)
        g_per, g_macro = _pb_macro(y_gold, s_gold)
        abort = bool(g_macro == g_macro and g_macro > _PB_ABORT)
        print(f"[probe-b] gold_macro={g_macro:.4f} abort_threshold={_PB_ABORT} abort={abort} gold_in_train=0", flush=True)
        for name, auc in zip(TARGETS, g_per):
            print(f"[probe-b] gold {name:22s} {auc:.4f}", flush=True)
        s_ho = _pb_predict(ho_ids)
        y_ho = (v4.loc[ho_ids, list(TARGETS)].to_numpy(dtype=float) >= 0.5).astype(float)
        h_per, h_macro = _pb_macro(y_ho, s_ho)
        print(f"[probe-b] holdout n={len(ho_ids)} macro={h_macro:.4f}", flush=True)
        for name, auc in zip(TARGETS, h_per):
            print(f"[probe-b] holdout {name:22s} {auc:.4f}", flush=True)
        ho_frame = _pb_pd.DataFrame(s_ho, columns=list(TARGETS))
        ho_frame.insert(0, "StudyInstanceUID", list(ho_ids))
        ho_frame.to_csv(_pb_work / "probe_holdout870.csv", index=False)
        g_frame = _pb_pd.DataFrame(s_gold, columns=list(TARGETS))
        g_frame.insert(0, "StudyInstanceUID", g_ids)
        g_frame.to_csv(_pb_work / "oof_gold_probe.csv", index=False)
        log_lines = [
            f"backbone=dinov2-base frozen=1 dim={_pb_dim}",
            f"unfreeze_last=0 trainable_param_flags={_pb_trainable}",
            f"fit_n={len(fit_ids)} holdout_n={len(ho_ids)} gold_n={len(g_ids)}",
            "gold_in_train=0",
            "holdout_in_train=0",
            f"gold_macro={g_macro:.4f}",
            f"abort_threshold={_PB_ABORT}",
            f"abort={abort}",
            f"holdout_macro={h_macro:.4f}",
            f"cache_s={_pb_cache_s:.1f}",
            "DO NOT SUBMIT",
        ]
        (_pb_work / "protocol_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if abort:
            print("[probe-b] ABORT gold macro > 0.94. Do not overlay. DO NOT SUBMIT.", flush=True)
        else:
            print("[probe-b] gold abort did not fire. License only from the local mill-870 join. DO NOT SUBMIT.", flush=True)
'''


def main() -> None:
    import shutil

    text = (SRC / "main.py").read_text(encoding="utf-8")
    needle = "run_dinov2()\n"
    idx = text.find(needle)
    if idx < 0:
        raise SystemExit("run_dinov2() not found")
    text = text[:idx] + PROBE_BLOCK
    if "\nrun_dinov2()\n" in text:
        raise SystemExit("run_dinov2() call still present after truncate")
    DST.mkdir(exist_ok=True)
    (DST / "main.py").write_text(text, encoding="utf-8")
    meta = """{
  "id": "vitaliecervinschi/rsna-knee-probe-b870",
  "title": "RSNA Knee probe b870",
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
    "tonylica/rsna-knee-bend-dinov3-0917-repro-assets"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "rsna-knee-abnormality-detection"
  ],
  "model_sources": [
    "metaresearch/dinov2/PyTorch/base/1"
  ],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
}
"""
    (DST / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    print(f"wrote {DST / 'main.py'} bytes={len(text)}")


if __name__ == "__main__":
    main()
