"""Score top-public ingredients vs mill V10 on gold58 (clean) and holdout870 (nartaa-contaminated).

Never Submits. Writes artifacts/top_public_dissect/scores.json and a text table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsna_knee.mill_holdout import holdout_ids_and_y  # noqa: E402

OUT = ROOT / "artifacts" / "top_public_dissect"
ID = "StudyInstanceUID"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
APEX_W = {"Baker's": 0.35, "Contusion": 0.30, "Medial OA": 0.25, "ACL": 0.25, "Medial Meniscus": 0.20,
          "MCL": 0.20, "Lateral Meniscus": 0.08, "Fracture": 0.08, "Lateral OA": 0.08, "PF OA": 0.08,
          "Effusion": 0.04, "Synovitis": 0.03}
NB = 1000


def rp(a):
    return pd.DataFrame(a).rank(pct=True, method="average").to_numpy()


def per_auc(y, s):
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) == 2:
            out[j] = roc_auc_score(y[:, j], s[:, j])
    return out


def blend(base, new, w):
    w = np.asarray(w, float).reshape(1, -1)
    return (1 - w) * rp(base) + w * rp(new)


def boot_gain(y, base, cand, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(y)
    pm, pf = [], []
    for _ in range(NB):
        t = rng.integers(0, n, n)
        a = per_auc(y[t], base[t]); b = per_auc(y[t], cand[t])
        pf.append(b - a); pm.append(np.nanmean(b) - np.nanmean(a))
    pm = np.array(pm); pf = np.array(pf)
    return {
        "macro_gain_point": float(np.nanmean(per_auc(y, cand)) - np.nanmean(per_auc(y, base))),
        "macro_gain_lo95": float(np.percentile(pm, 2.5)), "macro_gain_hi95": float(np.percentile(pm, 97.5)),
        "per_finding_gain_point": dict(zip(L, np.round(per_auc(y, cand) - per_auc(y, base), 4).tolist())),
        "per_finding_gain_lo95": dict(zip(L, np.round(np.nanpercentile(pf, 2.5, axis=0), 4).tolist())),
    }


def load(path, ids):
    df = pd.read_csv(path, dtype={ID: str}).set_index(ID)
    return df.loc[ids, L].to_numpy(float)


def evaluate(tag, y, base, arms, note):
    res = {"note": note, "n": int(len(y)), "positives": dict(zip(L, y.sum(0).astype(int).tolist())),
           "standalone": {}, "blends": {}}
    lines = [f"=== {tag} (n={len(y)}) {note}"]
    cols = {"mill_v10": base, **arms}
    for k, v in cols.items():
        pa = per_auc(y, v)
        res["standalone"][k] = {"macro": float(np.nanmean(pa)), "per_finding": dict(zip(L, np.round(pa, 4).tolist()))}
    lines.append(f"{'finding':18}" + "".join(f"{k[:12]:>13}" for k in cols))
    for j, f in enumerate(L):
        lines.append(f"{f:18}" + "".join(f"{res['standalone'][k]['per_finding'][f]:13.4f}" for k in cols))
    lines.append(f"{'MACRO':18}" + "".join(f"{res['standalone'][k]['macro']:13.4f}" for k in cols))
    for k, v in arms.items():
        for w in (0.15, 0.3, 0.5, 0.7, 1.0):
            cand = blend(base, v, [w] * 12)
            r = boot_gain(y, base, cand)
            res["blends"][f"mill+{k}@{w}"] = r
            lines.append(f"mill+{k}@{w:.2f} all-12: gain {r['macro_gain_point']:+.4f} LB95 {r['macro_gain_lo95']:+.4f} UB95 {r['macro_gain_hi95']:+.4f}")
    if "nartaa384" in arms and "convnext" in arms:
        apex = blend(arms["nartaa384"], arms["convnext"], [APEX_W[f] for f in L])
        r = boot_gain(y, base, apex)
        res["blends"]["apex_recipe(nartaa+convnext)_vs_mill"] = r
        res["standalone"]["apex_recipe"] = {"macro": float(np.nanmean(per_auc(y, apex)))}
        lines.append(f"APEX recipe standalone macro {np.nanmean(per_auc(y, apex)):.4f}; vs mill gain {r['macro_gain_point']:+.4f} LB95 {r['macro_gain_lo95']:+.4f}")
        r2 = boot_gain(y, arms["nartaa384"], apex)
        res["blends"]["apex_recipe_vs_nartaa_alone"] = r2
        lines.append(f"APEX recipe vs nartaa alone: gain {r2['macro_gain_point']:+.4f} LB95 {r2['macro_gain_lo95']:+.4f}")
        for w in (0.3, 0.5):
            tri = blend(blend(base, arms["nartaa384"], [w] * 12), arms["convnext"], [0.15] * 12)
            r3 = boot_gain(y, base, tri)
            res["blends"][f"mill+nartaa@{w}+convnext@0.15"] = r3
            lines.append(f"mill+nartaa@{w}+cnxt@0.15: gain {r3['macro_gain_point']:+.4f} LB95 {r3['macro_gain_lo95']:+.4f}")
    return res, lines


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID: str})
    gold = train[train[L].notna().all(axis=1)]
    gids = gold[ID].tolist(); yg = gold[L].to_numpy(float)
    nart = OUT / "nartaa384_eval928.csv"
    report, lines = {}, []
    arms_g = {"nartaa384": load(nart, gids)}
    if (OUT / "convnext_gold58.csv").is_file():
        arms_g["convnext"] = load(OUT / "convnext_gold58.csv", gids)
    report["gold58"], ln = evaluate("GOLD58", yg, load(ROOT / "data" / "mill_v10_gold58.csv", gids), arms_g,
                                    "radiologist gold labels; nartaa did NOT train on these (but used them for checkpoint selection)")
    lines += ln
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID: str})
    hids, yh = holdout_ids_and_y(train, v4)
    hids = list(hids)
    arms_h = {"nartaa384": load(nart, hids), "convnext_gate": load(ROOT / "artifacts" / "convnext_mil870" / "convnext_holdout870.csv", hids)}
    report["holdout870"], ln = evaluate("HOLDOUT870", yh, load(ROOT / "data" / "mill_v10_holdout870.csv", hids), arms_h,
                                        "LLM-soft v4>=0.5 labels; CONTAMINATED: nartaa 'alldata' trained on these studies")
    lines += ln
    oof = pd.read_csv(ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv", dtype={ID: str})
    oids = [u for u in hids if u in set(oof[ID])]
    ymap = dict(zip(hids, yh)); yo = np.stack([ymap[u] for u in oids])
    arms_o = {"nartaa384": load(nart, oids), "convnext": load(ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv", oids)}
    report["holdout_oof517"], ln = evaluate("HOLDOUT-OOF517", yo, load(ROOT / "data" / "mill_v10_holdout870.csv", oids), arms_o,
                                            "ConvNeXt is OOF here; nartaa still contaminated")
    lines += ln
    report["do_not_submit"] = True; report["keep"] = "55779066"
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
