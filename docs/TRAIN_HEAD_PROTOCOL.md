# Train-head protocol — frozen encoder + new linear head on steven v4

2026-10-08 measurement, 57 shared expert rows: keep steven v4, and copy the pilkwang Fracture column onto it (`rsna_knee.soft_labels.FRACTURE_GRAFT_FINDINGS`). Fracture gold AUC **0.7912 → 0.8713**, macro **0.8932 → 0.8999**. All-12 pilkwang replacement macro **0.8700**. Rank-average macro gain **+0.0003**. Known-mention mix loses Fracture (**0.7931 → 0.7833** on the 58). Gold stays out of the loss.

Gold **58 is val-only never train**. Train targets = stevenleehans **v4 blend**, with that Fracture graft.
Frozen backbone is **not** mill 6-slot 336/130 mm (r336 already scored **0.935**).
`unfreeze_last=0`. Abort if gold macro AUC **> 0.94**. Never read `Report` at test.
Keep mill **55779066 / 0.936**. Do not Submit if the abort fires.

Ruled out: V10, V41, CNN-tri, logfix, jointline, dino3s, tta-max, **r336ov 55856648 / 0.935**.

---

## 1. Data facts (measured locally, 2026-08-28)

Sources: `data/train.csv`, `data/train_series.csv`, `data/test.csv`.
LLM table: Kaggle `pilkwang/rsna-knee-llm-labels` → `report_labels_v2.csv` (train-only; test has no `Report`).

| Split | n | Notes |
| --- | ---: | --- |
| Train studies | **4407** | Unique `StudyInstanceUID`, 0 duplicates |
| Train series | **24371** | 1:1 with train studies (0 missing either way) |
| Series / study | mean 5.53, min 3, max 14 | Planes: Sag 9864 / Cor 8609 / Ax 5898 |
| Fluid_Sensitive = Fat_Suppression | 14010 FS / 10361 non-FS | No mixed FS flag in this table |
| Expert gold (all 12 findings non-null) | **58** | Binary {0,1}; no partial-label rows. Matches `data/gold58_ids.csv` (58×13). |
| Train with `Report` | **4407 / 4407** | Multilingual free text |
| LLM-soft rows | **4406** | Overlap 4406; **1** train UID has no LLM row |
| LLM covering non-gold | **4349** | = 4407 − 58; mill's documented train size |
| Public test studies | **3** | `test.csv` has **only** `StudyInstanceUID` |
| Local train DICOMs | **0** | Competition mount on Kaggle only |

The one UID missing from LLM **is gold** (still usable as val, not as a train target):

`1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`

### Gold-58 prevalence (n=58, all three planes present)

| Finding | pos | neg | prevalence |
| --- | ---: | ---: | ---: |
| ACL | 24 | 34 | 0.414 |
| MCL | **9** | 49 | 0.155 |
| Medial Meniscus | 26 | 32 | 0.448 |
| Lateral Meniscus | 23 | 35 | 0.397 |
| Medial OA | 15 | 43 | 0.259 |
| Lateral OA | 11 | 47 | 0.190 |
| PF OA | 21 | 37 | 0.362 |
| Effusion | 35 | 23 | 0.603 |
| Synovitis | 27 | 31 | 0.466 |
| Baker's | 12 | 46 | 0.207 |
| Contusion | 19 | 39 | 0.328 |
| Fracture | 18 | 40 | 0.310 |

MCL (9 pos) and Baker's (12 pos) have SE on AUC on the order of ~0.07. A gold fold-0 of 0.91/0.95 (dino3s) is **not** a hidden-test claim.

Gold series: 336 series, mean 5.79 / study (min 4, max 9). Every gold study has Sagittal + Coronal + Axial.

### LLM-soft (`report_labels_v2.csv`)

Per finding: score ∈ [0.08, 0.94], plus `__conf` and `__verdict` ∈ {YES, NO, UNK}.
Scores are rank-usable, not calibrated. UNK is common (Synovitis UNK 3712/4406; Fracture UNK 2486).

LLM vs gold on the **57** overlapping gold rows (not a model — a label-quality ceiling):

| Finding | LLM-vs-gold AUC | 0.5-agreement |
| --- | ---: | ---: |
| ACL | 0.997 | 0.842 |
| MCL | 0.976 | 0.842 |
| Medial Meniscus | 0.943 | 0.860 |
| Lateral Meniscus | 0.841 | 0.772 |
| Medial OA | 0.908 | 0.842 |
| Lateral OA | 0.789 | 0.842 |
| PF OA | 0.891 | 0.789 |
| Effusion | 0.830 | 0.719 |
| Synovitis | **0.694** | 0.702 |
| Baker's | 0.932 | 0.895 |
| Contusion | 0.768 | 0.614 |
| Fracture | 0.871 | 0.825 |
| **macro** | **0.870** | — |

Train on LLM scores (and optionally `score * conf` as sample weight). Do **not** harden UNK→0.

Public test / hidden test: **never** read `Report`.

---

## 2. Mill encoder (what not to steal)

Local `rsna_knee/encoder.py` is a **numpy 24-channel stub**. The scored mill lives in `kaggle_kernel_blend_script/main.py`.

### Frozen DINOv2-S feature dim

```573:590:kaggle_kernel_blend_script/main.py
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    ...
    bb = AutoModel.from_pretrained(str(p))
    ...
    dim = bb.config.hidden_size
    ...
    return Model(bb, dim, pool=pool, prior=prior)
```

| Piece | Mill V10 DINO |
| --- | --- |
| Backbone | HuggingFace `metaresearch/dinov2/pytorch/small/1` (`variant='small'`) |
| `hidden_size` | **384** (ViT-S/14) |
| Pool | `cls_mean` → CLS token + patch mean (`POOL_PARTS=2`) |
| SlotHead input dim | **384 × 2 = 768** |
| SlotHead | `LayerNorm → Linear(dim, 256) → GELU` + slot embedding + 12 queries |
| Crop / size | 130 mm, **336 px**, `GROUP=3` slices as channels |
| Member load | `build_model(m['config']['unfreeze_last'], variant=..., pool=..., prior=...)` then `load_state_dict` mill `.pt` |

`find_dinov2` currently keys off **any** directory containing `config.json`. Mill ASSET packages can also carry a `config.json`. That is the steal:

- Do **not** point `AutoModel.from_pretrained` at mill `rsna-knee-weights` / member dirs.
- Do **not** call `adopt_config_globals(mill_member_cfg)`.
- Do **not** `load_state_dict` mill SlotHead / mill member `state`.
- Require `'dinov2' in path.lower()` **and** `config.json` **and** the requested variant (`small` vs `base`).

A **new** head on frozen public weights is mill-incompatible even if the ViT-S architecture matches. Prefer **ViT-B** so the backbone is not mill's 384-d mill.

### Public DINOv2-B (nursrijan r336) — loadable?

| Source | What it is | Use? |
| --- | --- | --- |
| Model `metaresearch/dinov2/PyTorch/base/1` | Public ViT-B/14, `hidden_size=768`, own HF `config.json` | **Yes — frozen encoder** |
| Dataset `nursrijan/rsna-knee-dinov2-base-llm-models` | `r336_best.pt` / `r224_best.pt` each **348 014 603 B** (~DINOv2-B fp32) | Encoder fallback only |
| nursrijan train recipe | `RUNS=[{r224},{r336}]`, `EPOCHS=10`, `UNFREEZE_LAST=6`, `TIME_BUDGET=8h` | **No — do not clone** |

`r336_best.pt` is already-trained public mill-like weights (last-6 unfrozen SlotHead). A local inspect of that file (`cfg.img=336`, 6 slots, 12 logits, `prior=False`) scored **~0.786** gold-macro — well below mill’s documented **0.9167** on the same 58. Loading it as the **scored head** is not “train a head” and is not a lift.

If HF B is missing, load **backbone tensors only** from `r336_best.pt`, freeze them, attach a **new** head.

DINOv2-B + `cls_mean` → SlotHead / linear input **768 × 2 = 1536**.

**Pitfall:** mill `find_dinov2` is a `config.json` BFS. Mounting `metaresearch/dinov2/PyTorch/base/1` beside mill-small can make mill members load ViT-B and fail fingerprints. In a combined kernel, `find_dinov2('small')` / `find_dinov2('base')` must require the variant in the path. Never pass mill ASSET as `source=`.

---

## 3. Architecture (kernel, T4, internet OFF)

### Encoder — freeze 100%

1. Primary: public DINOv2-B @ **336**, ImageNet mean/std, 130 mm centre crop.
2. Fallback: public DINOv2-S @ 336, **new** head (not mill member weights). Same 384-d mill family — higher correlation risk.
3. `requires_grad=False` on **every** backbone param including last blocks and final LN.
4. Abort if `unfreeze_last > 0` or any backbone `requires_grad`.

### Input (keep mill-cheap, not co-plane-46)

Six public slots × **3** slices (`GROUP=3`) = **18** images / study.

Slots (public scheme, not mill recovered names):
`SAG_FLUID, COR_FLUID, AX_FLUID, SAG_STRUCT, COR_STRUCT, AX_STRUCT`.

Do **not** encode 46 slices (nursrijan co-plane). That is a 9 h encode budget, i.e. another long GPU burn.

Cache per study: `[6, 3, 336, 336]` uint8 **or** pooled tokens `[6, 1536]` float16 after one frozen pass. Prefer **write embeddings once**, then train the head on cached tokens so a bad head hyperparam does not re-decode DICOMs.

### Head (tiny; anti-overfit)

**A. Linear (default).** `LayerNorm(1536) → Dropout(0.3) → Linear(1536, 12)`.
Params ≈ 18 k. Strong L2 (`weight_decay=0.05` … `0.1`).

**B. Tiny attention (only if linear is dead on LLM CV).** Mill-style `SlotHead` hidden **128–256**, 12 queries, dropout 0.3, `weight_decay=0.05`. ~0.2–0.4 M params. Still freeze the ViT.

Not: 2-layer `TransformerEncoder` over 46 slices (co-plane). Too many params for this label budget.

Loss: `BCEWithLogits` on LLM **soft** scores for the 4349. Optional per-row weight = mean `__conf`. No pos-weight from the 58.

### Splits (non-negotiable)

| Set | n | Role |
| --- | ---: | --- |
| Train | **4349** | LLM-soft targets only |
| Val / gate | **58 gold** | **Never in the loss.** No `GOLD_WEIGHT=8`. No fine-tune on gold. |
| LLM GroupKFold | 5, grouped by study | Early-stop / WD / dropout **only**. Not the publish gate. |
| Publish gate | 58 gold vs mill on the **same** 58 | Per-finding AUC; overlay iff Δ and not mill-correlated |

Mill (raptor writeup) already used this split: “58 … were never trained on … 0.9167 macro-AUC on them.” Our 58 numbers are comparable **only** if we also hold them out.

### Mill compare in the **same** kernel

After the probe is trained:

1. Run mill infer **only** on the 58 gold study folders under `train_series/` (not the hidden test).
2. Write `oof_gold_mill.csv`, `oof_gold_probe.csv`, `auc_compare.csv`.
3. Per-finding Δ = probe AUC − mill AUC on n=58.
4. Overlay candidates: Δ > 0 **and** not a ruled-out family, **and** at most **1–2** findings, weight **≪ 0.50** (start 0.15).
5. If mill-58 would blow the wall, skip mill compare and **do not overlay** (fail-closed).

Public LB n=3 is not a gate.

---

## 4. Kernel time estimate (Tesla T4, prefer T4×2)

Probe 8 studies at start. Project `n_img / imgs_per_s`. Abort if projected encode > **2.8 h** or total > **4.0 h**.

| Mode | Images / study | Projected T4×2 | This protocol |
| --- | ---: | --- | --- |
| Frozen S @336, 6×3, linear head | 18 | **1.5–2.5 h** | Allowed fallback |
| Frozen B @336, 6×3, linear/tiny SlotHead | 18 | **2.5–4.0 h** | **Preferred** |
| Frozen B, 46 slices (co-plane) | ~46 | 5–8 h | Abort |
| Unfreeze last 6 of B (nursrijan r336) | 18 | ≥ 8 h | Abort |
| Full-finetune (dino3s family) | — | ~24 h | Abort |
| Mill 20-member on 58 gold only | mill | ~20–45 min | Optional same-kernel compare |
| Head fit on cached tokens | 0 extra DICOM | minutes | Always |

Encode 4349 + 58 ≈ 79 k images (18 / study). Conservative T4×2 rate for ViT-B@336 including DICOM decode: ~8 img/s → **~2.8 h encode** + 0.5 h mill-58 + 0.15 h head ≈ **3.4 h**. An 8-study probe at kernel start replaces this guess; abort if it projects above 4 h.

Head fit is not the clock. Decode + ViT-B is.

Hidden mill V10 on the **full** test is ~4.2 h by itself — do **not** run mill on hidden test in this train kernel.

Wall cap: `TIME_BUDGET = 4.0 * 3600`. If encode overruns, dump whatever embeddings exist and **do not** unfreeze to “catch up.”

---

## 5. Anti-overfit checklist

- 58 gold = **val only**. nursrijan `GOLD_WEIGHT=8` is the overfit path (dino3s gold 0.91/0.95 → public 0.908).
- Train labels = LLM-soft 4349, not gold, not `sample_submission` 0.5s.
- Strong L2 + dropout; linear head first.
- Do not fit 384-d / 1536-d probes **on 58 rows**.
- Group CV on LLM for hyperparameters; gold is a **once** sanity gate.
- Overlay ≪ 0.50, 1–2 findings, fail-closed. No 1.0 replace. No uniform 0.50-all-12.
- Internet OFF. No report text at infer.
- Do not TTA mill’s own DINO/CoAtNet (tta-max tied 0.936).

---

## 6. Go / no-go this session

Live CLI 2026-10-07 (`kaggle quota`): GPU **0.00 h used / 30.00 h remaining**, refresh **2026-10-10T00:00:00**. The 2026-09-05 refresh line below is the previous capture.

Previous capture (`kaggle quota`, 2026-09-05 session): GPU **0.00 h used / 30.00 h remaining**, refresh **2026-09-05T00:00:00**.
Last teacher-note capture: 23.06 h used / **6.94 h** remaining, refresh 2026-08-29 (week has rolled).

| Action | Verdict |
| --- | --- |
| Push a scored kernel now | **NO-GO** (explicit) |
| Start 24 h full-finetune / unfreeze-last-N | **NO-GO** |
| Train locally | **NO-GO** — train DICOMs are not on disk |
| Frozen linear/tiny-attn kernel **later**, after a 8-study probe projects &lt; 4 h | **GO** (quota is 30 h; budget is still 4 h, not 24 h) |
| Spend GPU **this session** | **NO-GO** — protocol + CPU dry-run only |

**This session: NO-GO.** Next GPU session: freeze encoder, train head on 4349 LLM-soft, gate on 58, abort if probe says another long burn.

---

## 7. Kernel mount list (when it is built — not now)

Competition: `rsna-knee-abnormality-detection` (train DICOMs live here).

Models:

- `metaresearch/dinov2/PyTorch/base/1` (primary encoder)
- `metaresearch/dinov2/PyTorch/small/1` (fallback encoder **and** mill infer if mill-58 is in-kernel)

Datasets:

- `pilkwang/rsna-knee-llm-labels`
- mill assets **only** if running mill-58 compare (`tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, mill DINO members). Do not load their `config.json` as the **probe** backbone.
- Optional: `nursrijan/rsna-knee-dinov2-base-llm-models` (backbone fallback; strip head).

Machine: NvidiaTeslaT4, internet OFF, GPU on.

Artifacts (working dir, not a leaderboard CSV unless a later scored overlay is approved):

- `embeddings_train.npy` / `embeddings_gold.npy`
- `probe_head.pt` (head only)
- `oof_gold_probe.csv`, `oof_gold_mill.csv`, `auc_compare.csv`
- `protocol_log.txt` with freeze proof (`requires_grad` count == 0 on backbone)

---

## 8. Local dry-run

`python scripts/train_probe_head.py` loads CSVs + LLM table if present, prints this split and the go/no-go, and **does not** pretend `train_series/**/*.dcm` exist.
