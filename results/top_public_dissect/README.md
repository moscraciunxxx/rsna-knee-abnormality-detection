# Top public notebooks: dissection and gate results (2026-10-07, PT)

Research only. Nothing submitted, nothing selected. Anchor 55779066 @ 0.936 stays selected.
Sources were pulled to `external_notebooks/<owner>__<slug>/` (ipynb, kernel-metadata.json, plus a `.py` dump).

## 1. What each notebook is

| Notebook (Kaggle-shown public) | Docker | Sources | What it really is |
|---|---|---|---|
| nartaa/rsna-knee-0949-anatomical-mirror (0.949) | 2757e0 (py3.13/torch2.11) | nartaa/rsna-knee-publication-swa-weights-20261007 | ONE CoAtNet checkpoint, `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (sha 7e5315da...). Author's private scored sub 56906959. |
| matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed (0.949) | **37c64f** (our anchor image) | same nartaa dataset | **Byte-identical copy** of the nartaa inference cell (diff = 0 lines). Title is misleading: no DINO, no stack. |
| sujanmajhisuzan/rsna-knee-apex-grandmaster-stack (0.950) | 2757e0 | nartaa dataset + goodpjw2008/rsna-knee-2-5d-convnext-reader | nartaa cell (+2 lines saving a backup CSV) then a per-finding **rank blend** with the goodpjw2008 ConvNeXt 2.5D reader (3 fold ckpts). |
| pjmathematician/rsna-knee-d4-blend (0.946) | 37c64f | ~17 datasets (raptor maxspan/native384, mattiaangeli CoAtNet resgated/d4/global96/repairv1, DINOv2-S tails, RadImageNet heads, pilkwang labels) + own legs | The public 0.943 "4-way CoAtNet consensus + DINO/RadImageNet base" stack (same lineage as our mill V10) plus a small blend of the author's own models. |

### nartaa 0.949 model (the one new ingredient)
- Arch: timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, Raptor MIL head with per-finding attention over 3-slice windows.
- Input: 96 slices in 5 slots (Sag FS 26, Sag non-FS 22, Cor FS 18, Cor non-FS 12, Axial 18), 140 mm crop, trained at 384 px full field; inference **native 320 px centre crop, K=94 windows**, fp16.
- TTA: plain + anatomical L/R mirror (sagittal: reverse slice channels; cor/ax: width flip), probability mean.
- Training: one seed-42 run, SWA of epochs 14/11/12, **"alldata" = all train studies** (no folds), labels "blendjev" = blend of stevenleehans + pilkwang LLM labels + riadmohamed42 JEV labels + their own report-derived labels, **plus 2,399 OAI knees as masked external supervision**. Gold58 used for checkpoint selection.
- A second checkpoint in the same dataset (`..._c08_r224_swa.pt`) is the 0.945 "efficient 224crop" notebook; same training data/family.

### Apex 0.950 blend weights (ConvNeXt reader weight w; nartaa gets 1-w), rank-pct, then re-ranked
Baker's 0.35, Contusion 0.30, Medial OA 0.25, ACL 0.25, Medial Meniscus 0.20, MCL 0.20, Lateral Meniscus 0.08, Fracture 0.08, Lateral OA 0.08, PF OA 0.08, Effusion 0.04, Synovitis 0.03. No other post-processing; falls back to pure nartaa on error.

### vs our mill V10 (`kaggle_kernel_blend_script/main.py`)
Mill V10 is the public DINOv2/v3 "bend" + RadImageNet heads + public Raptor CoAtNet v4/v5 stack (0.936 lineage). Nartaa differs in: (a) a retrained CoAtNet with 96 slices/K94 full coverage and native-320 crop, (b) mirror TTA, (c) a richer blended label set, (d) **OAI external supervision**. What plausibly accounts for 0.936 -> 0.949: the retrained single CoAtNet itself (two copies score 0.949 in two different docker images, so env is not the driver); the ConvNeXt reader adds only +0.001 (0.949 -> 0.950).

## 2. Rule / external-data risk
- Rules: "Freely & publicly available external data is allowed, including pre-trained models". Host (discussion 733965): click-through datasets OK; datasets needing institution-specific approvals / signed agreements "may present a meaningful accessibility barrier".
- OAI is distributed via NIMH NDA under a Data Use Certification (institutional sign-off). That is exactly the category the host flagged. The weights themselves are public on Kaggle (licence "other": research/education incl. this competition).
- Risk: LOW for leaderboard position, MEDIUM-HIGH for prize eligibility (winner must deliver reproducible training code; OAI part not reproducible by a non-NDA user). Thread 746792 could not be rendered by our fetcher; parent's earlier read: commenters question legality.

## 3. Contamination
- nartaa checkpoint is **alldata** -> it trained on all 870 holdout studies. Holdout numbers for it are not a valid license. Only gold58 is clean of training (but was used for its checkpoint selection).
- goodpjw2008 ConvNeXt reader: 3-fold; the existing `artifacts/convnext_oof870` (517 rows, left-out fold only) is clean.
- Caveat on our own base: mill V10 also contains public "full"-data CoAtNet arms and v4-label-trained heads.

## 4. Gate run (private kernel, pinned anchor env)
`vitaliecervinschi/rsna-knee-top-public-dissect-gate` v1 (private, never submit), docker 37c64f, log: python 3.12.13, torch 2.10.0+cu128, timm 1.0.26, 2xT4. Exact velciraptor inference code (embedded, hash-checked ckpt) on 928 train studies (gold58 + seed-2026 holdout870; ids match coatnet870_ids.csv), 0 fallbacks; ConvNeXt reader on gold58. Runtime 1,347 s (~0.37 GPU-h). Local run impossible: no train DICOMs on the MSI (only test_series), 6 GB laptop GPU.

Scores (`artifacts/top_public_dissect/scores.txt|json`, bootstrap 1000, seed 2026, all-12 rank blends onto mill):

| Set | mill V10 | nartaa 0.949 | ConvNeXt reader | Apex recipe |
|---|---|---|---|---|
| gold58 (radiologist labels, clean) | **0.9575** | 0.9229 | 0.9105 | 0.9240 |
| holdout870 (LLM v4 labels; nartaa contaminated) | 0.9262 | 0.9226 | 0.9439 (gate, fold-contaminated) | - |
| holdout OOF517 (ConvNeXt clean) | 0.9326 | 0.9286 | 0.9447 | 0.9341 |

Blended gain over mill, mill+nartaa all-12:
- gold58: w=0.15 -0.0019 (LB -0.0050); w=0.3 -0.0048 (LB -0.0102); w=0.5 -0.0113 (LB -0.0212).
- holdout870: w=0.15 +0.0018 (LB +0.0012); w=0.3 +0.0029 (LB +0.0017); w=0.5 +0.0029 (LB +0.0010).
- OOF517: w=0.3 +0.0026 (LB +0.0012); mill+nartaa@0.3+cnxt@0.15 +0.0070 (LB +0.0053).
- Apex recipe vs nartaa alone (OOF517): +0.0055 (LB +0.0044) — consistent with its +0.001 public gain.
Mean per-finding Spearman on holdout: nartaa-mill 0.922, nartaa-cnxt 0.891, mill-cnxt 0.909.

## 5. The key finding: our gates do not predict the public LB
Public (Kaggle-shown): nartaa 0.949 > mill 0.936 > ConvNeXt reader alone ~0.929 (author).
Holdout870: ConvNeXt 0.944 > mill 0.926 > nartaa 0.923. Gold58: mill 0.958 > nartaa 0.923 > ConvNeXt 0.911.
Both private yardsticks rank the 0.949 model BELOW mill, and the holdout ranks the 0.929 reader at the top. The holdout labels are LLM v4 labels (the same family mill's heads were fit to), and gold58 has been reused for mill selection for weeks. Every holdout/gold-licensed overlay so far scored <= 0.936 public (0.922, 0.935, 0.935, 0.908, 0.917 ...). Conclusion: the 4-point license bar cannot evaluate this candidate in either direction; for out-of-family models it is measuring label-family agreement, not radiologist truth.

## 6. Structural comparison (step C)
The top notebook is NOT a stack around something close to mill; it is the nartaa single CoAtNet (+ConvNeXt at small weights). So "overlay ingredients onto mill" means grafting a 0.949 model onto a 0.936 base at low weight, which on any of our yardsticks gives <= +0.003, while the public LB says the 0.949 model alone is +0.013 over mill. Reproducing the full nartaa/Apex pipeline (pinned 37c64f — already proven to run cleanly on our account in this env, ~20 min for 928 studies) is the higher-EV path for the LB, with the OAI compliance risk as the main cost.

## 7. Ranked options (none meets the current license bar; all need Vitaly's explicit OK to Submit)
1. Exact nartaa 0.949 reproduction, docker 37c64f (velciraptor bytes, which scored 0.949 on 37c64f). Expected public ~0.949. Risk: OAI/prize eligibility.
2. Apex reproduction (nartaa + ConvNeXt per-finding weights), 37c64f. Expected ~0.950. Same risk; adds the goodpjw2008 reader (pylibjpeg cp312 wheels need absolute paths).
3. Novel: rank blend nartaa (0.7) + mill V10 (0.3) or Apex + mill. Unproven on LB; diversity (Spearman 0.92) suggests small private upside; combined runtime of mill + nartaa must be checked against the 9 h limit.
Do not: overlay nartaa onto mill at w<=0.15 (yardsticks say ~0, public evidence says it throws away most of the 0.013).
