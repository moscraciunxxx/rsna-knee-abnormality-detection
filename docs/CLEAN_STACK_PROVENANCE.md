# Stack 0.944 + mill V10 provenance (verified 2026-10-13)

Question: is there a fully prize-safe (no-OAI, no gold-trained) public stack?
Answer: no. Neither `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` nor
mill V10 (`55779066`) is free of gold-fitted/gold-tuned weights. None of the
audited sources below uses OAI external images, so there is no external-data
rules violation, but several components fit or select directly on `gold58`.

Evidence pulled on disk:
- 0.944 stack `kernel-metadata.json` (`artifacts/_nextbets/goodpjw2008_rsna-knee-stack-2-5d-convnext-mil-lb-0-944/`);
- mill V10 `kaggle_kernel_blend_script/kernel-metadata.json`;
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` (`SOURCE_AND_LICENSES.md`, `bundle_manifest.json` in `artifacts/_mill_prov/tonylica_docs/`);
- `sofiaanjenje/rsna-knee-e11-train` + `e13-train` (`artifacts/_stack944_train/`);
- `pilkwang/rsna-knee-baseline-v1` (`artifacts/_mill_prov/mattia_manifests/pilkwang_baseline/`);
- `mattiaangeli` manifests & trainer scripts (`artifacts/_mill_prov/resgated_train/`, `artifacts/_mill_prov/mattia_manifests/`);
- `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` (`artifacts/_mill_prov/`);
- `goodpjw2008` reader code (`artifacts/provenance/reader_code/`).

## 1. goodpjw2008 0.944 stack sources

| Source | Verdict | Evidence (card + code) |
|---|---|---|
| `goodpjw2008/rsna-knee-2-5d-convnext-reader` | **CLEAN** | 3-fold ConvNeXt-Tiny (`cnxt_v0_fold{0,1,2}.pt`). `make_labels.py` builds v0 as mean of 4 public LLM tables (steven v4, pilkwang v2, lixin sol56, laymond qwen); gold58 gets `fold = -1` and is never trained on; zero OAI/external strings in `train.py`/`preprocess.py`. |
| `dreaddevelopment/raptor-knee-widedense` | **CLEAN** | CC0-1.0. Card + companion notebook (`knee-mri-twelve-findings-from-a-single-model.py`): CoAtNet-384 trained on 4,349 LLM-labelled competition studies; 58 gold held out (0.9167 macro AUC); explicitly states "No external data was used." |
| `dreaddevelopment/raptor-knee-maxspan` | **CLEAN (no OAI; gold58 used for eval only)** | CC0-1.0. Same author/pipeline (`raptor_ft_coatnet_v5_full_swa.pt`), 64 slices across 2–98% span on the 4,349 competition-study corpus, 140 mm crop, 336 px. No OAI or external data. |
| `dreaddevelopment/raptor-knee-native384` & `native384dense` | **CLEAN (no OAI; gold58 eval only)** | CC0-1.0. Same 4,349-study Raptor training pipeline (`v8_full_swa.pt`, `v10_full.pt`) with native 384 px slice storage; gold58 held out (0.9174). No external data. |
| `metaresearch/dinov2/PyTorch/small/1` | **CLEAN** | Apache-2.0. Official Meta self-supervised DINOv2 ViT-S/14 backbone; not competition-trained. |
| `marwanmath/resnet-50-radimagenet-marwan` | **CLEAN (encoder)** | CC-BY-NC-SA-4.0. Public RadImageNet ResNet-50 backbone weights (`ResNet50.pt`). |
| `mattiaangeli/opencv-python-headless-4120088-x86` | **CLEAN** | Offline OpenCV wheel, not a model. |
| `pilkwang/rsna-knee-llm-labels` | **CLEAN** | CC0-1.0. Train-time LLM report labels (`report_labels_v2.csv`). |
| `pilkwang/rsna-knee-weights` | **TAINTED (gold-trained, no OAI)** | CC0-1.0, 20 DINOv2-small checkpoints. Author's `pilkwang/rsna-knee-baseline-v1` cell 32 explicitly states: *"The annotated studies stay in training, at elevated weight, because they are the only labels read from the images rather than from text."* No OAI, but gold58 is in the training loss. |
| `antoinegg1/rsna-knee-e11-diverse-heads-v20` & `e9-radimagenet-heads-v15` | **TAINTED (gold-trained, no OAI)** | Training code (`sofiaanjenje/rsna-knee-e11-train` cell 23) fits LogisticRegression heads directly on the 58 gold rows (`exact_mask = train_df[target_name].notna()`, `fit_weight = 7.0` on `exact_idx`, plus PCA+LR fit on `exact_mask` only). |
| `sofiaanjenje/rsna-knee-e11-train` & `e13-train` | **TAINTED (gold-trained, no OAI)** | Same notebook code as above (`v52_e11_heads.pt` fit on gold58 at weight 7.0). |
| `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` | **TAINTED (gold-trained, no OAI)** | Upstream V52 RadImageNet heads adapted by E9/E11; trained on the competition split with gold58 included. |
| `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | **TAINTED (gold-tuned epoch selection, no OAI)** | CC0-1.0. `trainer_a6000_epochs01_06.py` + `coat_resgated_ep10_top3_manifest.json`: trained on 4,349 report-labelled competition studies (`smart_slice_cache384_v3_band0298`, no OAI), checkpoint selection criterion = *"top three official Gold58 macro-AUC by epoch 10"* (epochs 4, 6, 8). |
| `mattiaangeli/rsna-knee-coatnet-global96-top3` | **TAINTED (gold-tuned epoch selection, no OAI)** | CC0-1.0. `coatnet_global96_top3_manifest.json`: `"all_report_studies": 4349`, `"selector": "official Gold58 macro-AUC only"` (epochs 16, 23, 18). No OAI. |
| `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` | **TAINTED (gold-tuned, no OAI)** | CC0-1.0. `coatnet_d4_depthzone_swa_inference.py` inherits the Global96 parent and asserts the D4 SWA3 adapter's `gold_auc` match. No OAI. |
| `mattiaangeli/knee-mri-fold-weights` | **TAINTED (gold-calibrated in stack, no OAI)** | CC0-1.0. 5-fold DINOv3-small (`m_f0..m_f4.pt`), combined in `transformer_rad.py` with the gold-trained DINOv2/E13 branches and a frozen 12-output linear `calibration.json`. |

## 2. Mill V10 (`55779066`) sources — explicit verdict per source

| Mill V10 source | Verdict | Evidence |
|---|---|---|
| `dreaddevelopment/raptor-knee-widedense` | **CLEAN** | CC0-1.0. Trained on 4,349 competition studies with LLM report soft labels; gold58 held out; card explicitly states *"No external data was used."* |
| `dreaddevelopment/raptor-knee-maxspan` | **CLEAN (no OAI; gold58 eval only)** | CC0-1.0. Same 4,349-study competition corpus & author as widedense, widened to 64 slices across 2–98% span (`raptor_ft_coatnet_v5_full_swa.pt`). No OAI or external data. |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | **TAINTED (gold-trained & gold-tuned, NO OAI)** | `SOURCE_AND_LICENSES.md` + `bundle_manifest.json` prove it repackages: (1) `pilkwang/rsna-knee-weights` (20 DINOv2-S trained with gold58 at elevated weight), (2) `antoinegg1` E9 + `sofiaanjenje` E13 RadImageNet heads (fit on gold58 at weight 7.0), (3) `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` (epochs 4/6/8 selected on gold58), (4) `mattiaangeli/knee-mri-fold-weights` (5 DINOv3-S folds), and (5) `dreaddevelopment` Raptor v5/v8/v10. Zero OAI models inside, so it is **rules-compliant (no external data)**, but heavily gold-fitted. |
| `romantamrazov/rsna-knee-dinosaur-v5-train` | **TAINTED (gold-trained & gold-tuned, NO OAI)** | Pulled notebook (`artifacts/_nextbets/romantamrazov_rsna-knee-dinosaur-v5/rsna-knee-dinosaur-v5.ipynb`) is a fast-decode / dual-T4 wrapper around the exact same BTKD + D4 + resgated + E11/E13 community stack plus `goodpjw2008` reader. Contains zero OAI checkpoints (rules-compliant), but inherits the gold-trained DINOv2/RadImageNet heads and gold-selected CoAtNet checkpoints. |
| `metaresearch/dinov2/PyTorch/small/1` (DINOv2-S) | **CLEAN** | Apache-2.0. Official Meta pretrained ViT-S/14 initialization weights; no competition or OAI data. |

## 3. Summary & action

1. **External-data / OAI compliance:** Both `55779066` (mill V10, 0.936) and the 0.944 stack are **100% OAI-free**. Every checkpoint in both traces to RSNA competition DICOMs + LLM report labels + `gold58` (plus public ImageNet / RadImageNet / Meta DINOv2 backbones). Thus `55779066` IS rules-eligible as a no-OAI hedge.
2. **Gold-58 contamination:** Both stacks train or select on `gold58`, so under our strict no-gold-tuned rule they are tainted for validation, and we do NOT build `kaggle_kernel_clean_stack/`.
3. **Strictly clean (no-OAI AND no-gold) blend pool:** `goodpjw2008/rsna-knee-2-5d-convnext-reader` + `dreaddevelopment/raptor-knee-widedense` (+ `maxspan`) + our own 5-fold no-OAI CoAtNet.
