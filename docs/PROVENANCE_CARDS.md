# Public-model cards read 2026-10-12

Metadata from `kaggle datasets metadata`. Quotes are from those JSON descriptions.

- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache-2.0). "trained on report-derived soft labels; the 58 official gold-labelled studies were not used for training." Its `make_labels.py` builds v0 as the mean of four public LLM tables (steven v4, pilkwang v2, lixin sol56, laymond qwen); no OAI or external-image string in `train.py`/`make_labels.py`/`oof.py`/`preprocess.py` (`artifacts/provenance/reader_code/`). Public claim on the card: 0.929 alone. Clean for a no-OAI blend.
- `dreaddevelopment/raptor-knee-widedense` (CC0). "No external data was used." Gold 58 held out. Card states 0.9167 macro-AUC on those 58 and 0.924 public as one model.
- `dreaddevelopment/raptor-knee-maxspan` (CC0). Describes a 64-slice competition-study corpus. The card does not say "no external data" and does not mention OAI. Card states 0.928 public as one model.
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` (license name `other`). "full independent retraining of every checkpoint is not claimed." Upstream training sources "remain with their authors." Not a clean card.
- `romantamrazov/rsna-knee-dinosaur-v5-train`: CLI returned `403 Client Error: Forbidden`. Card not read.

Mill **55779066** still mounts the tonylica bundle and dinosaur-v5. Those two are not cleared. The ConvNeXt reader card is the one that can go into a no-OAI blend. Widedense is the public CoAtNet card that states no external data.

## Strict-clean pool check 2026-10-13

- `goodpjw2008/rsna-knee-2-5d-convnext-reader` — training labels: report-derived soft labels (v0 = mean of steven v4, pilkwang v2, lixin sol56, laymond qwen); gold58 not used for training. OAI strings: no. **CLEAN.**
- `dreaddevelopment/raptor-knee-widedense` — training labels: LLM report soft labels on 4,349 competition studies; gold58 held out. OAI strings: no (`No external data was used.`). **CLEAN.** Local holdout870 predictions: none.
- `dreaddevelopment/raptor-knee-maxspan` — its own card does not state the training labels or "no external data." The linked companion notebook's no-OAI sentence describes the 0.924 / 0.9167 widedense run, not this 0.928 checkpoint. Card OAI strings: no. **EXCLUDED** until the v5 training text is on the card.
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` — repackages gold58-trained DINOv2/RadImageNet heads and gold-selected CoAtNets. OAI strings: no. **EXCLUDED.**
- `romantamrazov/rsna-knee-dinosaur-v5-train` — card not readable (403). Notebook on disk wraps the gold-trained BTKD/E11/E13 stack. **EXCLUDED.**
- JEV HYBRID labels and any weights fit on gold58: **EXCLUDED.**

## Raptor cards read 2026-10-16

- `dreaddevelopment/raptor-knee-native384dense` (CC0-1.0). Card: CoAtNet 384 weights, “combining the geometry of the 0.928 model with a natively 384 pixel corpus.” Geometry stated on the card: 64 slices per study, span 2–98% of each series, 140 mm crop, stored at 384 pixels, scored on 62 windows. “On the 58 held-out labelled studies it scores 0.9174.” Training-label source is not stated. The words OAI and “external data” are not on the card. Gold58 is described as held out for that score, not as a training target. **Not dropped as OAI-trained or gold58-trained.** Not scored here: the local cache is the reader’s per-series volumes (≤64 slices, 384 px, 153.6 mm field, canonical orientation), not this 140 mm native-384 study stack.
- `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` (CC0-1.0). This is the residual-gated CoAtNet mounted by `TranBaDat2607/RSNA-Knee-Abnormality-Detection` (`experiments/phase2/kernels/clean-submit`). The dataset description is empty. The packaged trainer says: “all 4,349 report-labelled studies are used for optimization; the 58 fully labelled studies are used only for epoch selection.” The code keeps those 58 out of the train set and checks the split is 4349/58. Checkpoints `coat_resgated_gold_e04/e06/e08.pt` are the top three epochs by `official_gold58_macro_auc`. No OAI string in that trainer. **EXCLUDED from the holdout run: gold58 was the selection metric.**
