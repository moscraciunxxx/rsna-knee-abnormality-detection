# RSNA Knee Abnormality Detection (Kaggle, 2026)

This repo holds Vitalie Cervinschi's code, Kaggle kernels, validation tooling and experiment notes for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition.

## The competition

- Knee MRI studies, each with several DICOM series. Every study is scored on **12 findings**: ACL, MCL,
  Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's cyst,
  Contusion and Fracture.
- The metric is **macro ROC AUC** over the 12 findings.
- It was a **code competition**. Inference ran in an offline Kaggle notebook (T4 GPUs, internet off) on a hidden
  test set. The public leaderboard used about 30% of the test set and the private leaderboard used the other 70%.

## Final results

| Submission (Kaggle kernel) | Public LB | Private LB | Notes |
|---|---|---|---|
| `rsna-knee-hedge-958` v1 | 0.958 | **0.942** | Final pick 1. Raptor + CoatNet/R6 2.5D MIL rank blend (public-stack hedge) |
| `rsna-knee-clean-b03` v2 | 0.944 | 0.929 | Final pick 2. "Clean" ConvNeXt-reader blend with documented provenance |
| `rsna-knee-apex-copy` v1 | 0.950 | 0.932 | Exact copy of the public Apex stack, used as a reference (not selected) |
| `rsna-knee-hedge-958-h2` v1 | 0.958 | n/a | Hedge plus the nartaa s6fall leg at 1/3 weight. It tied the hedge, so it was not selected |

## Repo layout

```
rsna_knee/       Python package: DICOM/series loading, 2.5D encoders, metrics, soft labels, submission helpers
scripts/         Training, caching, holdout scoring, blending and staging scripts
tests/           pytest suite for the package and the blend/validation tooling
kernels/         Final Kaggle kernels (notebook + kernel-metadata.json)
  rsna-knee-hedge-958/   final pick 1 (0.958 / 0.942)
  hedge-958-h2/          hedge + s6fall third leg
  hedge-958-local/       local dry-run copy of the hedge (paths pointed at a local mount)
  hedge-958-out/         hedge run outputs: receipts, logs, per-leg CSVs on the 3 public sample studies
  clean-b03/             final pick 2 (0.944 / 0.929)
  train-headers/         kernel that extracts DICOM header fingerprints (output CSV not included)
kaggle_kernels/  Every other experiment kernel we pushed (gates, holdout scorers, overlays, cache builders, ...)
results/         Our own holdout / OOF prediction CSVs and score summaries (870-study holdout, gold58 subset)
tools/           Small Kaggle CLI helpers (status, pulls, watchers)
docs/            Plans, provenance cards and the running experiment log
```

Validation was done on a fixed 870-study holdout split from the training set (seed 2026), with macro AUC and
paired-bootstrap SDs (see `results/` and `docs/TRAIN_HEAD_PROTOCOL.md`).

## What is NOT included

To follow the competition rules and the licenses of the datasets involved, this repo does **not** include:

- competition data (DICOMs, train/test CSVs, labels, or anything derived from them such as header dumps),
- model weights or checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, ...),
- third-party Kaggle datasets, or local copies of other authors' public notebooks that we only pulled for study.

To run the kernels you need to attach the competition and the public datasets listed in each
`kernel-metadata.json` on Kaggle.

## Credits

The final submissions build on public work from the Kaggle community. Many thanks to:

- **goodpjw2008**: *RSNA Knee CoatNet R6 2.5D MIL (LB 0.958)* notebook (the base of the hedge kernel) and the
  `rsna-knee-2-5d-convnext-reader` dataset
- **nartaa**: Raptor SWA publication weights (`rsna-knee-publication-swa-weights-20261007`), the s6fall weights and
  solo script (`rsna-knee-s6fall-weights`), and the anatomical-mirror notebook
- **heliosli**: `rsna-orthofoundation-fold0-epoch7-20261004` and `rsna-knee-compact-r7cnxfsss`
- **rostigor**: `rsna-knee-r6-fe055406`
- **sujanmajhisuzan**: *RSNA Knee Apex Grandmaster Stack* (the Apex reference copy)
- **dreaddevelopment** and other authors credited in `docs/PROVENANCE_CARDS.md` and `docs/CLEAN_STACK_PROVENANCE.md`

Thanks also to RSNA and Kaggle for hosting the competition.
