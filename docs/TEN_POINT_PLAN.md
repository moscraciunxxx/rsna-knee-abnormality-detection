# 10-point plan: RSNA Knee Abnormality Detection

Competition URL: `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection`  
Extracted 2026-08-25 from the logged-in Kaggle CLI (`userHasEntered=True`) plus official RSNA pages. Full DICOM archive stays on Kaggle; this repo ships schema, metric, a 2.5D inference path, and smoke tests.

## Official facts (extracted)

| Field | Value | Source |
| --- | --- | --- |
| URL | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection | Kaggle / RSNA |
| Category | Research; prize **$77,000** (accuracy + first-time efficiency awards) | `kaggle competitions list`; RSNA 2026-08-05 news |
| Entry window | **2026-10-15** | Official Kaggle @kaggle post 2026-08-05 |
| Final window | **2026-10-22 23:59:00** | `kaggle competitions list` deadline |
| Teams (this extract) | 2387 entered | CLI |
| Submit mode | **Notebook-only** (`isKernelsSubmissionsOnly`). CSV API upload is rejected. | Public kernel metadata + Kaggle code-comp rules |
| Metric | **macro-averaged ROC AUC** over the 12 findings: \(\frac{1}{12}\sum_{i=0}^{11}\mathrm{AUC}_i\) | prvsiyan notebook quoting Evaluation; matches host scoring |
| Id column | `StudyInstanceUID` | `sample_submission.csv`, `test.csv` |
| Header (must match, including apostrophe) | `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture` | downloaded `data/sample_submission.csv` |
| Public test | 3 studies in `test.csv` / `sample_submission.csv` (placeholder 0.5) | downloaded CSVs |
| Train | 4407 studies, 24371 series; **every** study has a `Report`; **only 58** studies have expert 12-finding labels (rest NaN) | `data/train.csv`, `data/train_series.csv` |
| Series metadata | `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` ∈ {Sagittal, Coronal, Axial} | `train_series.csv` / `test_series.csv` |
| Images | DICOM under `{StudyInstanceUID}/{SeriesInstanceUID}/{SOPInstanceUID}.dcm` | competition file listing |
| Multimodal train | knee MRI **+ multilingual radiology reports** (dozen languages, 16–19 sites, 5 continents; >5000 exams advertised, 4407 in the current train table) | RSNA challenge page + train.csv |
| Test text | `test.csv` has **only** `StudyInstanceUID` — **no Report**. Reports at test time would be leakage / unavailable. | downloaded `test.csv` |
| Hidden test | larger; kernels must **discover study folders**, not hard-code the 3 public IDs or trust `test.csv` alone | code-competition convention + public kernels |

### The 12 findings (verbatim)

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

### Public sources reused in this repo

| Kind | Name / slug | URL | How we reuse it |
| --- | --- | --- | --- |
| Kaggle notebook | `[Baseline] RSNA Knee MRI — 2.5D ResNet34` (`barun2104/baseline-rsna-knee-mri-2-5d-resnet34`) | https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34 | **Adapted method (shipped):** resample each study to **24 depth slices as 2D-CNN channels**, 12 sigmoid heads, **macro ROC AUC**. Local smoke uses a numpy 2.5D encoder with the same slice-as-channel contract so it runs without GPU; a ResNet-34 `conv1` expansion is the Kaggle-GPU fill-in. |
| Kaggle dataset | `pilkwang/rsna-knee-llm-labels` | https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels | Train-time **LLM report-label / soft-label weak supervision** for the 4349 unlabeled reports (YES/NO/UNK + severity; rank-usable, not calibrated). Never used at test. |
| Kaggle dataset | `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets | Optional foundation-model slice embeddings (DINOv3) for a later rank-average ensemble with the 2.5D CNN. |
| Paper / prior method | Bien et al., **MRNet**, PLOS Medicine 2018 | https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699 | Slice-wise CNN + pooling across a series; logistic fusion of planes. Informs study-level pooling and plane-aware series grouping. |
| Paper / prior method | Astuto et al., 3D CNN knee MRI, *Radiology: AI* 2021 | https://pubs.rsna.org/doi/10.1148/ryai.2021200275 | Multi-tissue knee MRI grading (cartilage, BME, meniscus, ACL). Informs per-finding heads and why OA compartments must stay separate. |
| Extra public kernel (labels/eval quote) | `prvsiyan/rsna-knee-read-the-report-then-the-knee` | https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee | Confirms host score \(\frac{1}{12}\sum AUC_i\); lexicon/LLM report → 12 targets; **do not** send report text into the scored kernel. |

Pulled copies of the 2.5D and report notebooks live under `refs/`.

---

## The 10 points

### 1. Data contract and protocol-aware MRI ingest

Treat a **study** as the prediction unit (`StudyInstanceUID`). Load every series under that study; keep `Anatomical_Plane`, `Fluid_Sensitive`, and `Fat_Suppression`. Sagittal/coronal/axial and PD/T2-FS vs T1 are not interchangeable: ACL and menisci light up on fluid-sensitive sagittal; MCL and medial OA on coronal; PF OA and effusion on axial/FS. Window/rescale DICOM to a fixed 2D size (public baseline uses 224×224). Do not assume `test.csv` lists every hidden study.

### 2. Weak labels from multilingual reports (train only)

4407/4407 train studies have `Report`; 58/4407 have expert 12-way labels. Fit on **LLM-read report labels** (`pilkwang/rsna-knee-llm-labels` and/or `stevenleehans/rsna-knee-llm-report-labels`) as soft targets: YES/NO/UNK + severity → rank-usable scores. Fine-tune the 58 expert rows. Reports are Spanish, Dutch, and many others; a morphology lexicon fails silent in some languages — that is why the public LLM dataset exists. **Never** read reports at inference.

### 3. Cross-validation by site and language, not random rows

MRI vendors, languages, and sites (16–19 worldwide) dominate covariate shift. Split on study ID, grouped by inferred language / site proxy (report language, series metadata), so a language never leaks into its own validation. Macro ROC AUC on a 58-row expert fold is high-variance; report-soft-label CV is the workhorse, expert fold is the sanity check.

### 4. 2.5D encoder (the method we implement)

Adapt **`[Baseline] RSNA Knee MRI — 2.5D ResNet34`**: concatenate a study’s series along depth, **resample to 24 slices**, treat those slices as **input channels** of a 2D ResNet-34 (`conv1` 3→24 with ImageNet weight repeat × √(3/24)), 12-logit head, BCE-with-logits, sigmoid at export. This is the MRNet idea (slice CNN + pool) with ResNet-34 instead of AlexNet and with all 12 findings at once. Local smoke uses the same 24-channel contract with a numpy encoder (no GPU, no full DICOM dump). Optional upgrade path: DINOv3 slice embeddings from `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`.

### 5. Per-finding heads, not one “abnormal” bit

The 12 findings are clinically distinct (ACL tear ≠ medial OA ≠ Baker's cyst ≠ fracture). Share the 2.5D trunk; give each finding its own logit. Loss: `BCEWithLogitsLoss` (pos-weight from train prevalence). Do not collapse medial/lateral meniscus or the three OA compartments. `Baker's` keeps the apostrophe in every tensor name that touches CSV I/O.

### 6. Calibration is secondary; ranking is the score

Macro ROC AUC is invariant to strictly increasing maps per finding. Temperature scaling / isotonic still helps an efficiency-track blend and human-readable probabilities, but **do not** optimize Brier at the expense of order. Clip export to `[0, 1]`. Rank-average ensembles beat probability-average when models are miscalibrated (public DINO + RadImageNet blends).

### 7. Notebook-only inference: discover studies from folders

Kernels-only, internet off, ≤ competition runtime. Walk `test_series/` (or the hidden test image root), collect parent `StudyInstanceUID` folders, union with `test.csv` if present, **never** hard-code the three public IDs. Write `submission.csv` with the exact header including `Baker's`. A local CSV is the checkable artifact; a leaderboard click is not required in this repo.

### 8. Ensemble: 2.5D CNN ⊕ report-trained soft-label model ⊕ optional DINOv3

Rank-average: (a) 2.5D ResNet-34 on images, (b) a model trained on LLM soft labels then distilled to images, (c) optional DINOv3 / RadImageNet heads from public datasets. Plane-specific models (MRNet-style axial/coronal/sagittal) fused by logistic or rank mean. Efficiency track: one 2.5D ResNet-34, 24×224×224, no test-time report.

### 9. Leakage controls

- Reports exist in train and **not** in test — any report-in-the-forward-pass model scores 0.5 on hidden data.
- Do not use `StudyInstanceUID` as a feature.
- Group CV by study (multiple series ≠ multiple labels).
- Do not train on `sample_submission.csv` placeholders.
- Public LB has 3 studies; do not chase it. Trust grouped CV + the 58 expert labels.
- Social posts claiming a finished “2nd place” are not official; the contest is open through **2026-10-22**.

### 10. Metric, schema freeze, and shipped smoke path

Freeze `rsna_knee.schema.SUBMISSION_COLUMNS` to the sample header. Score with `macro_roc_auc` = mean of per-finding ROC AUC (skip a finding only if it is single-class in a split; the host averages all 12). The shipped inference entry encodes each discovered study with the **2.5D 24-channel** path and writes probabilities for all 12 findings. Two identical smoke launches on a tiny synthetic fixture must reproduce the schema (≥3 rows, values in `[0, 1]`).

---

## What this repo ships vs what stays on Kaggle

| Here | On Kaggle only |
| --- | --- |
| 10-point plan, schema, submission builder, macro ROC AUC, 2.5D encoder stub, smoke CLI, tests | Full train DICOMs, GPU ResNet-34 training, notebook submit |
| `data/sample_submission.csv`, `test.csv`, `test_series.csv`, `train.csv`, `train_series.csv` | `test_series/**.dcm`, train image tree |

## Medical reading (for model design, not clinical advice)

Internal derangement on MRI: ACL (often PD/T2 sagittal), MCL (coronal), menisci (sagittal + coronal), compartmental OA (medial/lateral tibiofemoral, patellofemoral), effusion and synovitis (FS fluid), Baker's cyst (popliteal axial), bone contusion, fracture. Multi-label overlap is expected (unhappy triad, OA + Baker's). That is why the host scores **macro-averaged ROC AUC**, not a single abnormal flag.
