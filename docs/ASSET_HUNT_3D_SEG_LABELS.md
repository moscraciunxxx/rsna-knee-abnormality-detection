# Asset hunt — 3D / seg→ROI / better labels

Hunt date: 2026-08-28. Competition: `rsna-knee-abnormality-detection` (kernels-only, Tesla T4, internet **OFF** at test, **no Report** at test).

**Method.** Public Kaggle dataset pages + pulled `refs/` notebooks + this repo’s `kernel-metadata.json`. `kaggle.exe datasets list/files` was **not** executed in this session (no shell). File trees are therefore **samples from descriptions and kernel sources**, not a CLI listing. Where a page had no file list, that is marked **unlisted**.

**Do not recommend (already scored / ruled out):** mill CoAtNet TTA, native384, jointline 90 mm mill-RadImageNet crop, DINOv3-S+ column replace.

**Local unused refs.** `umud` has **zero** hits in this repo. `pilkwang/rsna-knee-llm-labels` and `stevenleehans/rsna-knee-llm-report-labels` are named in `docs/TEN_POINT_PLAN.md` and a comment in `rsna_knee/encoder.py` as **train-time only**; they are **not** attached to any of our submit kernels. Mill CoAtNet already implements **per-finding `SlotHead`** over 2.5D 3-slice windows (not 3D conv).

---

## Avenue recap

| ID | Claim | What would count as TEST NOW |
| --- | --- | --- |
| **A** | 3D / true slice-attention inside a **new** backbone (not mill CoAtNet SlotHead) | Loadable 12-head 3D or MIL `.pt/.pth/.ckpt/.safetensors` + infer code, T4, internet off |
| **B** | Segmentation → ROI classify. Hinted corpus `dreaddevelopment/umud-seg-corpus-v1` (~2.7 GB). Meniscus 1–3 mm is why jointline 0.935 failed | Frozen **knee** seg model + tiny 12-head (or 2–4 finding) classifier, not a 40 h train |
| **C** | Better **labels**, not better crops. Mill already uses LLM-soft labels. Top 0.950–0.952 is almost certainly a private label/train stack | A **new 12-head image checkpoint** trained on a better public key, **or** a documented unused key that is worth a retrain — CSV alone cannot run at test |

---

## Inventory table

Legend: **T4-off** = loadable on scored kernel with internet off. **12-head** = competition 12 findings. **Theft-risk** = bare HF `config.json` / tokenizer / timm-hub download needed at test.

| asset | avenue | size | file-tree sample | weights? | masks / JSON / NIfTI? | CSV soft labels @ StudyInstanceUID? | 12-head vs encoder vs seg | T4-off? | theft-risk | GPU hours (est.) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dreaddevelopment/umud-seg-corpus-v1` | B (hinted) | 2 728 845 934 B (~2.54 GiB) | Ultrasound **TIFF** frames of skeletal muscle (UMUD aponeurosis challenge). **Images only.** | **No** | **No** (page: “masks and the trained segmenter are **not** included”) | No | **Wrong domain** (muscle US, not knee MRI) | Images would mount; **useless at test** | Low | 0 overlay / ∞ as knee ROI | **skip** — not knee, no masks, no weights |
| `pilkwang/rsna-knee-llm-labels` | C | 145 577 B | Unlisted; description: `StudyInstanceUID` + per-finding **YES/NO/UNK + severity + scores** (rank-usable, not calibrated) | No | No | **Yes** (12 findings) | labels only | CSV yes; **train-only** | Low | 0 overlay / 10–40 h retrain | **train-only** — unused here; mill already LLM-trained |
| `stevenleehans/rsna-knee-llm-report-labels` | C | 426 025 B | `llm_labels_v4_blend.csv` (recommended), `llm_labels_v2.csv`, `llm_labels_full.csv` | No | No | **Yes**, 4407 rows, 12 probs in `[0,1]`; `0.5` = not addressed | labels only | CSV yes; **train-only** | Low | 0 overlay / 10–40 h retrain | **train-only** — **best public unused key** (v4 **0.8927** vs 58 gold; regex 0.8136; Pilkwang 0.8700) |
| `lixin73/rsna-knee-llm-report-labels-sol56` | C | 266 784 B | Unlisted small CSV (GPT-5.6-Sol). Blended into stevenleehans v4 | No | No | **Yes** (claimed) | labels only | CSV yes; **train-only** | Low | 0 / included in v4 blend | **train-only** — weak alone (0.8347) but **low corr** (0.718) so it **helps the blend** |
| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | C | 3 026 684 B | `train_folds.csv`; `train_folds_with_pseudo.csv` (`fold`, `is_manual`, 12 GT + `pseudo_*`) | No | No | **Yes** | labels + 5-fold meta | CSV yes; **train-only** | Low | 0 / used by 2.5D baseline train | **train-only** — unused in our kernels |
| `barun2104/rsna-knee-mri-processed-3d-volumes` | A (false 3D) | 17 378 118 832 B (~16.2 GiB) | 24 371 series as compressed `.npz`, shape **`(24, 224, 224)` uint8** | **No** | npz volumes, not NIfTI masks | No (pairs with folds CSV) | **2.5D train tensors**, not a 3D CNN | Train mount only; **no hidden-test npz** | Low | Train 2.5D ResNet-34: ~4–12 h/fold | **skip overlay** — not weights; not true 3D; test DICOMs still required |
| `barun2104/baseline-rsna-knee-mri-2-5d-resnet34` (code) | A | n/a (notebook) | Trains ResNet-34 `conv1` 3→24, 12 sigmoid heads, AMP fold-0; writes `best_resnet34_fold0.pth` to **working**, not a dataset | Train-run artifact **not published** as a dataset | No | Uses barun pseudo CSV | 12-head **2.5D**, not 3D/MIL | No public ckpt to attach | Low | 4–12 h/fold T4 | **skip overlay** / **defer** only as a weak 2.5D train (CNN-tri already 0.917) |
| `pilkwang/rsna-knee-weights` | C / A | 1 657 083 403 B (~1.54 GiB) | Unlisted; tagged **pre-trained model**. Attached by `pilkwang/rsna-knee-baseline-v1` + 0.89× public clones | **Yes** (presumed `.pt`/DINO) | No | Used **with** llm-labels in those kernels | 12-head **or** encoder for public **~0.89** DINO baseline | **Likely yes** if no hub fetch | **Med** if HF `config.json` missing | 0 overlay | **skip overlay** — already a **weaker public 12-head family**; not unused 3D |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | — (used) | 2 624 081 118 B | DINO + RadImageNet mill assets (`config.json` gated in mill code, `ResNet50.pt`, `*_f*.pt`) | Yes | No | No | encoder + 12-head DINO members | Yes (mill V10) | **High** if `config.json` stripped | 0 | **already used** (0.936 mill) |
| `tonylica/rsna2026-models` | — | 1 329 626 434 B | Unlisted pre-trained model pack | Yes | No | No | encoder/12-head (public DINO family) | Likely | Med–high HF | 0 | **skip** — public DINO/RadImageNet lane, not 3D |
| `dreaddevelopment/raptor-knee-widedense` | — (used) | unlisted (CoAtNet `.pt`) | `raptor_ft_coatnet_v4_full.pt` + arch `coatnet_rmlp_2_rw_384` | Yes | No | Mill **trained on LLM labels** (not this CSV) | **12-head 2.5D CoAtNet + SlotHead** | Yes | Med (timm arch baked in `.pt`; no hub if weights complete) | 0 infer / 20–40 h retrain | **already used** — this **is** mill slice-attention, **not** avenue A |
| `dreaddevelopment/raptor-knee-maxspan` | — (used) | unlisted | `raptor_ft_coatnet_v5_full_swa.pt` | Yes | No | No | 12-head CoAtNet | Yes | Med | 0 | **already used** |
| `dreaddevelopment/raptor-knee-native384` / `native384dense` | — (used / ruled out) | unlisted | `raptor_ft_coatnet_v8_full_swa.pt`, `v10_full.pt` | Yes | No | No | 12-head CoAtNet | Yes | Med | 0 | **ruled out** (native384 tie 0.936) |
| `dreaddevelopment/raptor-knee-arms` | — (used) | unlisted | extra mill arms (cnn/swin/eff) | Yes | No | No | 12-head | Yes | Med | 0 | **already used** (CNN-tri 0.917) |
| mill `SlotHead` in our kernels | A (already have) | n/a | Per-finding attention over 42 three-slice windows | mill `.pt` | No | mill LLM train | 12-head **2.5D attention**, **not 3D conv** | Yes | — | 0 | **not avenue A** — do not TTA |
| `kgaooo/oai-tissue-segmentations` | B (domain transfer) | unlisted (OAI-scale) | `XXXXXXX_YYYY_ZZ.npz` masks; femur/tibia/patella, cartilage, **meniscus**; 46 824 OAI MRIs | **No** (masks only) | **Yes** npz masks; **not NIfTI**; **OAI IDs ≠ StudyInstanceUID** | No | **seg-only**, OAI DESS | Masks yes; **no RSNA test mapping** | Low | 20–80 h: domain-adapt seg + ROI cls | **train-only / skip overlay** — closest public meniscus masks, **wrong study IDs**, no frozen RSNA classifier |
| OAI / IWOAI / SKM-TEA / 3DReasonKnee (HF `rajpurkarlab/3DReasonKnee`) | B / A papers | large | External 3D boxes + nnU-Net stories; **not** this competition’s Kaggle mount | some off-Kaggle | yes (other corpora) | No | seg / VLM, not 12-head RSNA | **No** (internet off; not attached) | High | 40–100 h | **skip** for this code-comp unless re-hosted as a Kaggle dataset **with** weights |
| `tennogh/my-knee-mri` | — | 73 844 732 B | One personal ACL-rupture study in competition layout | No | DICOMs | No | n/a | n/a | Low | 0 | **skip** |
| MedicalNet / MONAI 3D / TotalSegmentator (generic) | A/B init | varies | 3D ImageNet-like / CT-heavy | encoder only | no RSNA masks | No | **encoder**, not 12-head | Only if **fully vendored** (no torch.hub) | **High** (`config.json`, hub) | 20–80 h train | **skip overlay** — no published RSNA-knee 12-head 3D finetune |
| `sofiaanjenje/rsna-knee-e11-train`, `e13-train`; `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`; `mattiaangeli/*`; `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`; `romantamrazov/rsna-knee-dinosaur-v3-train` | — (mill family) | unlisted | RadImageNet / DINO **heads** `.pt` | Yes | No | No | 12-head / heads-on-frozen encoder | Yes in mill | Med | 0 | **already used** (V10 0.936) |
| `gourabroy553/rsna-knee-hybrid-dino-radiomics` | A/B? | unlisted (page JS-empty) | Name says DINO + **radiomics** | unknown | unknown | unknown | unknown | unverified | unknown | unknown | **unverified** — not 3D MIL; treat as mill-adjacent until files listed |
| Public 3D/MIL **12-head RSNA-knee weights** | A | — | **Not found** on public dataset pages / attached kernels | **None** | — | — | — | **No** | — | — | **does not exist (public)** |

---

## Notebooks and discussions (hunt)

| item | what it is | 3D / MIL / seg / labels | use |
| --- | --- | --- | --- |
| `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` | 0.924 CoAtNet infer | **2.5D** 3-slice windows + **per-finding SlotHead** (explicitly “most of the benefit of a 3D model at the cost of a 2D one”). LLM-trained labels, 58 gold **held out** (0.9167). | **Used.** Not a 3D backbone. |
| `prvsiyan/head-and-shoulders-knees-and-toes` | 0.936 mill V10 | DINO + RadImageNet + dual CoAtNet | **Used.** |
| `barun2104/baseline-rsna-knee-mri-2-5d-resnet34` | train notebook | **2.5D ResNet-34**, 24 channels; **not** Conv3d / ABMIL | Train recipe only; no public ckpt dataset. |
| `pilkwang/rsna-knee-baseline-v1` | ~0.89 | DINOv2 + `rsna-knee-llm-labels` + `rsna-knee-weights` | Weaker public 12-head; labels unused **by us**. |
| `prvsiyan/rsna-knee-read-the-report-then-the-knee` | report→12 | Attaches **all three** public LLM keys | **Confirms reports are train-only.** |
| `karnakbaevarthur/rsna-knee-eda-to-2-5d` | EDA | 2.5D, not 3D | skip |
| `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` | DINO at meniscus px | Resolution story, **not** a frozen meniscus **seg** model | skip overlay (jointline already proved mill FOV ≠ 1–3 mm meniscus) |
| `mattiaangeli/bend-the-knee-to-dinov3-ensembled` | 0.917 | DINO ensemble | Used / dominated by mill |
| Discussion `733497` (@harsh2035) | fun extra MRI | spawned `tennogh/my-knee-mri` | skip |
| **No public discussion found** that ships 3D Conv / ABMIL 12-head weights for **this** contest | — | — | — |
| Literature (Astuto 3D CNN; CoPAS; Res3DKnee / TransMed; ABMIL on OAI **X-ray** patches) | papers | Real 3D/MIL **elsewhere** | **No Kaggle `.pt` for these 12 findings** |

---

## Local repo (unused vs used)

**Never used here**

- `umud` (no string matches)
- `pilkwang/rsna-knee-llm-labels`
- `stevenleehans/rsna-knee-llm-report-labels`
- `lixin73/rsna-knee-llm-report-labels-sol56`
- `barun2104/*` 3D-volume + folds
- `pilkwang/rsna-knee-weights` (referenced only inside **pulled** `refs/`, not our submit kernels)

**Used / ruled out**

- mill CoAtNet SlotHead + raptor `*.pt`
- DINO 20-member + RadImageNet e9/e11/e13 heads
- native384, jointline crop, dino3s replace, tta-max

---

## Explicit verdicts

### A — 3D / true slice-attention, new backbone

**DEAD** as a public overlay.

- There is **no** published 3D-conv or ABMIL **12-head** checkpoint for this competition.
- Mill’s “3D” is **2.5D CoAtNet + SlotHead** (already in V10). That is **not** a new backbone.
- The only public “3D” dataset (`barun2104/rsna-knee-mri-processed-3d-volumes`) is **24×224×224 npz for 2.5D CNN train**, CC-BY-NC-SA, **train series only**. The matching notebook **does not publish** `best_resnet34_fold0.pth`.
- MedicalNet / 3D U-Net / 3DReasonKnee are **not** T4-off 12-head drops.

A later **DEFER** (own train, not this hunt) would be: vendor a 3D encoder **without** hub, train 12-head on a **better** label key (avenue C), group-CV vs mill on the 58 gold **and** a held-out report-soft fold. Budget **≳20–80 T4-h**, not a 4 h blend. Do **not** start that by cloning mill CoAtNet.

### B — Segmentation → ROI classify (`umud-seg-corpus-v1`)

**DEAD**.

- `dreaddevelopment/umud-seg-corpus-v1` is **UMUD skeletal-muscle ultrasound TIFFs** (~2.54 GiB). Page states **masks and segmenter are omitted**. Not knee MRI, not meniscus 1–3 mm, not StudyInstanceUID.
- **No** frozen knee-MRI seg + tiny 12-head classifier exists on public Kaggle for this contest.
- Closest masks: `kgaooo/oai-tissue-segmentations` (OAI DESS meniscus/cartilage/bone npz). **Wrong IDs, wrong protocol, no classifier.** Domain-adapt + ROI train is a **multi-day** campaign; jointline already showed mill-FOV crops do not buy meniscus millimetres.

Do not attach umud to a scored kernel.

### C — Better labels, not better crops

**DEFER** (train-time only; **not** TEST NOW).

- Public unused keys **do exist** and **are** aligned to `StudyInstanceUID` + 12 findings.
- Best unused stack: **`stevenleehans/rsna-knee-llm-report-labels` v4 blend** (`llm_labels_v4_blend.csv`) = stevenleehans v2 ⊕ `lixin73/...-sol56`, **0.8927** vs 58 gold (regex 0.8136, Pilkwang 0.8700, steven v2 0.8873). `0.5` = unmentioned; loss weight `2·|p−0.5|`.
- Reports are **100% train / 0% test**. CSV overlay at inference is **illegal / 0.5**.
- Mill **already** trained CoAtNet on **its own** LLM labels (0.9167 on held-out 58). Public keys are **not** proven better than mill’s private labeller; they **are** unused **by us**.
- **No** public 12-head image model advertised as trained on the v4 key. `pilkwang/rsna-knee-weights` is the old ~0.89 DINO lane.

**How to DEFER (when we train, not this week):** train a **non-mill** encoder on v4 (or v4 ⊕ barun folds), freeze mill at 0.936, overlay **≪0.50** only on findings where private fold beats V10. Never put Report in the forward pass.

---

## TEST NOW / DEFER / DEAD (one line)

| avenue | verdict | why |
| --- | --- | --- |
| **A 3D/MIL new backbone** | **DEAD** | No public 3D/MIL 12-head weights; mill SlotHead is 2.5D; barun npz is not a ckpt |
| **B seg→ROI** | **DEAD** | umud v1 is muscle US without masks/weights; no frozen knee seg+cls |
| **C better labels** | **DEFER** | Unused CSVs are real (stevenleehans v4 0.8927) but **train-only**; need a **new** image train, not a 4 h overlay |

**Nothing in this hunt is a T4-off 4 h overlay that is uncorrelated with mill.**
)
