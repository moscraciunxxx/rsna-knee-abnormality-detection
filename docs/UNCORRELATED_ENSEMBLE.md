# Live correction 2026-10-07

CLI this session: best public still **0.936** (55779066). Page-1 #1 **Ilya Kolb 0.964** (gap **0.028**, not 0.016). Page-1 10th row **0.960**. GPU **30.00 h remaining**, refresh **2026-10-10T00:00:00**. userRank **2412** / 5366. The 0.952 and ~6.94 h lines below are the 2026-08-28 capture, not this week.

# Uncorrelated ensemble hunt — mill V10 55779066 / 0.936

Captured 2026-08-28 from this repo’s `refs/`, `kaggle_kernel_*`, `docs/TEACHER_NOTE.md`, and public Kaggle dataset/model pages. **No kernel was pushed.**

Anchor stays **`55779066` DINOsaur V10 = 0.936**. Overlay default is **0.0 (pure mill)** unless a private fold Δ on that finding is **> 0**. Each overlay arm **fail-closes to mill** if missing, NaN, dummy ranks `{0, 0.5, 1}`, or degenerate `std ≈ 0`. Overlay weight **≪ 0.50** on a **subset of findings**. Never resubmit:

| Forbidden | Evidence |
|-----------|----------|
| Uniform **0.50-all-12** weaker CNN | CNN-tri **55792478 / 0.917** |
| Mill DINO TTA / CoAtNet hflip | tta-max **55848021 / 0.936** (tie) |
| Mill RadImageNet crop millimetres | jointline **55822934 / 0.935** |
| DINOv3-S+ **1.0 replace** MCL + Baker's | dino3s **55832530 / 0.908** |
| Public four-view **native384** subsets | V41 **55782774 / 0.936**, logfix **55807100 / 0.936** |
| More mill CoAtNet | raptor v4 **55778197 / 0.924**; mill already dual CoAtNet |

Live gap at capture: **#1 0.952 − mill 0.936 = 0.016**. GPU quota at teacher note: **~6.94 h remaining**.

---

## 1. Local inventory (non-mill encoders)

Mill V10 itself is **DINOv2 ViT-S/14 @ 336** (20-member package in `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`) **+ RadImageNet ResNet-50** (e9 + e13, `marwanmath/resnet-50-radimagenet-marwan`) **+ dual CoAtNet** (`dreaddevelopment/raptor-knee-maxspan` v5 SWA + `raptor-knee-widedense` v4). Hidden size **384**. `find_dinov2()` in mill always mounts `metaresearch/dinov2/PyTorch/small/1`.

| Encoder | Where | Mill-correlated? | Scored here? | Loadable 12-finding heads? | Weight cap |
|---------|-------|------------------|--------------|----------------------------|------------|
| Mill V10 (S/14 + Rad-R50 + dual CoAtNet) | `kaggle_kernel_blend*`, `refs/prvsiyan__head-and-shoulders-knees-and-toes` | **anchor** | **55779066 / 0.936** | yes (keep) | 1.00 (anchor) |
| CoAtNet raptor v4 only | `kaggle_kernel_raptor`, `raptor-knee-widedense` | **yes** (same family as mill CoAtNet) | **55778197 / 0.924** | yes | **0.00** — do not add more CoAtNet |
| native384 CoAtNet v8/v10 | `raptor-knee-native384`, `native384dense`; `kaggle_kernel_hs_v41`, `kaggle_kernel_logfix` | **yes** (CoAtNet + mill pipeline) | V41 0.936; logfix 0.936 | yes | **0.00** — ruled out |
| ConvNeXt-V2-B 336 + EffNetV2-L 480 | `dreaddevelopment/raptor-knee-arms`; `kaggle_kernel_cnn_tri` | **medium–high** (same 2.5D raptor windows as mill CoAtNet; ImageNet not RadImageNet) | **55792478 / 0.917** at **0.50-all-12** | yes | **0.00** as a pair — already scored; do not retry 0.50-all-12 |
| CoAtNet inside `raptor-knee-arms` | same dataset (3rd file) | **yes** | mill already uses CoAtNet | yes | **0.00** |
| Mill RadImageNet e13 @ 90 mm | `kaggle_kernel_jointline` | **yes** (same encoder, tighter crop) | **55822934 / 0.935** | yes | **0.00** |
| DINOv3-S/16+ v13.1 | `surasan092/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model`; `kaggle_kernel_dino3s` | **high** (SSL ViT, mill 336/130 mm slots) | **55832530 / 0.908** @ w=1.0 MCL+Baker's | yes | **0.00** as 1.0 replace; not Arm B |
| Mill DINO window-max + CoAtNet hflip | `kaggle_kernel_ttamax` | **yes** (same mill ranks) | **55848021 / 0.936** tie | n/a | **0.00** |
| Fast 2.5D schema / ResNet-34 baseline | `kaggle_kernel_fast`, `refs/baseline-2p5d` | low (weak) | **55776456 / 0.491** | local smoke / public 2.5D | **0.00** — not a specialist |
| DINOv2-B backbone (Meta hub) | `metaresearch/dinov2/PyTorch/base/1` mounted in `refs/sakhawathossen__*` and `refs/report-then-knee` **not** mill kernels | **medium** (same SSL family, 768-d vs mill 384-d) | **not scored as overlay** | **backbone yes; 12-finding heads NO in-repo** | n/a until heads exist |
| MedicalNet 3D ResNet-50 | `royceplayz/rsna-medicalnet-resnet50-weights` (report-then-knee only) | **low** (3D medical pretrain ≠ mill 2.5D CoAtNet / 2D Rad-R50) | **not scored** | **backbone only** — no 12-finding RSNA head in-repo | n/a |
| BiomedCLIP / EffNet-B3 hub | report-then-knee model_sources | unknown / report-family | not as mill overlay | no 12-finding mill-schema heads here | 0.00 |
| 2.5D ResNet-34 public baseline | `refs/baseline-2p5d` | low | not mill-overlay scored | yes (weak ImageNet) | 0.00 |

Raptor gold-set comments (not live LB; from `kaggle_kernel_cnn_tri` / raptor source):

| Raptor arm | Gold (45/58-study panel) | In `raptor-knee-arms`? | Status |
|------------|--------------------------|------------------------|--------|
| coatnet384 | 0.9025 | CoAtNet file yes | mill / scored 0.924 alone |
| swinbase384 | 0.8825 | **no** in-repo mount | unused transformer, ImageNet, mill-windowed |
| effv2l480 | 0.8716 | **yes** | CNN-tri pair (scored) |
| ConvNeXt-V2-B 336 | (cnn-tri arm) | **yes** | CNN-tri pair (scored) |
| cnn336 | 0.8833 “former champion” | separate slug `dreaddevelopment/raptor-cnn336` (page exists) | unused raptor **conv**, **mill-correlated** 2.5D family, weaker than mill CoAtNet |
| cnbase384 / cnlarge384 / maxvit384 | 0.8754 / 0.8752 / 0.8438 | not mounted in this repo | unused; MaxViT ≈ CoAtNet hybrid |

**Refs notebooks (not our scored kernels):** mill clone (`prvsiyan` H&S), DINOv3 0.917 assets (`tonylica`, `mattiaangeli`), RadImageNet+DINO soup (`sakhawathossen` — **DINOv2-B hub + mill Rad-R50**, no new medical conv), report/LLM (`report-then-knee`, `aadigupta7686`, `pilkwang`).

---

## 2. Kaggle search — medical conv **not** mill CoAtNet, **not** CNN-tri pair

Sought: RadImageNet DenseNet/Inception, MedNet/MedicalNet, medical ConvNeXt, ResNet3D, Video Swin, **this competition**, with **12-finding heads**.

| Candidate | Slug / source | This-comp 12-finding heads? | Verdict |
|-----------|---------------|-----------------------------|---------|
| RadImageNet **ResNet-50** | `marwanmath/resnet-50-radimagenet-marwan` + mill e9/e13 heads | **yes — already inside mill** | mill-correlated; jointline already burned crop |
| RadImageNet DenseNet121 / InceptionV3 / Inception-ResNet-v2 | official RadImageNet zoo (Mei et al.) | **none found** as RSNA-knee 12-way datasets | unused backbone only |
| MedicalNet 3D ResNet-50 | `royceplayz/rsna-medicalnet-resnet50-weights` (~343 MB, Apache-2.0) | **no** (pretrain weights) | best **future** medical conv; not loadable this session |
| MedicalNet 3D zoo (generic) | `werus23/medicalnet`, `jirkaborovec/meidcalnet-pretrained-3d-resnet-weights` | no RSNA-knee 12-way | prior RSNA spine/brain, not this task |
| DINOv2-B r336 (nursrijan) | Kaggle user **Nur Srijan** (`kaggle.com/nursrijan`) exists; dataset listing **not scrapeable** this session (JS shell). **No `nursrijan` string in this repo.** | **not confirmed** | treat as **not loadable** until a dataset with B heads + 12 logits is attached |
| Meta DINOv2-B hub | `metaresearch/dinov2/PyTorch/base/1` | backbone; mill SlotHead is **S/384-d** | cannot drop mill heads onto B |
| ConvNeXt-V2 + EffNetV2-L | `dreaddevelopment/raptor-knee-arms` (~1.04 GB) | yes | **already scored 0.917** |
| raptor-cnn336 | `dreaddevelopment/raptor-cnn336` | likely 12-finding raptor ckpt | unused but **mill-family ImageNet conv**, gold 0.8833 |
| Video Swin / ResNet3D RSNA-knee | search | **none found** for this competition | unused |

---

## 3. Proposed 3-arm rank-average (on paper)

Fail-closed mix: `per_finding_rank_blend(mill, overlay, w)` with `w[f] = 0` unless private fold Δ>0. `fusion.py` already rejects uniform ≥0.50 on all 12.

| Arm | Identity | Role | Default w | Cap if fold Δ>0 |
|-----|----------|------|-----------|-----------------|
| **A** | mill V10 **55779066** | **anchor** | **1.00** | — |
| **B** | DINOv2-B r336 (**nursrijan** / Meta B + B-heads) | uncorrelated SSL width vs mill S | **0.00 — none loadable** | ≤ **0.20** on the **few** findings that beat mill on a **private** fold; never 1.0 replace |
| **C** | medical conv ≠ CoAtNet ≠ CNN-tri pair | domain-shift conv | **0.00 — none loadable** | ≤ **0.15** on those findings only |

Recommended per-finding vector **today**: all twelve **0.0** (exact mill CSV). Do not “try 0.15 anyway” — tta-max already showed mill-family 0.15 is a tie.

If heads appear later, **do not** overlay mill-saturated TTA findings first (menisci / Baker's / Fracture / ACL / MCL already mill-max/top2). Prefer findings mill pools with **mean** (PF OA, Effusion) **only if** the new encoder is **not** mill DINO/CoAtNet TTA.

---

## 4. Session decision

| Question | Answer |
|----------|--------|
| **Arm B** | **none loadable** — no DINOv2-B 12-finding checkpoint in-repo; mill heads are ViT-S/384-d; nursrijan heads not verified |
| **Arm C** | **none loadable** — MedicalNet / RadImageNet DenseNet / Inception / Video Swin / ResNet3D have **no** this-comp 12-finding heads; CNN-tri pair and mill CoAtNet/Rad-R50 are **out** |
| **Kernel this session?** | **No.** Quota is scarce; another mill-family or 0.50-all-12 clone cannot beat 0.936. Do not push. |

**Next (later goal, not this kernel):** (1) attach or train **DINOv2-B @ 336** 12-finding heads and score **private** fold vs mill, per finding; (2) train a **MedicalNet-3D or RadImageNet-DenseNet121** 12-way head on the same slots **without** mill CoAtNet windows; (3) only then a fail-closed tiny-weight overlay kernel.
