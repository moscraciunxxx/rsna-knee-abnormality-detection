# Next stage: no-OAI CoAtNet from ImageNet (Oct 12 2026)
Full version: box /workspace/grok_audit/NEXT_STAGE_NO_OAI.md (MSI copy below is the same plan, summarized).

- Fold-0 v2 COMPLETE but COLLAPSED (OOF std ~3e-5, loss flat 0.63), 7.2 GPU-h used; ~22.8 h left to reset Fri Oct 16 17:00 PT; deadline Thu Oct 22 16:59 PT.
- Step 0 (0 GPU-h, Oct 12-13): keep pruned walk (walk cost only 6 min). Fix collapse: GradScaler, accumulate to batch 8, AdamW head 2e-4 / backbone 5e-5, warmup+cosine, mean weighted BCE. In-kernel collapse alarm (pred std <1e-3 -> collapse.json, exit). Local tests/test_learns.py on RTX 3060 (16 studies, 150 steps, loss -30%, std >0.05). Train DICOMs are not on the MSI: download a 32-64 study subset.
- Step 1 (0 GPU-h, Oct 13): 5 CPU kernels, one per fold, cache 96x384 uint8 volumes (~12.5 GB/fold raw, <20 GB output cap) -> private dataset rsna-knee-cache96-384.
- Step 2 recipe: coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k (ImageNet, bundle weights), Raptor MIL head, train K=8 img 320, eval K=94, 10 epochs, SWA 7-10, mirror TTA. Measured 1.22 s/study with decode; expect 0.4-0.5 s cached (estimate), 25-30 min/epoch. Cap 4.5 h/fold, wall 5 h, checkpoint per epoch for resume.
- Labels: steven v4 (tuned_on_gold=False) + pilkwang confidence weights; gold58 weight 0; no JEV HYBRID.
- Validation: honest OOF on data/folds.csv (by study; no PatientID exists). Kill: OOF <0.86 at epoch 3 or <0.90 at end; folds 1-4 only if fold-0 >=0.92. fold0_gate mill comparison can't fire (no full mill OOF), so use absolute thresholds.
- Ensemble: 5-fold no-OAI CoAtNet + goodpjw2008 ConvNeXt reader IF its card shows no OAI (unverified). Other public models only with clean cards; unverified: dreaddevelopment raptor maxspan/widedense, tonylica bend-dinov3, romantamrazov dinosaur-v5, mattiaangeli sets. Target 0.94-0.95 public (goal, not measured).
- GPU: Oct 13 fold-0 v3 <=5 h; Oct 14-16 folds 1-4 <=18 h; reset +30 h Oct 16 17:00; Oct 17-18 inference 2-3 h/run x2; Oct 18-20 second family <=15 h; Oct 21 picks, 4 h spare.
- Second selection: 55779066 mill mounts the four unverified sources above, so it's only a no-OAI hedge if their cards are clean. 56929063 is OAI so it can't win a prize; consider making both final picks OAI-free.
- Ask: "OK to (1) run 5 CPU-only Kaggle kernels that build the cached slice dataset (0 GPU-h), and (2) once tests/test_learns.py passes locally, push vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai v3 on a T4 (max 5 GPU-h, auto-exit on collapse or OOF < 0.86 at epoch 3)? No submit."
