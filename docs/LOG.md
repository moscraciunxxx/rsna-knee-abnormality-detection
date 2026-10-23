2026-10-08T02:01:18Z submissions: 56929063 Apex PENDING blank; 56926403 PENDING; 56923311 PENDING; anchor 55779066 0.936 selected. GPU 10.59/19.41 refresh 2026-10-10T00:00:00Z. No push.
2026-10-08T07:39:13Z CLI: 56929063 COMPLETE 0.950; 56926403 COMPLETE 0.922; 56923311 COMPLETE 0.936; 55779066 0.936. GPU 13.20/16.80 refresh 2026-10-10T00:00:00Z. Wrote data/folds.csv n=4407 gold=58. No GPU push.

## 2026-10-12 ~06:30 UTC browser OAI pass
Read rules 2.6, discussions 733965, 734109, 741819, 743416, 747455, 747660, 748276, NDA query-download, submissions (56929063 still the only selection, 1/2), Your Work (no running kernel, newest 4 days ago), Settings GPU 00:00 / 30 hrs. Wrote docs/OAI_CLEARANCE.md. Did not post. Did not push a kernel. Did not change selections.

## 2026-10-12 ~07:40 UTC fold-0 push
Audited session 01a039bd from summary.json, resume_status.json, and chat_history.jsonl (706 lines). Standing goal text still names mill 55779066; live selection is 56929063. Pushed vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai v1 (ImageNet CoAtNet, no OAI). Did not submit. pytest test_raptor_model.py: 4 passed.

## 2026-10-12 ~07:35 UTC fold-0 v2
v1 ERROR in 22s: script kernel did not upload raptor_model.py. Bundled model, fast reader, and folds.csv into main.py. Pushed v2. Status immediately after push: RUNNING. Did not submit. Selection unchanged.

## 2026-10-12 ~07:45 UTC fold0 gate
Did not poll the running kernel. fold 0 is 882 studies, 11 gold. mill_v10_holdout870 overlaps 174. Wrote rsna_knee/fold0_gate.py. pytest test_fold0_gate.py passed. A partial mill file cannot kill the recipe.

## 2026-10-12 ~07:50 UTC JEV labels
Did not poll the kernel. Downloaded riadmohamed42/jev-knee-labels and karttikjangid05 comparison into data/external_labels. HYBRID is gold-tuned. Left v2 training on steven v4.

## 2026-10-12 ~07:55 UTC memory probe
Did not poll the kernel. coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k fp16 backward on RTX 3060 6GB: K=1 1430 MiB, K=2 2075, K=4 3349. Did not push.

## 2026-10-12 ~07:58 UTC walk prune
Did not poll. driver.py pruned_walk skips train_series/test_series/train_images/test_images. The running v2 bundle does not include that change.

## 2026-10-12 ~15:40 UTC v2 collapse and learn test
v2 COMPLETE. fold0_timing.json wall_hours 7.167, 6 epochs, 882 OOF rows. OOF std mean 2.55e-5. Losses 0.6377 to 0.6283. tests/test_learns.py passed after grad clip (K=4, 160 px, 150 steps). Did not push.

## 2026-10-12 ~15:55 UTC provenance cards
Read dataset metadata for the ConvNeXt reader, raptor maxspan, raptor widedense, and tonylica bend assets. dinosaur-v5 metadata returned 403. Wrote docs/PROVENANCE_CARDS.md. Did not push.

## 2026-10-12 ~23:30 UTC reader provenance
Downloaded goodpjw2008 reader train/make_labels/oof/preprocess scripts. v0 labels = mean of steven/pilkwang/lixin/qwen LLM tables; no OAI or external-image refs. Reader is clean for a no-OAI blend. Local test_series is empty (0 files). Did not push.

## 2026-10-12 ~23:35 UTC v3 pushed
Vitaly OK'd the two-part ask. Lane was clear (v2 COMPLETE). Pushed vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai v3 (T4, wall 5 h). CPU cache kernels deferred until v3 timing shows whether cache is needed.

## 2026-10-12 ~23:55 UTC cache kernels drafted
Wrote scripts/build_cache_kernels.py plus rsna_knee/cache_vol.py and tests/test_cache_vol.py (green). Generated kaggle_kernel_cache_f0..f4, CPU-only, internet off, docker 37c64f. Not pushed. v3 still running.

## 2026-10-13 ~09:00 UTC stack provenance
0.944 stack slug 404s live; used the Oct-7 pull (same version). 12 cards read; 8 empty. e11/e13 training code fits heads on gold58 at weight 7.0. Wrote docs/CLEAN_STACK_PROVENANCE.md: no copy. v3 COMPLETE (OOF 0.72 per Vitaly); output download broke, pt not fetched. No push.

## 2026-10-13 ~09:10 UTC task B start
v4 LRs in rsna_knee/raptor_train.py (1e-3/1e-4, accum 4); driver OOF subset 200, floors 0.80/0.92, wall 4 h. test_learns green at v4 LRs. v4 labels downloaded. Collector launched for 64 studies. No push.

## 2026-10-13 ~21:00 UTC v4 staged + mill V10 provenance complete
Vitaly OK'd v4 push (pick b). Updated driver.py + bundled main.py (481,931 B): head 1e-3, backbone 1e-4, accum 4, WALL_HOURS 4.0, per-epoch 200-study OOF logged, kill <0.80 at ep3 or std <1e-3. Pre-push check found vitaliecervinschi/casmi26-rows-fold0 RUNNING on T4; held push per 1-GPU rule. Completed full mill V10 + 0.944 stack provenance in docs/CLEAN_STACK_PROVENANCE.md (downloaded tonylica SOURCE_AND_LICENSES.md + bundle_manifest.json, mattiaangeli trainers/manifests, pilkwang baseline). All mill V10 sources are OAI-free; tonylica & dinosaur-v5 are tainted by gold58-trained heads.

## 2026-10-13 ~21:07 UTC v4 pushed
Pushed vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai v4 (T4, cap 4.0 GPU-h, no submit) per Vitaly's instruction (1-GPU rule is RSNA-only). Status immediately after push: KernelWorkerStatus.QUEUED. Updated WIN_STATUS.md lock to 18:10 PT; next check 15:10 PT.

## 2026-10-14 ~01:32 UTC v5 staged
v4 COMPLETE, collapsed. v5 lives in kaggle_kernel_nartaa_fold0/driver.py and main.py (486,563 B). Resume pin kernel version 3. CPU dry run of the bundle: RESUMED from v3 epoch 3, 20 steps, exit 0. raptor_train.py left at the v4 rates. Not pushed.

## 2026-10-14 ~02:50 UTC v5 push refused
One push of vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai. Kaggle: Maximum batch GPU session count of 2 reached. Holders: casmi26-val-c2f0-ours-nomvnf and casmi26-val-c2f0-ours-nomv, both RUNNING. Fold-0 kernel still v4 COMPLETE. Hash gate is in main.py (487,119 B): abort unless first-MB SHA-256 is eb0e7aa36def97fb. No submit.

## 2026-10-14 ~02:55 UTC elapsed-time correction
API last_run_time of ~72h is not the session clock. UI shows both casmi26-val runs at Running: 2 hours. GPU hard-stop is 12h, so a slot frees by ~06:00 PT. Do not wait for the Friday reset. Next push is on "slot free".

## 2026-10-14 ~03:10 UTC clean blend draft
CPU-only holdout870 scores, gold leak 0. Reader 0.9439, maxspan-v5 0.9056, equal rank blend of those two 0.9340. Widedense has no local predictions. Mill 0.9262 used only to check the label join. Drafted kaggle_kernel_clean_blend (CPU, docker 37c64f). No push, no submit.

## 2026-10-14 ~03:36 UTC v5 push aborted on v4 hash
Pushed vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai version 5 once. No submit. Log: first_mb_sha256 1c0e40b7fe4980fc then ABORT (expected eb0e7aa36def97fb). Status ERROR. kernel_sources pin /3 still mounted the v4 output. Clean blend left paused.

## 2026-10-14 ~03:42 UTC v5 resumed from the private v3 dataset
Uploaded vitaliecervinschi/rsna-knee-f0-v3-ckpt (private, fold0_last.pt only, first-MB eb0e7aa36def97fb, status ready). Pushed fold-0 kernel version 6 once, no submit. Log: hash eb0e7aa36def97fb, RESUMED from v3 epoch 3, s/iter 1.026, projected epoch 0.991 h.

## 2026-10-14 ~04:05 UTC clean reader draft
Holdout870, gold leak 0. Reader 0.9439 wins all 12 findings vs maxspan 0.9056. Equal rank blend 0.9340. Widedense unscored (no local file). Maxspan card does not state training labels, marked EXCLUDED. Draft kaggle_kernel_clean_blend is reader-only, CPU, docker 37c64f. v5 kernel not touched. No push.

## 2026-10-14 ~04:50 UTC reader run and v5 epoch 1
Pushed vitaliecervinschi/rsna-knee-clean-blend v1 once, CPU, internet off, docker 37c64f. No submit. Weights cnxt_v0_fold0/1/2.pt only, first-MB ee0da0c8f30a4017, 550788585adb4dde, 451a4eca59030230. submission.csv rows 3, 13 columns, 0 NaN. v5 epoch 4 resumed_epoch 1 loss 0.5661 subset_oof 0.7299 std 0.219542. Log shows cuda True, no GPU count.

## 2026-10-14 ~05:40 UTC reader GPU push refused
CPU v1: 28.1 s/study, about 1300 test studies, projected 10.1 h. T4 push of rsna-knee-clean-blend refused, 2 GPU sessions (v5 and casmi26-v4n-knobs1). No competition submit. v5 log at 22:40 PT still only resumed epoch 1.

## 2026-10-14 ~07:20 UTC v5 killed and reader quota
v5 COMPLETE. Exit kill at epoch 5: loss 0.5623, subset OOF 0.7517, std 0.253431, floor 0.78. Full OOF 882 macro 0.7503. Wall 2.70 h. CoAtNet path closed. Reader T4 push refused: weekly GPU quota 30 h reached. No submit. No selection change.

## 2026-10-14 ~07:36 UTC reader fold0 CPU submitted
Pushed rsna-knee-clean-blend v2, CPU, docker 37c64f, internet off, cnxt_v0_fold0.pt only. 13.302 s/study, 4.803 h for ~1300. submission.csv 3 rows, 0 NaN. Submitted once as 57172933 PENDING. No selection change. 3-fold T4 staged in kaggle_kernel_clean_blend_t4, not pushed.

## 2026-10-14 ~17:50 UTC clean plan, no GPU push
Two cards pass: reader (holdout870 0.9439, public fold0 0.922, author 3-fold 0.929) and widedense (no local holdout). Drafted fold34, infer5, and a widedense rank kernel that refuses to blend until a holdout file exists. Plan is 21 GPU-h for reader 3-fold then folds 3-4 then 5-fold infer. No push, no submit.

## 2026-10-14 ~23:25 UTC Friday queue locked
Vitaly queued 1 then 2-4. Folds 3 and 4 run in parallel. Kill a fold if step-50 projects > 9 h. 5-fold infer only if both OOF >= 0.92 and pred std > 1e-3; otherwise keep the 3-fold and submit nothing new. Widedense only if hours remain after that and a holdout870 score exists. Each push still needs an OK.

## 2026-10-16 ~15:40 UTC author OOF vs local fold 3
Same cache, labels, and k=12 eval. Author folds 0/1/2 macro 0.9523 / 0.9506 / 0.9495, mean 0.9508. Local fold 3 re-score 0.8808 (matches the training log; best epoch 12 was 0.8814). Gap 0.070. Not USABLE. 4-fold push will not run. 3-fold T4 push is scheduled for 17:00 PT. No submit.

## 2026-10-14 ~23:40 UTC cache kernels pushed
Reader preprocess is not the CoAtNet 96-slice stack. cache_vol.py now stores compressed per-series 384 volumes. test_cache_vol.py passed. Pushed CPU rsna-knee-cache-f0-no-oai through f4-no-oai, docker 37c64f, internet off. All RUNNING at 16:40 PT. No submit.
