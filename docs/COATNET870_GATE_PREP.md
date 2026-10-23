# CoAtNet 0.943 mill-870 private-gate prep (CPU inventory)

Built 2026-10-07 00:33 PT. Kernel is `kaggle_kernel_coatnet870_gate/`. Not pushed while probe-b870 is RUNNING. **No Submit. Never Submit holdout kernels.**

Push when the GPU lane is free: `kaggle kernels push -p kaggle_kernel_coatnet870_gate -t 14400`. Join with `scripts/pull_coatnet870_decision.ps1` (writes `artifacts/coatnet870/overlay_spec.json`, does not touch the vitb hold spec).

## Claim (author, not our CLI)

- Kernel: ctogaurav/rsna-knee-pure-coatnet-0946 (fork of 
yokucha/rsna-knee-d4-blend-0946-ours10 minus private ConvNeXt @10%).
- Author publicScore claim: **0.943** on 2026-09-28. **0.946** needs private ConvNeXt not in public metadata.
- Local pull: 
efs/ctogaurav__rsna-knee-pure-coatnet-0946/
- Excerpt: rtifacts/coatnet0943_claim_excerpt.txt

## Sources to attach (from kernel-metadata.json)

### dataset_sources (14)
- dreaddevelopment/raptor-knee-maxspan
- dreaddevelopment/raptor-knee-native384
- dreaddevelopment/raptor-knee-native384dense
- mattiaangeli/knee-mri-fold-weights
- mattiaangeli/opencv-python-headless-4120088-x86
- marwanmath/resnet-50-radimagenet-marwan
- mattiaangeli/rsna-knee-coat-resgated-ep10-top3
- mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2
- mattiaangeli/rsna-knee-coatnet-global96-top3
- antoinegg1/rsna-knee-e11-diverse-heads-v20
- antoinegg1/rsna-knee-e9-radimagenet-heads-v15
- pilkwang/rsna-knee-llm-labels
- prvsiyan/rsna-knee-v52-radimagenet-heads-20260812
- pilkwang/rsna-knee-weights

### kernel_sources
- sofiaanjenje/rsna-knee-e11-train
- sofiaanjenje/rsna-knee-e13-train

### model_sources
- metaresearch/dinov2/PyTorch/small/1

### Extra for scoring (add; mill-870 gate needs v4 labels)
- stevenleehans/rsna-knee-llm-report-labels (for llm_labels_v4_blend.csv + gold-58 exclusion)

## Notebook shape

- cells: 31 (23 code)
- code bytes: ~227115

## Mill-870 holdout contract (MUST match mill V10 gate)

Reuse the same seed-2026 LLM-soft holdout as scripts/build_mill_holdout_kernel.py / 
sna_knee.mill_holdout:

- HOLD_SEED=2026, HOLD_FRAC=0.2 on v4 rows **excluding gold 58**
- n=**870**, leaked_vs_gold58=0
- y = v4 ≥ 0.5 on those 870
- mill bar: macro **0.9262** (data/mill_v10_holdout870.csv)
- Output CSV: coatnet_holdout870.csv (870×13, header with Baker's)
- Also write submission.csv only as intermediate if the pipeline requires it — **DO NOT Submit**
- Abort / fail-closed if gold 58 intersects holdout IDs

## Required runtime shim (pipeline hard-codes test_series)

Cell 21 forbids train-series fallback (SERIES_ROOT = COMP / 'test_series').
Stages also read ROOT/'test.csv', 	est_series.csv, walk('test_series'), and RSNA_COMP_ROOT.

**Recommended shim** (mirror mill-holdout spirit):

1. Compute holdout ID tuple (identical to mill).
2. Build /kaggle/working/_holdout_comp/ with:
   - 	est.csv = holdout StudyInstanceUIDs
   - 	est_series.csv = 	rain_series.csv filtered to those UIDs
   - sample_submission.csv = 0.5 placeholder rows for those UIDs
   - 	est_series/<UID> → symlink to competition 	rain_series/<UID> (Linux Kaggle OK)
3. Set os.environ['RSNA_COMP_ROOT'] = str(HOLD_COMP) **before** stage imports/walks.
4. After full blend succeeds, copy final preds to coatnet_holdout870.csv, score vs v4≥0.5, print per-finding Δ vs mill if mill CSV mounted or recompute mill join locally after pull.
5. Cap wall ≤4 h; print abort and exit 0 without overlay license if over budget.

## Patch-point inventory (from pulled nb)

- cell3: dirs = [p for p in children if p.is_dir() and p.name not in ('train_series', 'test_series', 'train_images', 'test_images')]
- cell5: # Runtime integrity: no partial ensemble is ever published as submission.csv.
- cell5: for f in ['submission.csv','_pipeline_stage.csv','btkd_v559_complete.json','btkd_v559_complete.json']:
- cell7: os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),
- cell7: ], ['test.csv', 'test_series.csv', 'test_series'])
- cell7: os.environ['RSNA_COMP_ROOT']=str(ROOT)
- cell7: _meta = pd.read_csv(ROOT/'test_series.csv', dtype={'StudyInstanceUID':str})
- cell16: # Jittering here affected only the intermediate submission.csv that the
- cell16: test_series = pd.read_csv(ROOT / 'test_series.csv')
- cell16: plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
- cell16: hte = annotate(walk('test_series'))
- cell16: log(f'submission.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {sub.shape}')
- cell20: assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
- cell21: SERIES_ROOT = COMP / 'test_series'
- cell21: if not SERIES_ROOT.is_dir():
- cell21: raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')
- cell21: print('series root:', SERIES_ROOT)
- cell21: files=ordered_files(SERIES_ROOT/str(study)/series)
- cell21: sub_df = pd.read_csv(COMP / 'sample_submission.csv',dtype={'StudyInstanceUID':str})
- cell21: ser_csv = pd.read_csv(COMP / 'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
- cell26: tser=_rad_pd.read_csv(ROOT/'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
- cell26: headers=annotate(walk('test_series'))
- cell28: for value in [os.environ.get('RSNA_COMP_ROOT', ''),
- cell28: if _ke_os.environ.get('RSNA_COMP_ROOT'):
- cell28: _KE_NS['find_test_root'] = lambda: _ke_os.environ['RSNA_COMP_ROOT']
- cell28: series_root = root + '/test_series'
- cell28: test_series = _ke_pd.read_csv(root + '/test_series.csv')
- cell28: test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
- cell28: test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
- cell28: for key, frame in test_series.groupby('StudyInstanceUID')
- cell28: sample = root + '/sample_submission.csv'
- cell28: "competition_root=Path(os.environ['RSNA_COMP_ROOT']) "
- cell28: "if os.environ.get('RSNA_COMP_ROOT') "
- cell28: env['RSNA_COMP_ROOT'] = _KE_NS['find_test_root']()
- cell28: "competition_root=Path(os.environ['RSNA_COMP_ROOT']) "
- cell28: "if os.environ.get('RSNA_COMP_ROOT') "
- cell28: g96_env_vars['RSNA_COMP_ROOT'] = _KE_NS['find_test_root']()
- cell28: "competition_root=Path(os.environ['RSNA_COMP_ROOT']) "
- cell28: "if os.environ.get('RSNA_COMP_ROOT') "
- cell28: rv1_env_vars['RSNA_COMP_ROOT'] = _KE_NS['find_test_root']()
- cell30: # Publish submission.csv only after the entire BTKD + CoAt family graph succeeds.
- cell30: _final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')

## Weakest mill-870 findings (overlay candidates IF coat beats mill by >0.01)

- MCL 0.8836
- Effusion 0.8977
- Contusion 0.9070
- Synovitis 0.9075
- PF OA 0.9167

## Risk note (WIN_STATUS)

This blend is **mill-adjacent** (DINO + RadImageNet + multi CoAtNet). Private gate can still **tie** like V41/logfix/tta-max. Gate before any TEST Submit. Does not close gap to page-1 0.960 / #1 0.964 alone.

## Target paths for Grok Build

- kaggle_kernel_coatnet870_gate/ (main.py + kernel-metadata.json)
- scripts/build_coatnet870_gate_kernel.py (optional builder)
- scripts/pull_coatnet870_decision.ps1 / .py (join vs mill-870, write decision note; no Submit)

## Hard rules

- No competition Submit
- Never Submit holdout kernel
- Keep **55779066** until private license + TEST overlay strictly > 0.936
- Only one GPU kernel at a time (wait for probe-b870 COMPLETE before push)
