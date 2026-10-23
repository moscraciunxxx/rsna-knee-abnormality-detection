# ConvNeXt MIL reader mill-870 private-gate prep (CPU inventory)

Built 2026-10-07 ~08:30 PT while `vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov` v3 RUNNING.
**Not pushed. No Submit. Never Submit holdout kernels.** Keep **55779066 / 0.936**.

## Why this is bet #1 (independent of running coat overlay)

| Fact | Detail |
|------|--------|
| Author claim | `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` public **0.944** (stack 70% + reader 30%); reader alone **0.929** |
| New family | `goodpjw2008/rsna-knee-2-5d-convnext-reader` — ConvNeXt-Tiny 2.5D + study-level transformer; **not** in mill V10 and **not** in coatnet870-gate mounts |
| Independence | Coat vs mill Spearman mean ρ≈**0.984** on the same 870 (Effusion ρ=0.994) → coat overlays are mill-adjacent; this reader is a different encoder |
| GPU cost | Author: reader-only (`RUN_STACK=False`) ≈ **20 min**; design cap **≤1.5 h** for private 870 encode |
| License rule | Same as coat: finding Δ>0.01 vs mill-870, ≤2 findings, w≤0.15, never ruled-out families, fail-closed to mill |
| Expected Δ | 0 to +0.008 public if author 0.944 replicates as minority overlay; still short of page-1 0.960 |

## Assets (public, already listed)

Dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader`:
- `cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt`, `cnxt_v0_fold2.pt` (~123 MB each)
- `infer.py`, `knee.py`, `preprocess.py`, `train.py`, `make_labels.py`
- pylibjpeg wheels for offline DICOM decode

Pulled notebook meta: `artifacts/_nextbets/goodpjw2008_rsna-knee-stack-2-5d-convnext-mil-lb-0-944/`

## Private gate contract (must match mill V10)

- HOLD_SEED=2026, HOLD_FRAC=0.2 on v4 rows excluding gold 58 → n=**870**
- y = v4 ≥ 0.5; mill bar macro **0.9262** (`data/mill_v10_holdout870.csv`)
- Output: `convnext_holdout870.csv` (870×13, `Baker's` header)
- Delete any `submission.csv` before exit; write `DO_NOT_SUBMIT_HOLDOUT.txt`
- Cap wall ≤1.5 h; abort without license if over budget

## Kernel sketch (local, not pushed)

- Folder: `kaggle_kernel_convnext_mil870_gate/`
- Builder: `scripts/build_convnext_mil870_gate_kernel.py`
- Decision: `scripts/saturday_after_convnext.py` → `artifacts/convnext_mil870/overlay_spec.json`
- Pull: `scripts/pull_convnext_mil870_decision.ps1`
- Tests: `tests/test_convnext_mil870_gate.py`

When GPU lane is free (overlay COMPLETE or cancelled):  
`kaggle kernels push -p kaggle_kernel_convnext_mil870_gate -t 5400`  
Then `powershell -File scripts\pull_convnext_mil870_decision.ps1`.  
**Do not** push while coatnet MCL+Effusion TEST overlay is RUNNING.

## Kill criteria

- No finding beats mill-870 by >0.01 → stop / hold; keep 55779066
- Wins only on ruled-out sets (PF OA+Synovitis @0.20, native384, dino3s 1.0, jointline, uniform 0.50-all-12) → refuse
- Later TEST publicScore ≤0.936 → keep 55779066
- Never Submit the holdout kernel

## Ruled-out / lower priority (not this gate)

- Re-push coatnet870-gate or coatnet-test-v1
- Mill-family TTA / more CoAtNet clones (ρ≈0.98 with mill)
- mednet / v4head / dense12 / attn25d / pfoa-syn / probe-b / vitb870 (already stop/hold)
