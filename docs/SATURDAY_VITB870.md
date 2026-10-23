# Saturday T4 window — mill-870 ViT-B hunt (private-first)

Do **not** Submit `rsna-knee-vitb870` (train holdout IDs). Keep **55779066** unless a TEST overlay publicScore is strictly > 0.936.

## Pipelines (what each one is)

| Pipeline | What it is | Saturday role |
| --- | --- | --- |
| Mill V10 `55779066` | DINO 20 + RadImageNet + CoAtNet | Keep / overlay *anchor* |
| mill-870 CSV | V10 on seed-2026 LLM-soft holdout n=870 | **Private overlay bar** (not gold-58, not public 0.936) |
| vitb870 | r336 ViT-B on those **same 870 train UIDs** | License table only. **No Submit.** |
| mill870-ov | mill V10 **test** + r336 test, subset ≪0.50 | Only if mill-870 licenses findings **other than** PF OA+Synovitis @0.20 |
| B/C/dense12/v4head/mednet | ImageNet/Med3D trains | Dead on 870. Do not rerun. |
| r336ov 55856648 | PF OA+Synovitis @0.20 | Ruled out (0.935). |
| DINOsaur V5 | mill 2.0 | Do not attach. |
| Efficiency LB | Different prize | Ignore. |

## 2026-10-07 — version 1 is already RUNNING

`kaggle kernels push` returned **Kernel version 1 successfully pushed**. Status was **RUNNING**. Do **not** run the push script again while that session is alive (it now refuses RUNNING, QUEUED, and COMPLETE). When status is COMPLETE:

```powershell
powershell -File "D:\Coding Compete\RSNA Knee Abnormality Detection\scripts\pull_vitb870_decision.ps1"
```

That downloads `vitb_holdout870.csv` and runs the mill-870 decision. It does not push and it does not Submit.

## One command

```powershell
C:\Users\moscr\anaconda3\python.exe D:\Coding Compete\RSNA Knee Abnormality Detection\scripts\build_vitb870_kernel.py
powershell -File D:\Coding Compete\RSNA Knee Abnormality Detection\scripts\saturday_vitb870.ps1
```

If push says `Maximum weekly GPU quota of 30.00 hours reached`: stop. Do not restart a retry loop.

## After COMPLETE

1. `artifacts/saturday_vitb870/vitb_holdout870.csv` must exist (870×13, `Baker's`).
2. Script runs `saturday_after_vitb.py` → `kaggle_kernel_mill870_ov/overlay_spec.json`.
3. `decision=stop` or `hold`: keep 55779066, no Submit.
4. `decision=overlay`: `python scripts/build_licensed_overlay_kernel.py` then T4 push `kaggle_kernel_mill870_ov`. Submit **that** TEST kernel only if you still want the public shot — never the holdout kernel.

## Overlay law

- y = v4 ≥ 0.5 on the 870, gold 58 excluded
- new > mill by quoted Δ, ≤2 findings, weight ≪ 0.50
- Illegal: all-12 ≥0.50, V41, logfix, jointline, dino3s 1.0, r336ov PF OA+Synovitis
- Weakest mill-870 (candidates **if** ViT-B beats them): MCL 0.884, Effusion 0.898, Contusion 0.907
