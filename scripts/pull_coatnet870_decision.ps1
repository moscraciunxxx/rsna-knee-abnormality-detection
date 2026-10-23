# Download a COMPLETE coatnet870-gate run and write the mill-870 overlay decision.
# Does not push. Does not Submit. Train-holdout IDs stay off the leaderboard.
# Does not overwrite kaggle_kernel_mill870_ov/overlay_spec.json (vitb870 hold).
$ErrorActionPreference = "Stop"
$kaggle = "C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
$py = "C:\Users\moscr\anaconda3\python.exe"
$root = "D:\Coding Compete\RSNA Knee Abnormality Detection"
$slug = "vitaliecervinschi/rsna-knee-coatnet870-gate"
$out = Join-Path $root "artifacts\coatnet870"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$st = & $kaggle kernels status $slug 2>&1 | Out-String
Write-Host $st.Trim()
if ($st -notmatch "COMPLETE") {
    Write-Host "coatnet870-gate is not COMPLETE. Leave the GPU alone. Do not Submit."
    exit 3
}
& $kaggle kernels output $slug -p $out -o
$coat = Join-Path $out "coatnet_holdout870.csv"
if (-not (Test-Path $coat)) {
    Write-Host "coatnet_holdout870.csv missing in kernel output"
    Get-ChildItem $out | Select-Object Name, Length
    exit 1
}
& $py (Join-Path $root "scripts\saturday_after_coatnet.py") --coatnet $coat
Write-Host "If decision=stop or hold: keep 55779066. Do not Submit."
Write-Host "If decision=overlay: do not Submit the holdout kernel. A TEST overlay is a separate push."
Write-Host "NEVER competitions submit vitaliecervinschi/rsna-knee-coatnet870-gate."
