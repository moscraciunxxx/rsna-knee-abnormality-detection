# Download a COMPLETE probe-b870 run and write the mill-870 overlay decision.
# Does not push. Does not Submit. Train-holdout IDs stay off the leaderboard.
$ErrorActionPreference = "Stop"
$kaggle = "C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
$py = "C:\Users\moscr\anaconda3\python.exe"
$root = "D:\Coding Compete\RSNA Knee Abnormality Detection"
$slug = "vitaliecervinschi/rsna-knee-probe-b870"
$out = Join-Path $root "artifacts\probe_b870"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$st = & $kaggle kernels status $slug 2>&1 | Out-String
Write-Host $st.Trim()
if ($st -notmatch "COMPLETE") {
    Write-Host "probe-b870 is not COMPLETE. Leave GPU alone. Do not Submit."
    exit 3
}
& $kaggle kernels output $slug -p $out -o
$probe = Join-Path $out "probe_holdout870.csv"
if (-not (Test-Path $probe)) {
    Write-Host "probe_holdout870.csv missing in kernel output"
    Get-ChildItem $out | Select-Object Name, Length
    exit 1
}
& $py (Join-Path $root "scripts\saturday_after_probe_b.py") --probe $probe
Write-Host "If decision=stop or hold: keep 55779066. Do not Submit."
Write-Host "If decision=overlay: python scripts\build_licensed_overlay_kernel.py then push only kaggle_kernel_mill870_ov TEST kernel."
Write-Host "NEVER competitions submit vitaliecervinschi/rsna-knee-probe-b870."
