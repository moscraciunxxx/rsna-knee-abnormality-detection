# Saturday T4 window: encode ViT-B on mill-870, join, stop or overlay. Never Submit holdout IDs.
$ErrorActionPreference = "Stop"
$kaggle = "C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
$py = "C:\Users\moscr\anaconda3\python.exe"
$root = "D:\Coding Compete\RSNA Knee Abnormality Detection"
$out = Join-Path $root "artifacts\saturday_vitb870"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$slug = "vitaliecervinschi/rsna-knee-vitb870"
$existing = & $kaggle kernels status $slug 2>&1 | Out-String
if ($existing -match "RUNNING" -or $existing -match "QUEUED") {
    Write-Host "vitb870 already $existing"
    Write-Host "Do not push a second session. Wait, then run scripts\pull_vitb870_decision.ps1"
    exit 3
}
if ($existing -match "COMPLETE") {
    Write-Host "vitb870 already COMPLETE. Do not push again. Pull the license file."
    & powershell -NoProfile -File (Join-Path $root "scripts\pull_vitb870_decision.ps1")
    exit $LASTEXITCODE
}

Write-Host "==== rebuild vitb870 (truncate mill ensemble) ===="
& $py (Join-Path $root "scripts\build_vitb870_kernel.py")
Write-Host "==== probe GPU + push ===="
Set-Location (Join-Path $root "kaggle_kernel_vitb870")
$push = & $kaggle kernels push 2>&1 | Out-String
Write-Host $push
if ($push -match "weekly GPU quota") {
    Write-Host "QUOTA still blocked. Do not Submit. Keep 55779066."
    exit 2
}
if ($push -notmatch "successfully pushed") {
    Write-Host "push failed"
    exit 1
}
$slug = "vitaliecervinschi/rsna-knee-vitb870"
while ($true) {
    $st = & $kaggle kernels status $slug 2>&1 | Out-String
    Write-Host $st.Trim()
    if ($st -match "COMPLETE") { break }
    if ($st -match "ERROR" -or $st -match "CANCEL") {
        Write-Host "kernel failed; do not Submit"
        exit 1
    }
    Start-Sleep -Seconds 120
}
& $kaggle kernels output $slug -p $out -o
$vitb = Join-Path $out "vitb_holdout870.csv"
if (-not (Test-Path $vitb)) {
    Write-Host "vitb_holdout870.csv missing in kernel output"
    exit 1
}
Write-Host "==== mill-870 join (private bar) ===="
& $py (Join-Path $root "scripts\saturday_after_vitb.py") --vitb $vitb
Write-Host "Read overlay_spec.json. If decision=stop|hold: keep 55779066, do not Submit."
Write-Host "If decision=overlay: python scripts/build_licensed_overlay_kernel.py then kernels push kaggle_kernel_mill870_ov"
Write-Host "NEVER competitions submit the vitb870 kernel (train holdout IDs)."
