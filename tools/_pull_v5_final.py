"""One pull of the finished v5 log and small output files. No push."""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

slug = "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai"
dest = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\fold0_v5_final")
dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()
status = api.kernels_status(slug)
print("status", status.status, status.failure_message or "")
log = api.kernels_logs(slug) or ""
(dest / "v5.log").write_text(log, encoding="utf-8")
print("log_chars", len(log))
for line in log.splitlines():
    low = line.lower()
    if any(k in low for k in ("epoch ", "exit", "kill", "collapse", "abort", "wall", "saved", "oof ", "std", "resumed", "gpu")):
        print(line[:400])
api.kernels_output(
    slug,
    str(dest),
    file_pattern=r"(fold0_oof\.csv|fold0_timing\.json|fold0_oof_subset\.csv|collapse\.json)$",
    force=True,
    quiet=False,
)
