"""At 17:00 PT push the 3-fold T4 reader once. No competition submit.

Fold 3 is not USABLE, so the 4-fold variant is not pushed.
"""

from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

KAGGLE = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
KERNEL = r"D:\Coding Compete\RSNA Knee Abnormality Detection\kaggle_kernel_clean_blend_t4"
SLUG = "vitaliecervinschi/rsna-knee-clean-blend"
OUT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\clean_blend_t4_run")
LOG = Path(r"D:\rsna_cache\t4_push.log")
WHEN = datetime(2026, 10, 16, 17, 0, 0)


def log(msg: str) -> None:
    line = datetime.now().strftime("%H:%M:%S") + " " + msg
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")


def push_once() -> str:
    for attempt in range(1, 7):
        proc = run([KAGGLE, "kernels", "push", "-p", KERNEL])
        text = (proc.stdout or "") + (proc.stderr or "")
        log(f"push attempt {attempt} exit {proc.returncode}\n{text[-1500:]}")
        if proc.returncode == 0:
            return text
        if "quota" not in text.lower() and "gpu" not in text.lower():
            raise SystemExit(proc.returncode)
        time.sleep(600)
    raise SystemExit("push still refused after quota retries")


def status() -> str:
    proc = run([KAGGLE, "kernels", "status", SLUG])
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return text


def main() -> None:
    log("scheduled 3-fold T4 push. fold3 USABLE False. no 4-fold push. no submit.")
    delay = (WHEN - datetime.now()).total_seconds()
    if delay > 0:
        log(f"sleep {delay:.0f}s until 17:00")
        time.sleep(delay)
    push_once()
    deadline = time.time() + 90 * 60
    while time.time() < deadline:
        text = status()
        log("status " + text.replace("\n", " "))
        if "COMPLETE" in text:
            break
        if "ERROR" in text or "CANCEL" in text:
            raise SystemExit(text)
        time.sleep(60)
    else:
        raise SystemExit("kernel did not finish within 90 minutes")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = run([KAGGLE, "kernels", "output", SLUG, "-p", str(OUT)])
    log(f"output exit {proc.returncode} {(proc.stdout or '')[-500:]}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    frame_path = OUT / "submission.csv"
    with frame_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    nan = sum(cell == "" or cell.lower() == "nan" for row in body for cell in row)
    log(f"csv rows {len(body)} cols {len(header)} nan {nan} header {' '.join(header)}")
    logs = list(OUT.glob("*.log"))
    pace = ""
    for path in logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "seconds/study" in line:
                pace = line
    log("pace " + (pace or "MISSING"))
    if len(body) != 3 or len(header) != 13 or nan:
        raise SystemExit("csv check failed")
    log("3-fold T4 confirmed. no submit.")


if __name__ == "__main__":
    main()
