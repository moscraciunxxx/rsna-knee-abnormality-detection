from pathlib import Path

root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\kaggle_kernel_nartaa_fold0")
main = (root / "main.py").read_text(encoding="utf-8").replace("\r\n", "\n")
marker = "No-OAI fold-0 CoAtNet train"
idx = main.rfind(marker)
if idx < 0:
    raise SystemExit("marker missing")
start = main.rfind('"""', 0, idx)
if start < 0:
    raise SystemExit("docstring start missing")
head = main[:start]
drv = (root / "driver.py").read_text(encoding="utf-8").replace("\r\n", "\n")
prefix = "from __future__ import annotations\n"
if not drv.startswith(prefix):
    raise SystemExit("future import missing")
drv = drv[len(prefix):]
cut = drv.find("sys.path.insert")
end = drv.find("import _fastread")
if cut < 0 or end < cut:
    raise SystemExit("import block missing")
end = drv.find("\n", end) + 1
drv = drv[:cut] + drv[end:]
drv = drv.replace("_fastread.make_builder", "make_builder")
if "_fastread" in drv or "rsna_knee.raptor_train" in drv:
    raise SystemExit("side import survived")
for needle in (
    "eb0e7aa36def97fb",
    "RESUMED from v3 epoch",
    "/kaggle/input/rsna-knee-f0-v3-ckpt/fold0_last.pt",
    "first_mb_sha256",
    "s/iter",
):
    if needle not in drv:
        raise SystemExit("missing " + needle)
out = head + drv
if not out.endswith("\n"):
    out += "\n"
(root / "main.py").write_text(out, encoding="utf-8", newline="\n")
print("main_bytes", (root / "main.py").stat().st_size)
