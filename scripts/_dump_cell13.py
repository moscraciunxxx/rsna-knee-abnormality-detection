import json
from pathlib import Path

p = Path(
    r"D:\Coding Compete\RSNA Knee Abnormality Detection\.cache\dinov2b\kernels\rsna-knee-abnormality\rsna-knee-abnormality.ipynb"
)
nb = json.loads(p.read_text(encoding="utf-8"))
src = nb["cells"][13].get("source", "")
if isinstance(src, list):
    src = "".join(src)
out = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\.cache\dinov2b\cell13.py")
out.write_text(src, encoding="utf-8")
print("wrote", out, "len", len(src))
# print only load-related region
lines = src.splitlines()
for i, line in enumerate(lines):
    if any(s in line.lower() for s in ("r336", "r224", "custom", "variant='base'", "variant=\"base\"", "load_state")):
        start = max(0, i - 2)
        stop = min(len(lines), i + 8)
        print(f"\n--- around {i} ---")
        for j in range(start, stop):
            print(f"{j:4d}|{lines[j]}")
