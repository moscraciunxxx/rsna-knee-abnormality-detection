"""Push the five CPU cache kernels once each. No submit."""

import subprocess
import sys

root = r"D:\Coding Compete\RSNA Knee Abnormality Detection"
kaggle = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
for fold in range(5):
    folder = root + rf"\kaggle_kernel_cache_f{fold}"
    proc = subprocess.run([kaggle, "kernels", "push", "-p", folder], capture_output=True)
    sys.stdout.buffer.write(proc.stdout)
    sys.stdout.buffer.write(proc.stderr)
    print("FOLD", fold, "EXIT", proc.returncode, flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
