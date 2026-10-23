import subprocess
import sys

cmd = [
    r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe",
    "kernels",
    "push",
    "-p",
    r"D:\Coding Compete\RSNA Knee Abnormality Detection\kaggle_kernel_clean_blend",
]
proc = subprocess.run(cmd, capture_output=True)
sys.stdout.buffer.write(proc.stdout)
sys.stdout.buffer.write(proc.stderr)
print("EXIT", proc.returncode)
sys.exit(proc.returncode)
