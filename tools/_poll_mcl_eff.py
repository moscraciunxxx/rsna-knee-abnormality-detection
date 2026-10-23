import json, os, subprocess, sys
from pathlib import Path
os.environ["PYTHONIOENCODING"] = "utf-8"
kaggle = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
slug = "vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov"
root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
st = subprocess.run([kaggle, "kernels", "status", slug], capture_output=True, text=True)
print("STATUS", (st.stdout or st.stderr or "").strip())
p = subprocess.run([kaggle, "kernels", "logs", slug], capture_output=True)
raw = p.stdout or b""
text = raw.decode("utf-8", "replace")
(root / "artifacts" / "_mcl_eff_ov_logs.txt").write_text(text, encoding="utf-8")
print("LOG_BYTES", len(raw))
if len(raw) < 5:
    sys.exit(0)
try:
    ev = json.loads(text)
except Exception as e:
    print("JSON_FAIL", e)
    print(text[:800])
    sys.exit(0)
print("N_EVENTS", len(ev), "TMAX", ev[-1].get("time") if ev else None)
keys = ("mcl-eff-ov", "BOOT", "FileNotFound", "Traceback", "FAIL", "manifest", "START", "END mill", "END coat", "wrote", "cache", "test g", "COMPLETE", "OVERLAY", "ERROR")
for e in ev:
    d = e.get("data", "")
    if any(k in d for k in keys):
        t = e.get("time", 0)
        print(f"{t:7.1f} | {d[:230].rstrip()}")
