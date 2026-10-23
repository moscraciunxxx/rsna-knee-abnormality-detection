import base64, json, os, re, subprocess, zlib
from pathlib import Path

root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
os.environ["PYTHONIOENCODING"] = "utf-8"
kaggle = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
slug = "vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov"

# verify embedded mill patch
pull = root / "artifacts" / "_mcl_eff_pull" / "rsna-knee-coatnet-mcl-effusion-ov.py"
t = pull.read_text(encoding="utf-8")
m = re.search(r'_MILL_B64 = """(.*?)"""', t, re.S)
assert m, "no mill b64"
mill = zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8")
print("HAS_BUNDLE_ONLY", "['bundle_manifest.json']" in mill)
print("HAS_OLD_ASSET", "['bundle_manifest.json', 'knee-mri-fold-weights']" in mill)
idx = mill.find("ASSET = _boot_find_dir")
print("ASSET_BLOCK:\n", mill[idx:idx + 500])

st = subprocess.run([kaggle, "kernels", "status", slug], capture_output=True, text=True)
print("STATUS", (st.stdout or st.stderr or "").strip())
p = subprocess.run([kaggle, "kernels", "logs", slug], capture_output=True)
raw = p.stdout or b""
text = raw.decode("utf-8", "replace")
(root / "artifacts" / "_mcl_eff_ov_logs.txt").write_text(text, encoding="utf-8")
print("LOG_BYTES", len(raw))
if len(raw) > 5:
    try:
        ev = json.loads(text)
        print("N_EVENTS", len(ev), "TMAX", ev[-1].get("time") if ev else None)
        keys = ("mcl-eff-ov", "BOOT", "FileNotFound", "Traceback", "FAIL", "manifest", "START", "END mill", "wrote", "cache", "OVERLAY", "ERROR")
        for e in ev:
            d = e.get("data", "")
            if any(k in d for k in keys):
                print(f"{e.get('time', 0):7.1f} | {d[:230].rstrip()}")
    except Exception as e:
        print("JSON_FAIL", e)
        print(text[:800])

q = subprocess.run([kaggle, "quota"], capture_output=True, text=True)
print(q.stdout)

# show WIN_STATUS head
ws = (root / "WIN_STATUS.md").read_text(encoding="utf-8", errors="replace")
print("WIN_HEAD:\n", ws[:1800])
