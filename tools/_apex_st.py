import subprocess
K=r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
r=subprocess.run([K,"kernels","status","vitaliecervinschi/rsna-knee-apex-copy"],capture_output=True); print(r.stdout.decode("utf-8","replace").strip(), r.stderr.decode("utf-8","replace").strip()[:300])
