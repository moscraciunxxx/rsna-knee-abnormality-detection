import subprocess
K=r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
r=subprocess.run([K,"competitions","submissions","rsna-knee-abnormality-detection","--page-size","5"],capture_output=True); print(r.stdout.decode("utf-8","replace")[:3000])
