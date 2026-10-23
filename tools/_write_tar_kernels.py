"""Write five CPU kernels that tar one cache fold each. Not pushed by this script."""

import json
from pathlib import Path

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MAIN = '''"""Pack one cache fold into vols.tar. CPU only. Not a submission."""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

FOLD = __FOLD__


def find_vols() -> Path:
    root = Path("/kaggle/input")
    hits = [p for p in root.rglob("vols") if p.is_dir() and any(p.glob("*.npz"))]
    if len(hits) != 1:
        raise FileNotFoundError(f"vols dirs {len(hits)}")
    return hits[0]


def main() -> None:
    t0 = time.time()
    vols = find_vols()
    files = sorted(vols.glob("*.npz"))
    out = Path("/kaggle/working/vols.tar")
    with tarfile.open(out, "w") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
    print(
        f"fold {FOLD} files {len(files)} bytes {out.stat().st_size} seconds {time.time() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''

for fold in range(5):
    folder = ROOT / f"kaggle_kernel_cache_tar_f{fold}"
    folder.mkdir(exist_ok=True)
    (folder / "main.py").write_text(MAIN.replace("__FOLD__", str(fold)), encoding="utf-8")
    meta = {
        "id": f"vitaliecervinschi/rsna-cache-tar-f{fold}",
        "title": f"rsna-cache-tar-f{fold}",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [f"vitaliecervinschi/rsna-knee-cache-f{fold}-no-oai"],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": DOCKER,
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", folder)
