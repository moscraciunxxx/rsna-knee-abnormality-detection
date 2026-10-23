"""Pack one cache fold into vols.tar. CPU only. Not a submission."""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

FOLD = 3


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
