"""Flatten prvsiyan H&S V41 hybrid notebook into a T4-safe Kaggle script."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = Path(r"C:\Users\moscr\AppData\Local\Temp\rsna-hs2\head-and-shoulders-knees-and-toes.ipynb")
OUT_DIR = ROOT / "kaggle_kernel_hs_v41"
BLEND_BOOT = ROOT / "kaggle_kernel_blend_script" / "main.py"

BOOT_END = "# Hide P100 (sm_60) from current Kaggle torch wheels — T4 is sm_75."

PREAMBLE = r'''#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys
def _gpu_is_t4_or_newer():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT,
        )
        print("nvidia-smi:", out.strip(), flush=True)
        bad = ("P100" in out) or ("6.0" in out)
        good = any(x in out for x in ("T4", "V100", "A100", "A10", "L4", "7.5", "8.0", "8.6", "8.9", "9.0"))
        return (not bad) or good
    except Exception as e:
        print("nvidia-smi probe failed:", e, flush=True)
        return True
if not _gpu_is_t4_or_newer():
    print("Hiding pre-Volta GPU before importing torch", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ===== notebook cell =====

from pathlib import Path as _BootPath
import os as _boot_os

def _boot_find_test_root():
    cands = [
        _BootPath('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        _BootPath('/kaggle/input/rsna-knee-abnormality-detection'),
    ]
    for p in cands:
        if (p / 'test.csv').is_file():
            return p
    base = _BootPath('/kaggle/input')
    if base.is_dir():
        for child in base.iterdir():
            if not child.is_dir() or child.name.startswith('1.2.'):
                continue
            if (child / 'test.csv').is_file():
                return child
            for gchild in child.iterdir():
                if gchild.is_dir() and (gchild / 'test.csv').is_file():
                    return gchild
    raise FileNotFoundError('competition test.csv not mounted')

def _boot_find_dir(markers, extra=()):
    cands = list(extra)
    base = _BootPath('/kaggle/input')
    if base.is_dir():
        stack = [base]
        depth = 0
        while stack and depth < 80:
            cur = stack.pop(0)
            depth += 1
            try:
                kids = list(cur.iterdir())
            except OSError:
                continue
            names = {k.name for k in kids}
            if any(m in names for m in markers):
                return cur
            for k in kids:
                if k.is_dir() and k.name not in ('test_series', 'train_series', 'test_images') and not k.name.startswith('1.2.'):
                    stack.append(k)
    for p in cands:
        p = _BootPath(p)
        if p.is_dir():
            return p
    return None

ROOT = _boot_find_test_root()
COMP = ROOT
ASSET = _boot_find_dir(
    ['bundle_manifest.json', 'knee-mri-fold-weights'],
    extra=(
        '/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets',
        '/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets',
    ),
)
DINO = _boot_find_dir(
    ['config.json'],
    extra=(
        '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1',
        '/kaggle/input/dinov2/pytorch/small/1',
    ),
)
print('BOOT ROOT', ROOT, flush=True)
print('BOOT ASSET', ASSET, flush=True)
print('BOOT DINO', DINO, flush=True)
if ASSET is None:
    raise FileNotFoundError('DINOv3 reproduction assets not mounted')
if DINO is None:
    raise FileNotFoundError('DINOv2 small weights not mounted')

# Hide P100 (sm_60) from current Kaggle torch wheels — T4 is sm_75.
import torch as _boot_torch
if _boot_torch.cuda.is_available():
    _cap = _boot_torch.cuda.get_device_capability(0)
    _name = _boot_torch.cuda.get_device_name(0)
    print(f'BOOT GPU {_name} cap={_cap}', flush=True)
    if _cap[0] < 7:
        print('BOOT hiding incompatible GPU; this ensemble needs T4+', flush=True)
        _boot_os.environ['CUDA_VISIBLE_DEVICES'] = ''
        try:
            _boot_torch.cuda.is_available = lambda: False
        except Exception:
            pass

'''

FIND_WEIGHT_PATCH = '''
    direct = [f"/kaggle/input/raptor-knee-maxspan/{fname}",
              f"/kaggle/input/raptor-knee-native384dense/{fname}",
              f"/kaggle/input/raptor-knee-native384/{fname}",
              f"/kaggle/input/raptor-knee-widedense/{fname}",
              f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}",
              f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/{fname}",
              f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense/{fname}",
              f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384/{fname}",
              f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-widedense/{fname}",
              f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-arms/{fname}"]
'''

META = {
    "id": "vitaliecervinschi/rsna-knee-hs-v41-hybrid",
    "title": "RSNA Knee HS V41 hybrid",
    "code_file": "main.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-native384",
        "dreaddevelopment/raptor-knee-native384dense",
        "dreaddevelopment/raptor-knee-widedense",
        "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
    ],
    "kernel_sources": [
        "romantamrazov/rsna-knee-dinosaur-v3-train",
    ],
    "competition_sources": [
        "rsna-knee-abnormality-detection",
    ],
    "model_sources": [
        "metaresearch/dinov2/PyTorch/small/1",
    ],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
}


def strip_cell(src: str) -> str:
    src = re.sub(r"^from __future__ import annotations\s*", "", src)
    src = re.sub(r"^#!/usr/bin/env python3\s*", "", src)
    src = src.replace(
        "ASSET = Path('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets')\n"
        "ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')\n"
        "DINO = Path('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1')\n",
        "# ASSET / ROOT / DINO set by boot path discovery\n",
    )
    src = src.replace(
        "COMP = Path('/kaggle/input/rsna-knee-abnormality-detection')\n",
        "# COMP set by boot path discovery\n",
    )
    # broaden weight search: datasets/user/slug mounts
    old_direct = '''    direct = [f"/kaggle/input/raptor-knee-maxspan/{fname}",
              f"/kaggle/input/raptor-knee-native384dense/{fname}",
              f"/kaggle/input/raptor-knee-native384/{fname}",
              f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}"]'''
    if old_direct in src:
        src = src.replace(old_direct, FIND_WEIGHT_PATCH.rstrip())
    old_glob = '    for d in sorted(glob.glob("/kaggle/input/*/")):'
    new_glob = (
        '    _weight_roots = sorted(glob.glob("/kaggle/input/*/")) + '
        'sorted(glob.glob("/kaggle/input/*/*/")) + '
        'sorted(glob.glob("/kaggle/input/*/*/*/"))\n'
        '    for d in _weight_roots:'
    )
    if old_glob in src:
        src = src.replace(old_glob, new_glob)
    return src


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    parts = [PREAMBLE]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        parts.append(f"\n\n# ===== notebook cell {i} =====\n")
        parts.append(strip_cell(src))
        parts.append("\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(parts)
    # syntax guard: __future__ only at file start
    n_future = len(re.findall(r"^from __future__ import", body, flags=re.M))
    if n_future != 1:
        raise SystemExit(f"expected 1 __future__ import, found {n_future}")
    (OUT_DIR / "main.py").write_text(body, encoding="utf-8", newline="\n")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(META, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("wrote", OUT_DIR / "main.py", "bytes", len(body.encode("utf-8")))
    print("future imports", n_future)
    print("native384" in body, "find_weight patched", "datasets/dreaddevelopment/raptor-knee-native384" in body)


if __name__ == "__main__":
    main()
