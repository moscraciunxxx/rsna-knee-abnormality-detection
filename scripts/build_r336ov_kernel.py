"""Copy V10 mill and append DINOv2-B r336 overlay on PF OA + Synovitis."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
FUSION = ROOT / "rsna_knee" / "fusion.py"
DST = ROOT / "kaggle_kernel_r336ov"
STAGE = Path(__file__).with_name("r336ov_stage.py.inc")

OLD_DINO_BOOT = """DINO = _boot_find_dir(
    ['config.json'],"""
NEW_DINO_BOOT = """DINO = _boot_find_dir(
    ['pytorch_model.bin', 'model.safetensors'],"""


def main() -> None:
    mill = SRC.read_text(encoding="utf-8")
    if OLD_DINO_BOOT not in mill:
        raise SystemExit("dino boot marker missing")
    mill = mill.replace(OLD_DINO_BOOT, NEW_DINO_BOOT, 1)
    fusion = FUSION.read_text(encoding="utf-8")
    stage = STAGE.read_text(encoding="utf-8").replace("___FUSION_PY___", fusion)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "main.py").write_text(mill.rstrip() + "\n\n" + stage.lstrip(), encoding="utf-8")
    print("wrote", DST / "main.py", "bytes", (DST / "main.py").stat().st_size)


if __name__ == "__main__":
    main()
