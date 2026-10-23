"""Copy V10 mill script and append DINOv3-S+ MCL/Baker's replace stage."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
FUSION = ROOT / "rsna_knee" / "fusion.py"
DST_DIR = ROOT / "kaggle_kernel_dino3s"
STAGE_PATH = Path(__file__).with_name("dino3s_stage.py.inc")


def main() -> None:
    mill = SRC.read_text(encoding="utf-8")
    # v13.1 dataset also has config.json; mill boot must not treat it as DINOv2.
    mill = mill.replace(
        "DINO = _boot_find_dir(\n    ['config.json'],",
        "DINO = _boot_find_dir(\n    ['pytorch_model.bin', 'model.safetensors'],",
        1,
    )
    fusion = FUSION.read_text(encoding="utf-8")
    stage = STAGE_PATH.read_text(encoding="utf-8").replace("___FUSION_PY___", fusion)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / "main.py").write_text(mill.rstrip() + "\n\n" + stage.lstrip(), encoding="utf-8")
    print("wrote", DST_DIR / "main.py", "bytes", (DST_DIR / "main.py").stat().st_size)


if __name__ == "__main__":
    main()
