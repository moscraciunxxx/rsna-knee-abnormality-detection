"""Copy V10 mill and apply tta-max: DINO window-max + CoAtNet hflip on PF OA/Effusion."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
FUSION = ROOT / "rsna_knee" / "fusion.py"
DST = ROOT / "kaggle_kernel_ttamax"
STAGE = Path(__file__).with_name("ttamax_stage.py.inc")

HELPER = '''
def _ttamax_mix_coatnet(base_pred, flip_pred, weight=0.15):
    out = np.asarray(base_pred, dtype=np.float64).copy()
    flip = np.asarray(flip_pred, dtype=np.float64)
    for name in ("PF OA", "Effusion"):
        j = LAB.index(name)
        out[j] = (1.0 - weight) * out[j] + weight * flip[j]
    return out.astype(np.float32, copy=False)

'''

OLD_POOL = (
    "TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}"
)
NEW_POOL = (
    "TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean', "
    "'PF OA': 'max', 'Effusion': 'max'}"
)

OLD_INFER_SOLO = '''                primary_prediction = infer_probs(
                    primary_model,
                    primary_windows,
                    primary_device,
                )'''

NEW_INFER_SOLO = '''                primary_prediction = infer_probs(
                    primary_model,
                    primary_windows,
                    primary_device,
                )
                primary_prediction = _ttamax_mix_coatnet(
                    primary_prediction,
                    infer_probs(
                        primary_model,
                        primary_windows.flip(-1),
                        primary_device,
                    ),
                )'''

OLD_INFER_PAR = '''                primary_prediction = (
                    primary_future.result()
                )'''

NEW_INFER_PAR = '''                primary_prediction = (
                    primary_future.result()
                )
                primary_prediction = _ttamax_mix_coatnet(
                    primary_prediction,
                    infer_probs(
                        primary_model,
                        primary_windows.flip(-1),
                        primary_device,
                    ),
                )'''


def main() -> None:
    mill = SRC.read_text(encoding="utf-8")
    if OLD_POOL not in mill:
        raise SystemExit("TTA_TARGET_POOL marker missing")
    mill = mill.replace(OLD_POOL, NEW_POOL, 1)
    if "def infer_probs(model, xwins, device):" not in mill:
        raise SystemExit("infer_probs missing")
    mill = mill.replace(
        "    return _forward()\n\n\ndef rankpct(x):",
        "    return _forward()\n\n" + HELPER + "\ndef rankpct(x):",
        1,
    )
    if OLD_INFER_PAR not in mill:
        raise SystemExit("parallel infer marker missing")
    mill = mill.replace(OLD_INFER_PAR, NEW_INFER_PAR, 1)
    if OLD_INFER_SOLO not in mill:
        raise SystemExit("solo infer marker missing")
    mill = mill.replace(OLD_INFER_SOLO, NEW_INFER_SOLO, 1)
    fusion = FUSION.read_text(encoding="utf-8")
    stage = STAGE.read_text(encoding="utf-8").replace("___FUSION_PY___", fusion)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "main.py").write_text(mill.rstrip() + "\n\n" + stage.lstrip(), encoding="utf-8")
    print("wrote", DST / "main.py", "bytes", (DST / "main.py").stat().st_size)


if __name__ == "__main__":
    main()
