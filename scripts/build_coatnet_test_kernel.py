"""Test-mode copy of the coatnet870 pipeline.

The 870 gate measured this stack against mill (macro 0.9361 vs 0.9262) and
licensed only MCL+Effusion at 0.15 onto mill. A second full mill pass does not
fit beside this 6.9h stack inside a 9h notebook. This kernel is the stack
itself on the real test mount, not the 870 symlinks, and it keeps
submission.csv. Select it only if publicScore is strictly greater than 0.936.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_coatnet870_gate" / "main.py"
META = ROOT / "kaggle_kernel_coatnet870_gate" / "kernel-metadata.json"
DST = ROOT / "kaggle_kernel_coatnet_test"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    call = "\n_gate_run()\n"
    if call not in text:
        raise SystemExit("holdout _gate_run() call missing; refusing to build a test kernel")
    text = text.replace(
        call,
        "\nprint('[coatnet-test] real competition test mount. Not the 870 holdout. Select only if publicScore > 0.936.', flush=True)\n",
        1,
    )
    if "\n_gate_run()\n" in text:
        raise SystemExit("holdout staging call still present")
    marker = "# <<<COATNET870_TAIL"
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("holdout tail marker missing")
    text = text[:idx] + (
        "print('[coatnet-test] leaving submission.csv for the hidden test. "
        "Do not select unless publicScore > 0.936.', flush=True)\n"
    )
    if "removed submission.csv" in text:
        raise SystemExit("test kernel still deletes submission.csv")
    DST.mkdir(exist_ok=True)
    (DST / "main.py").write_text(text, encoding="utf-8")
    meta = json.loads(META.read_text(encoding="utf-8"))
    meta["id"] = "vitaliecervinschi/rsna-knee-coatnet-test-v1"
    meta["title"] = "RSNA Knee coatnet test v1"
    meta["enable_internet"] = False
    meta["enable_gpu"] = True
    meta["machine_shape"] = "NvidiaTeslaT4"
    meta["is_private"] = True
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DST / 'main.py'} bytes={len(text)}")


if __name__ == "__main__":
    main()
