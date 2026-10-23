"""Dump selected notebook cells to stdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    want = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if want is not None and i not in want:
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        print(f"\n{'=' * 80}\nCELL {i} {cell.get('cell_type')} {len(src)}\n{'=' * 80}")
        print(src)


if __name__ == "__main__":
    main()
