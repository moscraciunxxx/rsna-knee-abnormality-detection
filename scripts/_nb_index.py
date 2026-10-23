"""Print notebook cell index (local helper)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    nb = json.loads(path.read_text(encoding="utf-8"))
    print("cells", len(nb["cells"]), path.name)
    for i, cell in enumerate(nb["cells"]):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        head = src.replace("\n", " | ")[:140]
        print(f"--- {i} {cell.get('cell_type')} {len(src)} {head}")


if __name__ == "__main__":
    main()
