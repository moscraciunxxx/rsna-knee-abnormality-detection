"""Find r336 load / variant=base snippets in nursrijan notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\.cache\dinov2b\kernels")
NEEDLES = ("r336", "r224", "variant", "dinov2-base", "annot_auc", "find_dinov2", "load_state")


def main() -> None:
    for nb_path in sorted(ROOT.rglob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            low = src.lower()
            hits = [n for n in NEEDLES if n.lower() in low]
            if not hits:
                continue
            print("=" * 80)
            print(nb_path.name, "cell", i, "hits", hits, "len", len(src))
            for line in src.splitlines():
                if any(n.lower() in line.lower() for n in NEEDLES):
                    print(line[:220])


if __name__ == "__main__":
    main()
