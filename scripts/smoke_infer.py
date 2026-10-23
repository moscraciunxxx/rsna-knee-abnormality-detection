#!/usr/bin/env python
"""Launch the shipped 2.5D inference path on a tiny synthetic fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.infer import main as infer_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-studies", type=int, default=4)
    parser.add_argument(
        "--reuse-fixture",
        action="store_true",
        help="Do not rewrite the fixture (second identical launch)",
    )
    args = parser.parse_args(argv)
    flags = ["--input", args.input, "--output", args.output, "--n-studies", str(args.n_studies)]
    if not args.reuse_fixture:
        flags.append("--write-fixture")
    return infer_main(flags)


if __name__ == "__main__":
    raise SystemExit(main())
