#!/usr/bin/env python3
"""
Copies `shared/*.json` into the package so it ships inside the wheel.

`shared/` at the repo root is the single source of truth; the copy under
`src/petscii_core/data/` is a mirror that exists because a wheel cannot reach
outside its package. `tests/test_data.py` fails if the two drift, so a forgotten
sync is a red test rather than a silent parity break.
"""

import shutil
from pathlib import Path

FILES = ("palette.json", "charset.json", "subsets.json")
ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
DEST = ROOT / "src" / "petscii_core" / "data"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(SHARED / name, DEST / name)
        print(f"synced {name}")


if __name__ == "__main__":
    main()
