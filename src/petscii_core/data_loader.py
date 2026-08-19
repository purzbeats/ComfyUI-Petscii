"""
The shared data of `shared/` (core-spec §2, §6), mirrored into the package so it
ships inside the wheel. `sync_shared.py` refreshes it; `tests/test_data.py` fails
if it drifts from the repo copy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .oklab import srgb_to_oklab

DATA_DIR = Path(__file__).resolve().parent / "data"

PALETTE_SIZE = 16
GLYPHS = 128
CELL = 8
CELL_PIXELS = CELL * CELL


@lru_cache(maxsize=None)
def _load(name: str) -> Any:
    with (DATA_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def palette_rgb8() -> np.ndarray:
    """The Pepto palette as ``(16, 3)`` uint8 (core-spec §2)."""
    colors = _load("palette.json")["colors"]
    out = np.zeros((PALETTE_SIZE, 3), dtype=np.uint8)
    for entry in colors:
        out[entry["index"]] = entry["rgb"]
    return out


@lru_cache(maxsize=1)
def palette_names() -> tuple[str, ...]:
    colors = sorted(_load("palette.json")["colors"], key=lambda c: c["index"])
    return tuple(c["name"] for c in colors)


@lru_cache(maxsize=1)
def palette_oklab() -> np.ndarray:
    """The palette pre-converted to Oklab, ``(16, 3)`` float32."""
    return srgb_to_oklab(palette_rgb8().astype(np.float32) / 255.0)


@lru_cache(maxsize=2)
def charset_bytes(charset: int) -> np.ndarray:
    """
    Row bytes for all 256 screen codes of one set, ``(256, 8)`` uint8.

    Codes 128-255 are derived by bitwise inversion (core-spec §1).
    """
    glyphs = _load("charset.json")["sets"][charset]["glyphs"]
    low = np.array(glyphs, dtype=np.uint8)
    return np.concatenate([low, (~low) & 0xFF], axis=0)


@lru_cache(maxsize=2)
def glyph_masks(charset: int) -> np.ndarray:
    """
    Per-pixel 0/1 masks for codes 0-127, ``(128, 64)`` float32 — the ``M`` of
    core-spec §4.2. Inverse codes are handled by the algebra of §4.3, not by a
    second table.
    """
    rows = charset_bytes(charset)[:GLYPHS]
    bits = np.unpackbits(rows, axis=1)  # MSB first == leftmost pixel
    return bits.reshape(GLYPHS, CELL_PIXELS).astype(np.float32)


@lru_cache(maxsize=1)
def subset_names() -> tuple[str, ...]:
    return tuple(_load("subsets.json")["subsets"].keys())


@lru_cache(maxsize=None)
def subset_mask(name: str) -> np.ndarray:
    """
    The 256-entry boolean mask for a named subset (core-spec §6). Falls back to
    ``all`` for an unknown name so a stale workflow degrades instead of failing.
    """
    subsets = _load("subsets.json")["subsets"]
    entry = subsets.get(name) or subsets["all"]
    mask = np.zeros(256, dtype=bool)
    codes = np.asarray(entry["codes"], dtype=np.int64)
    mask[codes[(codes >= 0) & (codes < 256)]] = True
    return mask


def subset_description(name: str) -> str:
    subsets = _load("subsets.json")["subsets"]
    entry = subsets.get(name)
    return entry["description"] if entry else ""
