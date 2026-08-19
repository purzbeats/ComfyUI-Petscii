"""
The packaged copy of `shared/` must not drift from the repo copy.

`shared/` is the single source of truth for both ports (PLAN §1); the package
mirrors it so the data ships inside the wheel. A forgotten `sync_shared.py` is a
red test here rather than a silent parity break months later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from petscii_core.data_loader import (
    DATA_DIR,
    charset_bytes,
    glyph_masks,
    palette_names,
    palette_oklab,
    palette_rgb8,
    subset_mask,
    subset_names,
)

from .paths import shared_dir

SHARED = shared_dir()
FILES = ("palette.json", "charset.json", "subsets.json")


@pytest.mark.parametrize("name", FILES)
def test_packaged_copy_matches_shared(name: str) -> None:
    packaged = json.loads((DATA_DIR / name).read_text())
    source = json.loads((SHARED / name).read_text())
    assert packaged == source, f"{name} has drifted — run `python sync_shared.py`"


def test_palette_is_the_pepto_sixteen() -> None:
    assert palette_rgb8().shape == (16, 3)
    assert tuple(palette_rgb8()[0]) == (0, 0, 0)
    assert tuple(palette_rgb8()[1]) == (255, 255, 255)
    assert palette_names()[0] == "black"
    assert palette_names()[15] == "light grey"


def test_palette_oklab_matches_a_direct_conversion() -> None:
    from petscii_core.oklab import srgb_to_oklab

    direct = srgb_to_oklab(palette_rgb8().astype(np.float32) / 255.0)
    assert palette_oklab() == pytest.approx(direct, abs=1e-6)


def test_greys_order_by_lightness() -> None:
    lightness = palette_oklab()[:, 0]
    # black < dark grey < grey < light grey < white
    assert lightness[0] < lightness[11] < lightness[12] < lightness[15] < lightness[1]


def test_charset_has_both_sets() -> None:
    for charset in (0, 1):
        assert charset_bytes(charset).shape == (256, 8)
        assert glyph_masks(charset).shape == (128, 64)


def test_charset_blank_glyphs() -> None:
    """Space (32) and shifted space (96) are the only blanks in a C64 set."""
    for charset in (0, 1):
        rows = charset_bytes(charset)[:128]
        blanks = np.flatnonzero((rows == 0).all(axis=1))
        assert blanks.tolist() == [32, 96]


def test_subsets_are_present_and_ordered() -> None:
    names = subset_names()
    assert names == ("all", "blocks", "dither", "lines", "text")
    assert subset_mask("all").sum() == 256
    for name in names:
        assert 0 < subset_mask(name).sum() <= 256


def test_unknown_subset_falls_back_to_all() -> None:
    # A stale workflow should degrade, not fail.
    assert subset_mask("nonsense").sum() == 256


def test_blocks_are_rectangle_mosaics() -> None:
    """Every code in `blocks` is built from 4px-quantised rows."""
    rows = charset_bytes(0)
    for code in np.flatnonzero(subset_mask("blocks")):
        assert set(rows[code].tolist()) <= {0x00, 0x0F, 0xF0, 0xFF}


def test_dither_spans_the_coverage_ramp() -> None:
    """The dither subset exists to give an even tonal ramp (§6)."""
    rows = charset_bytes(0)
    codes = np.flatnonzero(subset_mask("dither"))
    coverage = np.array([int(np.unpackbits(rows[c]).sum()) for c in codes])
    assert coverage.min() == 0  # space
    assert coverage.max() == 64  # full block
    # At least six distinct levels, spread across the range.
    assert len(set(coverage.tolist())) >= 6


def test_text_is_letters_and_their_inverses() -> None:
    mask = subset_mask("text")
    assert mask[:64].all()
    assert mask[128:192].all()
    assert not mask[64:128].any()
