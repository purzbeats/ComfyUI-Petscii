"""
The parity contract (core-spec §7): this port must reproduce the fixtures frozen
by the TypeScript reference engine.

If this fails, per PLAN §10 the order is spec first, then the TS engine, then
regenerate the fixtures via `web/freeze.html`, then this port follows — never the
other way round.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from petscii_core import PetsciiFrame, Settings, compare_frames, convert, pre_adjust_to_oklab

from .paths import fixtures_dir
from petscii_core.engine import CELLS, SCREEN_H, SCREEN_W, frame_to_screen

FIXTURES = fixtures_dir()
NAMES = ["gradient", "photo", "portrait", "noise", "ui"]

Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to read the fixture PNGs")


def load_fixture(name: str) -> np.ndarray:
    with Image.open(FIXTURES / f"{name}.png") as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def load_expected(name: str) -> dict:
    return json.loads((FIXTURES / "expected" / f"{name}.json").read_text())


def expected_frame(expected: dict) -> PetsciiFrame:
    return PetsciiFrame(
        screen=np.array(expected["screen"], dtype=np.uint8),
        color=np.array(expected["color"], dtype=np.uint8),
        bg=expected["bg"],
        border=expected["border"],
        charset=expected["charset"],
    )


#: core-spec §9 reference settings, which the fixtures were frozen with.
REFERENCE = Settings()


@pytest.mark.parametrize("name", NAMES)
def test_fixture_is_320x200(name: str) -> None:
    image = load_fixture(name)
    assert image.shape == (SCREEN_H, SCREEN_W, 3)


@pytest.mark.parametrize("name", NAMES)
def test_matches_frozen_reference(name: str) -> None:
    image = load_fixture(name)
    expected = load_expected(name)
    actual = convert(image, REFERENCE)

    assert actual.bg == expected["bg"], f"{name}: background {actual.bg} != {expected['bg']}"
    assert actual.border == expected["border"]
    assert actual.charset == expected["charset"]

    oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
    result = compare_frames(oklab, expected_frame(expected), actual)
    assert result.passed, f"{name}: {result.describe()} (cells {result.divergent_cells[:8]})"


@pytest.mark.parametrize("name", NAMES)
def test_reports_a_clean_comparison(name: str, capsys) -> None:
    """Prints the parity figures, which is what makes a regression legible."""
    image = load_fixture(name)
    expected = load_expected(name)
    actual = convert(image, REFERENCE)
    oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
    result = compare_frames(oklab, expected_frame(expected), actual)
    with capsys.disabled():
        print(f"  {name:9s} {result.describe()}")
    assert result.divergent == 0


def test_comparator_rejects_a_different_background() -> None:
    image = load_fixture("photo")
    frame = convert(image, REFERENCE)
    oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
    wrong = PetsciiFrame(frame.screen.copy(), frame.color.copy(), (frame.bg + 1) % 16, frame.border, frame.charset)
    result = compare_frames(oklab, frame, wrong)
    assert not result.bg_matches
    assert not result.passed


def test_comparator_rejects_genuinely_wrong_cells() -> None:
    image = load_fixture("photo")
    frame = convert(image, REFERENCE)
    oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
    screen = frame.screen.copy()
    # Blank 50 cells: a real, large error rather than a near-tie.
    screen[np.arange(50) * 20] = 32
    broken = PetsciiFrame(screen, frame.color.copy(), frame.bg, frame.border, frame.charset)
    result = compare_frames(oklab, frame, broken)
    assert result.divergent > 0
    assert not result.passed


def test_comparator_excuses_an_invisible_swap() -> None:
    """A cell whose foreground equals the background paints solid bg whatever
    glyph it holds — §7 category 2, not a failure."""
    image = load_fixture("photo")
    frame = convert(image, REFERENCE)
    oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
    candidates = np.flatnonzero((frame.color == frame.bg) & (frame.screen != 32))
    if candidates.size == 0:
        pytest.skip("this fixture has no fg == bg cell")
    screen = frame.screen.copy()
    screen[candidates[0]] = 32
    swapped = PetsciiFrame(screen, frame.color.copy(), frame.bg, frame.border, frame.charset)
    result = compare_frames(oklab, frame, swapped)
    assert result.same_render == 1
    assert result.divergent == 0
    assert result.passed


def test_frame_shape() -> None:
    frame = convert(load_fixture("gradient"), REFERENCE)
    assert frame.screen.shape == (CELLS,)
    assert frame.color.shape == (CELLS,)
    assert frame.color.max() < 16
