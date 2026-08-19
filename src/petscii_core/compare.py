"""
The fixture comparator of core-spec §7 — the definition of "these two ports agree".

Mirrors `web/src/engine/compare.ts`. A mismatched cell is not automatically a
failure: float32 rounding decides which of two almost-equal glyph choices wins, and
wherever a cell's foreground equals the background *every* glyph paints solid
background, so the argmin is a genuine tie. What matters is the picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data_loader import CELL_PIXELS, charset_bytes
from .engine import CELLS, PetsciiFrame, cell_error_of, distance_tables, masked_sums

#: Relative-error tolerance for a near-tie (core-spec §7.3).
TIE_TOLERANCE = 1e-4
#: Minimum fraction of pixel-identical cells (core-spec §7).
RENDER_TARGET = 0.995

__all__ = ["ComparisonResult", "compare_frames", "renders_identically"]


@dataclass
class ComparisonResult:
    identical: int
    same_render: int
    tolerated: int
    divergent: int
    bg_matches: bool
    #: Cell indices that diverged, for error messages.
    divergent_cells: list[int] = field(default_factory=list)

    @property
    def render_fraction(self) -> float:
        return (self.identical + self.same_render) / CELLS

    @property
    def identical_fraction(self) -> float:
        return self.identical / CELLS

    @property
    def passed(self) -> bool:
        return self.bg_matches and self.divergent == 0 and self.render_fraction >= RENDER_TARGET

    def describe(self) -> str:
        return (
            f"{self.identical + self.same_render}/{CELLS} pixel-identical "
            f"({self.render_fraction * 100:.2f}%; {self.identical} exact, "
            f"{self.same_render} same-render), {self.tolerated} near-ties tolerated, "
            f"{self.divergent} divergent, bg {'matches' if self.bg_matches else 'DIFFERS'}"
        )


def renders_identically(
    charset: int,
    code_a: np.ndarray,
    fg_a: np.ndarray,
    code_b: np.ndarray,
    fg_b: np.ndarray,
    bg: int,
) -> np.ndarray:
    """
    Elementwise: do these two choices paint the same 64 pixels under ``bg``?

    Different choices very often *are* the same picture — see the module docstring.
    """
    codes = charset_bytes(charset)
    bits_a = np.unpackbits(codes[code_a], axis=1).reshape(-1, CELL_PIXELS).astype(bool)
    bits_b = np.unpackbits(codes[code_b], axis=1).reshape(-1, CELL_PIXELS).astype(bool)
    painted_a = np.where(bits_a, fg_a[:, None], bg)
    painted_b = np.where(bits_b, fg_b[:, None], bg)
    return (painted_a == painted_b).all(axis=1)


def compare_frames(
    oklab: np.ndarray,
    expected: PetsciiFrame,
    actual: PetsciiFrame,
) -> ComparisonResult:
    """
    Compares two conversions of the same source.

    ``oklab`` must be the working buffer both engines saw (core-spec §5); the
    comparator recomputes the errors itself rather than trusting either side.
    """
    if expected.charset != actual.charset:
        raise ValueError("cannot compare frames converted with different charsets")

    bg_matches = expected.bg == actual.bg
    exact = (expected.screen == actual.screen) & (expected.color == actual.color)

    # Errors are only comparable under a shared background; if the backgrounds
    # differ the frame has already failed, and scoring everything under the
    # expected one keeps the report useful.
    same_picture = renders_identically(
        expected.charset,
        expected.screen,
        expected.color,
        actual.screen,
        actual.color,
        expected.bg,
    )
    same_render = same_picture & ~exact

    unresolved = np.flatnonzero(~exact & ~same_picture)
    tolerated = 0
    divergent_cells: list[int] = []

    if unresolved.size:
        d, s = distance_tables(oklab)
        a = masked_sums(d, expected.charset)
        err_expected = cell_error_of(
            a, s, unresolved, expected.screen[unresolved].astype(np.int64), expected.color[unresolved], expected.bg
        )
        err_actual = cell_error_of(
            a, s, unresolved, actual.screen[unresolved].astype(np.int64), actual.color[unresolved], expected.bg
        )
        scale = np.maximum(np.abs(err_expected), np.abs(err_actual))
        scale = np.where(scale > 0, scale, np.finfo(np.float32).tiny)
        relative = np.abs(err_expected - err_actual) / scale
        near_tie = relative < TIE_TOLERANCE
        tolerated = int(near_tie.sum())
        divergent_cells = [int(c) for c in unresolved[~near_tie]]

    return ComparisonResult(
        identical=int(exact.sum()),
        same_render=int(same_render.sum()),
        tolerated=tolerated,
        divergent=len(divergent_cells),
        bg_matches=bg_matches,
        divergent_cells=divergent_cells,
    )
