"""
Cell-level Floyd-Steinberg error diffusion — core-spec §4.7. Stills only, and
mutually exclusive with temporal hysteresis.

Diffusion makes each cell depend on its predecessors, so this cannot reuse the
vectorized pipeline in `engine.py`: cells are chosen strictly row-major and the
residual is pushed into the Oklab working buffer before the next cell is scored.
Per §4.7 the background is picked on the un-dithered image and then locked.
"""

from __future__ import annotations

import numpy as np

from .data_loader import CELL, CELL_PIXELS, GLYPHS, charset_bytes, glyph_masks, palette_oklab
from .engine import (
    CELLS,
    COLS,
    ROWS,
    PetsciiFrame,
    Settings,
    distance_tables,
    glyph_minima,
    masked_sums,
    select_background,
)
from .oklab import OKLAB_MAX, OKLAB_MIN

#: Floyd-Steinberg weights at cell granularity (§4.7).
NEIGHBOURS = ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))


def convert_oklab_dithered(oklab: np.ndarray, settings: Settings) -> PetsciiFrame:
    """Converts with dithering. ``oklab`` is copied, not modified in place."""
    working = np.array(oklab, dtype=np.float32, copy=True)
    allowed = settings.subset_array()
    masks = glyph_masks(settings.charset)
    codes = charset_bytes(settings.charset)
    palette = palette_oklab()

    # §4.7: background comes from the un-dithered image, then stays fixed.
    if settings.bg_lock is not None:
        bg = int(settings.bg_lock) & 15
    else:
        d, s = distance_tables(working)
        a = masked_sums(d, settings.charset)
        bg, _ = select_background(a, s, glyph_minima(a, s), allowed)

    # Per-pixel palette index for every screen code, so the residual only needs a
    # lookup rather than a bit test per pixel.
    code_bits = np.unpackbits(codes, axis=1).reshape(256, CELL_PIXELS).astype(bool)

    screen = np.zeros(CELLS, dtype=np.uint8)
    color = np.zeros(CELLS, dtype=np.uint8)
    allowed_low = allowed[:GLYPHS]
    allowed_high = allowed[GLYPHS:]

    for cy in range(ROWS):
        for cx in range(COLS):
            cell = _cell_pixels(working, cx, cy)

            delta = cell[:, None, :] - palette[None, :, :]
            d = np.einsum("pkx,pkx->pk", delta, delta, dtype=np.float32)
            s = d.sum(axis=0, dtype=np.float32)
            a = masks @ d  # (128, 16)

            normal = np.where(allowed_low[:, None], a, np.inf)
            inverse = np.where(allowed_high[:, None], s[None, :] - a, np.inf)
            err_normal = normal.min(axis=1) + s[bg] - a[:, bg]
            err_inverse = inverse.min(axis=1) + a[:, bg]
            err = np.concatenate([err_normal, err_inverse])
            code = int(err.argmin())
            fg = int((normal if code < GLYPHS else inverse)[code % GLYPHS].argmin())

            index = cy * COLS + cx
            screen[index] = code
            color[index] = fg

            # Mean Oklab residual of what this cell actually paints (§4.7).
            painted = np.where(code_bits[code][:, None], palette[fg], palette[bg])
            residual = (cell - painted).mean(axis=0)

            for dx, dy, weight in NEIGHBOURS:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= COLS or ny >= ROWS:
                    continue
                _diffuse(working, nx, ny, residual * weight)

    return PetsciiFrame(
        screen=screen,
        color=color,
        bg=bg,
        border=bg if settings.border_lock is None else int(settings.border_lock) & 15,
        charset=settings.charset,
    )


def _cell_pixels(oklab: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """The 64 Oklab pixels of one cell, row-major, as ``(64, 3)``."""
    block = oklab[cy * CELL : (cy + 1) * CELL, cx * CELL : (cx + 1) * CELL]
    return block.reshape(CELL_PIXELS, 3)


def _diffuse(oklab: np.ndarray, cx: int, cy: int, delta: np.ndarray) -> None:
    """Adds ``delta`` to every pixel of one cell, clamped in Oklab (§4.7)."""
    view = oklab[cy * CELL : (cy + 1) * CELL, cx * CELL : (cx + 1) * CELL]
    np.clip(view + delta, OKLAB_MIN, OKLAB_MAX, out=view)


__all__ = ["convert_oklab_dithered", "NEIGHBOURS"]
