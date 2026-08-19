"""
Rendering a PETSCII frame back to pixels.

`np.kron` does the integer upscale, so nothing here rasterizes text or depends on
PIL — the glyph bitmasks are the only font data involved, exactly as in the web
renderer.
"""

from __future__ import annotations

import numpy as np

from .data_loader import CELL, CELL_PIXELS, charset_bytes, palette_rgb8
from .engine import COLS, ROWS, SCREEN_H, SCREEN_W, PetsciiFrame

__all__ = ["render_frame", "render_frames", "add_border", "scanline_overlay"]


def render_frame(frame: PetsciiFrame, scale: int = 1) -> np.ndarray:
    """
    Renders to ``(200 * scale, 320 * scale, 3)`` uint8.

    Nearest-neighbour by construction: any other filter would reintroduce exactly
    the resampling the pipeline exists to avoid.
    """
    codes = charset_bytes(frame.charset)
    # (1000, 64) foreground/background selector for the cells actually used.
    bits = np.unpackbits(codes[frame.screen], axis=1).reshape(-1, CELL_PIXELS).astype(bool)

    palette = palette_rgb8()
    fg = palette[frame.color]  # (1000, 3)
    pixels = np.where(bits[:, :, None], fg[:, None, :], palette[frame.bg][None, None, :])

    image = (
        pixels.reshape(ROWS, COLS, CELL, CELL, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(SCREEN_H, SCREEN_W, 3)
        .astype(np.uint8)
    )
    if scale > 1:
        image = np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))
    return image


def render_frames(frames: list[PetsciiFrame], scale: int = 1) -> np.ndarray:
    """Renders a sequence to ``(N, H, W, 3)`` uint8."""
    if not frames:
        return np.zeros((0, SCREEN_H * scale, SCREEN_W * scale, 3), dtype=np.uint8)
    return np.stack([render_frame(frame, scale) for frame in frames])


def add_border(image: np.ndarray, border_index: int, thickness: int) -> np.ndarray:
    """Surrounds a rendered image with the border colour."""
    if thickness <= 0:
        return image
    height, width = image.shape[:2]
    out = np.empty((height + thickness * 2, width + thickness * 2, 3), dtype=np.uint8)
    out[:] = palette_rgb8()[border_index & 15]
    out[thickness : thickness + height, thickness : thickness + width] = image
    return out


def scanline_overlay(image: np.ndarray, strength: float = 0.35, scale: int = 1) -> np.ndarray:
    """
    Darkens one row in every ``scale`` — the CRT look reduced to the one part that
    survives being a still. The full treatment is the web app's shader.
    """
    if strength <= 0 or scale < 2:
        return image
    out = image.astype(np.float32)
    out[scale - 1 :: scale] *= 1.0 - float(np.clip(strength, 0.0, 1.0))
    return np.rint(out).astype(np.uint8)
