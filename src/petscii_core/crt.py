"""
CRT treatment — a numpy port of the web app's post shader.

Kept deliberately faithful to that shader, in the same order and with the same
constants, so a still rendered here and a frame from the live app look like the
same monitor. Every offset is expressed in *native* screen pixels and scaled up
by the render scale, which is what makes the look identical at 1x and at 8x —
defining the glow radius or the scanline pitch in output pixels instead would
make it shrink as you rendered larger.

All zero intensities is a bypass, not a separate path: `apply_crt` returns the
input untouched when nothing is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CrtSettings", "apply_crt"]


@dataclass(frozen=True)
class CrtSettings:
    """Live CRT parameters. All zero (with brightness 1) is a flat image."""

    #: Darkening of alternate scan lines, and the aperture-grille tint with it.
    scanlines: float = 0.45
    #: Barrel distortion, as if the glass were curved.
    curvature: float = 0.35
    #: Phosphor bloom around bright cells.
    glow: float = 0.5
    #: Corner darkening.
    vignette: float = 0.4
    #: Sideways red/blue smear, the way composite video bleeds.
    chroma: float = 0.3
    #: Output level, applied last.
    brightness: float = 1.05

    @property
    def is_identity(self) -> bool:
        return (
            self.scanlines <= 0
            and self.curvature <= 0
            and self.glow <= 0
            and self.vignette <= 0
            and self.chroma <= 0
            and abs(self.brightness - 1.0) < 1e-6
        )


def _sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Bilinear sample at normalised coordinates, clamped at the edges.

    The shader gets this from the sampler; here it is explicit, and it is what
    makes the barrel distortion smooth rather than stair-stepped.

    The four corners are gathered with :func:`np.take` over a flat view rather
    than as ``image[y0, x0]``. Two-dimensional fancy indexing walks its index
    arrays as a pair and re-derives the offset for every element; ``take`` over
    ``(H*W, 3)`` gets one precomputed offset array and copies rows. It is the
    same four corners blended in the same order — bit-identical, and roughly
    three times faster, which matters because ``apply_crt`` calls this seven
    times per frame and the gather is almost the whole cost of the treatment.
    """
    height, width = image.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (width - 1)
    y = np.clip(v, 0.0, 1.0) * (height - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)

    # int64 rows: the product overflows int32 above ~2 gigapixels, and the
    # addition below would wrap silently rather than raise.
    flat = image.reshape(-1, image.shape[2])
    row0 = y0.astype(np.int64) * width
    row1 = y1.astype(np.int64) * width

    return (
        np.take(flat, row0 + x0, axis=0) * (1 - fx) * (1 - fy)
        + np.take(flat, row0 + x1, axis=0) * fx * (1 - fy)
        + np.take(flat, row1 + x0, axis=0) * (1 - fx) * fy
        + np.take(flat, row1 + x1, axis=0) * fx * fy
    ).astype(np.float32)


def apply_crt(image: np.ndarray, settings: CrtSettings, scale: int = 1) -> np.ndarray:
    """
    Applies the CRT treatment to an ``(H, W, 3)`` uint8 image.

    ``scale`` is the integer upscale the image was rendered at, so the effect
    sizes stay tied to the PETSCII grid rather than to the pixel count.
    """
    if settings.is_identity:
        return image

    source = np.asarray(image, dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    scale = max(1, int(scale))

    v_coords, u_coords = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )

    # Barrel: push outward from the centre, so the edges sample past the image.
    if settings.curvature > 0:
        cx = u_coords * 2.0 - 1.0
        cy = v_coords * 2.0 - 1.0
        r2 = cx * cx + cy * cy
        stretch = 1.0 + settings.curvature * 0.25 * r2
        u = cx * stretch * 0.5 + 0.5
        v = cy * stretch * 0.5 + 0.5
    else:
        u, v = u_coords, v_coords

    # Anything the curve pushed off the tube reads as unlit glass.
    outside = (u < 0.0) | (u > 1.0) | (v < 0.0) | (v > 1.0)

    colour = _sample(source, u, v)

    # One native pixel, in normalised units — the unit every offset is quoted in.
    texel_x = scale / width
    texel_y = scale / height

    if settings.chroma > 0:
        shift = texel_x * settings.chroma * 1.5
        red = _sample(source, u + shift, v)[..., 0]
        blue = _sample(source, u - shift, v)[..., 2]
        colour = np.stack([red, colour[..., 1], blue], axis=-1)

    if settings.glow > 0:
        ox = texel_x * 1.5
        oy = texel_y * 1.5
        blur = (
            _sample(source, u + ox, v + oy)
            + _sample(source, u - ox, v + oy)
            + _sample(source, u + ox, v - oy)
            + _sample(source, u - ox, v - oy)
        ) * 0.25
        # Add only where the surroundings are brighter, so dark cells stay dark.
        colour = colour + np.maximum(blur - colour, 0.0) * settings.glow
        colour = colour + blur * blur * settings.glow * 0.35

    if settings.scanlines > 0:
        # One dark band per character row, so the pitch follows the screen.
        line = v * (height / scale)
        wave = 0.5 + 0.5 * np.cos(line * 2.0 * np.pi)
        colour = colour * (1.0 - settings.scanlines * 0.6 * wave)[..., None]

        # Aperture grille: tint across triples of output columns.
        dim = 1.0 - settings.scanlines * 0.08
        grille = np.ones((1, width, 3), dtype=np.float32)
        columns = np.arange(width)
        for channel in range(3):
            grille[0, columns % 3 != channel, channel] = dim
        colour = colour * grille

    if settings.vignette > 0:
        cx = u * 2.0 - 1.0
        cy = v * 2.0 - 1.0
        falloff = np.clip(1.0 - settings.vignette * 0.6 * (cx * cx + cy * cy), 0.0, 1.0)
        colour = colour * falloff[..., None]

    colour = colour * settings.brightness
    colour[outside] = 0.0

    return np.rint(np.clip(colour, 0.0, 1.0) * 255.0).astype(np.uint8)
