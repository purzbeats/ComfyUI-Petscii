"""
The conversion engine — core-spec §4-5, pure numpy.

Mirrors the TypeScript reference in `web/src/engine/cpu/` step for step, including
the reduction that makes the background search affordable: because §4.3 adds only
background-constant terms, each glyph's best foreground and its cost are the same
for every candidate and can be found once (see :func:`glyph_minima`). In numpy
that reduction also happens to be the natural vectorization, which is a good sign
it is the right shape for the algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .data_loader import (
    CELL,
    CELL_PIXELS,
    GLYPHS,
    PALETTE_SIZE,
    charset_bytes,
    glyph_masks,
    palette_oklab,
    palette_rgb8,
    subset_mask,
)
from .oklab import OKLAB_MAX, OKLAB_MIN, srgb_to_oklab

COLS = 40
ROWS = 25
CELLS = COLS * ROWS
SCREEN_W = COLS * CELL
SCREEN_H = ROWS * CELL

__all__ = [
    "CELLS",
    "COLS",
    "ROWS",
    "SCREEN_H",
    "SCREEN_W",
    "PetsciiFrame",
    "Settings",
    "argmin_for_background",
    "cell_error_of",
    "convert",
    "convert_batch",
    "distance_tables",
    "frame_to_screen",
    "glyph_minima",
    "iter_convert_batch",
    "masked_sums",
    "pre_adjust_to_oklab",
    "select_background",
]


@dataclass(frozen=True)
class Settings:
    """Everything the conversion needs beyond the pixels (core-spec §4-5, §9)."""

    charset: int = 0
    subset: str = "all"
    #: Fixed background 0-15, or None to auto-select per §4.5.
    bg_lock: int | None = None
    #: Fixed border, or None to track ``bg``.
    border_lock: int | None = None
    #: Temporal hysteresis threshold per pixel; 0 disables (§4.6).
    eps: float = 0.0
    #: Cell-level Floyd-Steinberg; stills only (§4.7).
    dither: bool = False
    framing: str = "cover"
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0

    def subset_array(self) -> np.ndarray:
        return subset_mask(self.subset)


@dataclass
class PetsciiFrame:
    """One converted screen: true screen codes, one fg per cell, global bg/border."""

    #: Screen codes 0-255, row-major, ``(1000,)`` uint8.
    screen: np.ndarray
    #: Foreground palette indices 0-15, row-major, ``(1000,)`` uint8.
    color: np.ndarray
    bg: int
    border: int
    charset: int = 0

    def as_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """``(screen, color)`` reshaped to ``(25, 40)``."""
        return self.screen.reshape(ROWS, COLS), self.color.reshape(ROWS, COLS)

    def copy(self) -> PetsciiFrame:
        return replace(self, screen=self.screen.copy(), color=self.color.copy())


@dataclass
class HysteresisState:
    """Previous frame's choices, carried across frames to drive §4.6."""

    screen: np.ndarray = field(default_factory=lambda: np.zeros(CELLS, dtype=np.uint8))
    color: np.ndarray = field(default_factory=lambda: np.zeros(CELLS, dtype=np.uint8))
    bg: int = -1
    valid: bool = False


# ---------------------------------------------------------------- framing (§5)


def frame_to_screen(image: np.ndarray, mode: str = "cover", bars: Sequence[int] = (0, 0, 0)) -> np.ndarray:
    """
    Frames an arbitrary ``(H, W, 3)`` uint8 image to exactly 320 x 200 (core-spec §5).

    An already-320 x 200 input passes through untouched, which is what makes the
    fixtures a clean parity contract (core-spec §7).
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")
    src = image[:, :, :3].astype(np.float32)
    height, width = src.shape[:2]

    if (width, height) == (SCREEN_W, SCREEN_H):
        return np.rint(src).astype(np.uint8)

    target = SCREEN_W / SCREEN_H
    aspect = width / height

    sx, sy, sw, sh = 0, 0, width, height
    dx, dy, dw, dh = 0, 0, SCREEN_W, SCREEN_H

    if mode == "cover":
        if aspect > target:
            sw = max(1, int(round(height * target)))
            sx = (width - sw) // 2
        elif aspect < target:
            sh = max(1, int(round(width / target)))
            sy = (height - sh) // 2
    elif mode == "contain":
        if aspect > target:
            dh = max(1, int(round(SCREEN_W / aspect)))
            dy = (SCREEN_H - dh) // 2
        elif aspect < target:
            dw = max(1, int(round(SCREEN_H * aspect)))
            dx = (SCREEN_W - dw) // 2
    elif mode != "stretch":
        raise ValueError(f"unknown framing mode {mode!r}")

    out = np.empty((SCREEN_H, SCREEN_W, 3), dtype=np.float32)
    out[:] = np.asarray(bars, dtype=np.float32)
    out[dy : dy + dh, dx : dx + dw] = _box_resample(src[sy : sy + sh, sx : sx + sw], dw, dh)
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def _box_resample(src: np.ndarray, dw: int, dh: int) -> np.ndarray:
    """Area-average resample (core-spec §5), exact for arbitrary ratios."""
    sh, sw = src.shape[:2]
    rows = _axis_weights(sh, dh) @ src.reshape(sh, -1)
    rows = rows.reshape(dh, sw, 3)
    cols = _axis_weights(sw, dw) @ rows.transpose(1, 0, 2).reshape(sw, -1)
    return cols.reshape(dw, dh, 3).transpose(1, 0, 2)


def _axis_weights(src_len: int, dst_len: int) -> np.ndarray:
    """
    ``(dst_len, src_len)`` matrix whose rows average the exact source footprint of
    each destination sample. Building it once beats looping per pixel and keeps
    the arithmetic identical to the reference implementation's.
    """
    edges = np.arange(dst_len + 1, dtype=np.float64) * (src_len / dst_len)
    lo = edges[:-1][:, None]
    hi = edges[1:][:, None]
    pixels = np.arange(src_len, dtype=np.float64)[None, :]
    overlap = np.clip(np.minimum(hi, pixels + 1) - np.maximum(lo, pixels), 0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def pre_adjust_to_oklab(rgb: np.ndarray, settings: Settings) -> np.ndarray:
    """
    Applies the pre-adjust knobs and converts to the Oklab working buffer (§5).

    brightness/contrast/gamma act on non-linear sRGB in [0, 1]; saturation scales
    the Oklab chroma axes afterwards. The tone curve is built as a 256-entry LUT,
    matching the reference exactly — it is per-channel and channel-independent, so
    the LUT is not an approximation.
    """
    values = np.arange(256, dtype=np.float32) / 255.0
    values = values + settings.brightness
    values = (values - 0.5) * settings.contrast + 0.5
    values = np.clip(values, 0.0, 1.0)
    if settings.gamma != 1.0:
        values = values ** (1.0 / settings.gamma)
    lut = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)

    adjusted = lut[np.asarray(rgb, dtype=np.uint8)]
    lab = srgb_to_oklab(adjusted.astype(np.float32) / 255.0)
    if settings.saturation != 1.0:
        lab[..., 1:] *= settings.saturation
    return np.clip(lab, OKLAB_MIN, OKLAB_MAX)


# ------------------------------------------------------------- tables (§4.1-2)


def _cells_from_image(oklab: np.ndarray) -> np.ndarray:
    """``(200, 320, 3)`` Oklab to ``(1000, 64, 3)``, row-major within each cell."""
    return (
        oklab.reshape(ROWS, CELL, COLS, CELL, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(CELLS, CELL_PIXELS, 3)
    )


def distance_tables(oklab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ``D (1000, 64, 16)`` and ``S (1000, 16)`` of core-spec §4.1.

    The obvious spelling — broadcast every cell pixel against every palette entry,
    then contract — allocates a ``(1000, 64, 16, 3)`` float32 intermediate, 12 MB
    that has to be written and read back before a single distance exists. Walking
    the 16 palette entries instead keeps each intermediate at ``(1000, 64, 3)``,
    small enough to stay in cache, and more than halves the cost of the whole
    conversion. The arithmetic is unchanged, term for term and in the same order,
    so the result is bit-identical to the broadcast form — which is the only way
    a speedup is allowed to touch this function (core-spec §7).

    The expanded form ``|x|² - 2x·p + |p|²`` would be a further 8x as one GEMM,
    and is deliberately not used: it cancels catastrophically as the distance
    approaches zero, which is exactly where §4.4's ties are decided.
    """
    cells = _cells_from_image(oklab)
    palette = palette_oklab()
    d = np.empty((CELLS, CELL_PIXELS, PALETTE_SIZE), dtype=np.float32)
    for k in range(PALETTE_SIZE):
        delta = cells - palette[k]
        d[:, :, k] = (delta * delta).sum(axis=2, dtype=np.float32)
    return d, d.sum(axis=1, dtype=np.float32)


def masked_sums(d: np.ndarray, charset: int) -> np.ndarray:
    """``A (1000, 128, 16)`` of core-spec §4.2."""
    return np.einsum("cpk,gp->cgk", d, glyph_masks(charset), optimize=True).astype(np.float32)


def glyph_minima(a: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Best foreground per glyph for both branches of §4.3.

    ``argmin`` returns the first occurrence, which is the spec's "lowest colour
    index on a tie" (§4.4) for free.
    """
    min_normal = a.min(axis=2)
    fg_normal = a.argmin(axis=2).astype(np.uint8)
    # The inverse glyph paints f outside the mask: S[f] - A[g][f].
    inverse = s[:, None, :] - a
    min_inverse = inverse.min(axis=2)
    fg_inverse = inverse.argmin(axis=2).astype(np.uint8)
    return min_normal, fg_normal, min_inverse, fg_inverse


def _errors_for_background(
    a: np.ndarray,
    s: np.ndarray,
    min_normal: np.ndarray,
    min_inverse: np.ndarray,
    bg: int,
    allowed: np.ndarray,
) -> np.ndarray:
    """
    ``(1000, 256)`` cell error per screen code under one background (§4.3).

    Codes 0-127 occupy the first half and 128-255 the second, so the array index
    *is* the screen code and ``argmin`` breaks ties toward the lower one (§4.4).
    """
    a_bg = a[:, :, bg]
    normal = min_normal + s[:, bg : bg + 1] - a_bg
    inverse = min_inverse + a_bg
    err = np.concatenate([normal, inverse], axis=1)
    return np.where(allowed[None, :], err, np.inf)


def argmin_for_background(
    a: np.ndarray,
    s: np.ndarray,
    minima: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bg: int,
    allowed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ``(screen, color, error)`` for one background (§4.3-4.4)."""
    min_normal, fg_normal, min_inverse, fg_inverse = minima
    err = _errors_for_background(a, s, min_normal, min_inverse, bg, allowed)
    code = err.argmin(axis=1)
    error = err[np.arange(CELLS), code]
    fg = np.where(
        code < GLYPHS,
        fg_normal[np.arange(CELLS), code % GLYPHS],
        fg_inverse[np.arange(CELLS), code % GLYPHS],
    )
    return code.astype(np.uint8), fg.astype(np.uint8), error


def select_background(
    a: np.ndarray,
    s: np.ndarray,
    minima: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    allowed: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    The full 16-candidate search of core-spec §4.5.

    Returns the winning background and the per-candidate totals. ``argmin`` ties
    toward the lower index, as §4.5 requires.
    """
    min_normal, _, min_inverse, _ = minima
    # (1000, 128, 16) for each branch: every glyph against every candidate.
    normal = min_normal[:, :, None] + s[:, None, :] - a
    inverse = min_inverse[:, :, None] + a
    err = np.concatenate([normal, inverse], axis=1)
    err = np.where(allowed[None, :, None], err, np.inf)
    totals = err.min(axis=1).sum(axis=0, dtype=np.float64)
    return int(totals.argmin()), totals


def cell_error_of(
    a: np.ndarray,
    s: np.ndarray,
    cells: np.ndarray,
    codes: np.ndarray,
    colors: np.ndarray,
    bg: int,
) -> np.ndarray:
    """Error of explicit ``(code, fg)`` choices under ``bg`` — both §4.3 branches."""
    g = (codes & 0x7F).astype(np.int64)
    f = colors.astype(np.int64)
    a_bg = a[cells, g, bg]
    a_f = a[cells, g, f]
    normal = a_f + s[cells, bg] - a_bg
    inverse = s[cells, f] - a_f + a_bg
    return np.where(codes < GLYPHS, normal, inverse)


# --------------------------------------------------------------- conversion


def _convert_oklab(
    oklab: np.ndarray,
    settings: Settings,
    state: HysteresisState | None,
) -> PetsciiFrame:
    allowed = settings.subset_array()
    d, s = distance_tables(oklab)
    a = masked_sums(d, settings.charset)
    minima = glyph_minima(a, s)

    if settings.bg_lock is not None:
        bg = int(settings.bg_lock) & 15
    else:
        bg, _ = select_background(a, s, minima, allowed)

    screen, color, error = argmin_for_background(a, s, minima, bg, allowed)

    if state is not None:
        screen, color = _apply_hysteresis(a, s, state, bg, settings.eps, screen, color, error)

    return PetsciiFrame(
        screen=screen,
        color=color,
        bg=bg,
        border=bg if settings.border_lock is None else int(settings.border_lock) & 15,
        charset=settings.charset,
    )


def _apply_hysteresis(
    a: np.ndarray,
    s: np.ndarray,
    state: HysteresisState,
    bg: int,
    eps: float,
    screen: np.ndarray,
    color: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core-spec §4.6: keep the previous choice unless the new one wins by more than
    ``eps * 64``. A background change invalidates the state, so every cell
    re-decides. The state is refreshed either way — it has to be seeded for the
    frame after ``eps`` is turned back up.
    """
    if eps > 0 and state.valid and state.bg == bg:
        cells = np.arange(CELLS)
        prev_error = cell_error_of(a, s, cells, state.screen.astype(np.int64), state.color, bg)
        keep = ~(error < prev_error - eps * CELL_PIXELS)
        screen = np.where(keep, state.screen, screen).astype(np.uint8)
        color = np.where(keep, state.color, color).astype(np.uint8)

    state.screen = screen.copy()
    state.color = color.copy()
    state.bg = bg
    state.valid = True
    return screen, color


def convert(
    image: np.ndarray,
    settings: Settings | None = None,
    state: HysteresisState | None = None,
) -> PetsciiFrame:
    """
    Converts one ``(H, W, 3)`` uint8 image to a PETSCII frame (core-spec §4).

    Pass ``state`` to carry temporal hysteresis across frames; omit it for
    independent stills. Dithering (§4.7) lives in `dither.py` and is selected by
    ``settings.dither``.
    """
    settings = settings or Settings()
    if _adds_letterbox(image, settings):
        settings = replace(settings, bg_lock=_letterbox_background(image, settings))
    framed = frame_to_screen(image, settings.framing, _letterbox_rgb(settings))
    oklab = pre_adjust_to_oklab(framed, settings)
    if settings.dither:
        from .dither import convert_oklab_dithered

        return convert_oklab_dithered(oklab, settings)
    return _convert_oklab(oklab, settings, state)


def iter_convert_batch(
    images: Sequence[np.ndarray],
    settings: Settings | None = None,
    *,
    temporal: bool = False,
    bg_sample: int = 8,
) -> Iterator[PetsciiFrame]:
    """
    Converts a sequence of frames, yielding each as it is finished.

    With ``temporal=False`` each frame is independent — that is what a batch of
    unrelated stills wants. With ``temporal=True`` hysteresis carries across the
    batch in order and the background is voted once over evenly sampled frames
    then locked, because a background that flickers mid-clip is far more visible
    than one that is slightly wrong.

    This is the generator form so a caller can report progress, allow itself to
    be interrupted, or release each source frame between conversions. ``images``
    is indexed rather than iterated, so a lazy sequence over a video buffer never
    has to be materialised — see :func:`convert_batch` for the eager form.
    """
    settings = settings or Settings()
    count = len(images)
    if count == 0:
        return

    if temporal:
        locked = settings.bg_lock
        if locked is None:
            locked = vote_background(images, settings, sample=bg_sample)
        settings = replace(settings, bg_lock=locked)
        state: HysteresisState | None = HysteresisState()
    else:
        state = None

    for index in range(count):
        yield convert(images[index], settings, state)


def convert_batch(
    images: Iterable[np.ndarray],
    settings: Settings | None = None,
    *,
    temporal: bool = False,
    bg_sample: int = 8,
) -> list[PetsciiFrame]:
    """Eager :func:`iter_convert_batch` — the whole sequence as a list."""
    frames = images if isinstance(images, Sequence) else list(images)
    return list(iter_convert_batch(frames, settings, temporal=temporal, bg_sample=bg_sample))


def vote_background(
    frames: Sequence[np.ndarray],
    settings: Settings,
    sample: int = 8,
) -> int:
    """
    Picks one background for a whole clip by summing §4.5's per-candidate totals
    over evenly spaced frames. Sampling rather than scoring every frame keeps this
    from dominating the cost of a long clip; the totals are extremely stable.
    """
    allowed = settings.subset_array()
    count = min(len(frames), max(1, sample))
    indices = np.unique(np.linspace(0, len(frames) - 1, count).round().astype(int))

    totals = np.zeros(PALETTE_SIZE, dtype=np.float64)
    for index in indices:
        framed = frame_to_screen(frames[index], settings.framing, _letterbox_rgb(settings))
        oklab = pre_adjust_to_oklab(framed, settings)
        d, s = distance_tables(oklab)
        a = masked_sums(d, settings.charset)
        _, frame_totals = select_background(a, s, glyph_minima(a, s), allowed)
        totals += frame_totals
    return int(totals.argmin())


def _letterbox_rgb(settings: Settings) -> tuple[int, int, int]:
    index = 0 if settings.bg_lock is None else int(settings.bg_lock) & 15
    return tuple(int(v) for v in palette_rgb8()[index])  # type: ignore[return-value]


def _adds_letterbox(image: np.ndarray, settings: Settings) -> bool:
    """
    Whether ``contain`` framing will actually paint bars for this input.

    Only ``contain`` letterboxes, only a mismatched aspect produces bars, and a
    320 x 200 input short-circuits framing entirely (§5) — so the common cases,
    including every parity fixture, answer False and never pay for :func:`convert`'s
    second pass.
    """
    if settings.framing != "contain" or settings.bg_lock is not None:
        return False
    height, width = np.asarray(image).shape[:2]
    if (width, height) == (SCREEN_W, SCREEN_H):
        return False
    return abs(width / height - SCREEN_W / SCREEN_H) > 1e-9


def _letterbox_background(image: np.ndarray, settings: Settings) -> int:
    """
    The background to paint ``contain``'s bars in, for an auto background.

    Bar colour and background selection are circular: §4.5 picks the background
    from the framed image, but the bars are part of that image and have to be
    painted in *some* colour first. Resolving it takes one probe conversion over
    black bars to learn the background, which the real pass then both paints the
    bars in and locks. One round is enough — a second could only differ if the
    bars outvoted the picture, and bars that dominate the frame are already the
    degenerate case.
    """
    probe = frame_to_screen(image, settings.framing, _letterbox_rgb(settings))
    oklab = pre_adjust_to_oklab(probe, settings)
    d, s = distance_tables(oklab)
    a = masked_sums(d, settings.charset)
    chosen, _ = select_background(a, s, glyph_minima(a, s), settings.subset_array())
    return chosen


def charset_row_bytes(charset: int) -> np.ndarray:
    """Re-exported for the renderer and the `.petv` writer."""
    return charset_bytes(charset)
