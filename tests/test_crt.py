"""The CRT treatment — a port of the web app's post shader."""

from __future__ import annotations

import numpy as np
import pytest

from petscii_core import CrtSettings, apply_crt, crt, render_frame
from petscii_core.engine import SCREEN_H, SCREEN_W, Settings, convert

FLAT = CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.0)


def only(**effect) -> CrtSettings:
    """
    One effect on, everything else off.

    Each test isolates a single part of the treatment, and spelling out the five
    zeroes every time buried which one was actually under test.
    """
    off = {"scanlines": 0.0, "curvature": 0.0, "glow": 0.0, "vignette": 0.0, "chroma": 0.0}
    return CrtSettings(**{**off, "brightness": 1.0, **effect})


def checkerboard(scale: int = 4) -> np.ndarray:
    """A rendered PETSCII screen, which is what the CRT is designed to receive."""
    rng = np.random.default_rng(7)
    source = (rng.random((SCREEN_H, SCREEN_W, 3)) * 255).astype(np.uint8)
    return render_frame(convert(source, Settings()), scale)


class TestBypass:
    def test_all_zero_is_the_identity(self) -> None:
        image = checkerboard()
        assert apply_crt(image, FLAT, 4) is image

    def test_identity_is_detected_not_computed(self) -> None:
        assert FLAT.is_identity
        assert not CrtSettings().is_identity
        # Brightness alone is enough to make it do something.
        assert not only(brightness=1.2).is_identity


class TestShapeAndRange:
    @pytest.mark.parametrize("scale", [1, 2, 4])
    def test_preserves_shape_and_dtype(self, scale: int) -> None:
        image = checkerboard(scale)
        out = apply_crt(image, CrtSettings(), scale)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_stays_in_range(self) -> None:
        # Bright input with glow and brightness both up is where clipping bites.
        image = np.full((64, 96, 3), 250, dtype=np.uint8)
        out = apply_crt(image, CrtSettings(glow=2.0, brightness=2.0), 4)
        assert out.min() >= 0
        assert out.max() <= 255


class TestEffects:
    def test_scanlines_darken_alternate_rows(self) -> None:
        image = np.full((64, 96, 3), 200, dtype=np.uint8)
        out = apply_crt(image, only(scanlines=0.8), 4)
        # Row means must vary: that variation *is* the scanline.
        row_means = out.reshape(64, -1).mean(axis=1)
        assert row_means.max() - row_means.min() > 20

    def test_scanline_pitch_follows_the_scale(self) -> None:
        """
        The band spacing is one per character row, not one per output pixel, so a
        render at 8x has bands twice as far apart as one at 4x. Defining it in
        output pixels instead would make the look change with the scale.
        """
        image = np.full((256, 96, 3), 200, dtype=np.uint8)
        settings = CrtSettings(scanlines=0.8, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.0)

        def band_count(scale: int) -> int:
            # Count crossings of the mean, not rows below it — a symmetric wave
            # puts half the rows below the mean at *any* pitch, so that would
            # measure nothing.
            rows = apply_crt(image, settings, scale).reshape(256, -1).mean(axis=1)
            above = rows > rows.mean()
            return int(np.count_nonzero(above[1:] != above[:-1]))

        # 256 rows at 4x is twice as many bands as at 8x.
        assert band_count(4) == pytest.approx(band_count(8) * 2, rel=0.15)

    def test_vignette_darkens_the_corners_not_the_centre(self) -> None:
        image = np.full((128, 128, 3), 200, dtype=np.uint8)
        out = apply_crt(image, only(vignette=1.0), 4)
        centre = out[60:68, 60:68].mean()
        corner = out[:8, :8].mean()
        assert corner < centre * 0.75

    def test_curvature_blanks_the_corners(self) -> None:
        """The curve pushes the corners off the tube; that area reads as glass."""
        image = np.full((128, 128, 3), 220, dtype=np.uint8)
        out = apply_crt(image, only(curvature=1.0), 4)
        assert out[0, 0].max() == 0
        assert out[64, 64].min() > 100

    def test_chroma_separates_red_and_blue_at_an_edge(self) -> None:
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:, 48:] = 255  # a hard white edge down the middle
        out = apply_crt(image, only(chroma=1.0), 4)
        # Red and blue are shifted in opposite directions, so they no longer agree
        # in the column either side of the edge.
        assert not np.array_equal(out[:, 44:52, 0], out[:, 44:52, 2])

    def test_glow_brightens_around_a_bright_patch(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[56:72, 56:72] = 255
        settings = CrtSettings(scanlines=0, curvature=0, glow=1.5, vignette=0, chroma=0, brightness=1.0)
        out = apply_crt(image, settings, 4)
        halo = out[50:56, 56:72].mean()
        assert halo > image[50:56, 56:72].mean()

    def test_brightness_scales_the_result(self) -> None:
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        base = CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.0)
        up = CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.5)
        assert apply_crt(image, up, 4).mean() > apply_crt(image, base, 4).mean()


def test_is_deterministic() -> None:
    image = checkerboard()
    a = apply_crt(image, CrtSettings(), 4)
    b = apply_crt(image, CrtSettings(), 4)
    assert np.array_equal(a, b)


def _sample_by_fancy_index(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    The bilinear sampler written the obvious way, kept as the parity reference.

    `crt._sample` gathers with `np.take` over a flat view because it is three
    times faster. Speed is only allowed to touch this code if it changes nothing
    (core-spec §7), and "nothing" has to be checked against something — this is
    that something, and it is deliberately the slow, transparently-correct form.
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

    return (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy
        + image[y1, x1] * fx * fy
    ).astype(np.float32)


class TestGatherIsBitIdentical:
    """The `np.take` gather against the fancy-indexed form it replaced."""

    @pytest.mark.parametrize("scale", [1, 2, 4])
    def test_whole_treatment_matches_the_reference_sampler(
        self, scale: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = checkerboard(scale)
        fast = apply_crt(image, CrtSettings(), scale)

        monkeypatch.setattr(crt, "_sample", _sample_by_fancy_index)
        slow = apply_crt(image, CrtSettings(), scale)

        assert np.array_equal(fast, slow)

    @pytest.mark.parametrize(
        "effect",
        [
            {"curvature": 0.35},
            {"chroma": 0.3},
            {"glow": 0.5},
        ],
    )
    def test_each_sampling_effect_matches(
        self, effect: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only three of the six effects sample at all; the others are per-pixel
        # multiplies that never reach the gather.
        image = checkerboard(4)
        settings = only(**effect)
        fast = apply_crt(image, settings, 4)

        monkeypatch.setattr(crt, "_sample", _sample_by_fancy_index)
        slow = apply_crt(image, settings, 4)

        assert np.array_equal(fast, slow)

    def test_sampler_matches_off_the_edge(self) -> None:
        # Curvature pushes coordinates past 0 and 1, where the clamp decides the
        # result — the case a flat-index rewrite is most likely to get wrong.
        rng = np.random.default_rng(11)
        image = rng.random((37, 53, 3)).astype(np.float32)
        u = rng.uniform(-0.5, 1.5, size=(37, 53)).astype(np.float32)
        v = rng.uniform(-0.5, 1.5, size=(37, 53)).astype(np.float32)

        assert np.array_equal(crt._sample(image, u, v), _sample_by_fancy_index(image, u, v))
