"""The CRT treatment — a port of the web app's post shader."""

from __future__ import annotations

import numpy as np
import pytest

from petscii_core import CrtSettings, apply_crt, render_frame
from petscii_core.engine import SCREEN_H, SCREEN_W, Settings, convert

FLAT = CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.0)


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
        assert not CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.2).is_identity


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
        out = apply_crt(image, CrtSettings(scanlines=0.8, curvature=0, glow=0, vignette=0, chroma=0, brightness=1.0), 4)
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
        out = apply_crt(image, CrtSettings(scanlines=0, curvature=0, glow=0, vignette=1.0, chroma=0, brightness=1.0), 4)
        centre = out[60:68, 60:68].mean()
        corner = out[:8, :8].mean()
        assert corner < centre * 0.75

    def test_curvature_blanks_the_corners(self) -> None:
        """The curve pushes the corners off the tube; that area reads as glass."""
        image = np.full((128, 128, 3), 220, dtype=np.uint8)
        out = apply_crt(image, CrtSettings(scanlines=0, curvature=1.0, glow=0, vignette=0, chroma=0, brightness=1.0), 4)
        assert out[0, 0].max() == 0
        assert out[64, 64].min() > 100

    def test_chroma_separates_red_and_blue_at_an_edge(self) -> None:
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:, 48:] = 255  # a hard white edge down the middle
        out = apply_crt(image, CrtSettings(scanlines=0, curvature=0, glow=0, vignette=0, chroma=1.0, brightness=1.0), 4)
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
