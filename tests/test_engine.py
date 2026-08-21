"""Engine behaviour against core-spec §3-5, independent of the fixtures."""

from __future__ import annotations

import threading
import typing
from collections.abc import Sequence

import numpy as np
import pytest

from petscii_core import (
    CELLS,
    COLS,
    SCREEN_H,
    SCREEN_W,
    HysteresisState,
    Settings,
    convert,
    convert_batch,
    convert_workers,
    frame_to_screen,
    iter_convert_batch,
    palette_rgb8,
    pre_adjust_to_oklab,
    render_frame,
)
from petscii_core.data_loader import charset_bytes, glyph_masks, subset_mask
from petscii_core.engine import (
    _adds_letterbox,
    argmin_for_background,
    cell_error_of,
    distance_tables,
    glyph_minima,
    masked_sums,
    select_background,
)
from petscii_core.oklab import (
    linear_rgb_to_oklab,
    linear_to_srgb,
    oklab_to_linear_rgb,
    srgb_to_linear,
    srgb_to_oklab,
)

REFERENCE = Settings()


def solid(r: int, g: int, b: int) -> np.ndarray:
    image = np.empty((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    image[:] = (r, g, b)
    return image


def noise(seed: int, height: int = SCREEN_H, width: int = SCREEN_W) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


# ------------------------------------------------------------------- oklab


class TestOklab:
    #: Ottosson's published reference values, for linear-RGB inputs.
    REFERENCE_VALUES: typing.ClassVar = [
        ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.6279554, 0.2248631, 0.1258463)),
        ((0.0, 1.0, 0.0), (0.8664396, -0.2338874, 0.1794985)),
        ((0.0, 0.0, 1.0), (0.4520137, -0.0324454, -0.3115281)),
    ]

    def test_matches_published_values(self) -> None:
        for rgb, expected in self.REFERENCE_VALUES:
            got = linear_rgb_to_oklab(np.array(rgb, dtype=np.float32))
            # 4 places: the spec stores Oklab as float32 and the constants are float64.
            assert got == pytest.approx(expected, abs=1e-4)

    def test_round_trips(self) -> None:
        for rgb, _ in self.REFERENCE_VALUES:
            lab = linear_rgb_to_oklab(np.array(rgb, dtype=np.float32))
            back = oklab_to_linear_rgb(lab)
            assert back == pytest.approx(rgb, abs=1e-4)

    def test_transfer_function_round_trips_every_byte(self) -> None:
        values = np.arange(256, dtype=np.float32) / 255.0
        back = linear_to_srgb(srgb_to_linear(values))
        assert np.array_equal(np.rint(back * 255).astype(np.uint8), np.arange(256, dtype=np.uint8))

    def test_handles_negative_input_without_nan(self) -> None:
        """Dithering residuals push values out of gamut; cbrt must cope."""
        lab = linear_rgb_to_oklab(np.array([-0.2, 0.4, -0.05], dtype=np.float32))
        assert np.all(np.isfinite(lab))


# ------------------------------------------------------------------ charset


class TestCharset:
    def test_high_codes_are_bitwise_inverses(self) -> None:
        rows = charset_bytes(0)
        assert np.array_equal(rows[128:], (~rows[:128]) & 0xFF)

    def test_masks_agree_with_row_bytes(self) -> None:
        rows = charset_bytes(0)[:128]
        masks = glyph_masks(0).reshape(128, 8, 8)
        for x in range(8):
            assert np.array_equal(masks[:, :, x], ((rows >> (7 - x)) & 1).astype(np.float32))

    def test_space_and_its_inverse(self) -> None:
        rows = charset_bytes(0)
        assert np.all(rows[32] == 0x00)
        assert np.all(rows[160] == 0xFF)


# ------------------------------------------------------------ tables (§4.1-3)


class TestTables:
    def test_masked_sums_match_the_literal_definition(self) -> None:
        oklab = pre_adjust_to_oklab(noise(1), REFERENCE)
        d, _ = distance_tables(oklab)
        a = masked_sums(d, 0)
        masks = glyph_masks(0)
        for c in (0, 1, 39, 500, 999):
            for g in (0, 1, 32, 81, 127):
                expected = (d[c] * masks[g][:, None]).sum(axis=0)
                assert a[c, g] == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_cell_error_matches_a_direct_evaluation(self) -> None:
        image = noise(2)
        oklab = pre_adjust_to_oklab(frame_to_screen(image), REFERENCE)
        d, s = distance_tables(oklab)
        a = masked_sums(d, 0)
        rows = charset_bytes(0)

        cells_flat = (
            oklab.reshape(25, 8, 40, 8, 3).transpose(0, 2, 1, 3, 4).reshape(CELLS, 64, 3)
        )
        from petscii_core.data_loader import palette_oklab

        palette = palette_oklab()

        for cell, code, fg, bg in [(0, 1, 1, 0), (17, 90, 7, 6), (500, 200, 3, 11), (999, 255, 15, 2)]:
            bits = np.unpackbits(rows[code]).astype(bool)
            painted = np.where(bits[:, None], palette[fg], palette[bg])
            expected = float(((cells_flat[cell] - painted) ** 2).sum())
            got = float(
                cell_error_of(a, s, np.array([cell]), np.array([code]), np.array([fg]), bg)[0]
            )
            assert got == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_space_sums_nothing_and_its_inverse_sums_the_cell(self) -> None:
        oklab = pre_adjust_to_oklab(solid(90, 140, 200), REFERENCE)
        d, s = distance_tables(oklab)
        a = masked_sums(d, 0)
        assert a[0, 32] == pytest.approx(np.zeros(16), abs=1e-5)
        for bg in (0, 3, 12):
            err = cell_error_of(a, s, np.array([0]), np.array([160]), np.array([5]), bg)[0]
            assert err == pytest.approx(s[0, 5], rel=1e-5)


# ----------------------------------------------------------- argmin (§4.4-5)


class TestArgmin:
    @pytest.mark.parametrize("k", [0, 1, 5, 14])
    def test_reproduces_an_exact_palette_colour(self, k: int) -> None:
        rgb = palette_rgb8()[k]
        frame = convert(solid(*rgb), REFERENCE)
        rendered = render_frame(frame)
        assert np.array_equal(rendered, solid(*rgb))

    def test_honours_a_subset(self) -> None:
        settings = Settings(subset="blocks")
        frame = convert(noise(3), settings)
        allowed = np.flatnonzero(subset_mask("blocks"))
        assert set(frame.screen.tolist()) <= set(allowed.tolist())

    def test_ties_break_toward_the_lower_code_and_colour(self) -> None:
        # Grey 12 is an exact palette colour, so every glyph is free with fg == bg.
        # §4.4 forbids normalising that away, and the lowest code must win.
        oklab = pre_adjust_to_oklab(solid(108, 108, 108), REFERENCE)
        d, s = distance_tables(oklab)
        a = masked_sums(d, 0)
        minima = glyph_minima(a, s)
        screen, color, error = argmin_for_background(a, s, minima, 12, subset_mask("all"))
        assert screen[0] == 0
        assert color[0] == 12
        assert error[0] == pytest.approx(0.0, abs=1e-5)

    def test_background_lock_is_honoured(self) -> None:
        frame = convert(noise(4), Settings(bg_lock=6, border_lock=14))
        assert frame.bg == 6
        assert frame.border == 14

    def test_background_search_picks_the_exact_colour(self) -> None:
        for k in (0, 6, 9, 15):
            frame = convert(solid(*palette_rgb8()[k]), REFERENCE)
            assert np.array_equal(render_frame(frame)[0, 0], palette_rgb8()[k])
            assert frame.border == frame.bg

    def test_select_background_returns_all_sixteen_totals(self) -> None:
        oklab = pre_adjust_to_oklab(noise(5), REFERENCE)
        d, s = distance_tables(oklab)
        a = masked_sums(d, 0)
        bg, totals = select_background(a, s, glyph_minima(a, s), subset_mask("all"))
        assert totals.shape == (16,)
        assert bg == int(totals.argmin())


# ------------------------------------------------------------ hysteresis §4.6


class TestHysteresis:
    def test_eps_zero_is_exactly_the_stateless_result(self) -> None:
        """
        The precise claim of §4.6 at eps = 0: hysteresis does nothing. Asserting
        that a stateful run equals a stateless one is deterministic, unlike
        counting how many cells happen to move between two noise frames — cells
        average out over 64 pixels, so that count is not a reliable signal.
        """
        state = HysteresisState()
        settings = Settings(eps=0.0, bg_lock=0)
        convert(noise(10), settings, state)
        with_state = convert(noise(11), settings, state)
        stateless = convert(noise(11), settings)
        assert np.array_equal(with_state.screen, stateless.screen)
        assert np.array_equal(with_state.color, stateless.color)

    def test_large_eps_pins_the_previous_frame(self) -> None:
        state = HysteresisState()
        settings = Settings(eps=1.0, bg_lock=0)
        a = convert(noise(10), settings, state)
        b = convert(noise(11), settings, state)
        assert np.array_equal(a.screen, b.screen)
        assert np.array_equal(a.color, b.color)

    def test_first_frame_is_unaffected(self) -> None:
        settings = Settings(eps=1.0, bg_lock=0)
        with_state = convert(noise(12), settings, HysteresisState())
        without = convert(noise(12), settings)
        assert np.array_equal(with_state.screen, without.screen)

    def test_background_change_clears_the_state(self) -> None:
        state = HysteresisState()
        first = convert(solid(20, 20, 20), Settings(eps=1.0), state)
        second = convert(solid(230, 230, 230), Settings(eps=1.0), state)
        # A new background invalidates the state, so the bright frame must not be
        # held to the dark frame's choices. Both flat frames settle on code 0 with
        # fg == bg — the same glyph painting a different colour — so the colour is
        # where the re-decision shows.
        assert second.bg != first.bg
        assert not np.array_equal(second.color, first.color)


class TestBatch:
    def test_independent_batch_ignores_order(self) -> None:
        frames = [noise(20), noise(21), noise(22)]
        forward = convert_batch(frames, REFERENCE)
        backward = convert_batch(list(reversed(frames)), REFERENCE)
        assert np.array_equal(forward[0].screen, backward[2].screen)

    def test_temporal_batch_locks_one_background(self) -> None:
        frames = [solid(200, 40, 40), solid(40, 200, 40), solid(40, 40, 200)]
        out = convert_batch(frames, Settings(eps=0.002), temporal=True)
        assert len({f.bg for f in out}) == 1

    def test_temporal_batch_holds_cells_still(self) -> None:
        base = noise(30)
        # Two nearly identical frames: hysteresis should keep almost everything.
        nudged = base.copy()
        nudged[0:8, 0:8] = 0
        loose = convert_batch([base, nudged], Settings(eps=0.0), temporal=True)
        tight = convert_batch([base, nudged], Settings(eps=0.02), temporal=True)
        loose_moved = int((loose[0].screen != loose[1].screen).sum())
        tight_moved = int((tight[0].screen != tight[1].screen).sum())
        assert tight_moved <= loose_moved

    def test_empty_batch(self) -> None:
        assert convert_batch([], REFERENCE) == []

    def test_iter_form_yields_the_same_frames(self) -> None:
        """The generator is what the nodes drive; the list form must not diverge."""
        frames = [noise(60), noise(61), noise(62)]
        for settings, temporal in ((REFERENCE, False), (Settings(eps=0.002), True)):
            streamed = list(iter_convert_batch(frames, settings, temporal=temporal))
            eager = convert_batch(frames, settings, temporal=temporal)
            assert len(streamed) == len(eager)
            for a, b in zip(streamed, eager, strict=True):
                assert np.array_equal(a.screen, b.screen)
                assert np.array_equal(a.color, b.color)
                assert (a.bg, a.border) == (b.bg, b.border)

    def test_iter_form_is_lazy(self) -> None:
        """Nothing is converted until the caller asks for a frame."""
        touched: list[int] = []

        class Watched(Sequence):
            def __init__(self, frames):
                self._frames = frames

            def __len__(self):
                return len(self._frames)

            def __getitem__(self, index):
                touched.append(index)
                return self._frames[index]

        stream = iter_convert_batch(Watched([noise(70), noise(71)]), REFERENCE)
        assert touched == []
        next(stream)
        assert touched == [0]


# ------------------------------------------------------------- framing (§5)


class TestFraming:
    def test_320x200_passes_through_unchanged(self) -> None:
        image = noise(40)
        for mode in ("cover", "contain", "stretch"):
            assert np.array_equal(frame_to_screen(image, mode), image)

    def test_box_downscale_averages_each_block(self) -> None:
        # A checkerboard at exactly 2x averages to 100 in every 2x2 block.
        big = np.zeros((SCREEN_H * 2, SCREEN_W * 2, 3), dtype=np.uint8)
        ys, xs = np.mgrid[0 : SCREEN_H * 2, 0 : SCREEN_W * 2]
        big[(ys + xs) % 2 == 1] = 200
        out = frame_to_screen(big, "cover")
        assert np.all(out == 100)

    def test_cover_crops_the_long_axis(self) -> None:
        wide = np.zeros((200, 400, 3), dtype=np.uint8)
        wide[:, :200] = (200, 0, 0)
        wide[:, 200:] = (0, 0, 200)
        out = frame_to_screen(wide, "cover")
        # 400x200 is wider than 8:5, so 40px comes off each side; the centre split
        # stays centred.
        assert out[10, 10, 0] > 150
        assert out[10, -10, 2] > 150

    def test_contain_bars_take_the_chosen_background(self) -> None:
        """
        The bars are part of the image the background is chosen from, so painting
        them black regardless would show black bars around a picture whose
        background is something else entirely.
        """
        wide = np.zeros((200, 400, 3), dtype=np.uint8)
        wide[:, :] = (60, 60, 200)  # strongly blue, so black is not the winner
        out = convert(wide, Settings(framing="contain"))
        assert out.bg != 0
        # Every cell in the letterboxed strip paints nothing but the background.
        top_row_cells = out.screen[:COLS]
        assert np.all(out.color[:COLS] == out.bg) or np.all(top_row_cells == top_row_cells[0])

    def test_contain_bars_follow_an_explicit_background(self) -> None:
        wide = np.full((200, 400, 3), 180, dtype=np.uint8)
        out = convert(wide, Settings(framing="contain", bg_lock=11))
        assert out.bg == 11

    def test_cover_never_pays_for_the_letterbox_probe(self) -> None:
        """
        The second pass exists only for ``contain`` with an auto background; the
        fixtures and every ``cover`` input must take the single-pass path.
        """
        wide = np.zeros((200, 400, 3), dtype=np.uint8)
        assert not _adds_letterbox(wide, Settings(framing="cover"))
        assert not _adds_letterbox(wide, Settings(framing="contain", bg_lock=3))
        assert not _adds_letterbox(noise(80), Settings(framing="contain"))
        assert _adds_letterbox(wide, Settings(framing="contain"))

    def test_contain_letterboxes(self) -> None:
        wide = np.full((200, 400, 3), 180, dtype=np.uint8)
        out = frame_to_screen(wide, "contain", bars=(17, 23, 29))
        assert tuple(out[0, 0]) == (17, 23, 29)
        assert tuple(out[SCREEN_H // 2, 5]) != (17, 23, 29)

    def test_rejects_a_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            frame_to_screen(np.zeros((10, 10), dtype=np.uint8))


class TestPreAdjust:
    def test_identity_at_defaults(self) -> None:
        image = solid(123, 45, 200)
        got = pre_adjust_to_oklab(image, REFERENCE)
        expected = srgb_to_oklab(image.astype(np.float32) / 255.0)
        assert got[0, 0] == pytest.approx(expected[0, 0], abs=1e-6)

    def test_saturation_zero_collapses_chroma(self) -> None:
        got = pre_adjust_to_oklab(solid(200, 40, 40), Settings(saturation=0.0))
        assert got[..., 1] == pytest.approx(0.0, abs=1e-6)
        assert got[..., 2] == pytest.approx(0.0, abs=1e-6)

    def test_brightness_and_gamma_lift_lightness(self) -> None:
        base = pre_adjust_to_oklab(solid(100, 100, 100), REFERENCE)[0, 0, 0]
        brighter = pre_adjust_to_oklab(solid(100, 100, 100), Settings(brightness=0.2))[0, 0, 0]
        gamma_up = pre_adjust_to_oklab(solid(100, 100, 100), Settings(gamma=2.0))[0, 0, 0]
        assert brighter > base
        assert gamma_up > base


class TestDither:
    def test_is_a_no_op_on_an_exact_palette_colour(self) -> None:
        for k in (0, 5, 12):
            image = solid(*palette_rgb8()[k])
            plain = convert(image, REFERENCE)
            dithered = convert(image, Settings(dither=True))
            assert np.array_equal(plain.screen, dithered.screen)
            assert np.array_equal(plain.color, dithered.color)
            assert plain.bg == dithered.bg

    def test_improves_the_local_mean_on_a_gradient(self) -> None:
        # Error diffusion makes cells worse and neighbourhoods better, so the
        # metric has to be local.
        xs = np.linspace(0, 1, SCREEN_W, dtype=np.float32)
        ys = np.linspace(0, 1, SCREEN_H, dtype=np.float32)
        image = np.stack(
            [
                np.tile(30 + xs * 200, (SCREEN_H, 1)),
                np.tile((40 + ys * 170)[:, None], (1, SCREEN_W)),
                np.tile(200 - xs * 150, (SCREEN_H, 1)),
            ],
            axis=-1,
        ).astype(np.uint8)

        def local_error(rendered: np.ndarray) -> float:
            block = 32
            src = image.reshape(SCREEN_H // 8, 8, SCREEN_W // block, block, 3)
            got = rendered.reshape(SCREEN_H // 8, 8, SCREEN_W // block, block, 3)
            return float(np.abs(src.mean(axis=(1, 3)) - got.mean(axis=(1, 3))).mean())

        plain = render_frame(convert(image, REFERENCE))
        dithered = render_frame(convert(image, Settings(dither=True)))
        assert local_error(dithered) < local_error(plain)

    def test_breaks_banding_on_a_smooth_ramp(self) -> None:
        """
        Undithered, a smooth greyscale ramp collapses into a handful of flat bands.
        Diffusion trades those for texture that carries the sub-palette tone, so
        the code count goes *up*. (On a busy image the direction reverses — the
        undithered result chases hue detail while dithering collapses onto the
        block ramp — which is why the meaningful invariant is the local mean.)
        """
        xs = np.linspace(0, 255, SCREEN_W, dtype=np.float32)
        image = np.stack([np.tile(xs, (SCREEN_H, 1))] * 3, axis=-1).astype(np.uint8)
        plain = convert(image, REFERENCE)
        dithered = convert(image, Settings(dither=True))
        assert len(set(plain.screen.tolist())) < 10
        assert len(set(dithered.screen.tolist())) > len(set(plain.screen.tolist()))

    def test_stays_in_range_on_an_unrepresentable_ramp(self) -> None:
        image = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        image[: SCREEN_H // 2] = (255, 250, 245)
        image[SCREEN_H // 2 :] = (8, 6, 10)
        frame = convert(image, Settings(dither=True))
        assert frame.color.max() < 16
        assert frame.screen.max() <= 255


class TestRender:
    def test_scale_is_nearest_neighbour(self) -> None:
        frame = convert(noise(50), REFERENCE)
        one = render_frame(frame, 1)
        three = render_frame(frame, 3)
        assert three.shape == (SCREEN_H * 3, SCREEN_W * 3, 3)
        assert np.array_equal(three[::3, ::3], one)

    def test_renders_every_code_and_colour(self) -> None:
        from petscii_core import PetsciiFrame

        screen = (np.arange(CELLS) % 256).astype(np.uint8)
        color = ((np.arange(CELLS) * 7) % 16).astype(np.uint8)
        frame = PetsciiFrame(screen, color, bg=6, border=11, charset=0)
        image = render_frame(frame)
        assert image.shape == (SCREEN_H, SCREEN_W, 3)
        # Every pixel must be an exact palette colour.
        flat = image.reshape(-1, 3)
        palette = palette_rgb8()
        assert np.isin(flat, palette).all()


# ------------------------------------------------------- parallel conversion


class TestParallelBatch:
    """
    The pool in `iter_convert_batch` must be invisible from the outside.

    Every frame is converted independently and every intermediate is allocated
    per call, so the only ways threading could show up are a shared buffer or a
    reordered yield. Both would be intermittent in the wild, which is why these
    check equality over a batch rather than sampling.
    """

    @staticmethod
    def batch(count: int = 9) -> list[np.ndarray]:
        rng = np.random.default_rng(3)
        return [(rng.random((SCREEN_H, SCREEN_W, 3)) * 255).astype(np.uint8) for _ in range(count)]

    def test_parallel_matches_serial(self) -> None:
        images = self.batch()
        serial = convert_batch(images, Settings(), workers=1)
        parallel = convert_batch(images, Settings(), workers=4)

        assert len(serial) == len(parallel)
        for one, other in zip(serial, parallel, strict=True):
            assert np.array_equal(one.screen, other.screen)
            assert np.array_equal(one.color, other.color)
            assert (one.bg, one.border) == (other.bg, other.border)

    def test_frames_come_back_in_order(self) -> None:
        # Distinct flat colours: the converted screen of each is unmistakable,
        # so an out-of-order yield cannot pass by coincidence.
        images = [np.full((SCREEN_H, SCREEN_W, 3), v * 24, dtype=np.uint8) for v in range(8)]
        parallel = convert_batch(images, Settings(), workers=4)
        serial = convert_batch(images, Settings(), workers=1)
        assert [f.bg for f in parallel] == [f.bg for f in serial]

    def test_temporal_ignores_workers(self) -> None:
        """Hysteresis is a real dependency between frames — it may not be split."""
        images = self.batch(6)
        asked_for_threads = convert_batch(images, Settings(eps=0.02), temporal=True, workers=8)
        sequential = convert_batch(images, Settings(eps=0.02), temporal=True, workers=1)

        for one, other in zip(asked_for_threads, sequential, strict=True):
            assert np.array_equal(one.screen, other.screen)
            assert np.array_equal(one.color, other.color)

    def test_each_image_is_read_exactly_once(self) -> None:
        """
        The lazy-sequence contract holds on the pool too.

        `_TensorFrames` decodes on `__getitem__`, so an extra read is an extra
        tensor conversion — and the whole reason the engine indexes rather than
        iterates is that nothing should materialise a clip twice.
        """
        images = self.batch(7)

        class CountingSequence(Sequence):
            def __init__(self, items):
                self._items = items
                self.reads = []

            def __len__(self):
                return len(self._items)

            def __getitem__(self, index):
                self.reads.append(index)
                return self._items[index]

        lazy = CountingSequence(images)
        convert_batch(lazy, Settings(), workers=4)
        assert sorted(lazy.reads) == list(range(7))

    def test_abandoning_the_generator_shuts_the_pool_down(self) -> None:
        """A caller that stops early — an interrupt — must not leave threads running."""
        images = self.batch(24)
        stream = iter_convert_batch(images, Settings(), workers=4)
        assert next(stream) is not None
        assert next(stream) is not None
        stream.close()

        remaining = [t for t in threading.enumerate() if t.name.startswith("petscii-convert")]
        assert remaining == []

    def test_worker_count_is_bounded(self) -> None:
        assert convert_workers(1) == 1
        assert convert_workers(2) <= 2
        assert convert_workers(10_000) <= 8
        # An explicit request wins, and never lands below one.
        assert convert_workers(10_000, 3) == 3
        assert convert_workers(10_000, 0) == 1
