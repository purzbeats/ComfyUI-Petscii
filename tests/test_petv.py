"""`.petv` writer and reader — core-spec §8.1."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from petscii_core import PetsciiFrame, encode_petv, read_petv, write_petv
from petscii_core.engine import CELLS
from petscii_core.petv import DELTA_LIMIT, MAGIC, frames_from_stream


def frame(seed: int, bg: int = 6, border: int = 6) -> PetsciiFrame:
    rng = np.random.default_rng(seed)
    return PetsciiFrame(
        screen=rng.integers(0, 256, CELLS, dtype=np.uint8),
        color=rng.integers(0, 16, CELLS, dtype=np.uint8),
        bg=bg,
        border=border,
        charset=0,
    )


def nudge(base: PetsciiFrame, n: int) -> PetsciiFrame:
    """A copy of ``base`` with exactly ``n`` cells changed."""
    assert n <= CELLS
    screen = base.screen.copy()
    color = base.color.copy()
    # Cast out of uint8 for the arithmetic: numpy 2 refuses to assign 256 back.
    screen[:n] = ((screen[:n].astype(np.int16) + 1) % 256).astype(np.uint8)
    color[:n] = ((color[:n].astype(np.int16) + 1) % 16).astype(np.uint8)
    return PetsciiFrame(screen, color, base.bg, base.border, base.charset)


def test_header() -> None:
    data = encode_petv([frame(1)], fps=30, charset=1)
    assert data[:4] == MAGIC == b"PETV"
    assert data[4] == 1  # version
    assert data[5] == 1  # charset


def test_single_frame_round_trip() -> None:
    a = frame(7, bg=9, border=3)
    stream = read_petv(encode_petv([a], fps=30))
    assert len(stream.frames) == 1
    assert stream.frames[0].keyframe
    assert stream.frames[0].dt_ms == 0  # the first frame is always dt 0
    assert stream.frames[0].bg == 9
    assert stream.frames[0].border == 3
    assert np.array_equal(stream.frames[0].screen, a.screen)
    assert np.array_equal(stream.frames[0].color, a.color)


def test_delta_sequence_round_trips_cell_for_cell() -> None:
    base = frame(11)
    sequence = [base, nudge(base, 10), nudge(base, 40), nudge(base, 3)]
    stream = read_petv(encode_petv(sequence, fps=60))
    assert [f.keyframe for f in stream.frames] == [True, False, False, False]
    for expected, got in zip(sequence, stream.frames, strict=True):
        assert np.array_equal(got.screen, expected.screen)
        assert np.array_equal(got.color, expected.color)


def test_timing_follows_fps() -> None:
    base = frame(3)
    stream = read_petv(encode_petv([base, nudge(base, 5), nudge(base, 5)], fps=25))
    assert [f.dt_ms for f in stream.frames] == [0, 40, 40]
    assert stream.duration_ms == 80


def test_promotes_to_a_keyframe_past_the_delta_limit() -> None:
    base = frame(5)
    stream = read_petv(encode_petv([base, nudge(base, DELTA_LIMIT + 50)], fps=30))
    assert all(f.keyframe for f in stream.frames)


def test_never_deltas_across_a_background_change() -> None:
    # A new background repaints every cell, so a delta would be a lie even if few
    # codes moved.
    base = frame(13, bg=6)
    recoloured = PetsciiFrame(base.screen.copy(), base.color.copy(), 11, 11, 0)
    stream = read_petv(encode_petv([base, recoloured], fps=30))
    assert stream.frames[1].keyframe
    assert stream.frames[1].bg == 11


def test_deltas_are_much_smaller_than_keyframes() -> None:
    base = frame(17)
    deltas = encode_petv([base] + [nudge(base, 20 + i) for i in range(1, 60)], fps=60)
    keys = encode_petv([frame(100 + i) for i in range(60)], fps=60)
    assert len(deltas) < len(keys) / 4


def test_frames_from_stream_are_renderable() -> None:
    a = frame(23, bg=4, border=12)
    stream = read_petv(encode_petv([a], fps=30))
    back = next(iter(frames_from_stream(stream)))
    assert back.bg == 4
    assert back.border == 12
    assert back.charset == 0
    assert np.array_equal(back.screen, a.screen)


def test_write_to_disk(tmp_path) -> None:
    path = tmp_path / "clip.petv"
    size = write_petv(str(path), [frame(1), nudge(frame(1), 4)], fps=30)
    assert path.stat().st_size == size
    stream = read_petv(path.read_bytes())
    assert len(stream.frames) == 2


class TestRejection:
    def test_not_a_petv(self) -> None:
        with pytest.raises(ValueError, match=r"not a \.petv"):
            read_petv(b"\x01\x02\x03")

    def test_unsupported_version(self) -> None:
        data = bytearray(encode_petv([frame(2)], fps=30))
        data[4] = 9
        with pytest.raises(ValueError, match="version 9"):
            read_petv(bytes(data))

    def test_unknown_record(self) -> None:
        data = bytearray(encode_petv([frame(2)], fps=30))
        data[8] = 0x7F
        with pytest.raises(ValueError, match="record type"):
            read_petv(bytes(data))

    def test_truncated(self) -> None:
        data = encode_petv([frame(2)], fps=30)
        with pytest.raises(ValueError, match="truncated"):
            read_petv(data[:-20])

    def test_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            encode_petv([], fps=30)

    def test_bad_fps(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            encode_petv([frame(1)], fps=0)

    def test_wrong_cell_count(self) -> None:
        bad = PetsciiFrame(np.zeros(10, np.uint8), np.zeros(10, np.uint8), 0, 0, 0)
        with pytest.raises(ValueError, match="cells"):
            encode_petv([bad], fps=30)


def test_byte_layout_matches_the_spec() -> None:
    """
    Pins the wire format field by field, so a change to the writer that both ports
    would still agree on — but that breaks an existing file — fails here.
    """
    a = frame(31, bg=5, border=7)
    data = encode_petv([a, nudge(a, 2)], fps=50)

    magic, version, charset, flags, reserved = struct.unpack_from("<4sBBBB", data, 0)
    assert (magic, version, charset, flags, reserved) == (b"PETV", 1, 0, 0, 0)

    kind, dt, bg, border = struct.unpack_from("<BHBB", data, 8)
    assert (kind, dt, bg, border) == (0x01, 0, 5, 7)
    assert data[13 : 13 + CELLS] == a.screen.tobytes()
    assert data[13 + CELLS : 13 + CELLS * 2] == a.color.tobytes()

    offset = 13 + CELLS * 2
    kind, dt, bg, border = struct.unpack_from("<BHBB", data, offset)
    assert (kind, dt, bg, border) == (0x02, 20, 5, 7)  # 50 fps -> 20 ms
    (count,) = struct.unpack_from("<H", data, offset + 5)
    assert count == 2
    index, _char, _color = struct.unpack_from("<HBB", data, offset + 7)
    assert index == 0


def test_fractional_frame_rates_do_not_drift() -> None:
    """
    A whole-millisecond ``dt`` cannot represent 30 fps, and rounding the interval
    once and repeating it loses a third of a millisecond every frame — a second of
    drift per fifty seconds of playback. The gaps must be differences of rounded
    timestamps instead, so the error stays bounded however long the clip runs.
    """
    base = frame(11)
    sequence = [base] + [nudge(base, 5) for _ in range(299)]

    for fps in (30.0, 23.976, 59.94, 25.0, 24.0):
        stream = read_petv(encode_petv(sequence, fps=fps))
        ideal = (len(sequence) - 1) * 1000.0 / fps
        assert abs(stream.duration_ms - ideal) <= 1.0, f"{fps} fps drifted"


def test_thirty_fps_alternates_the_gap() -> None:
    base = frame(12)
    stream = read_petv(encode_petv([base] + [nudge(base, 3) for _ in range(6)], fps=30))
    assert [f.dt_ms for f in stream.frames] == [0, 33, 34, 33, 33, 34, 33]
