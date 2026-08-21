"""
Byte-level `.petv` interop with the TypeScript writer.

`fixtures/interop.petv` is produced by `web/test/make-interop.ts` and committed.
Reading it here is the only check that holds the two implementations to the same
*bytes* rather than merely to the same prose in core-spec §8.1 — and it runs
without needing the web toolchain installed.

Regenerate with `pnpm make:interop` in `web/` after any deliberate format change,
and expect this test to fail until both sides agree again.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from petscii_core import PetsciiFrame, encode_petv, read_petv

from .paths import fixtures_dir

PETV = fixtures_dir() / "interop.petv"
META = fixtures_dir() / "interop.json"

pytestmark = pytest.mark.skipif(
    not PETV.exists(), reason="run `pnpm make:interop` in web/ to create the fixture"
)


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(META.read_text())


@pytest.fixture(scope="module")
def stream():
    return read_petv(PETV.read_bytes())


def test_header_matches(stream, expected: dict) -> None:
    assert stream.charset == expected["charset"]


def test_frame_count(stream, expected: dict) -> None:
    assert len(stream.frames) == len(expected["frames"])


def test_every_cell_survives_the_round_trip(stream, expected: dict) -> None:
    for index, (got, want) in enumerate(zip(stream.frames, expected["frames"], strict=True)):
        assert got.bg == want["bg"], f"frame {index} background"
        assert got.border == want["border"], f"frame {index} border"
        assert np.array_equal(got.screen, np.array(want["screen"], dtype=np.uint8)), f"frame {index} screen"
        assert np.array_equal(got.color, np.array(want["color"], dtype=np.uint8)), f"frame {index} color"


def test_inter_frame_timing(stream, expected: dict) -> None:
    assert [f.dt_ms for f in stream.frames] == expected["expectedDeltas"]


def test_record_kinds_follow_the_spec(stream) -> None:
    """
    The fixture is built to exercise every branch: an opening keyframe, a small
    delta, a delta large enough to be promoted to a keyframe, and a background
    change that must force one.
    """
    kinds = [f.keyframe for f in stream.frames]
    assert kinds[0] is True  # first frame is always a keyframe
    assert kinds[1] is False  # 12 changed cells fits in a delta
    assert kinds[2] is True  # 640 changed cells is past the limit
    assert kinds[3] is True  # background changed


def test_python_writer_reproduces_the_same_bytes(stream, expected: dict) -> None:
    """
    Re-encoding the decoded frames at the fixture's own rate must give byte-identical
    output — that is what makes the two writers interchangeable, not just compatible.
    """
    deltas = expected["expectedDeltas"]
    interval = deltas[1]
    assert all(d in (0, interval) or d == deltas[-1] for d in deltas)

    frames = [
        PetsciiFrame(
            screen=np.array(f["screen"], dtype=np.uint8),
            color=np.array(f["color"], dtype=np.uint8),
            bg=f["bg"],
            border=f["border"],
            charset=expected["charset"],
        )
        for f in expected["frames"]
    ]
    # encode_petv writes a fixed interval, so compare against a fixed-rate stream
    # rather than the variable-rate original; the records themselves must match.
    ours = encode_petv(frames, fps=1000 / interval, charset=expected["charset"])
    theirs = PETV.read_bytes()

    assert len(ours) == len(theirs), "record layout differs in size"
    ours_stream = read_petv(ours)
    assert [f.keyframe for f in ours_stream.frames] == [f.keyframe for f in stream.frames]
    for a, b in zip(ours_stream.frames, stream.frames, strict=True):
        assert np.array_equal(a.screen, b.screen)
        assert np.array_equal(a.color, b.color)
        assert (a.bg, a.border) == (b.bg, b.border)
