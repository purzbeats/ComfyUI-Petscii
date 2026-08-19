"""
`.petv` writer and reader — core-spec §8.1.

Byte-for-byte the same format the web app records, so a clip converted in a Comfy
workflow plays back in the VJ app's player and feeds the deferred C64 and Looking
Glass exporters.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Sequence

import numpy as np

from .engine import CELLS, PetsciiFrame

MAGIC = b"PETV"
VERSION = 1
KEYFRAME = 0x01
DELTA = 0x02
#: Above this many changed cells a delta costs more than a keyframe.
DELTA_LIMIT = 500

_HEADER = struct.Struct("<4sBBBB")
_FRAME_HEAD = struct.Struct("<BHBB")
_DELTA_COUNT = struct.Struct("<H")

__all__ = [
    "MAGIC",
    "VERSION",
    "DELTA_LIMIT",
    "PetvFrame",
    "PetvStream",
    "write_petv",
    "read_petv",
    "encode_petv",
]


@dataclass
class PetvFrame:
    screen: np.ndarray
    color: np.ndarray
    bg: int
    border: int
    #: Milliseconds since the previous frame; 0 on the first.
    dt_ms: int
    keyframe: bool

    def to_frame(self, charset: int) -> PetsciiFrame:
        return PetsciiFrame(
            screen=self.screen,
            color=self.color,
            bg=self.bg,
            border=self.border,
            charset=charset,
        )


@dataclass
class PetvStream:
    charset: int
    flags: int
    frames: list[PetvFrame]

    @property
    def duration_ms(self) -> int:
        return sum(f.dt_ms for f in self.frames)


def encode_petv(
    frames: Sequence[PetsciiFrame],
    fps: float = 30.0,
    charset: int | None = None,
    flags: int = 0,
) -> bytes:
    """
    Encodes frames at a fixed rate.

    Realtime recording is variable-rate by nature, but a rendered clip has one
    known interval, so this takes ``fps`` rather than timestamps.
    """
    if not frames:
        raise ValueError("cannot encode an empty .petv")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    resolved_charset = frames[0].charset if charset is None else charset
    dt = int(round(1000.0 / fps))
    if dt > 0xFFFF:
        raise ValueError(f"fps {fps} gives a frame interval past the 65535 ms field")

    out = bytearray(_HEADER.pack(MAGIC, VERSION, resolved_charset & 0xFF, flags & 0xFF, 0))
    previous: PetsciiFrame | None = None

    for index, frame in enumerate(frames):
        _validate(frame)
        gap = 0 if index == 0 else dt
        changed = _changed_cells(previous, frame)
        if changed is None or len(changed) > DELTA_LIMIT:
            out += _pack_keyframe(frame, gap)
        else:
            out += _pack_delta(frame, gap, changed)
        previous = frame

    return bytes(out)


def write_petv(path: str, frames: Sequence[PetsciiFrame], fps: float = 30.0, **kwargs) -> int:
    """Writes a `.petv` and returns the byte count."""
    data = encode_petv(frames, fps, **kwargs)
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


def _validate(frame: PetsciiFrame) -> None:
    if frame.screen.shape != (CELLS,) or frame.color.shape != (CELLS,):
        raise ValueError(
            f"a .petv frame is {CELLS} cells, got screen {frame.screen.shape} "
            f"and color {frame.color.shape}"
        )


def _changed_cells(previous: PetsciiFrame | None, frame: PetsciiFrame) -> np.ndarray | None:
    """Indices whose code or colour moved, or None when there is no predecessor."""
    if previous is None:
        return None
    # A background change repaints everything, so never delta across one.
    if previous.bg != frame.bg:
        return None
    moved = (previous.screen != frame.screen) | (previous.color != frame.color)
    return np.flatnonzero(moved)


def _pack_keyframe(frame: PetsciiFrame, dt_ms: int) -> bytes:
    head = _FRAME_HEAD.pack(KEYFRAME, dt_ms, frame.bg & 15, frame.border & 15)
    return head + frame.screen.astype(np.uint8).tobytes() + frame.color.astype(np.uint8).tobytes()


def _pack_delta(frame: PetsciiFrame, dt_ms: int, changed: np.ndarray) -> bytes:
    head = _FRAME_HEAD.pack(DELTA, dt_ms, frame.bg & 15, frame.border & 15)
    body = np.empty((len(changed), 4), dtype=np.uint8)
    indices = changed.astype("<u2")
    body[:, 0:2] = indices.view(np.uint8).reshape(-1, 2)
    body[:, 2] = frame.screen[changed]
    body[:, 3] = frame.color[changed]
    return head + _DELTA_COUNT.pack(len(changed)) + body.tobytes()


def read_petv(data: bytes | BinaryIO) -> PetvStream:
    """Reads a whole `.petv`, materialising each frame (deltas applied)."""
    raw = data if isinstance(data, (bytes, bytearray)) else data.read()
    raw = bytes(raw)
    if len(raw) < _HEADER.size:
        raise ValueError("not a .petv stream")
    magic, version, charset, flags, _ = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ValueError("not a .petv stream")
    if version != VERSION:
        raise ValueError(f"unsupported .petv version {version}")

    frames: list[PetvFrame] = []
    screen = np.zeros(CELLS, dtype=np.uint8)
    color = np.zeros(CELLS, dtype=np.uint8)
    offset = _HEADER.size
    keyframe_size = _FRAME_HEAD.size + CELLS * 2

    while offset < len(raw):
        kind = raw[offset]
        if kind == KEYFRAME:
            if offset + keyframe_size > len(raw):
                raise ValueError(".petv truncated in a keyframe")
            _, dt_ms, bg, border = _FRAME_HEAD.unpack_from(raw, offset)
            body = offset + _FRAME_HEAD.size
            screen = np.frombuffer(raw, np.uint8, CELLS, body).copy()
            color = np.frombuffer(raw, np.uint8, CELLS, body + CELLS).copy()
            frames.append(PetvFrame(screen.copy(), color.copy(), bg, border, dt_ms, True))
            offset += keyframe_size
        elif kind == DELTA:
            head_end = offset + _FRAME_HEAD.size + _DELTA_COUNT.size
            if head_end > len(raw):
                raise ValueError(".petv truncated in a delta header")
            _, dt_ms, bg, border = _FRAME_HEAD.unpack_from(raw, offset)
            (count,) = _DELTA_COUNT.unpack_from(raw, offset + _FRAME_HEAD.size)
            end = head_end + count * 4
            if end > len(raw):
                raise ValueError(".petv truncated in a delta body")
            body = np.frombuffer(raw, np.uint8, count * 4, head_end).reshape(count, 4)
            indices = body[:, 0:2].copy().view("<u2").reshape(-1)
            if count and int(indices.max()) >= CELLS:
                raise ValueError(f".petv delta cell index {int(indices.max())} out of range")
            screen[indices] = body[:, 2]
            color[indices] = body[:, 3]
            frames.append(PetvFrame(screen.copy(), color.copy(), bg, border, dt_ms, False))
            offset = end
        else:
            raise ValueError(f"unknown .petv record type 0x{kind:02x}")

    return PetvStream(charset=charset, flags=flags, frames=frames)


def frames_from_stream(stream: PetvStream) -> Iterable[PetsciiFrame]:
    """The stream's frames as renderable :class:`PetsciiFrame` objects."""
    return (f.to_frame(stream.charset) for f in stream.frames)
