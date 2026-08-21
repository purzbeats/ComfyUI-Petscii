"""
PETSCII conversion engine — pure numpy, node-free.

Implements core-spec.md §3-5 so it can be imported and tested without ComfyUI or
torch. The node layer (`../nodes.py`) converts torch tensors to numpy at the
boundary and back.

The algorithm is normatively defined in `../../core-spec.md`; the frozen fixtures
in `../../fixtures/` are the contract this port has to meet, and `compare.py` is
the comparator that decides.
"""

from .compare import compare_frames
from .crt import CrtSettings, apply_crt
from .data_loader import (
    palette_names,
    palette_oklab,
    palette_rgb8,
    subset_description,
    subset_names,
)
from .engine import (
    CELLS,
    COLS,
    ROWS,
    SCREEN_H,
    SCREEN_W,
    HysteresisState,
    PetsciiFrame,
    Settings,
    convert,
    convert_batch,
    frame_to_screen,
    iter_convert_batch,
    pre_adjust_to_oklab,
    vote_background,
)
from .petv import PetvStream, encode_petv, frames_from_stream, read_petv, write_petv
from .render import add_border, render_frame, render_frames, scanline_overlay

__version__ = "0.1.0"

__all__ = [
    "CELLS",
    "COLS",
    "ROWS",
    "SCREEN_H",
    "SCREEN_W",
    "CrtSettings",
    "HysteresisState",
    "PetsciiFrame",
    "PetvStream",
    "Settings",
    "__version__",
    "add_border",
    "apply_crt",
    "compare_frames",
    "convert",
    "convert_batch",
    "encode_petv",
    "frame_to_screen",
    "frames_from_stream",
    "iter_convert_batch",
    "palette_names",
    "palette_oklab",
    "palette_rgb8",
    "pre_adjust_to_oklab",
    "read_petv",
    "render_frame",
    "render_frames",
    "scanline_overlay",
    "subset_description",
    "subset_names",
    "vote_background",
    "write_petv",
]
