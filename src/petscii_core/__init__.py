"""
PETSCII conversion engine — pure numpy, node-free.

Implements core-spec.md §3-5 so it can be imported and tested without ComfyUI or
torch. The node layer (`../nodes.py`) converts torch tensors to numpy at the
boundary and back.

The algorithm is normatively defined in `../../core-spec.md`; the frozen fixtures
in `../../fixtures/` are the contract this port has to meet, and `compare.py` is
the comparator that decides.
"""

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
    pre_adjust_to_oklab,
    vote_background,
)
from .render import add_border, render_frame, render_frames, scanline_overlay
from .petv import encode_petv, read_petv, write_petv
from .compare import compare_frames
from .data_loader import (
    palette_names,
    palette_oklab,
    palette_rgb8,
    subset_description,
    subset_names,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CELLS",
    "COLS",
    "ROWS",
    "SCREEN_W",
    "SCREEN_H",
    "Settings",
    "PetsciiFrame",
    "HysteresisState",
    "convert",
    "convert_batch",
    "vote_background",
    "frame_to_screen",
    "pre_adjust_to_oklab",
    "render_frame",
    "render_frames",
    "add_border",
    "scanline_overlay",
    "encode_petv",
    "read_petv",
    "write_petv",
    "compare_frames",
    "palette_rgb8",
    "palette_oklab",
    "palette_names",
    "subset_names",
    "subset_description",
]
