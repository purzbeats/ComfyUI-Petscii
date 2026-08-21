"""
ComfyUI-PETSCII nodes (v3 API).

The engine lives in `petscii_core` and knows nothing about ComfyUI or torch; this
layer is the boundary that converts tensors to numpy and back, and nothing else.
That split is what lets the parity tests run without ComfyUI installed.

Two things shape the code here more than anything else. A clip is a batch, and a
batch can be thousands of frames — so nothing materialises a whole sequence that
could be walked one frame at a time, and every loop over frames reports progress
and can be interrupted. And the conversion is the expensive half: `PETSCII`
carries cells, not pixels, so changing the look downstream never reconverts.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from comfy_api.latest import ComfyAPI, ComfyExtension, io, ui

# Relative, not absolute: dropping this pack into `custom_nodes/` does not put
# `src/` on sys.path, so `import petscii_core` would only work for a pip install.
from .petscii_core import (
    CELLS,
    COLS,
    ROWS,
    CrtSettings,
    PetsciiFrame,
    Settings,
    apply_crt,
    encode_petv,
    frames_from_stream,
    iter_convert_batch,
    palette_names,
    read_petv,
    render_frame,
    subset_description,
    subset_names,
)
from .petscii_core.render import add_border

PETSCII_TYPE = "PETSCII"
CATEGORY = "image/petscii"

_SUBSETS = list(subset_names())
_CHARSETS = ["0 — upper/graphics", "1 — lower/upper"]
_FRAMING = ["cover", "contain", "stretch"]
_BG_OPTIONS = ["auto"] + [f"{i} — {name}" for i, name in enumerate(palette_names())]
_BORDER_OPTIONS = ["bg"] + [f"{i} — {name}" for i, name in enumerate(palette_names())]

#: Widget copy that would otherwise be repeated verbatim on both convert nodes.
_TIP_SUBSET = (
    "Restricting the glyph vocabulary is the biggest stylistic lever. "
    + "; ".join(f"{name}: {subset_description(name)}" for name in _SUBSETS if subset_description(name))
)
_TIP_CHARSET = "Set 0 is uppercase and graphics — the one C64 art uses. Set 1 swaps in lowercase."
_TIP_BACKGROUND = (
    "The one background colour behind every cell. 'auto' scores all 16 and keeps the cheapest."
)
_TIP_BORDER = "Surrounds the screen. Never affects the conversion — see PETSCII Set Border."
_TIP_FRAMING = (
    "How a non-8:5 image reaches 320x200. cover crops to fill, contain letterboxes in the "
    "background colour, stretch distorts."
)
_TIP_BRIGHTNESS = "Added to sRGB before contrast and gamma."
_TIP_CONTRAST = "Scales around mid-grey, after brightness and before gamma."
_TIP_GAMMA = "Applied as an exponent of 1/gamma, so above 1 brightens midtones. Runs last of the three."
_TIP_SATURATION = "Scales the Oklab chroma axes after the tone curve. 0 is greyscale."
_TIP_PREVIEW = (
    "How many frames to write to the node preview. Each one is a PNG on disk, so a long clip "
    "with this turned up costs real time and space. 0 shows none."
)
_TIP_SCALE = "Integer nearest-neighbour upscale. Anything else would reintroduce the resampling this avoids."

api = ComfyAPI()


@dataclass
class PetsciiData:
    """
    The custom `PETSCII` datatype: a list of frames plus the settings that made
    them, so a downstream node can re-render or re-encode without reconverting.
    """

    frames: list[PetsciiFrame] = field(default_factory=list)
    charset: int = 0
    #: Frames per second when the source was a sequence; used by the writer.
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.frames)


# ------------------------------------------------------------------ helpers


def _index(choice: str) -> int:
    """`"9 — brown"` to `9`."""
    return int(choice.split("—")[0].strip())


def _settings(
    charset: str,
    subset: str,
    background: str,
    border: str,
    framing: str,
    brightness: float,
    contrast: float,
    gamma: float,
    saturation: float,
    dither: bool = False,
    eps: float = 0.0,
) -> Settings:
    return Settings(
        charset=_index(charset),
        subset=subset,
        bg_lock=None if background == "auto" else _index(background),
        border_lock=None if border == "bg" else _index(border),
        eps=eps,
        dither=dither,
        framing=framing,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        saturation=saturation,
    )


def _frame_to_numpy(frame: torch.Tensor) -> np.ndarray:
    """One ComfyUI IMAGE plane ``[H,W,C]`` float 0..1 to ``(H, W, 3)`` uint8."""
    clipped = frame[..., :3].clamp(0.0, 1.0).detach().cpu().numpy()
    return np.rint(clipped * 255.0).astype(np.uint8)


class _TensorFrames(Sequence):
    """
    A ComfyUI IMAGE batch seen as a sequence of ``(H, W, 3)`` uint8 arrays.

    Converting the whole batch up front would hold a second full copy of the clip
    in memory for as long as the conversion runs, on top of the float32 tensor it
    was copied from. The engine indexes its input rather than iterating it
    (:func:`iter_convert_batch`), so handing it this instead converts each frame
    at the moment it is needed and lets the previous one be collected.
    """

    def __init__(self, image: torch.Tensor) -> None:
        if image.dim() != 4:
            raise ValueError(f"expected a 4D IMAGE tensor [B,H,W,C], got {image.dim()}D")
        if image.shape[0] == 0:
            raise ValueError("the IMAGE input is an empty batch")
        self._image = image

    def __len__(self) -> int:
        return int(self._image.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
        return _frame_to_numpy(self._image[index])


def _paint(frame: PetsciiFrame, scale: int, border: int, crt: CrtSettings | None) -> np.ndarray:
    """One frame's cells to pixels, in the order the look depends on."""
    image = render_frame(frame, scale)
    # Border first, so the CRT's curvature and vignette bend the frame too
    # rather than stopping at the edge of the screen.
    if border > 0:
        image = add_border(image, frame.border, border * scale)
    if crt is not None:
        image = apply_crt(image, crt, scale)
    return image


def _memory_ceiling() -> tuple[int, str] | None:
    """
    A byte budget for the output batch, and the word for what it measures.

    Free memory is the number worth checking against, but only Linux reports it
    through `sysconf`: macOS has `SC_PHYS_PAGES` and no `SC_AVPHYS_PAGES`, and
    Windows has no `sysconf` at all. So this falls back to installed memory,
    which is a weaker bound but still a real one — a batch larger than the
    machine's RAM is never going to work, whatever else is running.

    Deliberately dependency-free. `psutil` would answer this everywhere, and
    taking a dependency to improve an error message is a bad trade for a pack
    whose entire install is numpy. Where nothing can be read, the allocation goes
    ahead and :func:`_allocate_batch` catches the failure instead.
    """
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None

    for name, kind in (("SC_AVPHYS_PAGES", "free"), ("SC_PHYS_PAGES", "installed")):
        try:
            pages = os.sysconf(name)
        except (AttributeError, ValueError, OSError):
            continue
        if pages > 0:
            return page * pages, kind
    return None


def _allocate_batch(count: int, height: int, width: int, scale: int) -> torch.Tensor:
    """
    The output IMAGE batch, or an error that says what to change.

    An IMAGE output is one contiguous float32 tensor — four bytes per channel per
    pixel per frame, and no node can hand back less than that. It grows with the
    square of the scale: a thousand frames is 0.8 GB at 1x and 49 GB at 8x. The
    failure was a bare allocation error thrown after the first frame had already
    been painted, which says nothing about which of the two knobs to reach for.

    Checked ahead of time against :func:`_memory_ceiling`, and caught either way.
    The up-front check is doing nearly all the work: both Linux overcommit and
    macOS's lazy mapping hand back a 46 GiB tensor on a 16 GiB machine without
    complaint, and only fail later, somewhere with no idea what to suggest.
    """
    needed = count * height * width * 3 * 4
    advice = (
        f"{count} frames at {width}x{height} (render scale {scale}) needs "
        f"{needed / 1024**3:.1f} GiB as one IMAGE batch. Lower the render scale — it costs "
        f"memory as the square — or send fewer frames at a time. To keep the cells without "
        f"paying for pixels, save the PETSCII output as .petv and render it in passes."
    )

    ceiling = _memory_ceiling()
    if ceiling is not None and needed > ceiling[0]:
        raise MemoryError(f"{advice} This machine has {ceiling[0] / 1024**3:.1f} GiB {ceiling[1]}.")
    try:
        return torch.empty((count, height, width, 3), dtype=torch.float32)
    except (MemoryError, RuntimeError) as exc:
        raise MemoryError(advice) from exc


def _store(out: torch.Tensor, index: int, painted: np.ndarray) -> None:
    """One painted frame into its slice of the output batch, as float 0..1."""
    out[index] = torch.from_numpy(painted).to(torch.float32).div_(255.0)


def _paint_workers(count: int) -> int:
    """
    How many threads to paint with.

    Painting is pure numpy over its own arrays, and numpy drops the GIL for the
    gathers and multiplies that dominate it, so threads genuinely scale here —
    measured 5.6x on eight of them for a CRT render. Processes would not: a
    painted 8x frame is 12 MB, and pickling that back from a worker costs more
    than painting it did.

    Capped at eight because the window below keeps two frames in flight per
    worker, and beyond that the memory held mid-render starts to matter more
    than the wall clock saved.
    """
    if count < 2:
        return 1
    return max(1, min(count, os.cpu_count() or 1, 8))


async def _render_all(
    data: PetsciiData,
    scale: int,
    border: int,
    crt: CrtSettings | None = None,
    *,
    report: bool = True,
) -> torch.Tensor:
    """
    Renders every frame into one preallocated IMAGE batch, painting in parallel.

    Collecting the frames in a list and stacking them at the end would hold the
    sequence three times over at the moment of the stack — the list, the stacked
    uint8 copy, and the float32 result. At a few hundred frames of scaled-up
    output that is gigabytes for no reason, so each frame is written straight
    into its slice of the final tensor and then dropped.

    Frame zero is painted alone, both to learn the output shape and to warm the
    charset and palette caches before any thread touches them. The rest go
    through a pool, drained strictly in order: out-of-order completion would buy
    very little when every frame costs the same, and in-order draining is what
    keeps the progress bar monotonic and the interrupt check somewhere obvious.
    """
    count = len(data.frames)
    first = _paint(data.frames[0], scale, border, crt)
    height, width = first.shape[:2]
    out = _allocate_batch(count, height, width, scale)
    _store(out, 0, first)
    del first
    if report:
        await _progress(1, count)
    if count == 1:
        return out

    workers = _paint_workers(count)
    if workers == 1:
        for index in range(1, count):
            _interrupted()
            _store(out, index, _paint(data.frames[index], scale, border, crt))
            if report:
                await _progress(index + 1, count)
        return out

    # Two frames in flight per worker: enough that no thread ever waits for the
    # main one to catch up, few enough that a long clip does not hold hundreds of
    # painted frames at once — which is the whole reason the output is
    # preallocated rather than collected.
    window = workers * 2
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="petscii-paint")
    queue: deque[tuple[int, Future]] = deque()
    submitted = 1

    def submit_up_to_the_window() -> None:
        nonlocal submitted
        while submitted < count and len(queue) < window:
            queue.append((submitted, pool.submit(_paint, data.frames[submitted], scale, border, crt)))
            submitted += 1

    try:
        submit_up_to_the_window()
        while queue:
            _interrupted()
            index, future = queue.popleft()
            _store(out, index, future.result())
            submit_up_to_the_window()
            if report:
                await _progress(index + 1, count)
    finally:
        # cancel_futures matters on the interrupt path: without it, shutdown
        # waits for every frame already queued to be painted and thrown away.
        pool.shutdown(wait=True, cancel_futures=True)
    return out


try:
    from comfy.model_management import throw_exception_if_processing_interrupted as _interrupted
except ImportError:  # pragma: no cover - only outside a running ComfyUI

    def _interrupted() -> None:
        """
        Lets a long conversion answer the interrupt button between frames.

        Resolved once at import rather than per frame. The fallback exists so this
        module can be imported by the tests that cover the tensor boundary; inside
        ComfyUI the real one is always what binds.
        """


async def _progress(done: int, total: int) -> None:
    """
    Drives the node's progress bar.

    Best-effort: a node run outside an executing context (a unit test, a direct
    call) has no progress state to update, and that is not a reason to fail a
    conversion that is otherwise going fine.
    """
    # Suppressed, not logged: a progress update that cannot land is not a problem
    # worth a line of output on every frame of a thousand-frame clip.
    with contextlib.suppress(Exception):
        await api.execution.set_progress(done, total)


def _preview(rendered: torch.Tensor, limit: int, cls: type[io.ComfyNode]) -> ui.PreviewImage | None:
    """
    The node preview, capped.

    Every previewed frame is a PNG written to the temp directory on every run, so
    an uncapped preview of a long clip costs more than the conversion did.
    """
    if limit <= 0 or rendered.shape[0] == 0:
        return None
    return ui.PreviewImage(rendered[:limit], cls=cls)


def _crt_settings(
    enabled: bool,
    scanlines: float,
    curvature: float,
    glow: float,
    vignette: float,
    chroma: float,
    brightness: float,
) -> CrtSettings | None:
    if not enabled:
        return None
    return CrtSettings(
        scanlines=scanlines,
        curvature=curvature,
        glow=glow,
        vignette=vignette,
        chroma=chroma,
        brightness=brightness,
    )


def _require_frames(petscii: PetsciiData) -> PetsciiData:
    if not petscii.frames:
        raise ValueError("PETSCII input holds no frames")
    return petscii


def _adjust_inputs() -> list[io.Input]:
    """The four pre-adjust knobs of core-spec §5, in the order they are applied."""
    return [
        io.Float.Input("brightness", default=0.0, min=-0.5, max=0.5, step=0.01, tooltip=_TIP_BRIGHTNESS),
        io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.01, tooltip=_TIP_CONTRAST),
        io.Float.Input("gamma", default=1.0, min=0.2, max=3.0, step=0.01, tooltip=_TIP_GAMMA),
        io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.01, tooltip=_TIP_SATURATION),
    ]


# -------------------------------------------------------------------- nodes


class PETSCIIConvert(io.ComfyNode):
    """Converts images to PETSCII, each frame independently."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIIConvert",
            display_name="PETSCII Convert",
            category=CATEGORY,
            description=(
                "Converts an image to a C64 PETSCII screen: optimal glyph and colour "
                "per cell in Oklab, with the background chosen by a full 16-candidate "
                "search. A batch is treated as independent stills — use PETSCII Video "
                "Convert for a sequence that should hold still between frames."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("charset", options=_CHARSETS, default=_CHARSETS[0], tooltip=_TIP_CHARSET),
                io.Combo.Input("subset", options=_SUBSETS, default="all", tooltip=_TIP_SUBSET),
                io.Combo.Input("background", options=_BG_OPTIONS, default="auto", tooltip=_TIP_BACKGROUND),
                io.Combo.Input("border", options=_BORDER_OPTIONS, default="bg", tooltip=_TIP_BORDER),
                io.Combo.Input("framing", options=_FRAMING, default="cover", tooltip=_TIP_FRAMING),
                *_adjust_inputs(),
                io.Boolean.Input(
                    "dither",
                    default=False,
                    tooltip="Cell-level Floyd-Steinberg. Roughly 3x slower, and stills only.",
                ),
                io.Int.Input("render_scale", default=1, min=1, max=8, tooltip=_TIP_SCALE),
                io.Int.Input("preview_frames", default=8, min=0, max=64, tooltip=_TIP_PREVIEW),
            ],
            outputs=[
                io.Image.Output(display_name="image", tooltip="The converted screens, rendered."),
                io.Custom(PETSCII_TYPE).Output(
                    display_name="petscii",
                    tooltip="The cells themselves. Feed PETSCII Render or Save .petv to avoid reconverting.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        charset: str,
        subset: str,
        background: str,
        border: str,
        framing: str,
        brightness: float,
        contrast: float,
        gamma: float,
        saturation: float,
        dither: bool,
        render_scale: int,
        preview_frames: int,
    ) -> io.NodeOutput:
        settings = _settings(
            charset, subset, background, border, framing,
            brightness, contrast, gamma, saturation, dither=dither, eps=0.0,
        )
        sources = _TensorFrames(image)
        if dither and len(sources) > 1:
            logging.warning(
                "PETSCII Convert: dithering %d frames as independent stills. Cell-level error "
                "diffusion cannot be made temporally stable (core-spec §4.7), so a sequence "
                "dithered this way will crawl — PETSCII Video Convert is the stable option.",
                len(sources),
            )

        frames: list[PetsciiFrame] = []
        total = len(sources)
        await _progress(0, total)
        for done, frame in enumerate(iter_convert_batch(sources, settings), start=1):
            frames.append(frame)
            _interrupted()
            await _progress(done, total)

        data = PetsciiData(frames=frames, charset=settings.charset)
        rendered = await _render_all(data, render_scale, 0, report=False)
        return io.NodeOutput(rendered, data, ui=_preview(rendered, preview_frames, cls))


class PETSCIIVideoConvert(io.ComfyNode):
    """Converts a batch as a sequence, with temporal hysteresis and one background."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIIVideoConvert",
            display_name="PETSCII Video Convert",
            category=CATEGORY,
            description=(
                "Converts an image batch as a sequence. Cells keep their previous "
                "choice unless a new one is better by more than eps, which is what "
                "stops the picture boiling between frames, and the background is "
                "voted once over sampled frames then locked."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Float.Input(
                    "eps",
                    default=0.002,
                    min=0.0,
                    max=0.02,
                    step=0.0001,
                    tooltip=(
                        "Per-pixel switch threshold: a cell only changes when the new choice "
                        "beats the old by more than eps x 64. Higher holds the picture stiller; "
                        "0.002-0.008 is the useful range; 0 disables."
                    ),
                ),
                io.Combo.Input("charset", options=_CHARSETS, default=_CHARSETS[0], tooltip=_TIP_CHARSET),
                io.Combo.Input("subset", options=_SUBSETS, default="all", tooltip=_TIP_SUBSET),
                io.Combo.Input("background", options=_BG_OPTIONS, default="auto", tooltip=_TIP_BACKGROUND),
                io.Combo.Input("border", options=_BORDER_OPTIONS, default="bg", tooltip=_TIP_BORDER),
                io.Combo.Input("framing", options=_FRAMING, default="cover", tooltip=_TIP_FRAMING),
                *_adjust_inputs(),
                io.Int.Input(
                    "background_sample",
                    default=8,
                    min=1,
                    max=64,
                    tooltip=(
                        "How many evenly spaced frames the background vote scores. The totals "
                        "are stable, so more than a handful rarely changes the answer and every "
                        "extra one costs a conversion."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=30.0,
                    min=1.0,
                    max=240.0,
                    step=0.1,
                    tooltip="Carried on the PETSCII output for the .petv writer. Does not affect conversion.",
                ),
                io.Int.Input("render_scale", default=1, min=1, max=8, tooltip=_TIP_SCALE),
                io.Int.Input("preview_frames", default=8, min=0, max=64, tooltip=_TIP_PREVIEW),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The converted screens, rendered."),
                io.Custom(PETSCII_TYPE).Output(
                    display_name="petscii",
                    tooltip="The cell stream, carrying fps for Save .petv.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        images: torch.Tensor,
        eps: float,
        charset: str,
        subset: str,
        background: str,
        border: str,
        framing: str,
        brightness: float,
        contrast: float,
        gamma: float,
        saturation: float,
        background_sample: int,
        fps: float,
        render_scale: int,
        preview_frames: int,
    ) -> io.NodeOutput:
        settings = _settings(
            charset, subset, background, border, framing,
            brightness, contrast, gamma, saturation, dither=False, eps=eps,
        )
        sources = _TensorFrames(images)
        total = len(sources)
        await _progress(0, total)

        frames: list[PetsciiFrame] = []
        stream = iter_convert_batch(sources, settings, temporal=True, bg_sample=background_sample)
        for done, frame in enumerate(stream, start=1):
            frames.append(frame)
            _interrupted()
            await _progress(done, total)

        data = PetsciiData(frames=frames, charset=settings.charset, fps=fps)
        rendered = await _render_all(data, render_scale, 0, report=False)
        return io.NodeOutput(rendered, data, ui=_preview(rendered, preview_frames, cls))


class PETSCIIRender(io.ComfyNode):
    """Re-renders PETSCII data without reconverting."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIIRender",
            display_name="PETSCII Render",
            category=CATEGORY,
            description=(
                "Renders PETSCII data to images at an integer scale, with an optional "
                "CRT treatment. Separate from conversion so you can change the look "
                "without paying for the conversion again."
            ),
            inputs=[
                io.Custom(PETSCII_TYPE).Input("petscii"),
                io.Int.Input("scale", default=3, min=1, max=8, tooltip=_TIP_SCALE),
                io.Int.Input(
                    "border",
                    default=0,
                    min=0,
                    max=32,
                    tooltip="Border thickness in screen pixels, before scaling.",
                ),
                io.Boolean.Input(
                    "crt",
                    default=False,
                    tooltip=(
                        "Scanlines, aperture grille, barrel distortion, phosphor glow, "
                        "vignette and chroma bleed. Effect sizes follow the PETSCII grid, "
                        "so the look is the same at any scale — but scale 3 or more is "
                        "needed for scanlines to have room to show."
                    ),
                ),
                io.Float.Input(
                    "crt_scanlines",
                    default=0.45, min=0.0, max=1.0, step=0.01,
                    tooltip="Darkening of alternate lines, and the aperture-grille tint with it.",
                ),
                io.Float.Input(
                    "crt_curvature",
                    default=0.35, min=0.0, max=1.0, step=0.01,
                    tooltip="Barrel distortion, as if the glass were curved.",
                ),
                io.Float.Input(
                    "crt_glow",
                    default=0.5, min=0.0, max=2.0, step=0.01,
                    tooltip="Phosphor bloom around bright cells.",
                ),
                io.Float.Input(
                    "crt_vignette",
                    default=0.4, min=0.0, max=1.0, step=0.01,
                    tooltip="Corner darkening.",
                ),
                io.Float.Input(
                    "crt_chroma",
                    default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="Sideways red/blue smear, the way composite video bleeds.",
                ),
                io.Float.Input(
                    "crt_brightness",
                    default=1.05, min=0.2, max=2.0, step=0.01,
                    tooltip=(
                        "Output level, applied last. Every intensity at 0 with this at 1 "
                        "is a true bypass."
                    ),
                ),
                io.Int.Input("preview_frames", default=8, min=0, max=64, tooltip=_TIP_PREVIEW),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    async def execute(
        cls,
        petscii: PetsciiData,
        scale: int,
        border: int,
        crt: bool,
        crt_scanlines: float,
        crt_curvature: float,
        crt_glow: float,
        crt_vignette: float,
        crt_chroma: float,
        crt_brightness: float,
        preview_frames: int,
    ) -> io.NodeOutput:
        _require_frames(petscii)
        if crt and scale < 3:
            logging.warning(
                "PETSCII Render: the CRT treatment at scale %d has no room for scanlines — "
                "they need at least 3 output pixels per screen pixel to show.",
                scale,
            )
        settings = _crt_settings(
            crt, crt_scanlines, crt_curvature, crt_glow, crt_vignette, crt_chroma, crt_brightness
        )
        rendered = await _render_all(petscii, scale, border, settings)
        return io.NodeOutput(rendered, ui=_preview(rendered, preview_frames, cls))


class PETSCIISetBorder(io.ComfyNode):
    """Retints the border of PETSCII data without reconverting."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIISetBorder",
            display_name="PETSCII Set Border",
            category=CATEGORY,
            description=(
                "Changes the border colour of already-converted PETSCII. The border never "
                "enters the error math (core-spec §4.5), so this is free — unlike changing "
                "the background, which decides every cell and needs a reconversion."
            ),
            inputs=[
                io.Custom(PETSCII_TYPE).Input("petscii"),
                io.Combo.Input(
                    "border",
                    options=_BORDER_OPTIONS,
                    default="bg",
                    tooltip="'bg' tracks each frame's background, as a stock C64 screen does.",
                ),
            ],
            outputs=[io.Custom(PETSCII_TYPE).Output(display_name="petscii")],
        )

    @classmethod
    def execute(cls, petscii: PetsciiData, border: str) -> io.NodeOutput:
        _require_frames(petscii)
        chosen = None if border == "bg" else _index(border)
        # A new list of new frames: the input is another node's cached output and
        # must not be mutated, or a second run would see the retinted copy.
        frames = [
            replace(frame, border=frame.bg if chosen is None else chosen)
            for frame in petscii.frames
        ]
        return io.NodeOutput(replace(petscii, frames=frames))


class PETSCIIInfo(io.ComfyNode):
    """Describes PETSCII data as text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIIInfo",
            display_name="PETSCII Info",
            category=CATEGORY,
            description=(
                "Reports what a PETSCII stream actually contains — frame count, background "
                "and border, how much of the picture moves between frames, and which glyphs "
                "the subset really used. Useful for tuning eps and for seeing whether a "
                "subset is as restrictive as it looks."
            ),
            inputs=[
                io.Custom(PETSCII_TYPE).Input("petscii"),
                io.Int.Input(
                    "top_glyphs",
                    default=8,
                    min=0,
                    max=64,
                    tooltip="How many of the most-used screen codes to list.",
                ),
            ],
            outputs=[io.String.Output(display_name="info")],
            # An output node so it shows its report on its own, without having to be
            # wired into something that terminates the branch.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, petscii: PetsciiData, top_glyphs: int) -> io.NodeOutput:
        _require_frames(petscii)
        frames = petscii.frames
        names = palette_names()
        screens = np.stack([f.screen for f in frames])
        colors = np.stack([f.color for f in frames])

        lines = [
            f"frames      {len(frames)}  ({COLS}x{ROWS} cells, {CELLS} per frame)",
            f"charset     {petscii.charset} — {'upper/graphics' if petscii.charset == 0 else 'lower/upper'}",
            f"fps         {petscii.fps:g}",
        ]

        backgrounds = sorted({int(f.bg) for f in frames})
        borders = sorted({int(f.border) for f in frames})
        lines.append("background  " + ", ".join(f"{b} — {names[b]}" for b in backgrounds))
        lines.append("border      " + ", ".join(f"{b} — {names[b]}" for b in borders))

        distinct = np.unique(screens)
        lines.append(f"glyphs used {len(distinct)} of 256")
        lines.append(f"colours     {len(np.unique(colors))} of 16 used as foreground")

        if len(frames) > 1:
            moved = (screens[1:] != screens[:-1]) | (colors[1:] != colors[:-1])
            per_frame = moved.sum(axis=1)
            lines.append(
                f"motion      {per_frame.mean():.1f} cells/frame changed "
                f"({per_frame.mean() / CELLS * 100:.1f}%), peak {int(per_frame.max())}"
            )

        if top_glyphs > 0:
            counts = np.bincount(screens.reshape(-1), minlength=256)
            order = np.argsort(counts)[::-1][:top_glyphs]
            total = float(screens.size)
            listed = ", ".join(
                f"{int(code)} ({counts[code] / total * 100:.1f}%)" for code in order if counts[code]
            )
            lines.append(f"top glyphs  {listed}")

        text = "\n".join(lines)
        return io.NodeOutput(text, ui=ui.PreviewText(text))


class PETSCIILoadPETV(io.ComfyNode):
    """Reads a `.petv` stream back into PETSCII data."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIILoadPETV",
            display_name="PETSCII Load .petv",
            category=CATEGORY,
            description=(
                "Loads a .petv stream (core-spec §8.1) from the input directory as PETSCII "
                "data — deltas applied, every frame materialised. Streams written by the web "
                "player or by Save .petv come back in as cells, so they can be re-rendered at "
                "any scale, given the CRT treatment, or exported as frames without ever being "
                "reconverted."
            ),
            inputs=[
                io.Combo.Input(
                    "petv",
                    options=_petv_files(),
                    tooltip="A .petv file in the ComfyUI input directory.",
                ),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=240.0,
                    step=0.1,
                    tooltip="0 derives the rate from the stream's own frame timings.",
                ),
            ],
            outputs=[
                io.Custom(PETSCII_TYPE).Output(display_name="petscii"),
                io.Int.Output(display_name="frame_count"),
            ],
        )

    @classmethod
    def validate_inputs(cls, petv: str) -> bool | str:
        import folder_paths

        if not petv:
            return "no .petv file selected — put one in the ComfyUI input directory"
        if not folder_paths.exists_annotated_filepath(petv):
            return f"{petv} is not in the input directory"
        return True

    @classmethod
    def fingerprint_inputs(cls, petv: str, **kwargs):
        """Re-read when the file changes underneath an unchanged filename."""
        import folder_paths

        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(petv))
        except OSError:
            return float("NaN")

    @classmethod
    def execute(cls, petv: str, fps: float) -> io.NodeOutput:
        import folder_paths

        path = folder_paths.get_annotated_filepath(petv)
        with open(path, "rb") as fh:
            stream = read_petv(fh.read())
        frames = list(frames_from_stream(stream))
        if not frames:
            raise ValueError(f"{petv} holds no frames")

        data = PetsciiData(frames=frames, charset=stream.charset, fps=fps or _stream_fps(stream))
        return io.NodeOutput(data, len(frames))


class PETSCIISavePETV(io.ComfyNode):
    """Writes a `.petv` stream to the ComfyUI output directory."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PETSCIISavePETV",
            display_name="PETSCII Save .petv",
            category=CATEGORY,
            description=(
                "Writes the cell stream as .petv (core-spec §8.1) — keyframes and "
                "deltas, true screen codes, one global background. This is the "
                "interchange format the web player reads and future C64 exporters "
                "will consume."
            ),
            inputs=[
                io.Custom(PETSCII_TYPE).Input("petscii"),
                io.String.Input("filename_prefix", default="petscii/petscii"),
                io.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.1,
                               tooltip="0 uses the rate carried by the PETSCII input."),
            ],
            outputs=[io.String.Output(display_name="path")],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, filename_prefix: str) -> bool | str:
        if not filename_prefix.strip():
            return "filename_prefix is empty"
        if os.path.isabs(filename_prefix) or ".." in filename_prefix.replace("\\", "/").split("/"):
            return "filename_prefix must stay inside the output directory"
        return True

    @classmethod
    def execute(cls, petscii: PetsciiData, filename_prefix: str, fps: float) -> io.NodeOutput:
        _require_frames(petscii)

        import folder_paths

        rate = fps if fps > 0 else petscii.fps
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(full_output_folder, exist_ok=True)
        name = f"{filename}_{counter:05}_.petv"
        path = os.path.join(full_output_folder, name)

        data = encode_petv(petscii.frames, fps=rate, charset=petscii.charset)
        with open(path, "wb") as fh:
            fh.write(data)

        result = {"filename": name, "subfolder": subfolder, "type": "output"}
        return io.NodeOutput(
            path,
            # "text" is what the frontend already knows how to show; "petv" is what
            # web/petscii.js turns into a download link. A client with neither still
            # gets the path on the node's output.
            ui={"text": [f"{len(petscii.frames)} frames, {len(data) / 1024:.1f} KiB\n{name}"],
                "petv": [result]},
        )


# ------------------------------------------------------------------ plumbing


def _petv_files() -> list[str]:
    """`.petv` files in the input directory, for the loader's combo."""
    try:
        import folder_paths

        directory = folder_paths.get_input_directory()
        return sorted(
            name
            for name in os.listdir(directory)
            if name.lower().endswith(".petv") and os.path.isfile(os.path.join(directory, name))
        )
    except (ImportError, OSError):
        return []


def _stream_fps(stream) -> float:
    """
    A playback rate for a stream that carries per-frame gaps instead of one.

    `.petv` is variable-rate by design (§8.1), so there is no single true answer;
    the mean gap is the one that makes a re-encode come out the same length as
    what was read.
    """
    gaps = [f.dt_ms for f in stream.frames[1:] if f.dt_ms > 0]
    if not gaps:
        return 30.0
    return round(1000.0 / (sum(gaps) / len(gaps)), 3)


class PetsciiExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PETSCIIConvert,
            PETSCIIVideoConvert,
            PETSCIIRender,
            PETSCIISetBorder,
            PETSCIIInfo,
            PETSCIILoadPETV,
            PETSCIISavePETV,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return PetsciiExtension()
