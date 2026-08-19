"""
ComfyUI-PETSCII nodes (v3 API).

The engine lives in `petscii_core` and knows nothing about ComfyUI or torch; this
layer is the boundary that converts tensors to numpy and back, and nothing else.
That split is what lets the parity tests run without ComfyUI installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io, ui

# Relative, not absolute: dropping this pack into `custom_nodes/` does not put
# `src/` on sys.path, so `import petscii_core` would only work for a pip install.
from .petscii_core import (
    CrtSettings,
    PetsciiFrame,
    Settings,
    apply_crt,
    convert,
    convert_batch,
    encode_petv,
    palette_names,
    render_frame,
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


def _to_numpy(image: torch.Tensor) -> list[np.ndarray]:
    """ComfyUI IMAGE ``[B,H,W,C]`` float 0..1 to a list of ``(H, W, 3)`` uint8."""
    if image.dim() != 4:
        raise ValueError(f"expected a 4D IMAGE tensor [B,H,W,C], got {image.dim()}D")
    clipped = image[..., :3].clamp(0.0, 1.0).detach().cpu().numpy()
    return [np.rint(frame * 255.0).astype(np.uint8) for frame in clipped]


def _to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """A list of ``(H, W, 3)`` uint8 back to a ComfyUI IMAGE batch."""
    stacked = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def _render_all(
    data: PetsciiData,
    scale: int,
    border: int,
    crt: CrtSettings | None = None,
) -> torch.Tensor:
    images = []
    for frame in data.frames:
        image = render_frame(frame, scale)
        # Border first, so the CRT's curvature and vignette bend the frame too
        # rather than stopping at the edge of the screen.
        if border > 0:
            image = add_border(image, frame.border, border * scale)
        if crt is not None:
            image = apply_crt(image, crt, scale)
        images.append(image)
    return _to_tensor(images)


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
                io.Combo.Input("charset", options=_CHARSETS, default=_CHARSETS[0]),
                io.Combo.Input(
                    "subset",
                    options=_SUBSETS,
                    default="all",
                    tooltip="Restricting the glyph vocabulary is the biggest stylistic lever.",
                ),
                io.Combo.Input("background", options=_BG_OPTIONS, default="auto"),
                io.Combo.Input("border", options=_BORDER_OPTIONS, default="bg"),
                io.Combo.Input("framing", options=_FRAMING, default="cover"),
                io.Float.Input("brightness", default=0.0, min=-0.5, max=0.5, step=0.01),
                io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.2, max=3.0, step=0.01),
                io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Boolean.Input(
                    "dither",
                    default=False,
                    tooltip="Cell-level Floyd-Steinberg. Much slower, and stills only.",
                ),
                io.Int.Input("render_scale", default=1, min=1, max=8),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Custom(PETSCII_TYPE).Output(display_name="petscii"),
            ],
        )

    @classmethod
    def execute(
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
    ) -> io.NodeOutput:
        settings = _settings(
            charset, subset, background, border, framing,
            brightness, contrast, gamma, saturation, dither=dither, eps=0.0,
        )
        frames = [convert(frame, settings) for frame in _to_numpy(image)]
        data = PetsciiData(frames=frames, charset=settings.charset)
        rendered = _render_all(data, render_scale, 0)
        return io.NodeOutput(rendered, data, ui=ui.PreviewImage(rendered, cls=cls))


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
                    tooltip="Per-pixel switch threshold. Higher holds the picture stiller; 0 disables.",
                ),
                io.Combo.Input("charset", options=_CHARSETS, default=_CHARSETS[0]),
                io.Combo.Input("subset", options=_SUBSETS, default="all"),
                io.Combo.Input("background", options=_BG_OPTIONS, default="auto"),
                io.Combo.Input("border", options=_BORDER_OPTIONS, default="bg"),
                io.Combo.Input("framing", options=_FRAMING, default="cover"),
                io.Float.Input("brightness", default=0.0, min=-0.5, max=0.5, step=0.01),
                io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.2, max=3.0, step=0.01),
                io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Int.Input(
                    "background_sample",
                    default=8,
                    min=1,
                    max=64,
                    tooltip="How many frames the background vote samples.",
                ),
                io.Float.Input("fps", default=30.0, min=1.0, max=240.0, step=0.1),
                io.Int.Input("render_scale", default=1, min=1, max=8),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Custom(PETSCII_TYPE).Output(display_name="petscii"),
            ],
        )

    @classmethod
    def execute(
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
    ) -> io.NodeOutput:
        settings = _settings(
            charset, subset, background, border, framing,
            brightness, contrast, gamma, saturation, dither=False, eps=eps,
        )
        frames = convert_batch(
            _to_numpy(images), settings, temporal=True, bg_sample=background_sample
        )
        data = PetsciiData(frames=frames, charset=settings.charset, fps=fps)
        rendered = _render_all(data, render_scale, 0)
        return io.NodeOutput(rendered, data, ui=ui.PreviewImage(rendered, cls=cls))


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
                io.Int.Input("scale", default=3, min=1, max=8),
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
                io.Float.Input("crt_scanlines", default=0.45, min=0.0, max=1.0, step=0.01),
                io.Float.Input("crt_curvature", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("crt_glow", default=0.5, min=0.0, max=2.0, step=0.01),
                io.Float.Input("crt_vignette", default=0.4, min=0.0, max=1.0, step=0.01),
                io.Float.Input("crt_chroma", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Float.Input("crt_brightness", default=1.05, min=0.2, max=2.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
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
    ) -> io.NodeOutput:
        if not petscii.frames:
            raise ValueError("PETSCII input holds no frames")
        settings = (
            CrtSettings(
                scanlines=crt_scanlines,
                curvature=crt_curvature,
                glow=crt_glow,
                vignette=crt_vignette,
                chroma=crt_chroma,
                brightness=crt_brightness,
            )
            if crt
            else None
        )
        rendered = _render_all(petscii, scale, border, settings)
        return io.NodeOutput(rendered, ui=ui.PreviewImage(rendered, cls=cls))


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
    def execute(cls, petscii: PetsciiData, filename_prefix: str, fps: float) -> io.NodeOutput:
        if not petscii.frames:
            raise ValueError("PETSCII input holds no frames")

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

        return io.NodeOutput(
            path,
            ui={"petv": [{"filename": name, "subfolder": subfolder, "type": "output"}]},
        )


class PetsciiExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PETSCIIConvert, PETSCIIVideoConvert, PETSCIIRender, PETSCIISavePETV]


async def comfy_entrypoint() -> ComfyExtension:
    return PetsciiExtension()
