"""
The node layer — the boundary that `petscii_core` deliberately knows nothing about.

Everything here is the wiring this pack owns and no other test touches: parsing
widget strings back into `Settings`, moving tensors to numpy and back, keeping the
preview from writing a PNG per frame, and the schemas agreeing with the `execute`
signatures they feed. That last one matters more than it looks — `_settings` takes
nine positional arguments and every convert node calls it positionally, so a pair
swapped there is silent and ships.

`comfy_api` is stubbed by `conftest.py` when a real one is absent; torch is not,
so this module skips whole when torch is missing.
"""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the node layer is the tensor boundary")

from petscii_core import PetsciiFrame, Settings, read_petv  # noqa: E402
from src import nodes  # noqa: E402
from src.nodes import (  # noqa: E402
    PETSCIIConvert,
    PetsciiData,
    PetsciiExtension,
    PETSCIIInfo,
    PETSCIILoadPETV,
    PETSCIIRender,
    PETSCIISavePETV,
    PETSCIISetBorder,
    PETSCIIVideoConvert,
)

CELLS = 1000

CONVERT_NODES = [PETSCIIConvert, PETSCIIVideoConvert]
ALL_NODES = [
    PETSCIIConvert,
    PETSCIIVideoConvert,
    PETSCIIRender,
    PETSCIISetBorder,
    PETSCIIInfo,
    PETSCIILoadPETV,
    PETSCIISavePETV,
]


def run(coro):
    """Drives one node's async execute to completion."""
    return asyncio.run(coro)


def image(batch: int = 1, height: int = 200, width: int = 320, seed: int = 0) -> torch.Tensor:
    """A deterministic IMAGE batch, `[B,H,W,C]` float 0..1."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((batch, height, width, 3), dtype=np.float32))


def frame(bg: int = 0, border: int = 0, value: int = 65) -> PetsciiFrame:
    return PetsciiFrame(
        screen=np.full(CELLS, value, dtype=np.uint8),
        color=np.full(CELLS, 1, dtype=np.uint8),
        bg=bg,
        border=border,
    )


# ------------------------------------------------------- schema / signature


@pytest.mark.parametrize("node", ALL_NODES, ids=lambda n: n.__name__)
def test_schema_inputs_match_execute_signature(node) -> None:
    """
    Every declared input must be a parameter of `execute`, and every parameter must
    be declared. ComfyUI calls `execute(**inputs)`, so a name that appears on only
    one side is a TypeError at run time and nothing catches it before then.
    """
    schema = node.define_schema()
    declared = {port.id for port in schema.inputs}
    parameters = set(inspect.signature(node.execute.__func__).parameters) - {"cls"}
    assert declared == parameters, f"{node.__name__}: {declared ^ parameters}"


@pytest.mark.parametrize("node", ALL_NODES, ids=lambda n: n.__name__)
def test_schema_is_well_formed(node) -> None:
    schema = node.define_schema()
    assert schema.node_id == node.__name__
    assert schema.category == nodes.CATEGORY
    assert schema.description
    assert schema.outputs
    ids = [port.id for port in schema.inputs]
    assert len(ids) == len(set(ids)), f"{node.__name__} declares a duplicate input"


def test_extension_registers_every_node() -> None:
    listed = run(PetsciiExtension().get_node_list())
    assert listed == ALL_NODES


# -------------------------------------------------------------- widget parsing


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("0 — upper/graphics", 0), ("1 — lower/upper", 1), ("9 — brown", 9), ("15 — light grey", 15)],
)
def test_index_parses_palette_choices(choice: str, expected: int) -> None:
    assert nodes._index(choice) == expected


def test_index_handles_every_generated_option() -> None:
    """The combo lists are built from the palette, so parsing must survive all of them."""
    for i, option in enumerate(nodes._BG_OPTIONS[1:]):
        assert nodes._index(option) == i
    for i, option in enumerate(nodes._BORDER_OPTIONS[1:]):
        assert nodes._index(option) == i


def test_settings_maps_every_widget_to_the_right_field() -> None:
    """
    Distinct values throughout, so a swapped pair cannot pass — which is the whole
    reason this test exists.
    """
    settings = nodes._settings(
        "1 — lower/upper",
        "blocks",
        "9 — brown",
        "4 — purple",
        "contain",
        brightness=0.125,
        contrast=1.25,
        gamma=1.75,
        saturation=0.5,
        dither=True,
        eps=0.004,
    )
    assert settings == Settings(
        charset=1,
        subset="blocks",
        bg_lock=9,
        border_lock=4,
        eps=0.004,
        dither=True,
        framing="contain",
        brightness=0.125,
        contrast=1.25,
        gamma=1.75,
        saturation=0.5,
    )


def test_settings_sentinels_become_none() -> None:
    """`auto` and `bg` are the two options that are not palette indices."""
    settings = nodes._settings(
        "0 — upper/graphics", "all", "auto", "bg", "cover", 0.0, 1.0, 1.0, 1.0
    )
    assert settings.bg_lock is None
    assert settings.border_lock is None


# ------------------------------------------------------------ tensor boundary


def test_tensor_frames_is_a_lazy_sequence() -> None:
    batch = image(batch=4)
    frames = nodes._TensorFrames(batch)
    assert len(frames) == 4
    first = frames[0]
    assert first.shape == (200, 320, 3)
    assert first.dtype == np.uint8


def test_tensor_frames_round_trips_values() -> None:
    batch = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32)
    assert nodes._TensorFrames(batch)[0].tolist() == [[[0, 128, 255]]]


def test_tensor_frames_clamps_and_drops_alpha() -> None:
    batch = torch.tensor([[[[-1.0, 2.0, 0.25, 0.9]]]], dtype=torch.float32)
    assert nodes._TensorFrames(batch)[0].tolist() == [[[0, 255, 64]]]


@pytest.mark.parametrize(
    ("bad", "message"),
    [(torch.zeros(200, 320, 3), "4D"), (torch.zeros(0, 8, 8, 3), "empty batch")],
)
def test_tensor_frames_rejects_bad_input(bad: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nodes._TensorFrames(bad)


def test_render_all_shapes_and_range() -> None:
    data = PetsciiData(frames=[frame(), frame(value=90)], charset=0)
    out = run(nodes._render_all(data, scale=2, border=0))
    assert out.shape == (2, 400, 640, 3)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_render_all_border_grows_the_image() -> None:
    data = PetsciiData(frames=[frame()], charset=0)
    out = run(nodes._render_all(data, scale=2, border=4))
    # Border thickness is in screen pixels, so it scales with the render:
    # 4 screen pixels at scale 2 is 8 output pixels on each of the four sides.
    assert out.shape == (1, 400 + 8 * 2, 640 + 8 * 2, 3)


def test_render_all_matches_frame_by_frame_painting() -> None:
    """The preallocated write must produce exactly what painting each frame gives."""
    frames = [frame(value=32), frame(value=160, bg=6, border=14)]
    data = PetsciiData(frames=frames, charset=0)
    out = run(nodes._render_all(data, scale=1, border=2))
    for index, source in enumerate(frames):
        expected = nodes._paint(source, 1, 2, None)
        assert np.array_equal(np.rint(out[index].numpy() * 255.0).astype(np.uint8), expected)


# --------------------------------------------------------- parallel painting


def test_render_all_threaded_matches_serial(monkeypatch) -> None:
    """
    The pool must not change a single pixel.

    Painting is per-frame and pure, so this is really a check that no thread is
    sharing a buffer with another — the failure mode would be intermittent, and
    an equality test over enough frames is what makes it deterministic.
    """
    frames = [frame(value=v, bg=v % 16, border=(v + 3) % 16) for v in range(9)]
    data = PetsciiData(frames=frames, charset=0)

    monkeypatch.setattr(nodes, "_paint_workers", lambda count: 1)
    serial = run(nodes._render_all(data, scale=2, border=2, report=False))

    monkeypatch.setattr(nodes, "_paint_workers", lambda count: 4)
    threaded = run(nodes._render_all(data, scale=2, border=2, report=False))

    assert torch.equal(serial, threaded)


def test_render_all_threaded_reports_every_frame_in_order(progress) -> None:
    data = PetsciiData(frames=[frame(value=v) for v in range(6)], charset=0)
    run(nodes._render_all(data, scale=1, border=0, report=True))
    assert progress == [(index + 1, 6) for index in range(6)]


def test_paint_workers_stays_within_the_batch() -> None:
    # One frame never pays for a pool, and a short batch never spins up more
    # threads than it has work for.
    assert nodes._paint_workers(1) == 1
    assert nodes._paint_workers(2) <= 2
    assert nodes._paint_workers(1000) <= 8


def test_render_all_threaded_propagates_an_interrupt(monkeypatch) -> None:
    """An interrupt must surface, not be swallowed by the pool's shutdown."""
    calls = {"n": 0}

    def interrupt_on_the_third_frame() -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt("stopped")

    monkeypatch.setattr(nodes, "_interrupted", interrupt_on_the_third_frame)
    monkeypatch.setattr(nodes, "_paint_workers", lambda count: 4)
    data = PetsciiData(frames=[frame(value=v) for v in range(40)], charset=0)

    with pytest.raises(KeyboardInterrupt):
        run(nodes._render_all(data, scale=1, border=0, report=False))


def test_render_all_threaded_surfaces_a_painting_failure(monkeypatch) -> None:
    """A frame that fails to paint must raise, not hang on the queue."""
    real = nodes._paint

    def fail_on_the_fifth(frame_, scale, border, crt):
        if frame_.bg == 5:
            raise ValueError("bad frame")
        return real(frame_, scale, border, crt)

    monkeypatch.setattr(nodes, "_paint", fail_on_the_fifth)
    monkeypatch.setattr(nodes, "_paint_workers", lambda count: 4)
    data = PetsciiData(frames=[frame(value=v, bg=v) for v in range(9)], charset=0)

    with pytest.raises(ValueError, match="bad frame"):
        run(nodes._render_all(data, scale=1, border=0, report=False))


# ------------------------------------------------------------------ preview
#
# Asserting on a real `ui.PreviewImage` would mean letting it write the PNGs this
# cap exists to prevent — and its internals differ from the stub's. So these
# intercept the constructor and check the only thing the node decides: how many
# frames it hands over.


@pytest.fixture
def previews(monkeypatch):
    """Records the tensor handed to each `ui.PreviewImage`, writing nothing."""
    seen: list[torch.Tensor] = []

    class Recorder:
        def __init__(self, image, cls=None, **kwargs):
            seen.append(image)

    monkeypatch.setattr(nodes.ui, "PreviewImage", Recorder)
    return seen


@pytest.fixture
def progress(monkeypatch):
    """Records every progress report, without needing an executing context."""
    seen: list[tuple[float, float]] = []

    class Execution:
        async def set_progress(self, value, max_value, **kwargs):
            seen.append((value, max_value))

    monkeypatch.setattr(nodes.api, "execution", Execution(), raising=False)
    return seen


@pytest.mark.parametrize(("limit", "expected"), [(0, None), (1, 1), (3, 3), (99, 5)])
def test_preview_is_capped(previews, limit: int, expected: int | None) -> None:
    rendered = torch.zeros((5, 8, 8, 3))
    preview = nodes._preview(rendered, limit, PETSCIIConvert)
    if expected is None:
        assert preview is None
        assert previews == []
    else:
        assert previews[0].shape[0] == expected


# ------------------------------------------------------------------ convert


def _convert_kwargs(**overrides):
    base = {
        "charset": "0 — upper/graphics",
        "subset": "all",
        "background": "auto",
        "border": "bg",
        "framing": "cover",
        "brightness": 0.0,
        "contrast": 1.0,
        "gamma": 1.0,
        "saturation": 1.0,
        "render_scale": 1,
        "preview_frames": 2,
    }
    base.update(overrides)
    return base


def test_convert_returns_image_and_cells(previews) -> None:
    result = run(PETSCIIConvert.execute(image=image(), dither=False, **_convert_kwargs()))
    rendered, data = result.args
    assert rendered.shape == (1, 200, 320, 3)
    assert isinstance(data, PetsciiData)
    assert len(data) == 1
    assert data.frames[0].screen.shape == (CELLS,)


def test_convert_render_scale_scales_only_the_image(previews) -> None:
    result = run(PETSCIIConvert.execute(image=image(), dither=False, **_convert_kwargs(render_scale=4)))
    rendered, data = result.args
    assert rendered.shape == (1, 800, 1280, 3)
    assert len(data) == 1


def test_convert_treats_a_batch_as_independent_stills(previews) -> None:
    batch = torch.cat([image(seed=1), image(seed=2)])
    result = run(PETSCIIConvert.execute(image=batch, dither=False, **_convert_kwargs()))
    _, data = result.args
    assert len(data) == 2
    assert not np.array_equal(data.frames[0].screen, data.frames[1].screen)


def test_convert_honours_locked_background_and_border(previews) -> None:
    result = run(
        PETSCIIConvert.execute(
            image=image(), dither=False,
            **_convert_kwargs(background="9 — brown", border="4 — purple"),
        )
    )
    _, data = result.args
    assert data.frames[0].bg == 9
    assert data.frames[0].border == 4


def test_convert_reports_progress_for_every_frame(progress, previews) -> None:
    batch = torch.cat([image(seed=i) for i in range(3)])
    run(PETSCIIConvert.execute(image=batch, dither=False, **_convert_kwargs()))
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_convert_preview_respects_the_cap(previews) -> None:
    batch = torch.cat([image(seed=i) for i in range(5)])
    run(PETSCIIConvert.execute(image=batch, dither=False, **_convert_kwargs(preview_frames=2)))
    assert previews[-1].shape[0] == 2

    off = run(PETSCIIConvert.execute(image=batch, dither=False, **_convert_kwargs(preview_frames=0)))
    assert off.ui is None
    assert len(previews) == 1, "a disabled preview must not build one at all"


def test_convert_dither_changes_the_result(previews) -> None:
    plain = run(PETSCIIConvert.execute(image=image(seed=7), dither=False, **_convert_kwargs()))
    dithered = run(PETSCIIConvert.execute(image=image(seed=7), dither=True, **_convert_kwargs()))
    assert not np.array_equal(plain.args[1].frames[0].screen, dithered.args[1].frames[0].screen)


def test_video_convert_locks_one_background_and_carries_fps(previews) -> None:
    batch = torch.cat([image(seed=i) for i in range(4)])
    result = run(
        PETSCIIVideoConvert.execute(
            images=batch, eps=0.002, background_sample=2, fps=24.0, **_convert_kwargs()
        )
    )
    rendered, data = result.args
    assert rendered.shape[0] == 4
    assert data.fps == 24.0
    assert len({f.bg for f in data.frames}) == 1


def test_video_convert_hysteresis_holds_cells_still(previews) -> None:
    """
    The same frame twice, plus noise: with eps high enough, cells that would
    otherwise re-decide keep their previous choice. That is the whole point of §4.6.
    """
    noisy = torch.cat([image(seed=3), image(seed=3) * 0.98 + 0.01])
    kwargs = _convert_kwargs()
    loose = run(PETSCIIVideoConvert.execute(images=noisy, eps=0.0, background_sample=2, fps=30.0, **kwargs))
    stiff = run(PETSCIIVideoConvert.execute(images=noisy, eps=0.02, background_sample=2, fps=30.0, **kwargs))

    def churn(data):
        a, b = data.frames
        return int(((a.screen != b.screen) | (a.color != b.color)).sum())

    assert churn(stiff.args[1]) < churn(loose.args[1])


# ------------------------------------------------------------------- render


def _render_kwargs(**overrides):
    base = {
        "scale": 2, "border": 0, "crt": False, "crt_scanlines": 0.45, "crt_curvature": 0.35,
        "crt_glow": 0.5, "crt_vignette": 0.4, "crt_chroma": 0.3, "crt_brightness": 1.05,
        "preview_frames": 1,
    }
    base.update(overrides)
    return base


def test_render_without_crt_is_the_plain_picture(previews) -> None:
    data = PetsciiData(frames=[frame(value=90)], charset=0)
    result = run(PETSCIIRender.execute(petscii=data, **_render_kwargs()))
    assert result.args[0].shape == (1, 400, 640, 3)


def test_render_crt_changes_the_picture(previews) -> None:
    data = PetsciiData(frames=[frame(value=90)], charset=0)
    plain = run(PETSCIIRender.execute(petscii=data, **_render_kwargs(scale=4)))
    crt = run(PETSCIIRender.execute(petscii=data, **_render_kwargs(scale=4, crt=True)))
    assert plain.args[0].shape == crt.args[0].shape
    assert not torch.equal(plain.args[0], crt.args[0])


def test_render_crt_at_zero_intensity_is_a_true_bypass(previews) -> None:
    data = PetsciiData(frames=[frame(value=90)], charset=0)
    kwargs = _render_kwargs(scale=4)
    plain = run(PETSCIIRender.execute(petscii=data, **kwargs))
    bypass = run(
        PETSCIIRender.execute(
            petscii=data,
            **_render_kwargs(
                scale=4, crt=True, crt_scanlines=0.0, crt_curvature=0.0, crt_glow=0.0,
                crt_vignette=0.0, crt_chroma=0.0, crt_brightness=1.0,
            ),
        )
    )
    assert torch.equal(plain.args[0], bypass.args[0])


def test_render_rejects_an_empty_stream() -> None:
    with pytest.raises(ValueError, match="no frames"):
        run(PETSCIIRender.execute(petscii=PetsciiData(), **_render_kwargs()))


# --------------------------------------------------------------- set border


def test_set_border_retints_without_touching_the_cells() -> None:
    source = PetsciiData(frames=[frame(bg=6, border=6)], charset=0)
    result = PETSCIISetBorder.execute(petscii=source, border="14 — light blue")
    out = result.args[0]
    assert out.frames[0].border == 14
    assert out.frames[0].bg == 6
    assert np.array_equal(out.frames[0].screen, source.frames[0].screen)


def test_set_border_bg_tracks_each_frame() -> None:
    source = PetsciiData(frames=[frame(bg=6, border=1), frame(bg=11, border=1)], charset=0)
    out = PETSCIISetBorder.execute(petscii=source, border="bg").args[0]
    assert [f.border for f in out.frames] == [6, 11]


def test_set_border_does_not_mutate_its_input() -> None:
    """
    The input is an upstream node's cached output. Retinting it in place would
    make a second run see the already-retinted copy.
    """
    source = PetsciiData(frames=[frame(bg=6, border=6)], charset=0)
    PETSCIISetBorder.execute(petscii=source, border="2 — red")
    assert source.frames[0].border == 6


# --------------------------------------------------------------------- info


def test_info_is_an_output_node() -> None:
    """It has to terminate its own branch, or its report never runs."""
    assert PETSCIIInfo.define_schema().is_output_node is True
    assert PETSCIISavePETV.define_schema().is_output_node is True


def test_info_reports_the_stream() -> None:
    data = PetsciiData(frames=[frame(bg=6, border=14)], charset=1, fps=24.0)
    text = PETSCIIInfo.execute(petscii=data, top_glyphs=4).args[0]
    assert "frames      1" in text
    assert "charset     1" in text
    assert "fps         24" in text
    assert "6 — blue" in text
    assert "14 — light blue" in text
    assert "top glyphs" in text


def test_info_reports_motion_only_for_a_sequence() -> None:
    still = PETSCIIInfo.execute(petscii=PetsciiData(frames=[frame()]), top_glyphs=0).args[0]
    assert "motion" not in still

    moving = PetsciiData(frames=[frame(value=1), frame(value=2)])
    assert "motion      1000.0 cells/frame changed (100.0%)" in (
        PETSCIIInfo.execute(petscii=moving, top_glyphs=0).args[0]
    )


# ---------------------------------------------------------------- .petv i/o


def test_save_petv_writes_a_readable_stream(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    data = PetsciiData(frames=[frame(value=1), frame(value=2)], charset=0, fps=30.0)

    result = PETSCIISavePETV.execute(petscii=data, filename_prefix="petscii/test", fps=0.0)
    path = result.args[0]
    stream = read_petv(pathlib.Path(path).read_bytes())
    assert len(stream.frames) == 2
    assert result.ui["petv"][0]["type"] == "output"
    assert "2 frames" in result.ui["text"][0]


def test_save_petv_uses_the_carried_fps(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    data = PetsciiData(frames=[frame(value=1), frame(value=2)], charset=0, fps=10.0)
    path = PETSCIISavePETV.execute(petscii=data, filename_prefix="a", fps=0.0).args[0]
    assert read_petv(pathlib.Path(path).read_bytes()).frames[1].dt_ms == 100


@pytest.mark.parametrize("prefix", ["", "   ", "/etc/passwd", "../../escape", "a/../../b"])
def test_save_petv_refuses_to_leave_the_output_directory(prefix: str) -> None:
    assert PETSCIISavePETV.validate_inputs(filename_prefix=prefix) is not True


def test_save_petv_accepts_ordinary_prefixes() -> None:
    assert PETSCIISavePETV.validate_inputs(filename_prefix="petscii/petscii") is True


def test_load_petv_round_trips_a_saved_stream(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    frames = [frame(value=1, bg=6), frame(value=2, bg=6)]
    saved = PETSCIISavePETV.execute(
        petscii=PetsciiData(frames=frames, charset=1, fps=25.0),
        filename_prefix="clip", fps=0.0,
    ).args[0]

    name = "loaded.petv"
    (tmp_path / "input" / name).write_bytes(pathlib.Path(saved).read_bytes())

    data, count = PETSCIILoadPETV.execute(petv=name, fps=0.0).args
    assert count == 2
    assert data.charset == 1
    assert data.fps == pytest.approx(25.0)
    for original, loaded in zip(frames, data.frames, strict=True):
        assert np.array_equal(original.screen, loaded.screen)
        assert np.array_equal(original.color, loaded.color)
        assert loaded.bg == original.bg


def test_load_petv_fps_override_wins(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    PETSCIISavePETV.execute(
        petscii=PetsciiData(frames=[frame()], charset=0, fps=30.0), filename_prefix="x", fps=0.0
    )
    name = "one.petv"
    (tmp_path / "input" / name).write_bytes(
        next((tmp_path / "output").rglob("*.petv")).read_bytes()
    )
    data, _ = PETSCIILoadPETV.execute(petv=name, fps=12.5).args
    assert data.fps == 12.5


def test_load_petv_rejects_a_missing_file(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    assert PETSCIILoadPETV.validate_inputs(petv="") is not True
    assert PETSCIILoadPETV.validate_inputs(petv="nope.petv") is not True


def test_load_petv_rejects_a_truncated_file(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    (tmp_path / "input" / "bad.petv").write_bytes(struct.pack("<4sBBBB", b"PETV", 1, 0, 0, 0) + b"\x01\x00")
    with pytest.raises(ValueError):
        PETSCIILoadPETV.execute(petv="bad.petv", fps=0.0)


def test_load_petv_registers_with_an_empty_input_directory(tmp_path, monkeypatch) -> None:
    """
    The fresh-install path: nobody has a .petv yet, so the combo has no options.
    The schema still has to build, or the whole pack fails to load.
    """
    _fake_folder_paths(monkeypatch, tmp_path)
    assert nodes._petv_files() == []
    schema = PETSCIILoadPETV.define_schema()
    assert [port.id for port in schema.inputs] == ["petv", "fps"]
    assert isinstance(PETSCIILoadPETV.validate_inputs(petv=""), str)


def test_petv_files_survives_a_missing_folder_paths(monkeypatch) -> None:
    """`folder_paths` only exists inside ComfyUI; the combo must not explode without it."""
    import sys

    monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    monkeypatch.setattr(nodes.os, "listdir", lambda _: (_ for _ in ()).throw(OSError()))
    assert nodes._petv_files() == []


def test_load_petv_lists_only_petv_files(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    for name in ("a.petv", "B.PETV", "photo.png", "notes.txt"):
        (tmp_path / "input" / name).write_bytes(b"")
    assert nodes._petv_files() == ["B.PETV", "a.petv"]


def test_load_petv_fingerprint_follows_the_file(tmp_path, monkeypatch) -> None:
    _fake_folder_paths(monkeypatch, tmp_path)
    target = tmp_path / "input" / "c.petv"
    target.write_bytes(b"")
    first = PETSCIILoadPETV.fingerprint_inputs(petv="c.petv", fps=0.0)
    import os

    os.utime(target, (0, 0))
    assert PETSCIILoadPETV.fingerprint_inputs(petv="c.petv", fps=0.0) != first


def _fake_folder_paths(monkeypatch, tmp_path):
    """
    A stand-in for ComfyUI's `folder_paths`, rooted in a temp directory.

    The node layer imports it lazily inside each function precisely so it can be
    absent — which is what makes stubbing it here enough.
    """
    import sys
    import types

    inputs = tmp_path / "input"
    outputs = tmp_path / "output"
    inputs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    module = types.ModuleType("folder_paths")
    module.get_input_directory = lambda: str(inputs)
    module.get_output_directory = lambda: str(outputs)
    module.get_annotated_filepath = lambda name, default_dir=None: str(inputs / name.split(" [")[0])
    module.exists_annotated_filepath = lambda name: (inputs / name.split(" [")[0]).is_file()

    def get_save_image_path(prefix, output_dir, width=0, height=0):
        prefix = prefix.replace("\\", "/")
        subfolder, _, filename = prefix.rpartition("/")
        folder = outputs / subfolder if subfolder else outputs
        folder.mkdir(parents=True, exist_ok=True)
        counter = len(list(folder.glob(f"{filename}_*"))) + 1
        return str(folder), filename, counter, subfolder, filename

    module.get_save_image_path = get_save_image_path
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return module


# ------------------------------------------------------- bundled workflows


def _workflow_files():
    from .paths import workflows_dir

    return sorted(workflows_dir().glob("*.json"))


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_bundled_workflow_matches_the_schemas(path) -> None:
    """
    The shipped examples must load against the nodes as they are now.

    `widgets_values` is positional, so adding an input silently shifts every value
    after it in a saved workflow — the graph still opens and quietly runs with the
    wrong settings. Counting widgets against the schema is what catches that.
    """
    import json

    by_id = {node.define_schema().node_id: node for node in ALL_NODES}
    graph = json.loads(path.read_text())

    for entry in graph["nodes"]:
        node = by_id.get(entry["type"])
        if node is None:
            continue  # a core ComfyUI node; not ours to check
        schema = node.define_schema()
        linked = {port["name"] for port in entry["inputs"]}
        widgets = [port for port in schema.inputs if port.id not in linked]
        assert len(widgets) == len(entry["widgets_values"]), (
            f"{path.name}: {entry['type']} has {len(entry['widgets_values'])} saved widget "
            f"values but the schema declares {len(widgets)} — re-save the workflow"
        )
        assert [port.display_name for port in schema.outputs] == [
            port["name"] for port in entry["outputs"]
        ], f"{path.name}: {entry['type']} output slots have drifted"


def test_every_workflow_input_file_is_shipped() -> None:
    """A LoadImage pointing at a file the repo does not ship is a broken example."""
    import json

    from .paths import workflows_dir

    available = {p.name for p in (workflows_dir().parent / "example_inputs").glob("*")}
    for path in _workflow_files():
        for entry in json.loads(path.read_text())["nodes"]:
            if entry["type"] == "LoadImage":
                name = entry["widgets_values"][0]
                assert name in available, f"{path.name} loads {name}, which is not in example_inputs/"
