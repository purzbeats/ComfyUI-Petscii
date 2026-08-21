"""
Test-time shims.

`petscii_core` is importable anywhere, which is the whole point of the split — but
`src/nodes.py` is the boundary and needs `torch` and `comfy_api` to import at all.
Requiring a ComfyUI checkout to test the boundary would mean it never gets tested,
so when the real `comfy_api` is absent a stub stands in. The stub is deliberately
thin: it records what the node layer asks for and returns it, which is enough to
check the wiring this pack owns — schemas, widget parsing, tensor shapes — without
pretending to check ComfyUI's own behaviour.

`torch` is not stubbed. The tensor boundary is the thing under test and a fake
tensor would test nothing, so `tests/test_nodes.py` skips without it.
"""

from __future__ import annotations

import pathlib
import sys
import types


def _install_comfy_api_stub() -> None:
    root = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")

    class Port:
        """One declared input or output; ``id`` is the execute() parameter name."""

        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

        @property
        def display_name(self):
            return self.kwargs.get("display_name")

        @property
        def tooltip(self):
            return self.kwargs.get("tooltip")

    def _spec(name: str):
        return type(name, (), {"NAME": name, "Input": type("Input", (Port,), {}),
                               "Output": type("Output", (Port,), {})})

    class Schema:
        def __init__(self, node_id, display_name=None, category=None, description=None,
                     inputs=None, outputs=None, is_output_node=False, **kwargs):
            self.node_id = node_id
            self.display_name = display_name
            self.category = category
            self.description = description
            self.inputs = list(inputs or [])
            self.outputs = list(outputs or [])
            self.is_output_node = is_output_node

    class NodeOutput:
        def __init__(self, *args, ui=None, **kwargs):
            self.args = args
            self.ui = ui

        def __getitem__(self, index):
            return self.args[index]

    class PreviewImage:
        def __init__(self, image, cls=None, **kwargs):
            self.image = image

    class PreviewText:
        def __init__(self, value, **kwargs):
            self.value = value

    class Execution:
        def __init__(self):
            self.calls: list[tuple[float, float]] = []

        async def set_progress(self, value, max_value, **kwargs):
            self.calls.append((value, max_value))

    class ComfyAPI:
        def __init__(self):
            self.execution = Execution()

    io = types.ModuleType("comfy_api.latest.io")
    for name in ("Image", "Float", "Int", "Boolean", "String", "Combo", "Mask", "Custom"):
        setattr(io, name, _spec(name))
    io.Custom = lambda name: _spec(name)
    io.ComfyNode = type("ComfyNode", (), {})
    io.Input = Port
    io.Schema = Schema
    io.NodeOutput = NodeOutput

    ui = types.ModuleType("comfy_api.latest.ui")
    ui.PreviewImage = PreviewImage
    ui.PreviewText = PreviewText

    latest.io = io
    latest.ui = ui
    latest.ComfyAPI = ComfyAPI
    latest.ComfyExtension = type("ComfyExtension", (), {})
    root.latest = latest

    sys.modules.setdefault("comfy_api", root)
    sys.modules.setdefault("comfy_api.latest", latest)
    sys.modules.setdefault("comfy_api.latest.io", io)
    sys.modules.setdefault("comfy_api.latest.ui", ui)


def _alias_core_under_src() -> None:
    """
    Make `petscii_core` and `src.petscii_core` the same module objects.

    `pythonpath` carries both `src/` (so the engine imports as `petscii_core`) and
    the repo root (so `src.nodes` can resolve its relative imports). Left alone
    that loads the engine twice under two names, and two `Settings` classes that
    look identical never compare equal — a confusing way for a correct test to
    fail. Aliasing before `src.nodes` is imported means its `from .petscii_core`
    finds what is already loaded.
    """
    import petscii_core  # noqa: F401

    src = types.ModuleType("src")
    src.__path__ = [str(pathlib.Path(__file__).resolve().parent.parent / "src")]
    sys.modules.setdefault("src", src)

    for name, module in list(sys.modules.items()):
        if name == "petscii_core" or name.startswith("petscii_core."):
            alias = f"src.{name}"
            sys.modules.setdefault(alias, module)
    setattr(sys.modules["src"], "petscii_core", sys.modules["petscii_core"])


try:  # pragma: no cover - depends on where the tests are run
    import comfy_api.latest  # noqa: F401
except ImportError:
    _install_comfy_api_stub()

_alias_core_under_src()
