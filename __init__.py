"""
ComfyUI-PETSCII — entry point.

ComfyUI awaits `comfy_entrypoint()`; everything real lives in `src/`. The import
of the node layer is deferred into that call rather than done at module level,
because the node layer needs torch and the engine (`petscii_core`) deliberately
does not — that is what lets the parity tests run without ComfyUI installed.
"""

#: Served to the frontend; `web/petscii.js` adds a download link to the .petv
#: save node. Declared here as well as in pyproject's `[tool.comfy] web` because
#: the pyproject route needs pydantic-settings, which is not always installed.
WEB_DIRECTORY = "web"


async def comfy_entrypoint():
    from .src.nodes import comfy_entrypoint as entrypoint

    return await entrypoint()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
