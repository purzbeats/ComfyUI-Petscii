"""
ComfyUI-PETSCII — entry point.

ComfyUI awaits `comfy_entrypoint()`; everything real lives in `src/`. The import
of the node layer is deferred into that call rather than done at module level,
because the node layer needs torch and the engine (`petscii_core`) deliberately
does not — that is what lets the parity tests run without ComfyUI installed.
"""


async def comfy_entrypoint():
    from .src.nodes import comfy_entrypoint as entrypoint

    return await entrypoint()


__all__ = ["comfy_entrypoint"]
