# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom node pack that converts images and video to Commodore 64 PETSCII.
It is a **port**: the normative algorithm is `core-spec.md`, and a TypeScript
reference implementation (in a separate monorepo, referred to as `web/`) is the
other port. Parity between the two is the central constraint on this codebase.

## Commands

```sh
uv venv && uv pip install -e ".[dev]"   # setup (torch is a test-only dev dep)
.venv/bin/python -m pytest -q            # full suite (254 tests, ~10s)
.venv/bin/python -m pytest -q --cov      # with the 100% gate CI enforces
.venv/bin/python -m pytest tests/test_parity.py -q          # one file
.venv/bin/python -m pytest tests/test_parity.py -k ui -q    # one case
python sync_shared.py                    # mirror shared/*.json into the package
uvx ruff check .                         # ruff is configured but not a dev dep

# Run the node tests against a real ComfyUI instead of the conftest stub:
PYTHONPATH=/path/to/ComfyUI .venv/bin/python -m pytest tests/test_nodes.py -q
```

There is no build step for development; ComfyUI loads the directory in place.
Packaging is hatchling (`uv build`), and `.comfyignore` keeps `fixtures/`,
`tests/`, `docs/` and `core-spec.md` out of the registry download while leaving
them in git.

## Architecture

Three layers, deliberately separated:

- **`src/petscii_core/`** — the engine. Pure numpy; imports neither ComfyUI nor
  torch. This is why the whole test suite runs anywhere. `engine.py` implements
  core-spec §4–5 (distance tables → masked glyph sums → per-cell argmin →
  background search → temporal hysteresis); `dither.py` is the mutually-exclusive
  offline path (§4.7); `render.py` paints cells to pixels; `crt.py` is a numpy
  port of the web app's post shader; `petv.py` reads/writes the `.petv`
  interchange format (§8.1); `compare.py` is the §7 parity comparator;
  `oklab.py` and `data_loader.py` are the color space and the frozen data.
- **`src/nodes.py`** — the only file that touches torch. It converts tensors to
  numpy at the boundary, builds `Settings`/`CrtSettings`, and does nothing else.
  Uses the ComfyUI **v3 API** (`comfy_api.latest`: `io.ComfyNode`,
  `define_schema`, `ComfyExtension`). Seven nodes plus a custom `PETSCII` datatype
  (`PetsciiData`) that carries cells between Convert and everything downstream, so
  a re-render never reconverts.
- **`web/petscii.js`** — one frontend extension, registered via `WEB_DIRECTORY`
  and `[tool.comfy] web`. It exists only because `.petv` is not a media type the
  frontend knows, so the save node would otherwise report success and show
  nothing.
- **`__init__.py`** — `comfy_entrypoint()` only, with the import of `src.nodes`
  deferred *inside* the call so the torch dependency never leaks into a bare
  `import petscii_core`. Do not hoist that import to module level.

Node imports are relative (`from .petscii_core import ...`) because dropping the
pack into `custom_nodes/` does not put `src/` on `sys.path`.

## Parity is the contract

`fixtures/` holds five frozen 320×200 inputs plus `expected/*.json` produced by
the TypeScript reference. `tests/test_parity.py` re-converts them and judges with
the core-spec §7 comparator: pass requires `bg` matching exactly, **zero
divergences**, and ≥99.5% of cells pixel-identical. "Same-render" cells (a
different `(glyph, color)` choice that paints the same 64 pixels — common when
foreground equals background) count as identical, because the picture is what
parity is about, not the argmin's float noise.

**When parity breaks, the order of repair is: fix `core-spec.md` first, then the
TS reference, then regenerate the fixtures, then this port.** Never edit a
fixture to match new behaviour.

`tests/test_interop.py` reads `fixtures/interop.petv`, written by the TypeScript
implementation, byte for byte — that is what proves both ports agree on the
format rather than on the prose describing it.

## Data flow for `shared/`

`shared/{palette,charset,subsets}.json` is the single source of truth for both
ports. `src/petscii_core/data/` is a mirror that exists only because a wheel
cannot reach outside its package. Edit `shared/`, run `python sync_shared.py`,
never edit the mirror directly — `tests/test_data.py` fails on drift.

## Scale is the constraint on the node layer

A clip is a batch and a batch can be thousands of frames, which is why:

- `_TensorFrames` presents the IMAGE tensor as a lazy `Sequence` of uint8 arrays.
  `iter_convert_batch` *indexes* rather than iterates precisely so this works —
  nothing ever materialises the whole clip a second time.
- `_render_all` writes each painted frame straight into a preallocated output
  tensor. Building a list and `np.stack`-ing it held the sequence three times over.
- `_preview` caps how many frames reach `ui.PreviewImage`, which writes one PNG to
  disk per frame on **every** execution.
- Every frame loop calls `_interrupted()` and reports progress. `_progress` is
  best-effort and suppresses failure; a node called outside an executing context
  (i.e. a test) has no progress state.

## Conventions worth preserving

- Float32 throughout the engine, squared Euclidean Oklab distance, no `sqrt`.
- Tie-breaks are normative (§4.4): lowest error, then lower glyph code, then
  lower color index. A cell whose foreground equals `bg` is **not** normalized.
- CRT effect sizes are in *native* screen pixels scaled by the render scale, so
  the look holds at 1x and 8x; all-zero intensities is a true bypass.
- `tests/paths.py` searches upward for `fixtures/` and `shared/` rather than
  counting parent directories, because this pack lives both standalone (repo
  root) and as a subdirectory of the development monorepo.
- Docstrings here explain *why* a shape was chosen, not what the code does.
  Match that register.
- Any speedup in the engine must be bit-identical, not merely close.
  `distance_tables` walks the 16 palette entries rather than broadcasting, which is
  2.4x faster with identical arithmetic; the expanded `|x|² - 2x·p + |p|²` form
  would be 8x faster again and is rejected because it cancels exactly where §4.4's
  ties are decided.
- `validate_inputs` signatures are introspected by ComfyUI — name the arguments,
  do not take `**kwargs`, or every input including linked ones gets passed.
  `fingerprint_inputs` is the opposite: it receives all inputs, so it needs
  `**kwargs`.
- `widgets_values` in a saved workflow is positional. Adding a node input shifts
  every later value in the bundled examples under `workflows/`, which still load
  and quietly run with the wrong settings — `tests/test_nodes.py` counts them
  against the schemas to catch exactly that.

## Coverage is a gate, not a report

The suite is at 100% and `[tool.coverage.report] fail_under` holds it there. The
argument for a lower number is always that some particular line is not worth
testing, and the ones here that genuinely are not — the `_interrupted` import
fallback — are marked `# pragma: no cover`. Coverage is deliberately off the
default `pytest` run so the local loop stays fast; `--cov` turns it on and CI
always passes it.

What that exercise actually found was not obscure branches. It was whole
exported functions the node layer happens not to call — `render_frames`,
`scanline_overlay`, `PetsciiFrame.as_grid` — and every one of the `.petv`
reader's refusals. The refusals matter most: `.petv` is what the two ports
exchange, and reading past a truncated record does not raise on its own, it
hands numpy whatever bytes came next and produces a frame.

## What is parallel, and what may not be

Painting (`_render_all`) and non-temporal conversion (`iter_convert_batch`) run on
a `ThreadPoolExecutor`. Threads, not processes: numpy drops the GIL for the dense
float32 work, and a painted 8x frame is 12 MB that a process pool would have to
pickle back. Both drain **in order** behind a window of two items per worker —
the window keeps a long clip from holding hundreds of painted frames at once, and
in-order draining is what keeps progress monotonic and leaves one obvious place
for the interrupt check.

The temporal path is sequential and must stay that way: hysteresis reads the
previous frame's choices, and the background is voted once over the clip and
locked. `iter_convert_batch` ignores `workers` when `temporal=True` rather than
quietly honouring it, and a test pins that.

The first item is always processed on the calling thread, both to learn the
output shape and to warm the `lru_cache`d charset and palette data before any
worker reaches it.

## The output batch has a ceiling

An IMAGE output is one contiguous float32 tensor and it grows with the square of
the render scale — 1000 frames is 0.8 GB at 1x and 46 GB at 8x. `_allocate_batch`
checks that up front, because the allocation is not a check: Linux overcommit and
macOS's lazy mapping both hand back the 46 GB tensor and report success. Reading
the budget is best-effort and dependency-free — Linux reports free memory through
`sysconf`, macOS has `SC_PHYS_PAGES` but no `SC_AVPHYS_PAGES` so it falls back to
installed, and Windows has no `sysconf` at all and the allocation just proceeds.
