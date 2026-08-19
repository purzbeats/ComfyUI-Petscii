# PETSCII Engine — Normative Core Spec

This document defines the conversion algorithm and file formats **exactly**, so the
TypeScript engine (`web/`) and the Python port (`comfy/`) produce matching output.
When code and this spec disagree, this spec wins; if the spec is wrong, fix the spec
first, then the ports.

## 1. Screen model

- Screen: **40 × 25 cells**, each cell **8 × 8 pixels** → 320 × 200 source resolution.
- Per cell: one **screen code** `g ∈ [0, 255]` and one **foreground color** `f ∈ [0, 15]`.
- Global: **background color** `bg ∈ [0, 15]`, **border color** `border ∈ [0, 15]`.
- Output of a conversion is `{ screen: Uint8Array(1000), color: Uint8Array(1000), bg, border }`,
  row-major (index = `row * 40 + col`).
- Two charsets: `0` = uppercase/graphics (default for art), `1` = lowercase/uppercase.
- Screen codes `128–255` are the **bitwise inverses** of codes `0–127`. Implementations
  MUST exploit this (see §4.4) but MUST output the true screen code either way.

## 2. Palette (Pepto, normative RGB)

Index → sRGB hex. Single source of truth is `shared/palette.json`; these values seed it.

```
 0 #000000 black        8 #6F4F25 orange
 1 #FFFFFF white        9 #433900 brown
 2 #68372B red         10 #9A6759 light red
 3 #70A4B2 cyan        11 #444444 dark grey
 4 #6F3D86 purple      12 #6C6C6C grey
 5 #588D43 green       13 #9AD284 light green
 6 #352879 blue        14 #6C5EB5 light blue
 7 #B8C76F yellow      15 #959595 light grey
```

## 3. Color space and distance

All error math happens in **Oklab**, float32.

sRGB u8 → linear: `c = (c8/255)`; `c ≤ 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4`.

Linear RGB → Oklab (Björn Ottosson's matrices):

```
l = 0.4122214708 r + 0.5363325363 g + 0.0514459929 b
m = 0.2119034982 r + 0.6806995451 g + 0.1073969566 b
s = 0.0883024619 r + 0.2817188376 g + 0.6299787005 b
l' = cbrt(l); m' = cbrt(m); s' = cbrt(s)
L = 0.2104542553 l' + 0.7936177850 m' - 0.0040720468 s'
a = 1.9779984951 l' - 2.4285922050 m' + 0.4505937099 s'
b = 0.0259040371 l' + 0.7827717662 m' - 0.8086757660 s'
```

**Distance = squared Euclidean** in (L, a, b). No sqrt anywhere.

## 4. Conversion algorithm

Input: a 320 × 200 sRGB image (see §5 for how arbitrary inputs become 320 × 200),
a charset bitmask table, a glyph-subset mask, optional previous-frame state.

### 4.1 Distance tables

For each cell `c` (1000), pixel `p` (64, row-major within the cell), palette color `k` (16):

```
D[c][p][k] = oklab_dist2(pixel(c, p), palette[k])     # float32, [1000][64][16]
S[c][k]    = Σ_p D[c][p][k]                            # float32, [1000][16]
```

### 4.2 Masked glyph sums

Charset bitmasks `M[g][p] ∈ {0,1}` for `g ∈ [0, 127]` (row-major, bit set = foreground).

```
A[c][g][k] = Σ_p M[g][p] * D[c][p][k]                  # [1000][128][16]
```

(numpy: `A = np.einsum('cpk,gp->cgk', D, M)`; WGSL: one thread per (c, g), loop k.)

### 4.3 Cell error

For background `bg`, glyph `g ∈ [0,127]`, foreground `f`:

```
err(c, g, f)      = A[c][g][f] + S[c][bg] - A[c][g][bg]        # normal glyph
err(c, g+128, f)  = (S[c][f] - A[c][g][f]) + A[c][g][bg]       # inverse glyph
```

### 4.4 Per-cell argmin

For each cell, over all glyphs allowed by the subset mask (applied to the full 0–255
range) and all 16 foregrounds, pick minimum error. **Tie-break: lowest error wins; on
exact float equality, lower glyph code wins, then lower color index.** A cell whose
chosen foreground equals `bg` SHOULD be normalized to `(space, bg)`... do **not** do
this normalization — output exactly the argmin winner. (Normalization breaks parity
and the C64 doesn't care.)

### 4.5 Background selection

- If `bg_lock` is set, use it.
- Else evaluate total frame error `E(bg) = Σ_c min_{g,f} err_bg(c, g, f)` for all 16
  candidates (reuses D/S/A — only §4.3–4.4 rerun per candidate) and pick argmin,
  tie-break lower index.
- **Realtime hysteresis**: re-evaluate bg only every `bg_interval` frames (default 30);
  switch only if `E(best) < E(current) * (1 - eps_bg)` with default `eps_bg = 0.10`.
- `border` defaults to `bg`; independently settable, never affects error.

### 4.6 Temporal hysteresis (video / realtime only)

State: previous frame's `(g, f)` per cell. For each cell:

```
keep previous choice unless err(c, g_new, f_new) < err(c, g_prev, f_prev) - eps * 64
```

where both errors are computed on the **current** frame's tables, and `eps` is the
per-pixel switch threshold, default `0.002`, live-controllable, `0` disables.
On bg change, hysteresis state is cleared (all cells re-decide).

### 4.7 Dithering (stills / offline only, default off)

Cell-level error diffusion, processed row-major: after choosing a cell, compute the
mean Oklab residual `r = mean_p(target_p - rendered_p)`, then add `r * w` to every
pixel of the not-yet-processed neighbor cells with weights: right `7/16`, below-left
`3/16`, below `5/16`, below-right `1/16` (Floyd–Steinberg at cell granularity),
clamped in Oklab. Mutually exclusive with temporal hysteresis.

`rendered_p` is the Oklab of the palette color that cell paints at pixel `p` — the
chosen foreground where the (possibly inverted) glyph mask is set, `bg` elsewhere.
Clamping is per component: `L → [0, 1]`, `a → [-0.5, 0.5]`, `b → [-0.5, 0.5]`.

Because diffusion makes each cell depend on its predecessors, `bg` cannot be chosen
from the dithered image. **With dithering on, `bg` is selected per §4.5 on the
un-dithered image, then locked for the diffusion pass.**

## 5. Input framing

Arbitrary input → 320 × 200: **center cover-crop** to 8:5 aspect, then **box-filter
(area average) downscale**. Modes `contain` (letterbox with bg-colored bars) and
`stretch` are options. **Fixture images are exactly 320 × 200** so resampling never
affects parity tests.

Pre-adjust runs on the framed 320 × 200 image, before conversion. The first three
knobs operate on non-linear sRGB normalized to `[0, 1]`, per channel, in this order:

```
v = v + brightness                        # brightness, default 0
v = (v - 0.5) * contrast + 0.5            # contrast,   default 1
v = clamp(v, 0, 1) ^ (1 / gamma)          # gamma,      default 1 (>1 brightens)
```

The result is clamped to `[0, 1]` and converted to Oklab (§3). Saturation then scales
the chroma axes in Oklab — `a *= saturation`, `b *= saturation` (default 1) — and the
components are clamped as in §4.7. That Oklab buffer is the input to §4.

Note `gamma` is applied as an exponent of `1 / gamma`, so values above 1 brighten
midtones; `brightness` and `contrast` are deliberately applied before it.

## 6. Charset data

`shared/charset.json`:

```json
{ "source": "Pet Me 64 (Kreative Software), rasterized 8px",
  "sets": [ { "name": "upper-graphics", "glyphs": [[8 bytes], ... 128 entries] },
            { "name": "lower-upper",    "glyphs": [ ... 128 entries] } ] }
```

Each glyph = 8 bytes, top row first, MSB = leftmost pixel (C64 char ROM layout).
Only codes 0–127 are stored; 128–255 are derived by inversion. Produced once by
`tools/charset-extract/` (browser page: load TTF → draw each glyph at 8px → threshold
at 50% alpha → JSON download), committed, then treated as frozen data. Spot-verify
against known shapes (e.g. code 0 = `@`, code 1 = `A`, code 32 = space) before freezing.

Glyph subsets are named lists of screen codes in `shared/subsets.json` (`all`,
`blocks`, `dither`, `lines`, `text`); curated by eye with the charset-viewer tool page.

## 7. Parity between ports

Float32 throughout; ports will still diverge on near-ties. The comparator is shared
by TS↔Python parity and by the GPU↔CPU check.

Convert the fixture input with reference settings, then per cell:

1. **Identical** — same `(screen, color)`.
2. **Same-render** — different choice, but the cell paints the same 64 pixels under
   the shared `bg`. This is not a curiosity: wherever a cell's foreground equals the
   background, *every* glyph renders as solid background, so the argmin is a genuine
   tie broken by float noise. The `ui` fixture is 40% such cells.
3. **Tolerated** — paints differently, but the two choices' errors (recomputed by
   the comparator, not taken from either port) differ by < 1e-4 relative.
4. Anything else is a **divergence** and fails.

Pass requires: `bg` matches exactly, **zero divergences**, and ≥ 99.5% of cells in
categories 1–2 — that is, ≥ 99.5% of the picture is pixel-identical.

> Earlier revisions set the bar at ≥ 99.5% of cells bit-identical in `(screen,
> color)`. That target is unmeetable *and* meaningless on flat imagery — the WebGPU
> engine hits only 60.4% on `ui` while rendering 100% of it identically to the CPU
> reference. Pixels are what parity is actually about.

Fixtures live in `fixtures/`: input PNGs (320 × 200) + `expected/*.json`
(`{settings, bg, border, screen: [...], color: [...]}`), generated once by the TS CPU
reference engine after visual sign-off, then frozen.

## 8. File formats

### 8.1 `.petv` — recorded PETSCII stream (our interchange format)

Little-endian. Header:

```
magic "PETV" | version u8 = 1 | charset u8 | flags u8 | reserved u8
```

Then frame records until EOF:

```
type u8:
  0x01 keyframe: dt_ms u16 | bg u8 | border u8 | screen[1000] | color[1000]
  0x02 delta:    dt_ms u16 | bg u8 | border u8 | count u16 |
                 count × { index u16 | char u8 | color u8 }
```

`dt_ms` = time since previous frame (first frame: 0). Writers SHOULD emit a keyframe
when delta count > 500. Realtime recording is variable-rate; consumers resample.

### 8.2 C64 export formats (stretch — specced later)

Real-hardware export (`.prg` stills, delta-animation player, `.d64`) is a deferred
stretch goal. The output model above (§1: true screen codes, 1000-byte screen +
1000-nibble color, global bg/border) is deliberately C64-faithful and `.petv` captures
everything a future exporter needs, so nothing in the core has to change when that
head gets planned.

## 9. Reference settings (used by fixtures)

```
charset=0  subset=all  bg=auto  border=bg  eps=0 (stills)  dither=off
framing=cover  brightness=0 contrast=1 gamma=1 saturation=1
```
