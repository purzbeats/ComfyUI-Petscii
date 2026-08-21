"""
Oklab conversion — core-spec §3. Float32 throughout; distance is squared
Euclidean with no sqrt anywhere.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OKLAB_MAX",
    "OKLAB_MIN",
    "linear_rgb_to_oklab",
    "linear_to_srgb",
    "oklab_to_linear_rgb",
    "srgb_to_linear",
    "srgb_to_oklab",
]

# Ottosson's matrices, core-spec §3.
_LMS_FROM_RGB = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

_LAB_FROM_LMS = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

_LMS_FROM_LAB = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)

_RGB_FROM_LMS = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)

#: Per-component clamp bounds shared with dithering (core-spec §4.7).
OKLAB_MIN = np.array([0.0, -0.5, -0.5], dtype=np.float32)
OKLAB_MAX = np.array([1.0, 0.5, 0.5], dtype=np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] to linear light, per core-spec §3."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Inverse of :func:`srgb_to_linear`."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1 / 2.4) - 0.055).astype(
        np.float32
    )


def linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB ``(..., 3)`` to Oklab ``(..., 3)``."""
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ _LMS_FROM_RGB.T.astype(np.float32)
    # cbrt rather than **(1/3): the latter is NaN for the negative values that
    # out-of-gamut inputs and dithering residuals can produce.
    lms = np.cbrt(lms)
    return (lms @ _LAB_FROM_LMS.T.astype(np.float32)).astype(np.float32)


def oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    """Oklab ``(..., 3)`` back to linear RGB ``(..., 3)``."""
    lab = np.asarray(lab, dtype=np.float32)
    lms = lab @ _LMS_FROM_LAB.T.astype(np.float32)
    lms = lms**3
    return (lms @ _RGB_FROM_LMS.T.astype(np.float32)).astype(np.float32)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] ``(..., 3)`` to Oklab ``(..., 3)``."""
    return linear_rgb_to_oklab(srgb_to_linear(srgb))


def clamp_oklab(lab: np.ndarray) -> np.ndarray:
    """Clamps to the §4.7 bounds, in place where possible."""
    return np.clip(lab, OKLAB_MIN, OKLAB_MAX, out=np.asarray(lab, dtype=np.float32))
