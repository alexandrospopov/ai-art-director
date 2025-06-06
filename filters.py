"""photo_adjustments.py

A lightweight toolbox of common photo‑editing primitives—contrast, exposure,
saturation, shadows/highlights, white‑balance (temperature/tint), per‑hue HSL
corrections—and creative effects such as vignette, film‑grain, and simple
median denoising.

Tweak‑cheat
-----------
For any multiplicative parameter (``factor``, ``amount``, ``strength``) a 10 %
increase is achieved by multiplying by **1.10** (or **0.90** to decrease). For
exposure this is ≈ **± 0.14 EV** because ``2**0.14 ≈ 1.10``.  Temp ±500 mired and
Tint ±20 roughly equal a 10 % slider nudge in Lightroom.

Dependencies:
    pip install pillow numpy
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance
from smolagents import tool

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a ``PIL.Image`` to a ``float32`` NumPy array in the `[0, 1]` range.

    Args:
        img (PIL.Image.Image): Input image in RGB mode.

    Returns:
        np.ndarray: Array of shape *(H, W, 3)* with values ∈ [0, 1].
    """
    return np.asarray(img).astype(np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    """Convert a `[0, 1]` NumPy array back to an 8‑bit ``PIL.Image``.

    Args:
        arr (np.ndarray): Image data scaled to the `[0, 1]` range.

    Returns:
        PIL.Image.Image: 8‑bit RGB image.
    """
    arr_uint8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)


# ---------------------------------------------------------------------------
# Basic global adjustments
# ---------------------------------------------------------------------------


@tool
def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust global contrast.

    Args:
        img (PIL.Image.Image): Input image.
        factor (float): Contrast multiplier. `1.0` leaves the image unchanged;
            values > 1 increase contrast and values < 1 flatten it.

    Returns:
        PIL.Image.Image: Contrast‑adjusted image.

    Notes:
        A 10 % boost corresponds to `factor *= 1.10` (or `0.90` to reduce 10 %).
    """
    return ImageEnhance.Contrast(img).enhance(factor)


@tool
def adjust_exposure(img: Image.Image, ev: float) -> Image.Image:
    """Adjust exposure by a given EV (Exposure Value) offset.

    Args:
        img (PIL.Image.Image): Input image.
        ev (float): Exposure compensation in stops. `+1` doubles brightness,
            `‑1` halves it.

    Returns:
        PIL.Image.Image: Exposure‑adjusted image.

    Notes:
        `±0.14 EV` is roughly a 10 % luminance change (`2**0.14 ≈ 1.10`).
    """
    return _to_image(_to_numpy(img) * (2.0**ev))


@tool
def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """Adjust global saturation.

    Args:
        img (PIL.Image.Image): Input image.
        factor (float): Saturation multiplier. Values > 1 intensify colour and
            values < 1 desaturate.

    Returns:
        PIL.Image.Image: Saturation‑adjusted image.

    Notes:
        `factor *= 1.10` (or `0.90`) yields a ± 10 % change.
    """
    return ImageEnhance.Color(img).enhance(factor)


# ---------------------------------------------------------------------------
# Shadows / Highlights
# ---------------------------------------------------------------------------


@tool
def adjust_shadows_highlights(
    img: Image.Image,
    shadow: float = 1.0,
    highlight: float = 1.0,
) -> Image.Image:
    """Lift shadows or tame highlights.

    Args:
        img (PIL.Image.Image): Input image.
        shadow (float, optional): Multiplier applied mainly to dark tones.
            Defaults to `1.0`. Values > 1 brighten shadows; < 1 darken them.
        highlight (float, optional): Multiplier applied mainly to bright tones.
            Defaults to `1.0`. Values < 1 recover detail; > 1 brighten further.

    Returns:
        PIL.Image.Image: Image with adjusted shadows/highlights.

    Notes:
        A 10 % shadow lift is `shadow *= 1.10`; a 10 % highlight cut is
        `highlight *= 0.90`.
    """
    arr = _to_numpy(img)
    lum = arr.mean(axis=2, keepdims=True)
    shadow_mask = np.clip(1.0 - lum * 2.0, 0.0, 1.0)
    highlight_mask = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)
    arr = arr * (shadow_mask * (shadow - 1.0) + 1.0)
    arr = arr * (highlight_mask * (highlight - 1.0) + 1.0)
    return _to_image(arr)


# ---------------------------------------------------------------------------
# White‑balance: Temperature & Tint
# ---------------------------------------------------------------------------


@tool
def adjust_temperature(img: Image.Image, delta: int) -> Image.Image:
    """Shift white‑balance temperature.

    Args:
        img (PIL.Image.Image): Input image.
        delta (int): Temperature shift in *mireds*. Positive values warm the
            image (yellow/red); negative values cool it (blue).

    Returns:
        PIL.Image.Image: Temperature‑adjusted image.

    Notes:
        ± 500 mired gives roughly a 10 % change on Lightroom’s Temp slider.
    """
    arr = _to_numpy(img)
    r_scale, b_scale = 1.0 + delta * 4e-4, 1.0 - delta * 4e-4
    return _to_image(arr * np.array([r_scale, 1.0, b_scale], dtype=np.float32))


@tool
def adjust_tint(img: Image.Image, delta: int) -> Image.Image:
    """Shift white‑balance tint between green and magenta.

    Args:
        img (PIL.Image.Image): Input image.
        delta (int): Tint shift. Positive values push toward magenta; negative
            values toward green.

    Returns:
        PIL.Image.Image: Tint‑adjusted image.

    Notes:
        ± 20 approximates a 10 % tweak of Lightroom’s Tint slider.
    """
    arr = _to_numpy(img)
    g_scale, rb_scale = 1.0 - delta * 5e-4, 1.0 + delta * 5e-4
    return _to_image(arr * np.array([rb_scale, g_scale, rb_scale], dtype=np.float32))


# ---------------------------------------------------------------------------
# RGB ⇄ HSL helpers
# ---------------------------------------------------------------------------


def _rgb_to_hsl(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB→HSL conversion.

    Args:
        arr (np.ndarray): Float array in `[0, 1]` with shape *(H, W, 3)*.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Hue, Saturation, Lightness
        arrays each in `[0, 1]` and shape *(H, W)*.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc, minc = arr.max(axis=2), arr.min(axis=2)
    li = (maxc + minc) / 2.0
    s = np.zeros_like(li)
    diff = maxc - minc
    mask = diff != 0
    lesser = li < 0.5
    s[mask & lesser] = diff[mask & lesser] / (maxc + minc)[mask & lesser]
    s[mask & ~lesser] = diff[mask & ~lesser] / (2.0 - maxc - minc)[mask & ~lesser]
    h = np.zeros_like(li)
    rc, gc, bc = (maxc - r) / (diff + 1e-20), (maxc - g) / (diff + 1e-20), (maxc - b) / (diff + 1e-20)
    h[maxc == r] = (bc - gc)[maxc == r]
    h[maxc == g] = 2.0 + (rc - bc)[maxc == g]
    h[maxc == b] = 4.0 + (gc - rc)[maxc == b]
    h = (h / 6.0) % 1.0
    h[~mask] = 0.0
    return h, s, li


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, li: np.ndarray) -> np.ndarray:
    """Vectorised HSL→RGB conversion.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.

    Returns:
        np.ndarray: Reconstructed RGB array in `[0, 1]`.
    """

    def _f(n: float) -> np.ndarray:
        k = (n + h * 12.0) % 12.0
        a = s * np.minimum(li, 1.0 - li)
        return li - a * np.clip(np.minimum(np.minimum(k - 3.0, 9.0 - k), 1.0), -1.0, 1.0)

    r, g, b = _f(0.0), _f(8.0), _f(4.0)
    return np.stack([r, g, b], axis=-1)


# Hue centres and +/- half‑widths (degrees) taken from Adobe’s HSL model
COLOR_RANGES: dict[str, tuple[float, float]] = {
    "red": (345, 15),  # 345° → 15° (wraps around 0)
    "orange": (15, 45),
    "yellow": (45, 75),
    "green": (75, 165),
    "aqua": (165, 195),
    "blue": (195, 255),
    "purple": (255, 285),
    "magenta": (285, 345),
}

ColorName = Literal[
    "red",
    "orange",
    "yellow",
    "green",
    "aqua",
    "blue",
    "purple",
    "magenta",
]


def _range_for(color: ColorName) -> tuple[float, float]:
    start, end = COLOR_RANGES[color]
    return start % 360, end % 360


@tool
def adjust_hue_color(img: Image.Image, color: ColorName, delta: float) -> Image.Image:
    """Shift the **hue** of a specific colour bucket.

    Args:
        img (PIL.Image.Image): Input RGB image.
        color (ColorName): Colour family to target [red, orange, yellow, green, aqua, blue, purple, magenta]
        delta (float): Hue shift *in degrees*.

    Returns:
        PIL.Image.Image: Image with adjusted hue for the selected colour.

    Notes:
        ± 10° is roughly a *10 %* tweak of the full hue wheel segment.
        Possible colors are: red, orange, yellow, green, aqua, blue, purple, magenta
    """
    return adjust_hsl_channel(img, _range_for(color), h_delta=delta)


@tool
def adjust_saturation_color(img: Image.Image, color: ColorName, factor: float) -> Image.Image:
    """Change **saturation** of a specific colour bucket.

    Args:
        img (PIL.Image.Image): Input image.
        color (ColorName): Colour family to target [red, orange, yellow, green, aqua, blue, purple, magenta]
        factor (float): Saturation multiplier.

    Returns:
        PIL.Image.Image: Image with adjusted saturation for the selected colour.

    Notes:
        Multiply ``factor`` by *1.10* (or *0.90*) for a ± 10 % change.
        Possible colors are: red, orange, yellow, green, aqua, blue, purple, magenta
    """
    return adjust_hsl_channel(img, _range_for(color), s_factor=factor)


@tool
def adjust_luminance_color(img: Image.Image, color: ColorName, factor: float) -> Image.Image:
    """Change **luminance** (Lightness) of a specific colour bucket.

    Args:
        img (PIL.Image.Image): Input image.
        color (ColorName): Colour family to target[red, orange, yellow, green, aqua, blue, purple, magenta]
        factor (float): Luminance multiplier.

    Returns:
        PIL.Image.Image: Image with adjusted luminance for the selected colour.

    Notes:
        Multiply ``factor`` by *1.10* (or *0.90*) for a ± 10 % luminance tweak.
        Possible colors are: red, orange, yellow, green, aqua, blue, purple, magenta
    """
    return adjust_hsl_channel(img, _range_for(color), l_factor=factor)


def adjust_hsl_channel(
    img: Image.Image,
    hue_range: tuple[float, float],
    h_delta: float = 0.0,
    s_factor: float = 1.0,
    l_factor: float = 1.0,
) -> Image.Image:
    """Adjust Hue, Saturation, or Lightness for pixels within a hue slice.

    Args:
        img (PIL.Image.Image): Input image.
        hue_range (Tuple[float, float]): Start and end hue in degrees `[0, 360)`.
            The range may wrap past 360° (e.g. `(350, 20)` selects reds).
        h_delta (float, optional): Hue shift in degrees. Defaults to `0.0`.
        s_factor (float, optional): Saturation multiplier. Defaults to `1.0`.
        l_factor (float, optional): Lightness multiplier. Defaults to `1.0`.

    Returns:
        PIL.Image.Image: Image with per‑hue HSL adjustment applied.

    Notes:
        Typical 10 % tweaks: `h_delta ≈ ±10°`, `s_factor *= 1.10`,
        `l_factor *= 1.10` (or `0.90`).
    """
    arr = _to_numpy(img)
    h, s, li = _rgb_to_hsl(arr)

    h_start, h_end = np.deg2rad(hue_range[0]), np.deg2rad(hue_range[1])
    h_rad = h * 2 * math.pi
    if hue_range[0] <= hue_range[1]:
        mask = (h_rad >= h_start) & (h_rad <= h_end)
    else:
        mask = (h_rad >= h_start) | (h_rad <= h_end)

    h_new = h.copy()
    s_new = s.copy()
    l_new = li.copy()

    h_new[mask] = (h[mask] + h_delta / 360.0) % 1.0
    s_new[mask] = np.clip(s[mask] * s_factor, 0.0, 1.0)
    l_new[mask] = np.clip(li[mask] * l_factor, 0.0, 1.0)

    return _to_image(_hsl_to_rgb(h_new, s_new, l_new))


# ---------------------------------------------------------------------------
# Creative effects
# ---------------------------------------------------------------------------


@tool
def add_vignette(img: Image.Image, strength: float = 0.5, softness: float = 0.5) -> Image.Image:
    """Add a radial vignette.

    Args:
        img (PIL.Image.Image): Input image.
        strength (float, optional): Corner darkening amount in `[0, 1]`. Defaults
            to `0.5`. `strength *= 1.10` boosts the vignette ~10 %.
        softness (float, optional): Edge fall‑off exponent. Larger values give
            smoother vignettes. Defaults to `0.5`.

    Returns:
        PIL.Image.Image: Vignetted image.
    """
    w, h = img.size
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = 1.0 - strength * ((r / r.max()) ** (softness * 4.0))
    return _to_image(_to_numpy(img) * mask[..., None])


@tool
def add_grain(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """Add monochromatic Gaussian grain.

    Args:
        img (PIL.Image.Image): Input image.
        amount (float, optional): Noise standard deviation in the `[0, 1]`
            domain. Defaults to `0.05`. `amount *= 1.10` ≈ +10 % more grain.

    Returns:
        PIL.Image.Image: Noisy image with film‑like grain.
    """
    noise = np.random.normal(0.0, amount, _to_numpy(img).shape).astype(np.float32)
    return _to_image(_to_numpy(img) + noise)


@tool
def save_image(img: Image.Image, path: str) -> None:
    """Save a PIL image to a file.

    Args:
        img (PIL.Image.Image): Image to save.
        path (str): File path where the image will be saved.
    """
    img.save(path, format="JPEG", quality=95)


@tool
def load_image(path: str) -> Image.Image:
    """Load an image from a file.

    Args:
        path (str): File path to the image.

    Returns:
        PIL.Image.Image: Loaded image in RGB mode.
    """
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Demo pipeline
# ---------------------------------------------------------------------------


def demo_all(input_path: str, output_dir: str | Path = "demo_out") -> dict[str, str]:
    """Run every adjustment once and save results.

    Args:
        input_path (str): Path to the source image file.
        output_dir (str | Path, optional): Directory to write results. Defaults
            to ``"demo_out"``.

    Returns:
        Dict[str, str]: Mapping of effect name to the saved file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    img = Image.open(input_path).convert("RGB")

    effects = {
        "contrast": adjust_contrast(img, 1.1),
        "exposure": adjust_exposure(img, 0.2),
        "saturation": adjust_saturation(img, 1.2),
        "shadows_highlights": adjust_shadows_highlights(img, 1.2, 0.9),
        "temperature": adjust_temperature(img, 300),
        "tint": adjust_tint(img, 15),
        "vignette": add_vignette(img, 0.6, 0.7),
        "grain": add_grain(img, 0.08),
        "blue_hue": adjust_hue_color(img, "blue", 10),
        "blue_saturation": adjust_saturation_color(img, "blue", 1.1),
        "blue_luminance": adjust_luminance_color(img, "blue", 1.1),
    }

    saved: dict[str, str] = {}
    stem = Path(input_path).stem
    for name, im in effects.items():
        file_path = output_path / f"{stem}_{name}.jpg"
        im.save(file_path, quality=95)
        saved[name] = str(file_path)

    return saved


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Run all photo adjustments on an image.")
    parser.add_argument("--input", "-i", type=str, help="Path to the input image file.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save the output images.",
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = tempfile.mkdtemp(prefix="photo_adjustments_")

    results = demo_all(args.input, args.output_dir)
    for effect, path in results.items():
        print(f"{effect}: {path}")
