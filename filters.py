from __future__ import annotations

import math
import os
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
            A factor of 1.1 is a lot. A factor of 1.02 is a delicate modification .


    Returns:
        PIL.Image.Image: Contrast‑adjusted image.

    """
    return ImageEnhance.Contrast(img).enhance(factor)


@tool
def adjust_exposure(img: Image.Image, ev: float) -> Image.Image:
    """Adjust exposure by a given EV (Exposure Value) offset.

    Args:
        img (PIL.Image.Image): Input image.
        ev (float): Exposure compensation in stops. `+1` doubles brightness,
            `‑1` halves it.
            a ev of 0.2 is a lot. a ev of 0.05 is a delicate modification .


    Returns:
        PIL.Image.Image: Exposure‑adjusted image.
    """
    return _to_image(_to_numpy(img) * (2.0**ev))


@tool
def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """Adjust global saturation.

    Args:
        img (PIL.Image.Image): Input image.
        factor (float): Saturation multiplier. Values > 1 intensify colour and
            values < 1 desaturate. `factor *= 1.10` (or `0.90`) yields a ± 10 % change.
            a factor of 1.5 is a lot. a factor of 1.1 is a delicate modification .

    Returns:
        PIL.Image.Image: Saturation‑adjusted image.

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
        Here, variations of 1, are a lot.
        Variations under 0.2 are not noticeable.

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


# @tool
def adjust_temperature(img: Image.Image, delta: int) -> Image.Image:
    """Shift white‑balance temperature.

    Args:
        img (PIL.Image.Image): Input image.
        delta (int): Temperature shift in *mireds*. Positive values warm the
            image (yellow/red); negative values cool it (blue).
            You should not beyond ± 700 mired.

    Returns:
        PIL.Image.Image: Temperature‑adjusted image.
    """
    arr = _to_numpy(img)
    r_scale, b_scale = 1.0 + delta * 4e-4, 1.0 - delta * 4e-4
    return _to_image(arr * np.array([r_scale, 1.0, b_scale], dtype=np.float32))


# @tool
def adjust_tint(img: Image.Image, delta: int) -> Image.Image:
    """Shift white‑balance tint between green and magenta.

    Args:
        img (PIL.Image.Image): Input image.
        delta (int): Tint shift. Positive values push toward magenta; negative
            values toward green.
            You should not go beyond ± 150. Changes under 40 are barely noticeable.

    Returns:
        PIL.Image.Image: Tint‑adjusted image.
    """
    arr = _to_numpy(img)
    g_scale, rb_scale = 1.0 - delta * 5e-4, 1.0 + delta * 5e-4
    return _to_image(arr * np.array([rb_scale, g_scale, rb_scale], dtype=np.float32))


# ---------------------------------------------------------------------------
# RGB ⇄ HSL helpers
# ---------------------------------------------------------------------------


@tool
def rgb_to_hsl(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB→HSL conversion.

    Args:
        img: PIL.Image.Image: 8‑bit RGB image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Hue, Saturation, Lightness
        arrays each in `[0, 1]` and shape *(H, W)*.
    """
    arr = _to_numpy(img)
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


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, li: np.ndarray) -> np.ndarray:
    """Vectorised HSL→RGB conversion.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.

    Returns:
        PIL.Image.Image: 8‑bit RGB image.
    """

    def _f(n: float) -> np.ndarray:
        k = (n + h * 12.0) % 12.0
        a = s * np.minimum(li, 1.0 - li)
        return li - a * np.clip(np.minimum(np.minimum(k - 3.0, 9.0 - k), 1.0), -1.0, 1.0)

    r, g, b = _f(0.0), _f(8.0), _f(4.0)
    return _to_image(np.stack([r, g, b], axis=-1))


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
def adjust_hue_color(
    h: np.ndarray, s: np.ndarray, li: np.ndarray, color: ColorName, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift the **hue** of a specific colour bucket.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.
        color (ColorName): Colour family to target [red, orange, yellow, green, aqua, blue, purple, magenta]
        delta (float): Hue shift *in degrees*. 15 degrees is good increment. 40 is a lot. 5 is a delicate modification .

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Hue, Saturation, Lightness
        arrays each in `[0, 1]` and shape *(H, W)*.

    """
    return adjust_hsl_channel(h, s, li, _range_for(color), h_delta=delta)


@tool
def adjust_saturation_color(
    h: np.ndarray, s: np.ndarray, li: np.ndarray, color: ColorName, factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change **saturation** of a specific colour bucket.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.
        color (ColorName): Colour family to target [red, orange, yellow, green, aqua, blue, purple, magenta]
        factor (float): Saturation multiplier. Factor under +/- 0.1 are delicate modifications.
            Factor of 0.2 and 1.8 are very strong variations.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Hue, Saturation, Lightness
        arrays each in `[0, 1]` and shape *(H, W)*.
    """
    return adjust_hsl_channel(h, s, li, _range_for(color), s_factor=factor)


def adjust_luminance_color(
    h: np.ndarray, s: np.ndarray, li: np.ndarray, color: ColorName, factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change **luminance** (Lightness) of a specific colour bucket.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.
        color (ColorName): Colour family to target[red, orange, yellow, green, aqua, blue, purple, magenta]
        factor (float): Luminance multiplier. The allowed maximum is 0.1 variation. 0.05 is a delicate modication.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Hue, Saturation, Lightness
        arrays each in `[0, 1]` and shape *(H, W)*.
    """
    return adjust_hsl_channel(h, s, li, _range_for(color), l_factor=factor)


def adjust_hsl_channel(
    h: np.ndarray,
    s: np.ndarray,
    li: np.ndarray,
    hue_range: tuple[float, float],
    h_delta: float = 0.0,
    s_factor: float = 1.0,
    l_factor: float = 1.0,
) -> Image.Image:
    """Adjust Hue, Saturation, or Lightness for pixels within a hue slice.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.
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

    return h_new, s_new, l_new


# ---------------------------------------------------------------------------
# Creative effects
# ---------------------------------------------------------------------------


@tool
def add_vignette(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """Add a radial vignette.

    Args:
        img (PIL.Image.Image): Input image.
        strength (float, optional): Corner darkening amount in `[0, 1]`. Defaults
            to `0.5`. `strength *= 1.10` boosts the vignette ~10 %.
        A strength of 1 is the maximum a lot. Under 0.2 is a delicate modification .
    Returns:
        PIL.Image.Image: Vignetted image.
    """
    softness = 0.5
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
            domain.
            An amount of 0.01 is a delicate modification . The max is 0.1.
            In classic usage 0.02 is a good start.
    """
    noise = np.random.normal(0.0, amount, _to_numpy(img).shape).astype(np.float32)
    return _to_image(_to_numpy(img) + noise)


@tool
def save_image(h: np.ndarray, s: np.ndarray, li: np.ndarray, output_directory: str) -> None:
    """Save an HSL image as a JPEG file in the specified directory.

    The image will be saved with a filename of the form "trial_N.jpeg", where N is the
    current count of JPEG files in the directory.

    Args:
        h (np.ndarray): Hue channel `[0, 1]`.
        s (np.ndarray): Saturation channel `[0, 1]`.
        li (np.ndarray): Lightness channel `[0, 1]`.
        output_directory (str): Path to the output directory.

    Returns:
        str: The full path to the saved image file.
    """
    img = hsl_to_rgb(h, s, li)
    nb_iter = str(len([f for f in os.listdir(output_directory) if f.endswith(".jpeg")]))
    output_path = os.path.join(output_directory, f"trial_{nb_iter}.jpeg")
    img.save(output_path, format="JPEG", quality=95)


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


def demo_all(input_path: str, output_dir: str | Path = "demo_out") -> None:
    """Run every adjustment once and save results.

    Args:
        input_path (str): Path to the source image file.
        output_dir (str | Path, optional): Directory to write results. Defaults
            to ``"demo_out"``.

    Returns:
        Dict[str, str]: Mapping of effect name to the saved file path.
    """
    img = load_image(path=input_path)

    # Apply global RGB adjustments in sequence
    # 2. Reduced exposure (moderately)
    img = adjust_exposure(img=img, ev=0.05)  # Decreased from 0.1 as per feedback

    # 1. Reduced contrast (significantly)
    img = adjust_contrast(img=img, factor=1.03)  # From 1.1 to 1.03 (delicate contrast increase)

    # 3. Reduced global saturation
    img = adjust_saturation(img=img, factor=1.1)  # From 1.2 to 1.1 (moderate enhancement)

    # 5. Subtler shadows/highlights
    img = adjust_shadows_highlights(img=img, shadow=1.05, highlight=0.95)  # From 1.1  # From 0.9 to 0.95

    # 7. Remove vignette
    img = add_vignette(img=img, strength=0.0)  # Set to min per feedback

    # 8. Remove grain
    img = add_grain(img=img, amount=0.0)  # No grain

    # Convert to HSL for color-specific modifications
    h, s, li = rgb_to_hsl(img)

    # 3(a) Reduced blue saturation
    h, s, li = adjust_saturation_color(h=h, s=s, li=li, color="blue", factor=1.3)  # From 1.5 to 1.3

    # 3(b) Reduced red & yellow saturation
    for color in ["red", "yellow"]:
        h, s, li = adjust_saturation_color(h=h, s=s, li=li, color=color, factor=1.1)  # From 1.2 to 1.1

    # 4. Adjusted hue (reduced intensity)
    # Red/orange toward amber
    h, s, li = adjust_hue_color(h=h, s=s, li=li, color="red", delta=10)  # From 15° to 10°
    h, s, li = adjust_hue_color(h=h, s=s, li=li, color="orange", delta=10)  # From 15° to 10°

    # Blue cooling (reduced effect)
    h, s, li = adjust_hue_color(h=h, s=s, li=li, color="blue", delta=-10)  # From -15° to -10°

    # 6. Adjusted luminance (reduced effect)
    # Blue luminance boost (reduced)
    # h, s, li = adjust_luminance_color(
    #     h=h, s=s, li=li,
    #     color='blue',
    #     factor=1.05  # From 1.1 to 1.05
    # )

    # # Orange luminance (adjusted)
    # h, s, li = adjust_luminance_color(
    #     h=h, s=s, li=li,
    #     color='orange',
    #     factor=0.95  # From 0.9 to 0.95
    # )

    # Save the processed image
    print(output_dir)
    save_image(h, s, li, output_directory=output_dir)


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

    demo_all(args.input, args.output_dir)
