import cv2
import numpy as np
from PIL import Image

def apply_filters(image: np.ndarray) -> list[np.ndarray]:
    """Applies a series of filters to the input image.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list[np.ndarray]: List of filtered images in BGR format.
    """
    filtered_images = []

    # Filter 1: Contrast adjustment
    filtered_images.append(adjust_contrast(image))

    # Filter 2: Saturation boost
    filtered_images.append(adjust_saturation(image))

    # Filter 3: Exposure adjustment
    filtered_images.append(adjust_exposure(image))

    # Filter 4: Denoised
    filtered_images.append(denoise_image(image))

    # Filter 5: Vignette effect
    filtered_images.append(apply_vignette(image))

    return filtered_images

def adjust_contrast(image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Adjusts the contrast of the image.

    Args:
        image (np.ndarray): Input image in BGR format.
        alpha (float, optional): Contrast control (1.0-3.0). 1.0 means no change. Defaults to 1.5.

    Returns:
        np.ndarray: Contrast adjusted image in BGR format.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_saturation(image: np.ndarray, saturation_scale: float = 1.0) -> np.ndarray:
    """Adjusts the saturation of the image.

    Args:
        image (np.ndarray): Input image in BGR format.
        saturation_scale (float, optional): Saturation scale factor. 1.0 means no change. Defaults to 1.0.

    Returns:
        np.ndarray: Saturation adjusted image in BGR format.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[:, :, 1] *= saturation_scale
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_exposure(image: np.ndarray, beta: int = 50) -> np.ndarray:
    """Adjusts the exposure (brightness) of the image.

    Args:
        image (np.ndarray): Input image in BGR format.
        beta (int, optional): Brightness control. Positive values increase brightness, negative decrease. Defaults to 50.

    Returns:
        np.ndarray: Exposure adjusted image in BGR format.
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

def denoise_image(image: np.ndarray, h: int = 10) -> np.ndarray:
    """Denoises the image using Non-local Means Denoising algorithm.

    Args:
        image (np.ndarray): Input image in BGR format.
        h (int, optional): Filter strength. Higher h value removes noise better but removes details. Defaults to 10.

    Returns:
        np.ndarray: Denoised image in BGR format.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crops the image to the specified rectangle.

    Args:
        image (np.ndarray): Input image in BGR format.
        x (int): Top-left x-coordinate.
        y (int): Top-left y-coordinate.
        width (int): Width of the crop rectangle.
        height (int): Height of the crop rectangle.

    Returns:
        np.ndarray: Cropped image in BGR format.
    """
    return image[y:y+height, x:x+width]

def apply_vignette(image: np.ndarray, level: int = 2) -> np.ndarray:
    """Applies a vignette effect to the image.

    Args:
        image (np.ndarray): Input image in BGR format.
        level (int, optional): Intensity of the vignette effect. Defaults to 2.

    Returns:
        np.ndarray: Image with vignette effect applied in BGR format.
    """
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/level)
    kernel_y = cv2.getGaussianKernel(rows, rows/level)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = np.copy(image)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette

if __name__ == "__main__":
    # Load a test image
    test_image = Image.open("toa-heftiba-Xmn-QXsVL4k-unsplash.jpg")
    # Convert PIL Image to numpy array (BGR format for OpenCV)
    test_image_np = np.array(test_image)
    test_image_np = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR)
    
    # Apply all filters
    filtered_images = apply_filters(test_image_np)
    
    # Save results
    for i, filtered_img in enumerate(filtered_images):
        # Convert back to RGB for saving
        rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_img).save(f"filter_{i+1}_result.jpg")
        print(f"Saved filter_{i+1}_result.jpg")