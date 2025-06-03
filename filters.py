import cv2
import numpy as np
from PIL import Image

def apply_filters(image):
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

def adjust_contrast(image, alpha=1.5):
    """
    Adjusts the contrast of the image.
    :param image: Input image (numpy array).
    :param alpha: Contrast control (1.0-3.0). 1.0 means no change.
    :return: Contrast adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_saturation(image, saturation_scale=1.0):
    """
    Adjusts the saturation of the image.
    :param image: Input image (numpy array).
    :param saturation_scale: Saturation scale factor. 1.0 means no change.
    :return: Saturation adjusted image.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[:, :, 1] *= saturation_scale
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_exposure(image, beta=50):
    """
    Adjusts the exposure (brightness) of the image.
    :param image: Input image (numpy array).
    :param beta: Brightness control. Positive values increase brightness, negative decrease.
    :return: Exposure adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

def denoise_image(image, h=10):
    """
    Denoises the image using Non-local Means Denoising algorithm.
    :param image: Input image (numpy array).
    :param h: Filter strength. Higher h value removes noise better but removes details.
    :return: Denoised image.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def crop_image(image, x, y, width, height):
    """
    Crops the image to the specified rectangle.
    :param image: Input image (numpy array).
    :param x: Top-left x-coordinate.
    :param y: Top-left y-coordinate.
    :param width: Width of the crop rectangle.
    :param height: Height of the crop rectangle.
    :return: Cropped image.
    """
    return image[y:y+height, x:x+width]

def apply_vignette(image, level=2):
    """
    Applies a vignette effect to the image.
    :param image: Input image (numpy array).
    :param level: Intensity of the vignette effect.
    :return: Image with vignette effect applied.
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