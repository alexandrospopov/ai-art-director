import numpy as np
import cv2

def score_image(pil_img):
    img = np.array(pil_img.convert("L"))
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance

def evaluate_filters(images):
    scores = [score_image(img) for img in images]
    best_index = int(np.argmax(scores))
    reasons = [f"Sharpness score: {s:.2f}" for s in scores]
    return best_index, reasons