# src/image_io.py

import cv2
import os
from config import IMAGE_SIZE

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def discover_images(image_dir):
    """
    Automatically discover images in a directory.
    Returns a sorted list of image file paths.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if len(images) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    return sorted(images)


def load_image(path):
    """
    Load an image in grayscale and resize.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")

    img = cv2.resize(img, IMAGE_SIZE)
    return img
