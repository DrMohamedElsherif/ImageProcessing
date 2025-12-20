# main.py

import matplotlib.pyplot as plt
import os

from src.image_io import load_image, discover_images
from src.fourier_transform import compute_fft, inverse_fft
from src.visualization import (
    show_image,
    show_magnitude,
    show_phase,
    show_color_spectrum
)
from src.editor import interactive_lowpass_filter
from config import FIGSIZE_TRIPLE, IMAGE_DIR


def process_image(image_path):
    print(f"\nProcessing: {os.path.basename(image_path)}")

    img = load_image(image_path)
    F_shift, magnitude, phase = compute_fft(img)

    plt.figure(figsize=FIGSIZE_TRIPLE)
    plt.subplot(1, 3, 1)
    show_image(img, "Original")

    plt.subplot(1, 3, 2)
    show_magnitude(magnitude)

    plt.subplot(1, 3, 3)
    show_phase(phase)
    plt.show()

    show_color_spectrum(magnitude, phase)
    plt.show()

    reconstructed = inverse_fft(F_shift)
    plt.figure(figsize=(5, 5))
    show_image(reconstructed, "Reconstructed")
    plt.show()

    # Interactive editor (optional per image)
    interactive_lowpass_filter(img)


def main():
    image_paths = discover_images(IMAGE_DIR)

    print(f"Discovered {len(image_paths)} images:")
    for p in image_paths:
        print(" -", os.path.basename(p))

    for image_path in image_paths:
        process_image(image_path)


if __name__ == "__main__":
    main()
