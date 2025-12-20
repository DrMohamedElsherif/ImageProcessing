# src/visualization.py

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(img, title="Image"):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")


def show_magnitude(magnitude):
    mag_log = np.log(1 + magnitude)
    plt.imshow(mag_log, cmap="gray")
    plt.title("Magnitude Spectrum (log scale)")
    plt.axis("off")


def show_phase(phase):
    plt.imshow(phase, cmap="gray")
    plt.title("Phase Spectrum")
    plt.axis("off")


def show_color_spectrum(magnitude, phase):
    mag_norm = magnitude / magnitude.max()
    phase_norm = (phase + np.pi) / (2 * np.pi)

    hsv = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    hsv[..., 0] = phase_norm
    hsv[..., 2] = mag_norm

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
    plt.title("Color-coded Fourier Spectrum")
    plt.axis("off")
