# src/editor.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.fourier_transform import compute_fft, inverse_fft

def interactive_lowpass_filter(image):
    """
    Interactive radial low-pass filter editor.
    """
    F_shift, magnitude, phase = compute_fft(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.25)

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    filtered_display = ax[1].imshow(image, cmap="gray")
    ax[1].set_title("Filtered")
    ax[1].axis("off")

    slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
    cutoff_slider = Slider(slider_ax, "Cutoff", 1, crow, valinit=30)

    def update(val):
        r = int(cutoff_slider.val)
        Y, X = np.ogrid[:rows, :cols]
        mask = ((X - ccol)**2 + (Y - crow)**2 <= r*r).astype(float)

        F_mod = mask * magnitude * np.exp(1j * phase)
        img_filtered = inverse_fft(F_mod)
        filtered_display.set_data(img_filtered)
        fig.canvas.draw_idle()

    cutoff_slider.on_changed(update)
    plt.show()
