# src/fourier_transform.py

import numpy as np

def compute_fft(image):
    """
    Compute 2D FFT and return shifted spectrum, magnitude, and phase.
    """
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)
    magnitude = np.abs(F_shift)
    phase = np.angle(F_shift)
    return F_shift, magnitude, phase


def inverse_fft(F_shift):
    """
    Reconstruct image from shifted FFT.
    """
    F_ishift = np.fft.ifftshift(F_shift)
    img_back = np.fft.ifft2(F_ishift)
    return np.real(img_back)
