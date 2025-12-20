# src/reconstruction.py

import numpy as np

def reconstruct_from_mag_phase(magnitude, phase):
    """
    Reconstruct image from magnitude and phase.
    """
    F = magnitude * np.exp(1j * phase)
    F_ishift = np.fft.ifftshift(F)
    img = np.fft.ifft2(F_ishift)
    return np.real(img)
