import os
import cv2
import numpy as np
import re

# ---------------------------
# Paths
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(current_dir, "img")
output_folder = os.path.join(current_dir, "spectra")

os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Helper to clean filenames
# ---------------------------
def clean_filename(filename):
    # Remove problematic characters for file system and URLs
    filename = re.sub(r"[^\w\-.]", "_", filename)
    return filename

# ---------------------------
# Process images
# ---------------------------
for filename in os.listdir(input_folder):
    # Skip hidden/system files
    if filename.startswith("."):
        print(f"Skipping hidden/system file: {filename}")
        continue

    # Only process image files
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
        print(f"Skipping non-image file: {filename}")
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, 0)

    if img is None:
        print(f"Warning: Cannot read {filename}, skipping...")
        continue

    # Compute Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Normalize and save
    out = (255 * magnitude / magnitude.max()).astype(np.uint8)
    clean_name = clean_filename(filename)
    cv2.imwrite(os.path.join(output_folder, clean_name), out)

print("Done! Fourier images saved.")
