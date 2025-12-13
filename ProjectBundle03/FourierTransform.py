from pathlib import Path
import re
import cv2
import numpy as np

def sanitize_filename(name: str) -> str:
    """
    Replace characters that are unsafe for filesystems or URLs.
    """
    return re.sub(r"[^\w\-.]", "_", name)

def fourier_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude spectrum of the Fourier transform of
    a real-valued 2D image. The result is logarithmically scaled
    and returned in 8-bit form.
    """
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(f_shift))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return magnitude.astype(np.uint8)

def process_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Process all image files in `input_dir` and write their Fourier
    magnitude spectra to `output_dir`. Non-image files and hidden
    entries are ignored.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for entry in input_dir.iterdir():
        if entry.name.startswith("."):
            continue

        if entry.suffix.lower() not in valid_suffixes:
            continue

        img = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to read {entry.name}; skipping.")
            continue

        spectrum = fourier_magnitude(img)
        out_name = sanitize_filename(entry.name)
        cv2.imwrite(str(output_dir / out_name), spectrum)

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    input_dir = base / "img"
    output_dir = base / "spectra"

    process_directory(input_dir, output_dir)
    print("Processing complete.")
