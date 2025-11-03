# mosaic.py
# =============================================================
# 🧩 Photo Mosaic Generator (with optional GPU acceleration)
# =============================================================
# Usage:
#   python mosaic.py <input_image> <output_image_name> <num_cols> <num_rows> [method]
#
# Example:
#   python mosaic.py sample_images/Mohamed\ A.\ Elsherif-\ Photo.JPG mosaic_animals.jpg 60 40 avg
#
# Description:
#   - Automatically downloads Animals-10 dataset via kagglehub (if not already)
#   - Builds a photo mosaic using those images as tiles
#   - Supports GPU acceleration (MPS/CUDA) for color matching
# =============================================================

import os
import sys

# -------------------------------------------------------------
# 🌍 IMPORTS FROM SETUP.PY
# -------------------------------------------------------------
from setup import torch, np, PIL, Image, transforms, random, shutil, kagglehub, get_best_device

# -------------------------------------------------------------
# 📦 AUTO-DOWNLOAD AND PREPARE DATASET
# -------------------------------------------------------------
def get_animals10_dataset(local_folder="./Datasets/Animals10"):
    """Downloads Animals-10 dataset via kagglehub if not already present."""
    target_path = os.path.join(local_folder, "raw")
    os.makedirs(target_path, exist_ok=True)

    # Check if dataset already exists
    if any(os.scandir(target_path)):
        print(f"✅ Dataset already available at: {target_path}")
        return target_path

    print("📥 Downloading Animals-10 dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
    print(f"✅ Dataset downloaded to: {dataset_path}")

    # Copy all images into target folder
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(root, f)
                dst = os.path.join(target_path, f)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    print(f"✅ All images copied to {target_path}")
    return target_path

# -------------------------------------------------------------
# 🧠 COLOR UTILITIES
# -------------------------------------------------------------
def get_average_color(img):
    img = img.resize((16, 16))
    arr = np.array(img)
    return tuple(np.mean(arr.reshape(-1, 3), axis=0))

def get_brightness(color):
    r, g, b = color
    return 0.299*r + 0.587*g + 0.114*b

def choose_best_match_gpu(target_colors, tile_colors, device):
    target_tensor = torch.tensor(target_colors, dtype=torch.float32, device=device)
    tile_tensor = torch.tensor(tile_colors, dtype=torch.float32, device=device)
    dists = torch.cdist(target_tensor.unsqueeze(0), tile_tensor.unsqueeze(0)).squeeze(0)
    return torch.argmin(dists, dim=1).cpu().numpy()

# -------------------------------------------------------------
# 🧩 MOSAIC CREATION
# -------------------------------------------------------------
def create_mosaic(input_path, output_filename, num_cols, num_rows, method="avg", output_base_folder="./Outputs"):
    """Create a mosaic using the Animals-10 dataset automatically."""
    device = get_best_device()
    print(f"🔧 Using device: {device}")

    # Automatically get dataset folder
    tiles_folder = get_animals10_dataset()

    # --- Prepare output folder ---
    input_folder_name = os.path.basename(os.path.dirname(input_path))
    output_folder = os.path.join(output_base_folder, f"Output_{input_folder_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(output_filename))

    # --- Load source image ---
    src_img = Image.open(input_path).convert("RGB")
    src_w, src_h = src_img.size
    print(f"📷 Source image: {input_path} ({src_w}x{src_h})")

    # --- Load tile images ---
    tile_paths = [os.path.join(root, f)
                  for root, _, files in os.walk(tiles_folder)
                  for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not tile_paths:
        raise ValueError(f"No images found in tile folder: {tiles_folder}")

    tiles = [Image.open(p).convert("RGB") for p in tile_paths]
    tile_colors = np.array([get_average_color(t) for t in tiles])
    print(f"🧩 Loaded {len(tiles)} tiles from Animals-10 dataset")

    # --- Resize source image ---
    cell_w, cell_h = src_w // num_cols, src_h // num_rows
    src_small = src_img.resize((num_cols, num_rows))
    src_pixels = np.array(src_small).reshape(-1, 3)

    print(f"⚙️ Matching using method: {method}")

    # --- Match colors ---
    if method.lower() == "avg":
        try:
            tile_idxs = choose_best_match_gpu(src_pixels, tile_colors, device)
            print("✅ GPU acceleration used")
        except Exception as e:
            print(f"⚠️ GPU matching failed ({e}), using CPU fallback")
            diffs = np.linalg.norm(tile_colors[None, :, :] - src_pixels[:, None, :], axis=2)
            tile_idxs = np.argmin(diffs, axis=1)
    elif method.lower() == "brightness":
        src_b = np.apply_along_axis(get_brightness, 1, src_pixels)
        tile_b = np.apply_along_axis(get_brightness, 1, tile_colors)
        diffs = np.abs(src_b[:, None] - tile_b[None, :])
        tile_idxs = np.argmin(diffs, axis=1)
    elif method.lower() == "random":
        tile_idxs = np.random.randint(0, len(tiles), len(src_pixels))
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Build mosaic ---
    mosaic_img = Image.new("RGB", (cell_w * num_cols, cell_h * num_rows))
    print("🧱 Building mosaic...")
    for i, idx in enumerate(tile_idxs):
        row, col = divmod(i, num_cols)
        mosaic_img.paste(tiles[idx].resize((cell_w, cell_h)), (col*cell_w, row*cell_h))
    mosaic_img.save(output_path)
    print(f"✅ Mosaic saved to: {output_path}")

# -------------------------------------------------------------
# 🧭 MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python mosaic.py <input_image> <output_image_name> <num_cols> <num_rows> [method]")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image_name = sys.argv[2]
    num_cols = int(sys.argv[3])
    num_rows = int(sys.argv[4])
    method = sys.argv[5] if len(sys.argv) > 5 else "avg"

    create_mosaic(input_image, output_image_name, num_cols, num_rows, method)
