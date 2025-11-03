# mosaic.py

import os
import sys
import random
from setup import torch, np, PIL, get_best_device

Image = PIL.Image  # alias

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

def create_mosaic(input_path, tiles_folder, output_filename, num_cols, num_rows, method="avg", output_base_folder="./Outputs"):
    device = get_best_device()
    print(f"🔧 Using device: {device}")

    input_folder_name = os.path.basename(os.path.dirname(input_path))
    output_folder = os.path.join(output_base_folder, f"Output_{input_folder_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(output_filename))

    src_img = Image.open(input_path).convert("RGB")
    src_w, src_h = src_img.size
    print(f"📷 Source image: {input_path} ({src_w}x{src_h})")

    tile_paths = [os.path.join(tiles_folder, f) for f in os.listdir(tiles_folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not tile_paths:
        raise ValueError(f"No images found in tile folder: {tiles_folder}")

    tiles = [Image.open(p).convert("RGB") for p in tile_paths]
    tile_colors = np.array([get_average_color(t) for t in tiles])
    print(f"🧩 Loaded {len(tiles)} tiles")

    cell_w, cell_h = src_w // num_cols, src_h // num_rows
    src_small = src_img.resize((num_cols, num_rows))
    src_pixels = np.array(src_small).reshape(-1, 3)

    print(f"⚙️ Matching using '{method}'")
    if method.lower() == "avg":
        try:
            tile_idxs = choose_best_match_gpu(src_pixels, tile_colors, device)
            print("✅ GPU acceleration used")
        except Exception as e:
            print(f"⚠️ GPU failed ({e}), using CPU")
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

    mosaic_img = Image.new("RGB", (cell_w * num_cols, cell_h * num_rows))
    print("🧱 Building mosaic...")
    for i, idx in enumerate(tile_idxs):
        row, col = divmod(i, num_cols)
        mosaic_img.paste(tiles[idx].resize((cell_w, cell_h)), (col*cell_w, row*cell_h))
    mosaic_img.save(output_path)
    print(f"✅ Mosaic saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python mosaic.py <input_image> <tiles_folder> <output_image_name> <num_cols> <num_rows> [method]")
        sys.exit(1)
    create_mosaic(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else "avg")
