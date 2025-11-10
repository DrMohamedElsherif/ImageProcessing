# recreate_mosaic.py
# Command Line Syntax: python recreate_mosaic.py <input_image_path> <output_image_name> <target_tiles>

import sys
from setup import torch, F, transforms, Image, get_best_device, make_output_path

def recreate_mosaic_conv2d(input_image_path, output_filename, target_tiles, output_base_folder="./Outputs"):
    img = Image.open(input_image_path).convert("RGB")
    width, height = img.size
    print(f"📷 Loaded image: {input_image_path} ({width}x{height})")

    aspect_ratio = width / height
    num_rows = int((target_tiles / aspect_ratio) ** 0.5)
    num_cols = int(aspect_ratio * num_rows)
    tile_size = min(width // num_cols, height // num_rows)
    print(f"🔧 Grid: {num_cols}×{num_rows}, Tile: {tile_size}×{tile_size}")

    device = get_best_device()
    print(f"⚙️ Using device: {device}")

    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    kernel = torch.ones((img_tensor.shape[1], 1, tile_size, tile_size), device=device) / (tile_size ** 2)
    filtered = F.conv2d(img_tensor, kernel, stride=tile_size, groups=img_tensor.shape[1])
    mosaic = F.interpolate(filtered, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='nearest')
    mosaic_img = transforms.ToPILImage()(mosaic.squeeze(0).cpu())

    output_path = make_output_path(input_image_path, output_filename, output_base_folder)
    mosaic_img.save(output_path)
    print(f"✅ Mosaic saved to: {output_path}")
    print(f"📏 Final grid: {num_cols} × {num_rows} → {num_cols*num_rows} total tiles")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python recreate_mosaic.py <input_image> <output_image_name> <target_tiles>")
        sys.exit(1)

    recreate_mosaic_conv2d(sys.argv[1], sys.argv[2], int(sys.argv[3]))
