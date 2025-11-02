# convert.py
# Usage: python convert.py <input_image> <output_image>
# Converts an RGB image to grayscale using BT.601 weights via PyTorch Conv2d

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def color_to_gray(input_path, output_path, output_base_folder="./Outputs"):
    # --- Device selection ---
    device = get_best_device()
    print(f"🔧 Using device: {device}")

    # --- Prepare output folder ---
    input_folder_name = os.path.basename(os.path.dirname(input_path))
    output_folder = os.path.join(output_base_folder, f"Output_{input_folder_name}")
    os.makedirs(output_folder, exist_ok=True)

    # Ensure output_path uses the folder we just created
    output_path = os.path.join(output_folder, os.path.basename(output_path))

    # --- BT.601 weights for grayscale ---
    weights = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]], device=device)  # (1,3,1,1)
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False).to(device)
    conv.weight.data = weights

    # --- Load image ---
    img = Image.open(input_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # --- Convert to grayscale ---
    with torch.no_grad():
        gray_tensor = conv(img_tensor)
    gray_tensor = gray_tensor.squeeze(0).squeeze(0).cpu()

    # --- Save grayscale image ---
    gray_img = transforms.ToPILImage()(gray_tensor)
    gray_img.save(output_path)
    print(f"✅ Saved grayscale image to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_image> <output_image_filename>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image_name = sys.argv[2]
    color_to_gray(input_image, output_image_name)
