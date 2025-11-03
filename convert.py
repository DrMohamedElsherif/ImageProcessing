# convert.py

import os
import sys
from setup import torch, PIL, transforms, get_best_device
import torch.nn as nn

def color_to_gray(input_path, output_path, output_base_folder="./Outputs"):
    device = get_best_device()
    print(f"🔧 Using device: {device}")

    input_folder_name = os.path.basename(os.path.dirname(input_path))
    output_folder = os.path.join(output_base_folder, f"Output_{input_folder_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(output_path))

    weights = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]], device=device)
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False).to(device)
    conv.weight.data = weights

    img = PIL.Image.open(input_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        gray_tensor = conv(img_tensor).squeeze(0).squeeze(0).cpu()

    transforms.ToPILImage()(gray_tensor).save(output_path)
    print(f"✅ Saved grayscale image to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_image> <output_image_filename>")
        sys.exit(1)

    color_to_gray(sys.argv[1], sys.argv[2])
