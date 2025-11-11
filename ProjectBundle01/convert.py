# convert.py
# Command Line Syntax: python convert.py <input_image_path> <output_image_filename>
# My usage: python convert.py sample_images/MariaVonLinden.jpg MariaVonLinden_gray.jpg

import os
import sys

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from setup import torch, PIL, transforms, get_best_device, nn, make_output_path

def color_to_gray(input_path, output_filename, output_base_folder=None):
    device = get_best_device()
    print(f"🔧 Using device: {device}")

    output_path = make_output_path(input_path, output_filename, output_base_folder)

    # Define 1x1 conv weights for grayscale conversion
    weights = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]], device=device)
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False).to(device)
    conv.weight.data = weights

    img = PIL.Image.open(input_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        gray_tensor = conv(img_tensor).squeeze(0).squeeze(0).cpu()

    transforms.ToPILImage()(gray_tensor).save(output_path)
    print(f"Saved grayscale image to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_image> <output_image_filename>")
        sys.exit(1)

    color_to_gray(sys.argv[1], sys.argv[2])