from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- Color Space Utilities ---------------- #
def srgb_to_linear(img_srgb):
    img = img_srgb.astype(np.float32) / 255.0
    mask = img <= 0.04045
    img_linear = np.empty_like(img)
    img_linear[mask] = img[mask] / 12.92
    img_linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return img_linear

def linear_to_srgb(img_linear):
    mask = img_linear <= 0.0031308
    img_srgb = np.empty_like(img_linear)
    img_srgb[mask] = img_linear[mask] * 12.92
    img_srgb[~mask] = 1.055 * (img_linear[~mask] ** (1/2.4)) - 0.055
    return np.clip(img_srgb * 255.0, 0, 255).astype(np.uint8)

# ---------------- Correction Implementations ---------------- #
def white_balance_grey_world(img_srgb):
    img_linear = srgb_to_linear(img_srgb)
    avg = np.mean(img_linear, axis=(0,1))
    gray = np.mean(avg)
    gains = gray / np.maximum(avg, 1e-8)
    img_balanced_linear = img_linear * gains
    return linear_to_srgb(img_balanced_linear)

def white_balance_white_patch(img_srgb):
    img_linear = srgb_to_linear(img_srgb)
    max_vals = np.max(img_linear, axis=(0,1))
    gains = 1.0 / np.maximum(max_vals, 1e-8)
    img_balanced_linear = img_linear * gains
    return linear_to_srgb(img_balanced_linear)

def white_balance_grey_edge(img_srgb):
    img_linear = srgb_to_linear(img_srgb)
    img_f = img_linear.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    avg = np.mean(grad, axis=(0,1))
    gray = np.mean(avg)
    gains = gray / np.maximum(avg, 1e-8)
    img_balanced_linear = img_f * gains
    return linear_to_srgb(img_balanced_linear)

ALGO = {
    "grey": "Grey-World",
    "white_patch": "White-Patch",
    "grey_edge": "Grey-Edge"
}

def apply_algorithm(img_srgb, algo_code):
    cpu_funcs = {
        "grey": white_balance_grey_world,
        "white_patch": white_balance_white_patch,
        "grey_edge": white_balance_grey_edge
    }
    func = cpu_funcs.get(algo_code, lambda x: x)
    return func(img_srgb)

# ---------------- Flask Routes ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    input_file = None
    output_file = None
    method_display = None

    if request.method == "POST":
        file = request.files["image"]
        algo = request.form.get("algorithm", "")

        if not algo:
            return render_template("corrector.html", input=None, result=None, method=None)

        # Save uploaded file
        input_filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)
        input_file = input_filename

        # Read image
        img = cv2.imread(input_path)

        # Apply selected algorithm
        result = apply_algorithm(img, algo)

        # Save output in output/
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_{algo}{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, result)
        output_file = output_filename

        method_display = ALGO.get(algo, algo)

    return render_template(
        "corrector.html",
        input=input_file,
        result=output_file,
        method=method_display
    )

# Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Serve corrected files from output folder
@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
