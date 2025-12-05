
from flask import Flask, render_template, jsonify, send_from_directory
import os
import random
import re

app = Flask(__name__)

# ---------------------------------
# Absolute paths to project folders
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
SPEC_DIR = os.path.join(BASE_DIR, "spectra")

# ---------------------------------
# Helper to clean filenames
# ---------------------------------
def clean_filename(filename):
    return re.sub(r"[^\w\-.]", "_", filename)

# ---------------------------------
# Serve image and spectrum files
# ---------------------------------
@app.route("/img/<path:filename>")
def serve_img(filename):
    return send_from_directory(IMG_DIR, filename)

@app.route("/spectra/<path:filename>")
def serve_spectra(filename):
    return send_from_directory(SPEC_DIR, filename)

# ---------------------------------
# Home page
# ---------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------
# Game cards endpoint
# ---------------------------------
@app.route("/cards")
def get_cards():
    images = [clean_filename(f) for f in os.listdir(IMG_DIR) if not f.startswith(".")]
    spectra = [clean_filename(f) for f in os.listdir(SPEC_DIR) if not f.startswith(".")]

    cards = []
    card_id = 0

    for filename in images:
        if filename in spectra:
            cards.append({
                "id": card_id,
                "type": "image",
                "file": f"/img/{filename}",
                "pair_key": filename
            })
            card_id += 1
            cards.append({
                "id": card_id,
                "type": "fourier",
                "file": f"/spectra/{filename}",
                "pair_key": filename
            })
            card_id += 1

    random.shuffle(cards)
    return jsonify(cards)

# ---------------------------------
# Run server
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
