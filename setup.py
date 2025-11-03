# setup.py
# ============================================================
# 🧩 Cross-Platform PyTorch Environment Setup (GPU/CPU Adaptive)
# ============================================================

import os
import sys
import platform
import subprocess

# --- Lazy imports (to make them available to other scripts) ---
import importlib

# ============================================================
# 📦 INSTALL HELPERS
# ============================================================
def install(*packages):
    """Quietly install one or more pip packages."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages],
        check=True
    )

def detect_platform():
    system = platform.system().lower()
    if "darwin" in system:
        return "mac"
    elif "linux" in system:
        return "linux"
    elif "windows" in system:
        return "windows"
    else:
        return "unknown"

# ============================================================
# 🧠 DEVICE SELECTION
# ============================================================
def get_best_device():
    """Return the best available device: CUDA, MPS, or CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ============================================================
# 🚀 SETUP FUNCTION
# ============================================================
def main():
    """Set up the environment and verify installation."""
    print("🚀 Setting up environment for PyTorch with GPU/CPU support...\n")

    system = detect_platform()
    print(f"💻 Detected OS: {system.upper()}")

    print("\n📦 Installing PyTorch and dependencies...")
    if system == "mac":
        install("torch", "torchvision", "torchaudio")
    elif system == "linux":
        install(
            "torch==2.4.1+cu121",
            "torchvision==0.19.1+cu121",
            "torchaudio==2.4.1+cu121",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        )
    else:
        install("torch", "torchvision", "torchaudio")

    print("\n📦 Installing supporting libraries...")
    install(
        "numpy<2.0,>=1.24",
        "Pillow<12.0",
        "wrapt<2.0",
        "numba>=0.57,<0.62",
        "dynamax==0.1.8",
        "streamlit>=1.30,<1.42",
    )

    print("\n🔍 Verifying installation...\n")
    import torch, torchvision, numpy, PIL
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ TorchVision: {torchvision.__version__}")
    print(f"✅ NumPy: {numpy.__version__}")
    print(f"✅ Pillow: {PIL.__version__}")

    if system == "mac":
        print(f"✅ MPS available: {torch.backends.mps.is_available()}")
    elif system == "linux":
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    else:
        print("✅ CPU-only build verified.")

    print("\n📝 Exporting requirements.txt...")
    req = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open("requirements.txt", "w") as f:
        f.write(req.stdout)
    print("✅ requirements.txt saved.\n🎉 Setup complete!")

# ============================================================
# 🌍 SHARED IMPORTS (for other scripts)
# ============================================================
# Dynamically import the libraries only once
# 🌍 SHARED IMPORTS (for other scripts)
try:
    import importlib
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    PIL = importlib.import_module("PIL")
    Image = importlib.import_module("PIL.Image")  # PIL.Image module
    transforms = importlib.import_module("torchvision.transforms")
    random = importlib.import_module("random")

    # KaggleHub optional
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ModuleNotFoundError:
        print("📦 kagglehub not found. Installing quietly...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        kagglehub = importlib.import_module("kagglehub")

    import shutil
except ModuleNotFoundError:
    print("⚠️ Missing modules. Run `python setup.py` first to install dependencies.")
    raise


# ============================================================
# 🧩 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
