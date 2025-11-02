# ============================================================
# 🧩 Cross-Platform PyTorch Environment Setup (GPU/CPU Adaptive)
# ============================================================
# Works on macOS (MPS), Linux (CUDA), or Windows (CPU fallback)
# Automatically installs compatible dependencies and exports requirements.txt
# ============================================================

import os
import sys
import platform
import subprocess

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

def main():
    print("🚀 Setting up environment for PyTorch with GPU/CPU support...\n")

    system = detect_platform()
    print(f"💻 Detected OS: {system.upper()}")

    # ------------------------------------------------------------
    # 🔧 Install PyTorch based on OS
    # ------------------------------------------------------------
    print("\n📦 Installing PyTorch and related libraries...")

    if system == "mac":
        # Apple Silicon / Intel Mac — MPS backend
        install("torch", "torchvision", "torchaudio")

    elif system == "linux":
        # Linux — CUDA 12.1 build (compatible with most modern GPUs)
        install(
            "torch==2.4.1+cu121",
            "torchvision==0.19.1+cu121",
            "torchaudio==2.4.1+cu121",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        )

    elif system == "windows":
        # Windows — CPU-only build (since CUDA setup can vary)
        install("torch", "torchvision", "torchaudio")

    else:
        print("⚠️ Unknown system type; installing CPU-only version of PyTorch.")
        install("torch", "torchvision", "torchaudio")

    # ------------------------------------------------------------
    # 📚 Supporting packages
    # ------------------------------------------------------------
    print("\n📦 Installing supporting libraries...")
    install(
        "numpy<2.0,>=1.24",
        "Pillow<12.0",
        "wrapt<2.0",
        "numba>=0.57,<0.62",
        "dynamax==0.1.8",
        "streamlit>=1.30,<1.42",
    )

    # ------------------------------------------------------------
    # ✅ Verify Installation
    # ------------------------------------------------------------
    print("\n🔍 Verifying environment setup...\n")
    import torch, torchvision, numpy, PIL

    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ TorchVision: {torchvision.__version__}")
    print(f"✅ NumPy: {numpy.__version__}")
    print(f"✅ Pillow: {PIL.__version__}")

    # Hardware acceleration check
    if system == "mac":
        print(f"✅ MPS available: {torch.backends.mps.is_available()}")
    elif system == "linux":
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    else:
        print("✅ CPU-only build verified.")

    # ------------------------------------------------------------
    # 📝 Export installed packages
    # ------------------------------------------------------------
    print("\n📝 Exporting exact versions to requirements.txt...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True
    )

    req_path = os.path.join(os.getcwd(), "requirements.txt")
    with open(req_path, "w") as f:
        f.write(result.stdout)

    print(f"✅ requirements.txt saved at: {req_path}")
    print("\n🎉 Setup complete! Your environment is ready.\n")

if __name__ == "__main__":
    main()
