# 🧩 ProjectBundle 01 — PyTorch Image Processing

This repository contains a set of **PyTorch-based image processing scripts** for:
1. **Grayscale conversion**
2. **Photomosaic creation** using the *Animals-10* dataset
3. **Mosaic recreation** using convolution operations  

The project automatically configures its environment for **CPU, CUDA (GPU), or Apple MPS**.

---

## 📁 Project Structure

```bash
ProjectBundle01/
├── convert.py                 # Grayscale conversion using PyTorch
├── mosaic.py                  # Photomosaic creation with Animals-10 dataset
├── recreate_mosaic.py         # Mosaic recreation using convolution
├── setup.py                   # Environment setup and dependency management
├── requirements.txt           # Generated dependency list
├── sample_images/             # Example input images
│   └── MariaVonLinden.jpg
├── Datasets/                  # Auto-created dataset storage
│   └── Animals10/
│       └── raw/               # Animals-10 images (downloaded automatically)
└── Outputs/                   # All processed images
    └── Output_sample_images/
```
---

## 🚀 Setup
### 1. Navigate to the project directory:

```bash
cd ProjectBundle01
```

### 2. Run the setup script to automatically: 
- 📦 **Install and import** all supporting libraries required by every script in the assignment  
- 🧩 **Define helper functions** that are shared and reused across scripts

```bash
python setup.py
```
---

## 🛠️ Script Usage
### 1. Grayscale Conversion

Converts a color image to **grayscale** using a **1×1 convolution layer** implemented in **PyTorch**.

**Command Syntax:**
```bash
python convert.py <input_image_path> <output_image_filename>
```
**Example:**
```bash
python convert.py sample_images/MariaVonLinden.jpg MariaVonLinden_gray.jpg
```

### 2. Photomosaic Creation (Animals-10 Dataset)

Creates a **photomosaic** version of an image using animal pictures from the **Animals-10** dataset.  
If not already present, the dataset will be **downloaded automatically** using `kagglehub`.

**Command Syntax:**
```bash
python mosaic.py <input_image_path> <output_image_name> <num_cols> <num_rows> [method]
```
**Example:**
```bash
# Color-based mosaic
python mosaic.py sample_images/MariaVonLinden.jpg MariaVonLinden_mosaic.jpg 60 40 avg

# Brightness-based mosaic
python mosaic.py sample_images/MariaVonLinden.jpg mosaic_bright.jpg 25 20 brightness

# Random tile mosaic
python mosaic.py sample_images/MariaVonLinden.jpg mosaic_random.jpg 15 10 random
```

### 3. Mosaic Recreation using Convolution

Recreates the source image as a **block-style mosaic** using `torch.nn.functional.conv2d` to simulate pixel averaging.

**Command Syntax:**
```bash
python recreate_mosaic.py <input_image_path> <output_image_name> <target_tiles>
```
**Example:**
```bash
python recreate_mosaic.py sample_images/MariaVonLinden.jpg MariaVonLinden_mosaic.jpg 400
```
---

## 🧑‍🏫 Example Workflow for Evaluation

To reproduce the results, follow these steps:

```bash
# Set up the dependencies and helper functions (Mandatory Run)
python setup.py

# Convert to grayscale
python convert.py sample_images/MariaVonLinden.jpg grayscale_Maria.jpg

# Create a photomosaic using Animals-10 dataset
python mosaic.py sample_images/MariaVonLinden.jpg mosaic_Maria.jpg 40 30 avg

# Recreate mosaic using convolution
python recreate_mosaic.py sample_images/MariaVonLinden.jpg convMosaic_Maria.jpg 1000
```
All resulting images will appear in: ``` Outputs/Output_sample_images/ ```

---

<div>
    
**Author:** Mohamed Elsherif 

**Course:** Image Processing

**Date:** 11.11.2025 
    
</div>

---




