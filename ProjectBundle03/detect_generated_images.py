"""
detect_generated_images.py

Purpose:
  Tools to compare gradients between pairs of images (real vs generated)
  - compute edge / gradient maps (Sobel, Scharr, Laplacian, Prewitt)
  - compute gradient magnitude and orientation
  - visualize side-by-side and produce difference maps
  - compute quantitative metrics between gradients (L1/L2, cosine, histogram KL, SSIM)
  - batch mode: process a folder of pairs (or pairs listed in CSV)
  - export CSV of metrics and sample visualizations

Usage (examples):
  # single pair -> show plots and save outputs
  python detect_generated_images.py --real real.jpg --fake fake.jpg --outdir results/

  # batch mode: CSV with columns: real,fake
  python detect_generated_images.py --pairs pairs.csv --outdir results/

  # compute features for all pairs and write features.csv
  python detect_generated_images.py --pairs pairs.csv --outdir results/ --metrics_only

Dependencies:
  pip install -r requirements.txt
  requirements.txt content:
    numpy
    opencv-python
    matplotlib
    scikit-image
    pandas
    tqdm
    scikit-learn

Design notes:
  - Gradients are computed per-channel and combined. We compute both Sobel (x,y), gradient magnitude, laplacian.
  - Visual outputs: normalized gradient maps, difference of magnitudes, heatmaps.
  - Metrics: mean absolute diff, L2 norm, SSIM on gradient maps, cosine similarity of flattened maps, histogram KL.
  - Optional: extract simple handcrafted features (moments, kurtosis, sparsity) for training a classifier.

Caveat: this is an analysis and tooling script. It does not attempt to train a SOTA detector; it's a reproducible starting point.

Author: Generated for your project by assistant
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.preprocessing import normalize


# ----------------------- Core gradient utilities -----------------------

def read_image_rgb(path, target_size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def to_gray(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def sobel_gradients(img_gray):
    # returns gx, gy, grad_mag
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, mag


def scharr_gradients(img_gray):
    gx = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, mag


def laplacian(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    return lap


def normalize_for_vis(x):
    # scale to 0..1
    x = np.array(x, dtype=np.float32)
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)
    if hi - lo < 1e-6:
        return np.clip((x - lo), 0, 1)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)


# ----------------------- Metrics -----------------------

def l1_l2_metrics(a, b):
    diff = a - b
    return np.mean(np.abs(diff)), np.linalg.norm(diff.ravel()), np.mean(diff)


def cosine_similarity(a, b):
    fa = a.ravel().astype(np.float32)
    fb = b.ravel().astype(np.float32)
    # small eps
    eps = 1e-12
    denom = (np.linalg.norm(fa) * np.linalg.norm(fb)) + eps
    return float(np.dot(fa, fb) / denom)


def ssim_metric(a, b):
    # expects single-channel float images normalized roughly 0..1 or larger
    try:
        a_n = normalize_for_vis(a)
        b_n = normalize_for_vis(b)
        s, _ = ssim(a_n, b_n, full=True, data_range=1.0)
        return float(s)
    except Exception:
        return np.nan


def hist_kl(a, b, bins=256):
    # compute histograms and KL divergence (small smoothing)
    a_flat = np.clip(a.ravel(), a.min(), a.max()).astype(np.float32)
    b_flat = np.clip(b.ravel(), b.min(), b.max()).astype(np.float32)
    # compute hist edges based on combined
    try:
        hist_a, edges = np.histogram(a_flat, bins=bins, density=True)
        hist_b, _ = np.histogram(b_flat, bins=edges, density=True)
        # smoothing
        hist_a += 1e-8
        hist_b += 1e-8
        return float(entropy(hist_a, hist_b))
    except Exception:
        return np.nan


# ----------------------- Feature extraction -----------------------

def extract_gradient_features(img_rgb, method='sobel'):
    """Return a dict of features derived from gradients.
    Features are computed per grayscale gradient magnitude and per-channel mean magnitudes.
    """
    gray = to_gray(img_rgb)
    if method == 'sobel':
        gx, gy, mag = sobel_gradients(gray)
    elif method == 'scharr':
        gx, gy, mag = scharr_gradients(gray)
    elif method == 'laplacian':
        mag = laplacian(gray)
    else:
        raise ValueError('Unknown method')

    feat = {}
    mag_flat = mag.ravel()
    feat['mag_mean'] = float(np.mean(mag_flat))
    feat['mag_std'] = float(np.std(mag_flat))
    feat['mag_skew'] = float(((mag_flat - mag_flat.mean())**3).mean())
    feat['mag_kurtosis'] = float(((mag_flat - mag_flat.mean())**4).mean())
    feat['mag_sparsity'] = float(np.sum(mag_flat < (np.percentile(mag_flat, 25))) / mag_flat.size)

    # per-color-channel edges using sobel on each channel
    for i, ch in enumerate(['R', 'G', 'B']):
        c = img_rgb[..., i].astype(np.float32)
        gx_c = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        mag_c = np.sqrt(gx_c**2 + gy_c**2).ravel()
        feat[f'{ch}_mag_mean'] = float(np.mean(mag_c))
        feat[f'{ch}_mag_std'] = float(np.std(mag_c))
    return feat


# ----------------------- Visualization -----------------------

def visualize_pair(real, fake, outpath=None, method='sobel', show=False):
    # real and fake are RGB uint8 arrays
    gray_r = to_gray(real)
    gray_f = to_gray(fake)

    if method == 'sobel':
        gx_r, gy_r, mag_r = sobel_gradients(gray_r)
        gx_f, gy_f, mag_f = sobel_gradients(gray_f)
    elif method == 'scharr':
        gx_r, gy_r, mag_r = scharr_gradients(gray_r)
        gx_f, gy_f, mag_f = scharr_gradients(gray_f)
    elif method == 'laplacian':
        mag_r = laplacian(gray_r)
        mag_f = laplacian(gray_f)
        gx_r = gy_r = gx_f = gy_f = np.zeros_like(mag_r)
    else:
        raise ValueError('Unknown method')

    mag_diff = mag_r - mag_f

    # create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    axes[0].imshow(real.astype(np.uint8))
    axes[0].set_title('Real (RGB)')
    axes[0].axis('off')

    axes[1].imshow(fake.astype(np.uint8))
    axes[1].set_title('Fake (RGB)')
    axes[1].axis('off')

    axes[2].imshow(normalize_for_vis(np.abs(mag_r)), cmap='gray')
    axes[2].set_title('Real Grad Magnitude')
    axes[2].axis('off')

    axes[3].imshow(normalize_for_vis(np.abs(mag_f)), cmap='gray')
    axes[3].set_title('Fake Grad Magnitude')
    axes[3].axis('off')

    axes[4].imshow(normalize_for_vis(np.abs(mag_diff)), cmap='hot')
    axes[4].set_title('Abs Diff (mag)')
    axes[4].axis('off')

    # show gx, gy difference heatmaps
    axes[5].imshow(normalize_for_vis(np.abs(gx_r - gx_f)), cmap='viridis')
    axes[5].set_title('gx abs diff')
    axes[5].axis('off')

    axes[6].imshow(normalize_for_vis(np.abs(gy_r - gy_f)), cmap='viridis')
    axes[6].set_title('gy abs diff')
    axes[6].axis('off')

    # overlay difference as color
    overlay = np.stack([normalize_for_vis(mag_diff), normalize_for_vis(gx_r - gx_f), normalize_for_vis(gy_r - gy_f)], axis=-1)
    axes[7].imshow(np.clip(overlay, 0, 1))
    axes[7].set_title('Overlay diff (mag,gx,gy)')
    axes[7].axis('off')

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------- Pair processing -----------------------

def process_pair(real_path, fake_path, outdir, methods=('sobel', 'scharr', 'laplacian')):
    real = read_image_rgb(real_path)
    fake = read_image_rgb(fake_path, target_size=(real.shape[1], real.shape[0]))

    pair_name = f"{Path(real_path).stem}__{Path(fake_path).stem}"
    os.makedirs(outdir, exist_ok=True)

    metrics = {
        'pair': pair_name,
        'real': str(real_path),
        'fake': str(fake_path),
    }

    for method in methods:
        # compute gradients
        gray_r = to_gray(real)
        gray_f = to_gray(fake)
        if method == 'sobel':
            gx_r, gy_r, mag_r = sobel_gradients(gray_r)
            gx_f, gy_f, mag_f = sobel_gradients(gray_f)
        elif method == 'scharr':
            gx_r, gy_r, mag_r = scharr_gradients(gray_r)
            gx_f, gy_f, mag_f = scharr_gradients(gray_f)
        elif method == 'laplacian':
            mag_r = laplacian(gray_r)
            mag_f = laplacian(gray_f)
            gx_r = gy_r = gx_f = gy_f = np.zeros_like(mag_r)
        else:
            continue

        # compute metrics on magnitudes
        m_l1, m_l2, m_mean = l1_l2_metrics(mag_r, mag_f)
        metrics[f'{method}_mag_l1'] = m_l1
        metrics[f'{method}_mag_l2'] = m_l2
        metrics[f'{method}_mag_mean_diff'] = m_mean
        metrics[f'{method}_mag_ssim'] = ssim_metric(mag_r, mag_f)
        metrics[f'{method}_mag_cosine'] = cosine_similarity(mag_r, mag_f)
        metrics[f'{method}_mag_histkl'] = hist_kl(mag_r, mag_f)

        # also metrics on gx and gy
        metrics[f'{method}_gx_cosine'] = cosine_similarity(gx_r, gx_f)
        metrics[f'{method}_gy_cosine'] = cosine_similarity(gy_r, gy_f)

        # save a visualization
        vis_path = os.path.join(outdir, f'{pair_name}__{method}.png')
        visualize_pair(real, fake, outpath=vis_path, method=method, show=False)

    # feature vectors for real/fake
    feat_r = extract_gradient_features(real, method='sobel')
    feat_f = extract_gradient_features(fake, method='sobel')
    # add differences of selected features
    for k in feat_r.keys():
        metrics[f'real_{k}'] = feat_r[k]
        metrics[f'fake_{k}'] = feat_f[k]
        metrics[f'diff_{k}'] = feat_r[k] - feat_f[k]

    return metrics


# ----------------------- Batch and CLI -----------------------

def load_pairs_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if not {'real', 'fake'}.issubset(df.columns):
        raise ValueError('CSV must contain columns: real,fake')
    return list(df[['real', 'fake']].itertuples(index=False, name=None))


def write_metrics(metrics_list, out_csv):
    df = pd.DataFrame(metrics_list)
    df.to_csv(out_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pairs', type=str, help='CSV file with columns real,fake (paths)')
    g.add_argument('--real', type=str, help='real image path (single pair mode)')
    p.add_argument('--fake', type=str, help='fake image path (single pair mode)')
    p.add_argument('--outdir', type=str, default='detect_results', help='output directory')
    p.add_argument('--metrics_only', action='store_true', help='do not save visualizations')
    p.add_argument('--method', type=str, choices=['sobel', 'scharr', 'laplacian', 'all'], default='all')
    p.add_argument('--resize', type=int, nargs=2, metavar=('W','H'), help='resize images to W H for processing')
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    methods = ('sobel','scharr','laplacian') if args.method == 'all' else (args.method,)

    pairs = []
    if args.pairs:
        pairs = load_pairs_from_csv(args.pairs)
    else:
        if not args.fake:
            raise ValueError('If using --real, you must also provide --fake')
        pairs = [(args.real, args.fake)]

    results = []
    for real_path, fake_path in tqdm(pairs, desc='pairs'):
        try:
            metrics = process_pair(real_path, fake_path, outdir, methods=methods)
            results.append(metrics)
        except Exception as e:
            print(f'Error processing pair {real_path},{fake_path}: {e}', file=sys.stderr)

    out_csv = os.path.join(outdir, 'metrics.csv')
    write_metrics(results, out_csv)
    print('Wrote metrics to', out_csv)


if __name__ == '__main__':
    main()


# ----------------------- README and next steps -----------------------
"""
README / Next steps (copy into your project README.md):

1) Install:
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy opencv-python matplotlib scikit-image pandas tqdm scikit-learn

2) Run single pair (preview):
   python detect_generated_images.py --real ./data/real1.jpg --fake ./data/fake1.jpg --outdir out/

3) Batch mode: create pairs.csv with header: real,fake
   python detect_generated_images.py --pairs pairs.csv --outdir out/

4) Experiments to run:
  - Use generated "look-alike" images: keep same camera framing, pose and prompt the generator (or use image2image) to produce a close recreation. Compare gradient metrics distribution across many pairs.
  - Collect negative control: random unrelated real images vs generated images.
  - Compare across generation engines (Stable Diffusion, Midjourney, DALL-E) and multiple seeds.
  - Train a simple ML classifier (logistic regression / random forest) on the handcrafted gradient features in metrics.csv to see separability.

5) Evaluation tips:
  - Use ROC AUC to measure classifier ability to separate real vs fake from gradient features.
  - Use statistical tests (Mann-Whitney U) to test whether distributions of a metric (e.g. sobel_mag_l1) differ significantly.
  - Visualize distributions with violin/box plots.

6) Possible improvements:
  - Compute gradients on high-frequency enhanced images (apply highpass filter first).
  - Use patch-level features (compute gradients per 64x64 patch and pool stats).
  - Train a small CNN on gradient images (two-channel: sobel_x & sobel_y) for detection.

7) Reproducibility:
  - Record image resizing and preprocessing consistently.
  - Save random seeds used in generation.

"""
