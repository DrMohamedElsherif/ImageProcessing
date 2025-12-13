"""
GRADIENT-BASED REAL/FAKE IMAGE COMPARISON
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import normalize
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE GRADIENT UTILITIES 
# ============================================================================

def read_image_rgb(path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Robust image loading with resizing support"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def convert_color_space(img: np.ndarray, color_space: str = 'RGB') -> np.ndarray:
    """Convert between color spaces (from my code)"""
    if color_space == 'RGB':
        return img
    elif color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif color_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def compute_gradients(img: np.ndarray, method: str = 'sobel', 
                      return_components: bool = False) -> Union[np.ndarray, Tuple]:
    """
    Compute gradients using specified method (enhanced version)
    
    Args:
        img: Input image (single channel or multi-channel)
        method: 'sobel', 'scharr', 'laplacian', 'prewitt', or 'roberts'
        return_components: If True, return (gx, gy, magnitude)
    
    Returns:
        Gradient magnitude or tuple of components
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    if method == 'sobel':
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    elif method == 'scharr':
        gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    elif method == 'laplacian':
        lap = cv2.Laplacian(img, cv2.CV_32F)
        if return_components:
            return lap, lap, lap  # Return same for all components
        return lap
    elif method == 'prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gx = cv2.filter2D(img, -1, kernel_x)
        gy = cv2.filter2D(img, -1, kernel_y)
    elif method == 'roberts':
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(img, -1, kernel_x)
        gy = cv2.filter2D(img, -1, kernel_y)
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    
    magnitude = np.sqrt(gx**2 + gy**2)
    
    if return_components:
        return gx, gy, magnitude
    return magnitude


def normalize_for_vis(x: np.ndarray, percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """Normalize array for visualization"""
    x = np.array(x, dtype=np.float32)
    lo = np.nanpercentile(x, percentile_range[0])
    hi = np.nanpercentile(x, percentile_range[1])
    if hi - lo < 1e-6:
        return np.clip((x - lo), 0, 1)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)


# ============================================================================
# COMPARISON METRICS 
# ============================================================================

class ComparisonMetrics:
    """Comprehensive comparison metrics between two images"""
    
    @staticmethod
    def l1_l2_metrics(a: np.ndarray, b: np.ndarray) -> Dict:
        """Compute L1, L2, and mean difference metrics"""
        diff = a - b
        metrics = {
            'l1': np.mean(np.abs(diff)),
            'l2': np.linalg.norm(diff.ravel()),
            'mean_diff': np.mean(diff),
            'abs_mean_diff': np.mean(np.abs(diff)),
            'max_diff': np.max(np.abs(diff))
        }
        return metrics
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        """Compute cosine similarity between flattened arrays"""
        fa = a.ravel().astype(np.float32)
        fb = b.ravel().astype(np.float32)
        denom = (np.linalg.norm(fa) * np.linalg.norm(fb)) + eps
        return float(np.dot(fa, fb) / denom)
    
    @staticmethod
    def ssim_metric(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
        """Compute Structural Similarity Index"""
        try:
            a_n = normalize_for_vis(a)
            b_n = normalize_for_vis(b)
            s, _ = ssim(a_n, b_n, full=True, data_range=data_range)
            return float(s)
        except Exception:
            return np.nan
    
    @staticmethod
    def histogram_kl_divergence(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
        """Compute KL divergence between histograms"""
        a_flat = np.clip(a.ravel(), a.min(), a.max()).astype(np.float32)
        b_flat = np.clip(b.ravel(), b.min(), b.max()).astype(np.float32)
        
        try:
            # Use combined range for consistent binning
            min_val = min(a_flat.min(), b_flat.min())
            max_val = max(a_flat.max(), b_flat.max())
            
            hist_a, _ = np.histogram(a_flat, bins=bins, range=(min_val, max_val), density=True)
            hist_b, _ = np.histogram(b_flat, bins=bins, range=(min_val, max_val), density=True)
            
            # Add small smoothing factor
            hist_a += 1e-8
            hist_b += 1e-8
            
            return float(entropy(hist_a, hist_b))
        except Exception:
            return np.nan
    
    @staticmethod
    def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> Dict:
        """Compute all comparison metrics"""
        metrics = {}
        
        # Basic difference metrics
        diff_metrics = ComparisonMetrics.l1_l2_metrics(a, b)
        metrics.update(diff_metrics)
        
        # Similarity metrics
        metrics['cosine_similarity'] = ComparisonMetrics.cosine_similarity(a, b)
        metrics['ssim'] = ComparisonMetrics.ssim_metric(a, b)
        metrics['histogram_kl'] = ComparisonMetrics.histogram_kl_divergence(a, b)
        
        # Correlation
        a_flat = a.ravel()
        b_flat = b.ravel()
        if len(a_flat) > 1:
            metrics['pearson_corr'] = float(np.corrcoef(a_flat, b_flat)[0, 1])
        
        return metrics


# ============================================================================
# STATISTICAL FEATURE EXTRACTION 
# ============================================================================

class StatisticalFeatureExtractor:
    """Extract comprehensive statistical features from images/gradients"""
    
    def __init__(self, color_spaces: List[str] = None, 
                 gradient_methods: List[str] = None):
        self.color_spaces = color_spaces or ['RGB', 'HSV', 'LAB']
        self.gradient_methods = gradient_methods or ['sobel', 'scharr', 'laplacian']
    
    def extract_image_features(self, img: np.ndarray) -> Dict:
        """Extract comprehensive features from image"""
        features = {}
        
        # Process each color space
        for color_space in self.color_spaces:
            img_cs = convert_color_space(img, color_space)
            prefix = f"{color_space}_"
            
            # Extract features from each channel
            for ch in range(img_cs.shape[2]):
                channel = img_cs[:, :, ch].astype(np.float32)
                channel_features = self._extract_channel_features(channel)
                
                # Add to features dict
                for feat_name, feat_value in channel_features.items():
                    features[f"{prefix}ch{ch}_{feat_name}"] = feat_value
            
            # Also extract gradient-based features
            gray = to_gray(img_cs) if img_cs.shape[2] > 1 else img_cs
            gradient_features = self.extract_gradient_features(gray, prefix=prefix)
            features.update(gradient_features)
        
        return features
    
    def _extract_channel_features(self, channel: np.ndarray) -> Dict:
        """Extract statistical features from single channel"""
        channel_flat = channel.flatten()
        
        features = {
            'mean': float(np.mean(channel_flat)),
            'std': float(np.std(channel_flat)),
            'min': float(np.min(channel_flat)),
            'max': float(np.max(channel_flat)),
            'median': float(np.median(channel_flat)),
            'range': float(np.ptp(channel_flat)),
            'variance': float(np.var(channel_flat)),
            'skewness': float(skew(channel_flat)),
            'kurtosis': float(kurtosis(channel_flat, fisher=True)),
            'entropy': self._compute_entropy(channel),
            'energy': float(np.sum(channel_flat**2)),
            'rms': float(np.sqrt(np.mean(channel_flat**2))),
            'mean_abs_dev': float(np.mean(np.abs(channel_flat - np.mean(channel_flat))))
        }
        
        # Histogram-based features
        hist, _ = np.histogram(channel_flat, bins=256, density=True)
        hist += 1e-10  # Avoid log(0)
        features['hist_entropy'] = float(-np.sum(hist * np.log2(hist)))
        features['hist_energy'] = float(np.sum(hist**2))
        
        return features
    
    def _compute_entropy(self, image: np.ndarray) -> float:
        """Compute entropy of image"""
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist + 1e-10  # Avoid log(0)
        entropy_val = -np.sum(hist * np.log2(hist))
        return float(entropy_val)
    
    def extract_gradient_features(self, img_gray: np.ndarray, prefix: str = '') -> Dict:
        """Extract gradient-based features"""
        features = {}
        
        for method in self.gradient_methods:
            # Compute gradients
            grad_mag = compute_gradients(img_gray, method=method)
            
            # Extract statistical features from gradient magnitude
            grad_features = self._extract_channel_features(grad_mag)
            
            # Add method prefix
            for feat_name, feat_value in grad_features.items():
                features[f"{prefix}{method}_{feat_name}"] = feat_value
        
        return features


# ============================================================================
# DETECTION 
# ============================================================================

class DetectionScorer:
    """Score likelihood of image being AI-generated"""
    
    def __init__(self, weights: Optional[Dict] = None):
        # Default weights for different feature categories
        self.weights = weights or {
            'gradient_stats': 0.35,
            'color_stats': 0.25,
            'comparison_metrics': 0.25,
            'texture_features': 0.15
        }
    
    def compute_detection_score(self, real_features: Dict, fake_features: Dict, 
                                comparison_metrics: Dict) -> Dict:
        """
        Compute comprehensive detection score
        
        Returns:
            Dictionary with overall score, component scores, and decision
        """
        scores = {}
        
        # 1. Gradient statistics differences
        grad_score = self._compute_gradient_score(real_features, fake_features)
        scores['gradient_score'] = grad_score
        
        # 2. Color statistics differences
        color_score = self._compute_color_score(real_features, fake_features)
        scores['color_score'] = color_score
        
        # 3. Comparison metrics
        comp_score = self._compute_comparison_score(comparison_metrics)
        scores['comparison_score'] = comp_score
        
        # 4. Texture/pattern differences
        texture_score = self._compute_texture_score(real_features, fake_features)
        scores['texture_score'] = texture_score
        
        # 5. Overall weighted score (0-100)
        overall_score = (
            grad_score * self.weights['gradient_stats'] +
            color_score * self.weights['color_stats'] +
            comp_score * self.weights['comparison_metrics'] +
            texture_score * self.weights['texture_features']
        ) * 100
        
        scores['overall_score'] = overall_score
        
        # 6. Decision with confidence levels
        decision, confidence = self._make_decision(overall_score)
        scores['decision'] = decision
        scores['confidence'] = confidence
        
        return scores
    
    def _compute_gradient_score(self, real_feats: Dict, fake_feats: Dict) -> float:
        """Score based on gradient feature differences"""
        grad_keys = [k for k in real_feats.keys() if any(m in k for m in ['sobel', 'scharr', 'laplacian'])]
        
        if not grad_keys:
            return 0.5  # Neutral score if no gradient features
        
        differences = []
        for key in grad_keys:
            if key in fake_feats:
                diff = abs(real_feats[key] - fake_feats[key])
                # Normalize by real value (avoid division by zero)
                norm_diff = diff / (abs(real_feats[key]) + 1e-10)
                differences.append(min(norm_diff, 1.0))  # Cap at 1.0
        
        if not differences:
            return 0.5
        
        avg_diff = np.mean(differences)
        # Convert to 0-1 score (higher = more likely fake)
        return min(avg_diff * 2, 1.0)
    
    def _compute_color_score(self, real_feats: Dict, fake_feats: Dict) -> float:
        """Score based on color statistics differences"""
        color_keys = [k for k in real_feats.keys() if any(cs in k for cs in ['RGB', 'HSV', 'LAB'])]
        
        if not color_keys:
            return 0.5
        
        differences = []
        for key in color_keys:
            if 'hist_' not in key and key in fake_feats:  # Exclude histogram features
                real_val = real_feats[key]
                fake_val = fake_feats[key]
                
                # Handle different scales for different features
                if 'std' in key or 'variance' in key:
                    diff = abs(real_val - fake_val) / (abs(real_val) + 1e-10)
                else:
                    diff = abs(real_val - fake_val) / max(abs(real_val), 1e-10)
                
                differences.append(min(diff, 1.0))
        
        if not differences:
            return 0.5
        
        avg_diff = np.mean(differences)
        return min(avg_diff * 1.5, 1.0)
    
    def _compute_comparison_score(self, metrics: Dict) -> float:
        """Score based on direct comparison metrics"""
        score_components = []
        
        # SSIM: lower = more different
        if 'ssim' in metrics and not np.isnan(metrics['ssim']):
            ssim_score = 1.0 - metrics['ssim']  # Convert to difference score
            score_components.append(ssim_score)
        
        # Cosine similarity: lower = more different
        if 'cosine_similarity' in metrics:
            cos_score = 1.0 - ((metrics['cosine_similarity'] + 1) / 2)  # Convert -1..1 to 0..1
            score_components.append(cos_score)
        
        # KL divergence: higher = more different
        if 'histogram_kl' in metrics and not np.isnan(metrics['histogram_kl']):
            kl_score = min(metrics['histogram_kl'] / 10.0, 1.0)  # Normalize
            score_components.append(kl_score)
        
        # L1 difference: higher = more different
        if 'l1' in metrics:
            l1_score = min(metrics['l1'] / 50.0, 1.0)  # Normalize (adjust based on data range)
            score_components.append(l1_score)
        
        if not score_components:
            return 0.5
        
        return np.mean(score_components)
    
    def _compute_texture_score(self, real_feats: Dict, fake_feats: Dict) -> float:
        """Score based on texture/entropy differences"""
        texture_keys = [k for k in real_feats.keys() if 'entropy' in k or 'energy' in k]
        
        if not texture_keys:
            return 0.5
        
        differences = []
        for key in texture_keys:
            if key in fake_feats:
                real_val = real_feats[key]
                fake_val = fake_feats[key]
                
                # Relative difference for entropy/energy
                diff = abs(real_val - fake_val) / (abs(real_val) + 1e-10)
                differences.append(min(diff, 1.0))
        
        if not differences:
            return 0.5
        
        return np.mean(differences)
    
    def _make_decision(self, score: float) -> Tuple[str, str]:
        """Convert score to decision with confidence"""
        if score >= 70:
            return "FAKE", "HIGH"
        elif score >= 55:
            return "FAKE", "MEDIUM"
        elif score >= 45:
            return "INCONCLUSIVE", "LOW"
        elif score >= 30:
            return "REAL", "MEDIUM"
        else:
            return "REAL", "HIGH"


# ============================================================================
# VISUALIZATION 
# ============================================================================

class ComprehensiveVisualizer:
    """Create comprehensive visualizations combining both approaches"""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 12)):
        self.figsize = figsize

    def create_comprehensive_comparison(self, real_img: np.ndarray, fake_img: np.ndarray,
                                       real_gradients: Dict, fake_gradients: Dict,
                                       comparison_results: Dict, detection_scores: Dict,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive visualization with multiple panels"""
        
        # Create figure with subplots - Reduced from 4 rows to 2 rows
        fig = plt.figure(figsize=(20, 6))  # Reduced height from 12 to 6
        
        # Define grid layout - Changed from 4,6 to 2,6
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
        
        # 1. Original Images
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(real_img.astype(np.uint8))
        ax1.set_title('Real Image', fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(fake_img.astype(np.uint8))
        ax2.set_title('AI-Generated Image', fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        # 2. Gradient Magnitude Comparison
        real_grad = np.mean(list(real_gradients.values()), axis=0) \
                    if len(real_gradients) > 1 else list(real_gradients.values())[0]
        fake_grad = np.mean(list(fake_gradients.values()), axis=0) \
                    if len(fake_gradients) > 1 else list(fake_gradients.values())[0]
        
        ax3 = fig.add_subplot(gs[0, 4])
        ax3.imshow(normalize_for_vis(real_grad), cmap='hot')
        ax3.set_title('Real Gradients', fontsize=9)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 5])
        ax4.imshow(normalize_for_vis(fake_grad), cmap='hot')
        ax4.set_title('Fake Gradients', fontsize=9)
        ax4.axis('off')
        
        # 3. Gradient Difference Maps (from provided code)
        ax5 = fig.add_subplot(gs[1, 0])
        grad_diff = np.abs(real_grad - fake_grad)
        ax5.imshow(normalize_for_vis(grad_diff), cmap='coolwarm')
        ax5.set_title('Gradient Magnitude Diff', fontsize=9)
        ax5.axis('off')
        
        # 4. Component differences (if available)
        if 'gx_real' in real_gradients and 'gx_fake' in fake_gradients:
            ax6 = fig.add_subplot(gs[1, 1])
            gx_diff = np.abs(real_gradients['gx_real'] - fake_gradients['gx_fake'])
            ax6.imshow(normalize_for_vis(gx_diff), cmap='viridis')
            ax6.set_title('Gx Difference', fontsize=9)
            ax6.axis('off')
            
            ax7 = fig.add_subplot(gs[1, 2])
            gy_diff = np.abs(real_gradients['gy_real'] - fake_gradients['gy_fake'])
            ax7.imshow(normalize_for_vis(gy_diff), cmap='viridis')
            ax7.set_title('Gy Difference', fontsize=9)
            ax7.axis('off')
        
        # 5. Histogram Comparison
        ax8 = fig.add_subplot(gs[1, 3:5])
        real_hist, bins = np.histogram(real_grad.flatten(), bins=50, density=True)
        fake_hist, _ = np.histogram(fake_grad.flatten(), bins=bins, density=True)
        ax8.plot(bins[:-1], real_hist, alpha=0.7, label='Real', linewidth=2)
        ax8.plot(bins[:-1], fake_hist, alpha=0.7, label='Fake', linewidth=2)
        ax8.set_title('Gradient Histograms', fontsize=10)
        ax8.set_xlabel('Gradient Magnitude')
        ax8.set_ylabel('Density')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 6. Statistical Comparison Bar Chart
        ax9 = fig.add_subplot(gs[1, 5])
        stats_to_plot = ['mean', 'std', 'entropy', 'skewness']
        real_stats = []
        fake_stats = []
        
        print("\nStatistical Values Used for Comparison Bar Chart (Averaged RGB):")

        for stat in stats_to_plot:
            real_vals = [
                comparison_results.get(f'channel_{ch}', {}).get(stat, {}).get('real', 0)
                for ch in range(3)
            ]
            fake_vals = [
                comparison_results.get(f'channel_{ch}', {}).get(stat, {}).get('fake', 0)
                for ch in range(3)
            ]

            real_val = np.mean(real_vals)
            fake_val = np.mean(fake_vals)

            print(f"  {stat.upper()}:")
            print(f"    REAL: {real_val:.4f} (channels: {[round(v,4) for v in real_vals]})")
            print(f"    FAKE: {fake_val:.4f} (channels: {[round(v,4) for v in fake_vals]})")

            real_stats.append(real_val)
            fake_stats.append(fake_val)

        
        x = np.arange(len(stats_to_plot))
        width = 0.35
        
        ax9.bar(x - width/2, real_stats, width, label='Real', alpha=0.8)
        ax9.bar(x + width/2, fake_stats, width, label='Fake', alpha=0.8)
        ax9.set_title('Statistical Comparison', fontsize=10)
        ax9.set_xticks(x)
        ax9.set_xticklabels(stats_to_plot, rotation=45)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle('Real Vs Fake Image Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig


# ============================================================================
# PROCESSING PIPELINE 
# ============================================================================

class ComprehensiveDetectionPipeline:
    """Main pipeline combining all components"""
    
    def __init__(self, 
                 color_spaces: List[str] = None,
                 gradient_methods: List[str] = None,
                 output_dir: str = 'detection_results'):
        
        self.color_spaces = color_spaces or ['RGB', 'HSV', 'LAB']
        self.gradient_methods = gradient_methods or ['sobel', 'scharr', 'laplacian']
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.feature_extractor = StatisticalFeatureExtractor(
            color_spaces=self.color_spaces,
            gradient_methods=self.gradient_methods
        )
        self.metrics_calculator = ComparisonMetrics()
        self.detection_scorer = DetectionScorer()
        self.visualizer = ComprehensiveVisualizer()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def process_single_pair(self, real_path: Union[str, Path], 
                           fake_path: Union[str, Path],
                           pair_name: Optional[str] = None) -> Dict:
        """Process a single pair of real/fake images"""
        
        print(f"Processing: {real_path} vs {fake_path}")
        
        # Load images
        real_img = read_image_rgb(real_path)
        fake_img = read_image_rgb(fake_path, target_size=(real_img.shape[1], real_img.shape[0]))
        
        # Extract features
        print("  Extracting features...")
        real_features = self.feature_extractor.extract_image_features(real_img)
        fake_features = self.feature_extractor.extract_image_features(fake_img)
        
        # Compute gradients for visualization
        real_gradients = {}
        fake_gradients = {}
        gray_real = to_gray(real_img)
        gray_fake = to_gray(fake_img)
        
        for method in ['sobel']:  # Use sobel for main visualization
            gx_real, gy_real, mag_real = compute_gradients(gray_real, method, return_components=True)
            gx_fake, gy_fake, mag_fake = compute_gradients(gray_fake, method, return_components=True)
            
            real_gradients.update({
                'mag_real': mag_real,
                'gx_real': gx_real,
                'gy_real': gy_real
            })
            fake_gradients.update({
                'mag_fake': mag_fake,
                'gx_fake': gx_fake,
                'gy_fake': gy_fake
            })

            # --------------------------------------------------
        # PRINT GRADIENT VALUES USED FOR VISUALIZATION
        # --------------------------------------------------
        print("\nGradient Values Used for Plotting (Sobel Magnitude):")

        real_mag = real_gradients['mag_real']
        fake_mag = fake_gradients['mag_fake']

        print("  REAL IMAGE:")
        print(f"    Mean: {np.mean(real_mag):.4f}")
        print(f"    Std:  {np.std(real_mag):.4f}")
        print(f"    Min:  {np.min(real_mag):.4f}")
        print(f"    Max:  {np.max(real_mag):.4f}")

        print("  FAKE IMAGE:")
        print(f"    Mean: {np.mean(fake_mag):.4f}")
        print(f"    Std:  {np.std(fake_mag):.4f}")
        print(f"    Min:  {np.min(fake_mag):.4f}")
        print(f"    Max:  {np.max(fake_mag):.4f}")

        
        # Compute comparison metrics
        print("  Computing comparison metrics...")
        comparison_metrics = self.metrics_calculator.compute_all_metrics(gray_real, gray_fake)
        
        # Create comparison results structure
        comparison_results = {}
        for i in range(3):  # RGB channels
            real_channel = real_img[:, :, i].astype(np.float32)
            fake_channel = fake_img[:, :, i].astype(np.float32)
            
            real_stats = self.feature_extractor._extract_channel_features(real_channel)
            fake_stats = self.feature_extractor._extract_channel_features(fake_channel)
            
            channel_results = {}
            for stat_name in real_stats.keys():
                real_val = real_stats[stat_name]
                fake_val = fake_stats[stat_name]
                
                channel_results[stat_name] = {
                    'real': real_val,
                    'fake': fake_val,
                    'absolute_diff': fake_val - real_val,
                    'percent_diff': ((fake_val - real_val) / (abs(real_val) + 1e-10)) * 100
                }
            
            comparison_results[f'channel_{i}'] = channel_results
        
        # Compute detection score
        print("  Computing detection score...")
        detection_scores = self.detection_scorer.compute_detection_score(
            real_features, fake_features, comparison_metrics
        )
        
        # Generate pair name if not provided
        if pair_name is None:
            pair_name = f"{Path(real_path).stem}_{Path(fake_path).stem}"
        
        # Create visualization
        print("  Creating visualization...")
        vis_path = self.output_dir / f"{pair_name}_analysis.png"
        self.visualizer.create_comprehensive_comparison(
            real_img, fake_img,
            real_gradients, fake_gradients,
            comparison_results, detection_scores,
            save_path=str(vis_path)
        )
        
        # Prepare results
        results = {
            'pair_name': pair_name,
            'real_path': str(real_path),
            'fake_path': str(fake_path),
            'real_features': real_features,
            'fake_features': fake_features,
            'comparison_metrics': comparison_metrics,
            'comparison_results': comparison_results,
            'detection_scores': detection_scores,
            'visualization_path': str(vis_path),
            'decision': detection_scores['decision'],
            'confidence': detection_scores['confidence'],
            'overall_score': detection_scores['overall_score']
        }
        
        # Print summary
        print(f"\n  Results for {pair_name}:")
        print(f"    Decision: {detection_scores['decision']}")
        print(f"    Confidence: {detection_scores['confidence']}")
        print(f"    Overall Score: {detection_scores['overall_score']:.1f}/100")
        print(f"    Visualization saved to: {vis_path}")
        
        return results
    
    def process_batch(self, pairs: List[Tuple[str, str]], 
                     csv_path: Optional[str] = None) -> pd.DataFrame:
        """Process multiple pairs and save results to CSV"""
        
        print(f"Processing batch of {len(pairs)} pairs...")
        
        all_results = []
        
        for idx, (real_path, fake_path) in enumerate(tqdm(pairs, desc="Processing pairs")):
            try:
                pair_name = f"pair_{idx:03d}_{Path(real_path).stem}"
                results = self.process_single_pair(real_path, fake_path, pair_name)
                
                # Extract key metrics for CSV
                csv_row = {
                    'pair_name': results['pair_name'],
                    'real_path': results['real_path'],
                    'fake_path': results['fake_path'],
                    'decision': results['decision'],
                    'confidence': results['confidence'],
                    'overall_score': results['overall_score'],
                    'gradient_score': results['detection_scores'].get('gradient_score', 0) * 100,
                    'color_score': results['detection_scores'].get('color_score', 0) * 100,
                    'comparison_score': results['detection_scores'].get('comparison_score', 0) * 100,
                    'texture_score': results['detection_scores'].get('texture_score', 0) * 100,
                    'ssim': results['comparison_metrics'].get('ssim', np.nan),
                    'cosine_similarity': results['comparison_metrics'].get('cosine_similarity', np.nan),
                    'histogram_kl': results['comparison_metrics'].get('histogram_kl', np.nan),
                    'l1_diff': results['comparison_metrics'].get('l1', np.nan)
                }
                
                all_results.append(csv_row)
                
            except Exception as e:
                print(f"Error processing pair {real_path} vs {fake_path}: {e}")
                continue
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_results)
        
        if csv_path is None:
            csv_path = self.output_dir / "batch_results.csv"
        
        df.to_csv(csv_path, index=False)
        print(f"\nBatch results saved to: {csv_path}")
        
        # Print summary statistics
        print("\nBatch Summary Statistics:")
        print(f"Total pairs processed: {len(all_results)}")
        print(f"Fake detections: {len(df[df['decision'] == 'FAKE'])}")
        print(f"Real detections: {len(df[df['decision'] == 'REAL'])}")
        print(f"Inconclusive: {len(df[df['decision'] == 'INCONCLUSIVE'])}")
        print(f"Average score: {df['overall_score'].mean():.1f} ± {df['overall_score'].std():.1f}")
        
        return df


# ============================================================================
# CLI INTERFACE 
# ============================================================================

def load_pairs_from_csv(csv_path: str) -> List[Tuple[str, str]]:
    """Load image pairs from CSV file"""
    df = pd.read_csv(csv_path)
    if not {'real', 'fake'}.issubset(df.columns):
        raise ValueError('CSV must contain columns: real, fake')
    return list(df[['real', 'fake']].itertuples(index=False, name=None))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Gradient-Based Fake Image Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usages:
  # Single pair analysis
  python detect_fake.py --real real.jpg --fake fake.jpg
  
  # Batch processing from CSV
  python detect_fake.py --csv pairs.csv
  
  # Custom output directory
  python detect_fake.py --real real.jpg --fake fake.jpg --outdir ./my_results
  
  # Specific color spaces
  python detect_fake.py --real real.jpg --fake fake.jpg --colorspaces RGB HSV
  
  # Debug mode
  python detect_fake.py --real real.jpg --fake fake.jpg --debug
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, 
                           help='CSV file with columns: real, fake (paths)')
    input_group.add_argument('--real', type=str, 
                           help='Real image path (for single pair mode)')
    
    parser.add_argument('--fake', type=str,
                       help='Fake image path (required for single pair mode)')
    
    # Processing options
    parser.add_argument('--outdir', type=str, default='detection_results',
                       help='Output directory for results (default: detection_results)')
    parser.add_argument('--colorspaces', type=str, nargs='+',
                       default=['RGB', 'HSV', 'LAB'],
                       help='Color spaces to analyze (default: RGB HSV LAB)')
    parser.add_argument('--gradient-methods', type=str, nargs='+',
                       default=['sobel', 'scharr', 'laplacian'],
                       help='Gradient methods to use (default: sobel scharr laplacian)')
    
    # Output options
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate summary, no detailed analysis')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser.parse_args()


def main():
    """Main CLI entry point"""
    args = parse_args()
    
    # Validate arguments
    if args.real and not args.fake:
        #parser.error("--real requires --fake")
        print("Error: --real requires --fake")
    
    # Load pairs
    if args.csv:
        pairs = load_pairs_from_csv(args.csv)
        print(f"Loaded {len(pairs)} pairs from {args.csv}")
    else:
        pairs = [(args.real, args.fake)]
    
    # Initialize pipeline
    pipeline = ComprehensiveDetectionPipeline(
        color_spaces=args.colorspaces,
        gradient_methods=args.gradient_methods,
        output_dir=args.outdir
    )
    
    # Process pairs
    if len(pairs) == 1:
        # Single pair - detailed analysis
        real_path, fake_path = pairs[0]
        results = pipeline.process_single_pair(real_path, fake_path)
        
        # Print detailed results
        if args.debug:
            print("\nDetailed Results:")
            print(f"Decision: {results['decision']} ({results['confidence']} confidence)")
            print(f"Overall Score: {results['overall_score']:.1f}/100")
            print(f"\nKey Metrics:")
            for metric_name, metric_value in results['comparison_metrics'].items():
                if isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.4f}")
            
    else:
        # Batch processing
        csv_path = Path(args.outdir) / "batch_results.csv"
        df = pipeline.process_batch(pairs, csv_path=str(csv_path))
        
        # Generate summary report
        summary_path = Path(args.outdir) / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("FAKE IMAGE DETECTION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total pairs analyzed: {len(df)}\n")
            f.write(f"Fake detections: {len(df[df['decision'] == 'FAKE'])}\n")
            f.write(f"Real detections: {len(df[df['decision'] == 'REAL'])}\n")
            f.write(f"Inconclusive: {len(df[df['decision'] == 'INCONCLUSIVE'])}\n")
            f.write(f"\nAverage detection score: {df['overall_score'].mean():.1f}\n")
            f.write(f"Score standard deviation: {df['overall_score'].std():.1f}\n")
            f.write(f"\nBest performing pair: {df.loc[df['overall_score'].idxmax(), 'pair_name']} "
                   f"(score: {df['overall_score'].max():.1f})\n")
            f.write(f"Worst performing pair: {df.loc[df['overall_score'].idxmin(), 'pair_name']} "
                   f"(score: {df['overall_score'].min():.1f})\n")
        
        print(f"\nSummary report saved to: {summary_path}")


# ============================================================================
# QUICK START
# ============================================================================

def quick_start_example():
    """Quick start example for users"""
    print("""
QUICK START GUIDE:
------------------
1. For single pair analysis:
   python detect_fake.py --real /path/to/real.jpg --fake /path/to/fake.jpg

2. For batch processing:
   Create a CSV file (pairs.csv) with columns: real,fake
   Then run:
   python detect_fake.py --csv pairs.csv

3. Other options:
   python detect_fake.py --real real.jpg --fake fake.jpg \\
        --colorspaces RGB LAB \\
        --gradient-methods sobel laplacian \\
        --outdir ./my_results

OUTPUT FILES:
-------------
- detection_results/              # Output directory
  ├── pair_analysis.png          # Visualization
  ├── batch_results.csv          # Batch results
  └── summary_report.txt         # Summary statistics
    """)


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if help is requested
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        quick_start_example()
        print("\n" + "="*60 + "\n")
    
    # Run main program
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)