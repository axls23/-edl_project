import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot to avoid tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import base64
from io import BytesIO
from utils import format_percentage
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from modules.feature_extractors import DeepLearningExtractor
from sklearn.manifold import TSNE

class VisualizationGenerator:
    def __init__(self):
        """Initialize visualization generator"""
        self.setup_matplotlib()
        # UI-inspired palette (matches dark theme in static/css/style.css)
        self.palette = {
            'bg': '#0b0b12',
            'surface': '#0f1117',
            'surface2': '#11141c',
            'text': '#e5e7eb',
            'muted': '#9aa4b2',
            'border': '#1f2937',
            'ring': '#334155',
            'accent': '#8b5cf6',   # purple
            'accent2': '#06b6d4',  # cyan
            'accent_soft': '#a78bfa',
            'accent2_soft': '#22d3ee',
            'success': '#22c55e',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }
    
    def setup_matplotlib(self):
        """Configure matplotlib for web display"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.5
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.autolayout'] = True
        # Dark theme defaults
        plt.rcParams['figure.facecolor'] = '#0b0b12'
        plt.rcParams['axes.facecolor'] = '#0f1117'
        plt.rcParams['axes.edgecolor'] = '#334155'
        plt.rcParams['axes.labelcolor'] = '#e5e7eb'
        plt.rcParams['xtick.color'] = '#9aa4b2'
        plt.rcParams['ytick.color'] = '#9aa4b2'
        plt.rcParams['text.color'] = '#e5e7eb'
        plt.rcParams['grid.color'] = '#1f2937'
        plt.rcParams['savefig.facecolor'] = '#0b0b12'
        plt.rcParams['savefig.edgecolor'] = '#0b0b12'
        plt.rcParams['legend.facecolor'] = '#0f1117'
        plt.rcParams['legend.edgecolor'] = '#334155'

    def _style_axes(self, axes):
        """Apply consistent dark styling to axes or list of axes."""
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax in axes:
            if ax is None:
                continue
            ax.set_facecolor(self.palette['surface'])
            for spine in ax.spines.values():
                spine.set_color(self.palette['ring'])
            ax.tick_params(colors=self.palette['muted'])
            ax.yaxis.label.set_color(self.palette['text'])
            ax.xaxis.label.set_color(self.palette['text'])
            ax.title.set_color(self.palette['text'])
    
    def generate_visualizations(self, image1, image2, results, image1_path, image2_path):
        """Generate all visualizations for the similarity results"""
        visualizations = {}
        
        try:
            # 1. Basic similarity overview
            visualizations['overview'] = self._create_overview_visualization(results)
            
            # 2. Method comparison bar chart
            visualizations['method_comparison'] = self._create_method_comparison(results)
            
            # 3. Side-by-side comparison
            visualizations['side_by_side'] = self._create_side_by_side_comparison(
                image1, image2, results
            )
            
            # 4. Detailed breakdown
            visualizations['detailed_breakdown'] = self._create_detailed_breakdown(results)
            
            # 5. Confidence visualization
            visualizations['confidence'] = self._create_confidence_visualization(results)

            # 6. SSIM heatmap overlay
            visualizations['ssim_heatmap'] = self._create_ssim_heatmap(image1, image2)

            # 7. Edge and LBP difference maps
            visualizations['edge_lbp_diffs'] = self._create_edge_lbp_diffs(image1, image2)

            # 8. Method attribution (score x weight)
            visualizations['method_attribution'] = self._create_method_attribution(results)

            # 9. RGB histogram overlay with correlations
            visualizations['hist_overlay'] = self._create_hist_overlay(image1, image2)

            # 10. Feature-space t-SNE (lightweight using color+LBP features)
            visualizations['tsne_features'] = self._create_tsne_features(image1, image2)

            # 11. Grad-CAM overlays (best-effort; may return None if TF ops unavailable)
            gradcam_img = self._create_gradcam_overlays(image1, image2)
            if gradcam_img is not None:
                visualizations['gradcam_overlays'] = gradcam_img
            
            return visualizations
            
        except Exception as e:
            print(f"Visualization generation error: {e}")
            return {'error': str(e)}
        finally:
            # Ensure all matplotlib resources are cleaned up
            plt.close('all')
    
    def cleanup(self):
        """Cleanup matplotlib resources"""
        plt.close('all')
    
    def _create_overview_visualization(self, results):
        """Create overview visualization with main similarity score"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor(self.palette['bg'])
            self._style_axes([ax2])
            
            # Main similarity score
            ensemble_score = results.get('ensemble', {}).get('score', 0)
            confidence = results.get('ensemble', {}).get('confidence', 0)
            
            # Circular progress bar
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_facecolor(self.palette['surface'])
            
            # Background circle
            circle_bg = patches.Circle((0, 0), 1, fill=False, color=self.palette['ring'], linewidth=8, alpha=0.7)
            ax1.add_patch(circle_bg)
            
            # Progress circle
            theta = np.linspace(0, 2 * np.pi * ensemble_score, 100)
            x_circle = np.cos(theta)
            y_circle = np.sin(theta)
            ax1.plot(x_circle, y_circle, color=self.palette['accent2'], linewidth=8)
            
            # Score text with better positioning
            ax1.text(0, 0.1, f'{format_percentage(ensemble_score)}', 
                    ha='center', va='center', fontsize=28, fontweight='bold', color=self.palette['accent'])
            ax1.text(0, -0.4, f'Confidence: {confidence:.2f}', 
                    ha='center', va='center', fontsize=14, color=self.palette['muted'], fontweight='bold')
            ax1.text(0, -0.6, 'Overall Similarity', 
                    ha='center', va='center', fontsize=12, color=self.palette['muted'])
            
            # Method scores breakdown
            methods = []
            scores = []
            colors = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']]

            weight_map = {
                'deep_learning': 0.50,
                'cv_methods': 0.25, 
                'probabilistic': 0.10,
                'perceptual_hash': 0.15
            }
            
            for method_key in ['deep_learning', 'perceptual_hash', 'cv_methods', 'probabilistic']:
                if method_key in results:
                    # Convert key to display name
                    display_name = method_key.replace('_', ' ').title()
                    score = results[method_key].get('score', 0)
                    weight = weight_map.get(method_key, 0.25)
                    methods.append(f"{display_name}\n({weight:.0%})")
                    scores.append(score)
            
            bars = ax2.bar(methods, scores, color=colors[:len(methods)], width=0.6, edgecolor=self.palette['border'])
            ax2.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
            ax2.set_title('Method Contributions', fontsize=16, fontweight='bold', pad=20)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.5)
            
            # Add value labels on bars with better positioning
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{score:.3f}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold', color=self.palette['text'],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=self.palette['surface2'], edgecolor=self.palette['ring'], alpha=0.9))
            
            plt.tight_layout(pad=3.0)
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Overview visualization error: {e}")
            return None
    
    def _create_method_comparison(self, results):
        """Create detailed method comparison visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            fig.patch.set_facecolor(self.palette['bg'])
            self._style_axes(list(axes))
            
            # Deep Learning details
            if 'deep_learning' in results:
                ax = axes[0]
                dl_data = results['deep_learning']
                score = dl_data.get('score', 0)
                bars = ax.bar(['ResNet50\nCosine\nSimilarity'], [score], color=self.palette['accent'], width=0.6, edgecolor=self.palette['border'])
                ax.set_title('Deep Learning Features', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                ax.text(0, score + 0.05, f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=self.palette['text'])
                ax.grid(True, alpha=0.5)
            
            # Perceptual Hash details
            if 'perceptual_hash' in results:
                ax = axes[1]
                ph_data = results['perceptual_hash']['details']
                methods = ['pHash', 'dHash', 'aHash']
                scores = [
                    ph_data.get('phash_similarity', 0),
                    ph_data.get('dhash_similarity', 0),
                    ph_data.get('ahash_similarity', 0)
                ]
                colors = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft']]
                bars = ax.bar(methods, scores, color=colors, width=0.6, edgecolor=self.palette['border'])
                ax.set_title('Perceptual Hashing', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=self.palette['text'])
                ax.grid(True, alpha=0.5)
            
            # CV Methods details
            if 'cv_methods' in results:
                ax = axes[2]
                cv_data = results['cv_methods']['details']
                methods = ['SSIM', 'Histogram', 'Edge', 'LBP']
                scores = [
                    cv_data.get('ssim', 0),
                    cv_data.get('histogram', 0),
                    cv_data.get('edge_similarity', 0),
                    cv_data.get('lbp_similarity', 0)
                ]
                colors = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']]
                bars = ax.bar(methods, scores, color=colors, width=0.6, edgecolor=self.palette['border'])
                ax.set_title('Computer Vision Methods', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=self.palette['text'])
                ax.grid(True, alpha=0.5)
            
            # Probabilistic details
            if 'probabilistic' in results:
                ax = axes[3]
                prob_data = results['probabilistic']['details']
                methods = ['KL Div', 'JS Div', 'Wasserstein']
                scores = [
                    prob_data.get('kl_similarity', 0),
                    prob_data.get('js_similarity', 0),
                    prob_data.get('wasserstein_similarity', 0)
                ]
                colors = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft']]
                bars = ax.bar(methods, scores, color=colors, width=0.6, edgecolor=self.palette['border'])
                ax.set_title('Probabilistic Analysis', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=self.palette['text'])
                ax.grid(True, alpha=0.5)
            
            plt.tight_layout(pad=3.0)
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Method comparison visualization error: {e}")
            return None
    
    def _create_side_by_side_comparison(self, image1, image2, results):
        """Create side-by-side image comparison"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor(self.palette['bg'])
            for ax in axes:
                ax.set_facecolor(self.palette['surface'])
            
            # Original images
            axes[0].imshow(image1)
            axes[0].set_title('Image 1', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(image2)
            axes[1].set_title('Image 2', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Difference image - resize to same dimensions if needed
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]

            if h1 != h2 or w1 != w2:
                # Use the larger dimensions
                target_h = max(h1, h2)
                target_w = max(w1, w2)
                img1_resized = cv2.resize(image1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img2_resized = cv2.resize(image2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                diff = cv2.absdiff(img1_resized, img2_resized)
            else:
                diff = cv2.absdiff(image1, image2)

            axes[2].imshow(diff)
            axes[2].set_title('Difference Map', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Add similarity score overlay
            ensemble_score = results.get('ensemble', {}).get('score', 0)
            fig.suptitle(f'Overall Similarity: {format_percentage(ensemble_score)}', 
                        fontsize=16, fontweight='bold', y=0.95, color=self.palette['text'])
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Side-by-side comparison error: {e}")
            return None
    
    def _create_detailed_breakdown(self, results):
        """Create detailed statistical breakdown"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.patch.set_facecolor(self.palette['bg'])
            for row in axes:
                for ax in row:
                    ax.set_facecolor(self.palette['surface'])
                    self._style_axes(ax)
            
            # Score distribution
            ax = axes[0, 0]
            methods = []
            scores = []
            weights = []
            
            weight_map = {
                'deep_learning': 0.50,
                'cv_methods': 0.25,
                'probabilistic': 0.10,
                'perceptual_hash': 0.15
            }
            
            for method_key in ['deep_learning', 'perceptual_hash', 'cv_methods', 'probabilistic']:
                if method_key in results:
                    display_name = method_key.replace('_', ' ').title()
                    methods.append(display_name)
                    scores.append(results[method_key].get('score', 0))
                    weights.append(weight_map.get(method_key, 0.25))
            
            colors = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']]
            bars = ax.bar(methods, scores, color=colors, width=0.6, edgecolor=self.palette['border'])
            ax.set_title('Individual Method Scores', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Similarity Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.5)
            
            # Add weight annotations with better positioning
            for bar, weight, score in zip(bars, weights, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'Weight: {weight:.0%}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', color=self.palette['text'],
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=self.palette['surface2'], edgecolor=self.palette['ring'], alpha=0.9))
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{score:.3f}', ha='center', va='center', 
                       fontsize=11, fontweight='bold', color=self.palette['text'])
            
            # Confidence analysis
            ax = axes[0, 1]
            confidence = results.get('ensemble', {}).get('confidence', 0)
            bars = ax.bar(['Ensemble\nConfidence'], [confidence], color=self.palette['accent2'], width=0.6, edgecolor=self.palette['border'])
            ax.set_title('Ensemble Confidence', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Confidence Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.text(0, confidence + 0.05, f'{confidence:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color=self.palette['text'])
            ax.grid(True, alpha=0.5)
            
            # Score distribution histogram
            ax = axes[1, 0]
            all_scores = [score for score in scores if score > 0]
            if all_scores:
                ax.hist(all_scores, bins=8, alpha=0.8, color=self.palette['accent'], edgecolor=self.palette['border'], width=0.1)
                ax.set_title('Score Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Similarity Score', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                mean_score = np.mean(all_scores)
                ax.axvline(mean_score, color=self.palette['accent2'], linestyle='--', linewidth=2,
                          label=f'Mean: {mean_score:.3f}')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.5)
            
            # Method weights pie chart
            ax = axes[1, 1]
            if weights and scores:
                colors_pie = [self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']]
                wedges, texts, autotexts = ax.pie(weights, labels=methods, autopct='%1.1f%%', 
                                                 startangle=90, colors=colors_pie)
                ax.set_title('Method Weights', fontsize=14, fontweight='bold', pad=20)
                ax.set_facecolor(self.palette['surface'])
                
                # Improve text readability
                for text in texts:
                    text.set_fontsize(10)
                    text.set_fontweight('bold')
                for autotext in autotexts:
                    autotext.set_color(self.palette['text'])
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
            
            plt.tight_layout(pad=3.0)
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Detailed breakdown error: {e}")
            return None

    def _create_method_attribution(self, results):
        """Bar chart of score x weight contributions and simple agreement index."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            fig.patch.set_facecolor(self.palette['bg'])
            self._style_axes(ax)

            weight_map = {
                'deep_learning': 0.50,
                'cv_methods': 0.25,
                'probabilistic': 0.10,
                'perceptual_hash': 0.15
            }
            methods = []
            contributions = []
            for m in ['deep_learning', 'perceptual_hash', 'cv_methods', 'probabilistic']:
                if m in results:
                    s = results[m].get('score', 0)
                    w = weight_map.get(m, 0)
                    methods.append(m.replace('_',' ').title())
                    contributions.append(s * w)

            bars = ax.bar(methods, contributions, color=[self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']][:len(methods)], edgecolor=self.palette['border'])
            ax.set_title('Method Attribution (Score × Weight)', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Contribution')
            ax.set_ylim(0, 1.0)
            for b, c in zip(bars, contributions):
                ax.text(b.get_x()+b.get_width()/2., c + 0.02, f"{c:.3f}", ha='center', va='bottom', fontsize=10, color=self.palette['text'])

            # Agreement index
            scores = [results[m]['score'] for m in ['deep_learning','perceptual_hash','cv_methods','probabilistic'] if m in results]
            if len(scores) >= 2:
                agreement = max(0.0, min(1.0, 1 - float(np.std(scores))))
                ax.text(0.95, 0.95, f"Agreement: {agreement:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=11, color=self.palette['text'], bbox=dict(boxstyle="round,pad=0.3", facecolor=self.palette['surface2'], edgecolor=self.palette['ring']))

            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Attribution visualization error: {e}")
            return None

    def _create_ssim_heatmap(self, image1, image2):
        """Generate SSIM heatmap (hot=diff, cool=same)."""
        try:
            # Ensure same size
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            if (h1, w1) != (h2, w2):
                H, W = max(h1, h2), max(w1, w2)
                img1 = cv2.resize(image1, (W, H), interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(image2, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                img1, img2 = image1, image2

            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            gray1 = img_as_float(gray1)
            gray2 = img_as_float(gray2)
            # Explicit args for newer scikit-image: set channel_axis=None and data_range
            score, diff = ssim(gray1, gray2, full=True, data_range=1.0, channel_axis=None)
            diff = (1 - diff)  # now high = more different

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            fig.patch.set_facecolor(self.palette['bg'])
            ax.imshow(diff, cmap='inferno')
            ax.axis('off')
            ax.set_title(f'SSIM Difference Heatmap (score={score:.3f})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"SSIM heatmap error: {e}")
            return None

    def _create_edge_lbp_diffs(self, image1, image2):
        """Edge and LBP difference maps side-by-side."""
        try:
            # Resize to same size
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            H, W = max(h1, h2), max(w1, w2)
            img1 = cv2.resize(image1, (W, H), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(image2, (W, H), interpolation=cv2.INTER_LINEAR)

            g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            ediff = cv2.absdiff(e1, e2)

            from skimage.feature import local_binary_pattern
            lbp1 = local_binary_pattern(g1, 8, 1, method='uniform')
            lbp2 = local_binary_pattern(g2, 8, 1, method='uniform')
            ldiff = np.abs(lbp1 - lbp2)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor(self.palette['bg'])
            for ax in axes:
                ax.set_facecolor(self.palette['surface'])
                ax.axis('off')
            axes[0].imshow(ediff, cmap='magma')
            axes[0].set_title('Edge Difference', fontsize=14, fontweight='bold')
            im = axes[1].imshow(ldiff, cmap='viridis')
            axes[1].set_title('LBP Difference', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Edge/LBP diff error: {e}")
            return None

    def _create_hist_overlay(self, image1, image2):
        """Overlay RGB histograms and show per-channel correlation."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            fig.patch.set_facecolor(self.palette['bg'])
            self._style_axes(ax)

            colors = ['r','g','b']
            corr_texts = []
            for i, c in enumerate(colors):
                h1 = cv2.calcHist([image1],[i],None,[256],[0,256]).flatten()
                h2 = cv2.calcHist([image2],[i],None,[256],[0,256]).flatten()
                h1 /= (np.sum(h1) + 1e-10)
                h2 /= (np.sum(h2) + 1e-10)
                ax.plot(h1, color=c, alpha=0.7, label=f'Img1-{c.upper()}')
                ax.plot(h2, color=c, alpha=0.7, linestyle='--', label=f'Img2-{c.upper()}')
                # correlation
                num = np.sum(h1*h2)
                den = np.sqrt(np.sum(h1*h1) * np.sum(h2*h2)) + 1e-10
                corr = float(num/den)
                corr_texts.append(f'{c.upper()}={corr:.2f}')

            ax.set_title('RGB Histogram Overlay  (corr: ' + ', '.join(corr_texts) + ')', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Histogram overlay error: {e}")
            return None

    def _create_tsne_features(self, image1, image2):
        """t-SNE of lightweight color+LBP features for both images."""
        try:
            # Build features similar to ProbabilisticExtractor
            def build_features(img):
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(g, 8, 1, method='uniform').reshape(-1,1)
                cols = img.reshape(-1,3)
                feats = np.column_stack([cols, lbp])
                if len(feats) > 4000:
                    idx = np.random.choice(len(feats), 4000, replace=False)
                    feats = feats[idx]
                return feats

            f1 = build_features(image1)
            f2 = build_features(image2)
            X = np.vstack([f1, f2]).astype(np.float32)
            labels = np.array([0]*len(f1) + [1]*len(f2))

            tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', n_iter=500, random_state=42)
            Y = tsne.fit_transform(X)

            fig, ax = plt.subplots(1,1, figsize=(7,6))
            fig.patch.set_facecolor(self.palette['bg'])
            ax.scatter(Y[labels==0,0], Y[labels==0,1], s=5, c=self.palette['accent'], alpha=0.6, label='Image 1')
            ax.scatter(Y[labels==1,0], Y[labels==1,1], s=5, c=self.palette['accent2'], alpha=0.6, label='Image 2')
            ax.set_title('Feature-space (t-SNE)')
            ax.legend()
            ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"t-SNE error: {e}")
            return None

    def _create_gradcam_overlays(self, image1, image2):
        """Generate Grad-CAM overlays for both images. Best-effort; may return None."""
        try:
            extractor = DeepLearningExtractor()
            heat1 = extractor.gradcam_heatmap(image1)
            heat2 = extractor.gradcam_heatmap(image2)
            if heat1 is None or heat2 is None:
                return None

            def overlay(img, heat):
                heatmap = (heat * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlayed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
                return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

            o1 = overlay(image1, heat1)
            o2 = overlay(image2, heat2)

            fig, axes = plt.subplots(1,2, figsize=(12,5))
            fig.patch.set_facecolor(self.palette['bg'])
            for ax in axes:
                ax.axis('off')
            axes[0].imshow(o1); axes[0].set_title('Grad-CAM Image 1', fontsize=12, fontweight='bold')
            axes[1].imshow(o2); axes[1].set_title('Grad-CAM Image 2', fontsize=12, fontweight='bold')
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None
    
    def _create_confidence_visualization(self, results):
        """Create confidence and uncertainty visualization"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor(self.palette['bg'])
            self._style_axes(axes)
            
            # Confidence intervals
            ax = axes[0]
            methods = []
            scores = []
            uncertainties = []
            
            for method in ['deep_learning', 'perceptual_hash', 'cv_methods', 'probabilistic']:
                if method in results:
                    methods.append(method.replace('_', ' ').title())
                    score = results[method].get('score', 0)
                    scores.append(score)
                    # Calculate uncertainty based on confidence (inverse relationship)
                    confidence = results[method].get('confidence', 0.5)
                    uncertainty = (1 - confidence) * 0.2  # Scale to reasonable range
                    uncertainties.append(uncertainty)
            
            y_pos = np.arange(len(methods))
            bars = ax.barh(y_pos, scores, xerr=uncertainties, 
                          color=[self.palette['accent'], self.palette['accent2'], self.palette['accent_soft'], self.palette['accent2_soft']][:len(methods)], edgecolor=self.palette['border'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(methods)
            ax.set_xlabel('Similarity Score')
            ax.set_title('Method Scores with Uncertainty')
            ax.set_xlim(0, 1)
            
            # Ensemble confidence
            ax = axes[1]
            ensemble_score = results.get('ensemble', {}).get('score', 0)
            confidence = results.get('ensemble', {}).get('confidence', 0)
            
            # Create confidence gauge
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Background
            ax.plot(theta, r, color=self.palette['ring'], linewidth=10)
            
            # Confidence arc
            conf_theta = np.linspace(0, np.pi * confidence, 50)
            conf_r = np.ones_like(conf_theta)
            ax.plot(conf_theta, conf_r, self.palette['accent2'], linewidth=10)
            
            # Score point
            score_theta = np.pi * ensemble_score
            ax.plot(score_theta, 1, marker='o', color=self.palette['accent'], markersize=15)
            
            ax.set_xlim(0, np.pi)
            ax.set_ylim(0, 1.2)
            ax.set_title(f'Ensemble Score: {format_percentage(ensemble_score)}\nConfidence: {confidence:.2f}', color=self.palette['text'])
            ax.axis('off')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Confidence visualization error: {e}")
            return None
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close(fig)
            # Clear the current figure to release all resources
            plt.clf()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Figure to base64 conversion error: {e}")
            plt.close('all')  # Close all figures on error
            return None
