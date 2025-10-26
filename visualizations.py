import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
import base64
from io import BytesIO
import json
from utils import format_percentage

class VisualizationGenerator:
    def __init__(self):
        """Initialize visualization generator"""
        self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Configure matplotlib for web display"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.autolayout'] = True
    
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
            
            return visualizations
            
        except Exception as e:
            print(f"Visualization generation error: {e}")
            return {'error': str(e)}
    
    def _create_overview_visualization(self, results):
        """Create overview visualization with main similarity score"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Main similarity score
            ensemble_score = results.get('ensemble', {}).get('score', 0)
            confidence = results.get('ensemble', {}).get('confidence', 0)
            
            # Circular progress bar
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_aspect('equal')
            ax1.axis('off')
            
            # Background circle
            circle_bg = patches.Circle((0, 0), 1, fill=False, color='lightgray', linewidth=8)
            ax1.add_patch(circle_bg)
            
            # Progress circle
            theta = np.linspace(0, 2 * np.pi * ensemble_score, 100)
            x_circle = np.cos(theta)
            y_circle = np.sin(theta)
            ax1.plot(x_circle, y_circle, color='#dc2626', linewidth=8)
            
            # Score text with better positioning
            ax1.text(0, 0.1, f'{format_percentage(ensemble_score)}', 
                    ha='center', va='center', fontsize=28, fontweight='bold', color='#dc2626')
            ax1.text(0, -0.4, f'Confidence: {confidence:.2f}', 
                    ha='center', va='center', fontsize=14, color='gray', fontweight='bold')
            ax1.text(0, -0.6, 'Overall Similarity', 
                    ha='center', va='center', fontsize=12, color='#666')
            
            # Method scores breakdown
            methods = []
            scores = []
            colors = ['#dc2626', '#f59e0b', '#ef4444', '#f97316']
            
            for i, (method, weight) in enumerate([('Deep Learning', 0.35), 
                                                ('Perceptual Hash', 0.20), 
                                                ('CV Methods', 0.25), 
                                                ('Probabilistic', 0.20)]):
                if method.lower().replace(' ', '_') in results:
                    method_key = method.lower().replace(' ', '_')
                    score = results[method_key].get('score', 0)
                    methods.append(f"{method}\n({weight:.0%})")
                    scores.append(score)
            
            bars = ax2.bar(methods, scores, color=colors[:len(methods)], width=0.6)
            ax2.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
            ax2.set_title('Method Contributions', fontsize=16, fontweight='bold', pad=20)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars with better positioning
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{score:.3f}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
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
            
            # Deep Learning details
            if 'deep_learning' in results:
                ax = axes[0]
                dl_data = results['deep_learning']
                score = dl_data.get('score', 0)
                bars = ax.bar(['ResNet50\nCosine\nSimilarity'], [score], color='#dc2626', width=0.6)
                ax.set_title('Deep Learning Features', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                ax.text(0, score + 0.05, f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
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
                colors = ['#dc2626', '#f59e0b', '#ef4444']
                bars = ax.bar(methods, scores, color=colors, width=0.6)
                ax.set_title('Perceptual Hashing', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
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
                colors = ['#dc2626', '#f59e0b', '#ef4444', '#f97316']
                bars = ax.bar(methods, scores, color=colors, width=0.6)
                ax.set_title('Computer Vision Methods', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
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
                colors = ['#dc2626', '#f59e0b', '#ef4444']
                bars = ax.bar(methods, scores, color=colors, width=0.6)
                ax.set_title('Probabilistic Analysis', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Similarity Score', fontsize=12)
                ax.set_ylim(0, 1.1)
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Method comparison visualization error: {e}")
            return None
    
    def _create_side_by_side_comparison(self, image1, image2, results):
        """Create side-by-side image comparison"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original images
            axes[0].imshow(image1)
            axes[0].set_title('Image 1', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(image2)
            axes[1].set_title('Image 2', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Difference image
            diff = cv2.absdiff(image1, image2)
            axes[2].imshow(diff)
            axes[2].set_title('Difference Map', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Add similarity score overlay
            ensemble_score = results.get('ensemble', {}).get('score', 0)
            fig.suptitle(f'Overall Similarity: {format_percentage(ensemble_score)}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Side-by-side comparison error: {e}")
            return None
    
    def _create_detailed_breakdown(self, results):
        """Create detailed statistical breakdown"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Score distribution
            ax = axes[0, 0]
            methods = []
            scores = []
            weights = []
            
            for method, weight in [('Deep Learning', 0.35), ('Perceptual Hash', 0.20), 
                                 ('CV Methods', 0.25), ('Probabilistic', 0.20)]:
                method_key = method.lower().replace(' ', '_')
                if method_key in results:
                    methods.append(method)
                    scores.append(results[method_key].get('score', 0))
                    weights.append(weight)
            
            colors = ['#dc2626', '#f59e0b', '#ef4444', '#f97316']
            bars = ax.bar(methods, scores, color=colors, width=0.6)
            ax.set_title('Individual Method Scores', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Similarity Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Add weight annotations with better positioning
            for bar, weight, score in zip(bars, weights, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'Weight: {weight:.0%}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='white', alpha=0.8))
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{score:.3f}', ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='white')
            
            # Confidence analysis
            ax = axes[0, 1]
            confidence = results.get('ensemble', {}).get('confidence', 0)
            bars = ax.bar(['Ensemble\nConfidence'], [confidence], color='#dc2626', width=0.6)
            ax.set_title('Ensemble Confidence', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Confidence Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.text(0, confidence + 0.05, f'{confidence:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Score distribution histogram
            ax = axes[1, 0]
            all_scores = [score for score in scores if score > 0]
            if all_scores:
                ax.hist(all_scores, bins=8, alpha=0.7, color='#dc2626', edgecolor='black', width=0.1)
                ax.set_title('Score Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Similarity Score', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                mean_score = np.mean(all_scores)
                ax.axvline(mean_score, color='#f59e0b', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_score:.3f}')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Method weights pie chart
            ax = axes[1, 1]
            if weights and scores:
                colors_pie = ['#dc2626', '#f59e0b', '#ef4444', '#f97316']
                wedges, texts, autotexts = ax.pie(weights, labels=methods, autopct='%1.1f%%', 
                                                 startangle=90, colors=colors_pie)
                ax.set_title('Method Weights', fontsize=14, fontweight='bold', pad=20)
                
                # Improve text readability
                for text in texts:
                    text.set_fontsize(10)
                    text.set_fontweight('bold')
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
            
            plt.tight_layout(pad=3.0)
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"Detailed breakdown error: {e}")
            return None
    
    def _create_confidence_visualization(self, results):
        """Create confidence and uncertainty visualization"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
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
                          color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(methods)])
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
            ax.plot(theta, r, 'lightgray', linewidth=10)
            
            # Confidence arc
            conf_theta = np.linspace(0, np.pi * confidence, 50)
            conf_r = np.ones_like(conf_theta)
            ax.plot(conf_theta, conf_r, '#2E86AB', linewidth=10)
            
            # Score point
            score_theta = np.pi * ensemble_score
            ax.plot(score_theta, 1, 'ro', markersize=15)
            
            ax.set_xlim(0, np.pi)
            ax.set_ylim(0, 1.2)
            ax.set_title(f'Ensemble Score: {format_percentage(ensemble_score)}\nConfidence: {confidence:.2f}')
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
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Figure to base64 conversion error: {e}")
            return None

