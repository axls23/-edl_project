"""
Simplified and modular similarity engine using existing libraries
"""
import numpy as np
from datetime import datetime
from modules.feature_extractors import (
    DeepLearningExtractor, 
    PerceptualHashExtractor, 
    ComputerVisionExtractor, 
    ProbabilisticExtractor
)
from modules.report_generator import ReportGenerator

class SimilarityEngineV2:
    """Simplified similarity engine using modular components"""
    
    def __init__(self):
        """Initialize with modular extractors"""
        self.deep_learning = DeepLearningExtractor()
        self.perceptual_hash = PerceptualHashExtractor()
        self.cv_methods = ComputerVisionExtractor()
        self.probabilistic = ProbabilisticExtractor()
        self.report_generator = ReportGenerator()
        
        # Ensemble weights - prioritizing semantic similarity
        # Deep learning gets highest weight as it understands content semantically
        self.weights = {
            'deep_learning': 0.50,      # Increased: Best for semantic understanding
            'cv_methods': 0.25,         # Maintained: Good for structural similarity
            'probabilistic': 0.10,      # Decreased: Supporting role
            'perceptual_hash': 0.15     # Decreased: Best for near-duplicates only
        }
    
    def compute_similarity(self, image1, image2):
        """Compute similarity using modular approach"""
        results = {}
        timestamp = datetime.now().isoformat()
        
        try:
            # 1. Deep Learning Analysis
            dl_score, dl_details = self.deep_learning.compute_similarity(image1, image2)
            
            # Build detailed proof string with category information
            category_info = ""
            if 'categories1' in dl_details and 'categories2' in dl_details:
                cat1_top = dl_details['categories1'][0][0] if dl_details['categories1'] else 'unknown'
                cat2_top = dl_details['categories2'][0][0] if dl_details['categories2'] else 'unknown'
                category_info = f" | Categories: {cat1_top} vs {cat2_top} (overlap: {dl_details.get('category_overlap', 0)})"
            
            results['deep_learning'] = {
                'score': dl_score,
                'details': dl_details,
                'method': 'ResNet50 Feature Extraction + Category Matching',
                'proof': f"Feature similarity: {dl_details.get('feature_similarity', dl_score):.4f} + Category bonus: {dl_details.get('category_bonus', 0):.4f}{category_info}",
                'confidence': self._calculate_confidence(dl_score)
            }
            
            # 2. Perceptual Hashing Analysis
            ph_score, ph_details = self.perceptual_hash.compute_similarity(image1, image2)
            results['perceptual_hash'] = {
                'score': ph_score,
                'details': ph_details,
                'method': 'Multi-hash Comparison (pHash, dHash, aHash)',
                'proof': f"Hamming distances - pHash: {ph_details.get('phash_distance', 0)}, dHash: {ph_details.get('dhash_distance', 0)}, aHash: {ph_details.get('ahash_distance', 0)}",
                'confidence': self._calculate_confidence(ph_score)
            }
            
            # 3. Computer Vision Analysis
            cv_score, cv_details = self.cv_methods.compute_similarity(image1, image2)
            results['cv_methods'] = {
                'score': cv_score,
                'details': cv_details,
                'method': 'SSIM + Histogram + Edge + LBP Analysis',
                'proof': f"SSIM: {cv_details.get('ssim', 0):.4f}, Histogram: {cv_details.get('histogram', 0):.4f}, Edge: {cv_details.get('edge_similarity', 0):.4f}, LBP: {cv_details.get('lbp_similarity', 0):.4f}",
                'confidence': self._calculate_confidence(cv_score)
            }
            
            # 4. Probabilistic Analysis
            prob_score, prob_details = self.probabilistic.compute_similarity(image1, image2)
            results['probabilistic'] = {
                'score': prob_score,
                'details': prob_details,
                'method': 'GMM + KL/JS Divergence + Wasserstein Distance',
                'proof': f"KL Div: {prob_details.get('kl_divergence', 0):.4f}, JS Div: {prob_details.get('js_divergence', 0):.4f}, Wasserstein: {prob_details.get('wasserstein_distance', 0):.4f}",
                'confidence': self._calculate_confidence(prob_score)
            }
            
            # 5. Ensemble Score
            ensemble_score = self._compute_ensemble_score(results)
            ensemble_confidence = self._calculate_ensemble_confidence(results)
            results['ensemble'] = {
                'score': ensemble_score,
                'confidence': ensemble_confidence,
                'method': 'Semantic-Weighted Ensemble (50% DL + 25% CV + 10% Prob + 15% PH)',
                'proof': f"Final score: {ensemble_score:.4f} with confidence: {ensemble_confidence:.4f}",
                'statistical_significance': self._calculate_statistical_significance(results)
            }
            
            
            # 7. Generate Analysis Report
            results['analysis_report'] = self.report_generator.generate_analysis_report(results, timestamp)
            results['timestamp'] = timestamp
            
            return results
            
        except Exception as e:
            raise Exception(f"Error computing similarity: {str(e)}")
    
    def _compute_ensemble_score(self, results):
        """Compute weighted ensemble score"""
        scores = []
        weights = []
        
        for method, weight in self.weights.items():
            if method in results and 'score' in results[method]:
                scores.append(results[method]['score'])
                weights.append(weight)
        
        if not scores:
            return 0.0
        
        return float(np.average(scores, weights=weights))
    
    def _calculate_confidence(self, score):
        """Calculate confidence for individual scores based on score magnitude and stability"""
        # Confidence based on how close the score is to 0.5 (uncertainty)
        # Scores closer to 0.5 are less confident, scores at extremes are more confident
        distance_from_uncertainty = abs(score - 0.5)
        confidence = 2 * distance_from_uncertainty  # Scale to 0-1 range
        return max(0.1, min(1.0, confidence))  # Ensure minimum confidence
    
    def _calculate_ensemble_confidence(self, results):
        """Calculate ensemble confidence"""
        scores = [results[m]['score'] for m in self.weights.keys() if m in results]
        if len(scores) < 2:
            return 0.0
        
        std_dev = np.std(scores)
        confidence = 1 - std_dev
        return max(0, min(1, confidence))
    
    def _calculate_statistical_significance(self, results):
        """Calculate statistical significance"""
        scores = [results[m]['score'] for m in self.weights.keys() if m in results]
        if len(scores) < 2:
            return 0.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / (mean_score + 1e-10)
        significance = 1 - cv
        return max(0, min(1, significance))
    
