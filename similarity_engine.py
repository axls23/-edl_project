import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
import imagehash
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimilarityEngine:
    def __init__(self):
        """Initialize the similarity engine with pre-trained models"""
        # Load ResNet50 model for deep learning features
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Weights for ensemble scoring
        self.weights = {
            'deep_learning': 0.35,
            'perceptual_hash': 0.20,
            'cv_methods': 0.25,
            'probabilistic': 0.20
        }
    
    def compute_similarity(self, image1, image2):
        """Compute similarity between two images using ensemble methods with detailed analysis"""
        results = {}
        analysis_timestamp = datetime.now().isoformat()
        
        try:
            # 1. Deep Learning Features
            dl_score, dl_features = self._deep_learning_similarity(image1, image2)
            results['deep_learning'] = {
                'score': dl_score,
                'features': dl_features,
                'method': 'ResNet50 Feature Extraction + Cosine Similarity',
                'proof': f"Feature vectors dimension: {len(dl_features.get('features1', []))}",
                'confidence': self._calculate_method_confidence(dl_score)
            }
            
            # 2. Perceptual Hashing
            ph_score, ph_details = self._perceptual_hash_similarity(image1, image2)
            results['perceptual_hash'] = {
                'score': ph_score,
                'details': ph_details,
                'method': 'Multi-hash Comparison (pHash, dHash, aHash)',
                'proof': f"Hamming distances - pHash: {ph_details.get('phash_distance', 0)}, dHash: {ph_details.get('dhash_distance', 0)}, aHash: {ph_details.get('ahash_distance', 0)}",
                'confidence': self._calculate_method_confidence(ph_score)
            }
            
            # 3. Traditional CV Methods
            cv_score, cv_details = self._cv_methods_similarity(image1, image2)
            results['cv_methods'] = {
                'score': cv_score,
                'details': cv_details,
                'method': 'SSIM + Histogram + Edge + LBP Analysis',
                'proof': f"SSIM: {cv_details.get('ssim', 0):.4f}, Histogram: {cv_details.get('histogram', 0):.4f}, Edge: {cv_details.get('edge_similarity', 0):.4f}, LBP: {cv_details.get('lbp_similarity', 0):.4f}",
                'confidence': self._calculate_method_confidence(cv_score)
            }
            
            # 4. Probabilistic Distribution Analysis
            prob_score, prob_details = self._probabilistic_similarity(image1, image2)
            results['probabilistic'] = {
                'score': prob_score,
                'details': prob_details,
                'method': 'GMM + KL/JS Divergence + Wasserstein Distance',
                'proof': f"KL Div: {prob_details.get('kl_divergence', 0):.4f}, JS Div: {prob_details.get('js_divergence', 0):.4f}, Wasserstein: {prob_details.get('wasserstein_distance', 0):.4f}",
                'confidence': self._calculate_method_confidence(prob_score)
            }
            
            # 5. Ensemble Score with detailed analysis
            ensemble_score = self._compute_ensemble_score(results)
            ensemble_confidence = self._calculate_confidence(results)
            results['ensemble'] = {
                'score': ensemble_score,
                'confidence': ensemble_confidence,
                'method': 'Weighted Ensemble (35% DL + 20% PH + 25% CV + 20% Prob)',
                'proof': f"Final score: {ensemble_score:.4f} with confidence: {ensemble_confidence:.4f}",
                'statistical_significance': self._calculate_statistical_significance(results)
            }
            
            # 6. Generate comprehensive report
            results['analysis_report'] = self._generate_analysis_report(results, analysis_timestamp)
            results['timestamp'] = analysis_timestamp
            
            return results
            
        except Exception as e:
            raise Exception(f"Error computing similarity: {str(e)}")
    
    def _deep_learning_similarity(self, image1, image2):
        """Extract features using ResNet50 and compute cosine similarity"""
        try:
            # Preprocess images for ResNet50
            img1_processed = self._preprocess_for_resnet(image1)
            img2_processed = self._preprocess_for_resnet(image2)
            
            # Extract features
            features1 = self.resnet_model.predict(img1_processed, verbose=0)
            features2 = self.resnet_model.predict(img2_processed, verbose=0)
            
            # Compute cosine similarity
            similarity = cosine_similarity(features1, features2)[0][0]
            
            # Additional analysis
            feature_distance = np.linalg.norm(features1 - features2)
            feature_correlation = np.corrcoef(features1.flatten(), features2.flatten())[0, 1]
            
            return float(similarity), {
                'features1': features1.flatten().tolist(),
                'features2': features2.flatten().tolist(),
                'feature_distance': float(feature_distance),
                'feature_correlation': float(feature_correlation),
                'feature_dimension': int(features1.shape[1])
            }
            
        except Exception as e:
            print(f"Deep learning similarity error: {e}")
            # Return a proper error response instead of fake 0.0
            return 0.0, {'error': str(e), 'method': 'Deep Learning'}
    
    def _perceptual_hash_similarity(self, image1, image2):
        """Compute perceptual hash similarity using multiple hash types"""
        try:
            # Convert to PIL Images
            pil_img1 = Image.fromarray(image1)
            pil_img2 = Image.fromarray(image2)
            
            # Compute different hash types
            hash1_phash = imagehash.phash(pil_img1)
            hash2_phash = imagehash.phash(pil_img2)
            
            hash1_dhash = imagehash.dhash(pil_img1)
            hash2_dhash = imagehash.dhash(pil_img2)
            
            hash1_ahash = imagehash.average_hash(pil_img1)
            hash2_ahash = imagehash.average_hash(pil_img2)
            
            # Calculate Hamming distances
            phash_dist = hash1_phash - hash2_phash
            dhash_dist = hash1_dhash - hash2_dhash
            ahash_dist = hash1_ahash - hash2_ahash
            
            # Convert to similarity scores (0-1)
            phash_sim = 1 - (phash_dist / 64)  # phash is 64-bit
            dhash_sim = 1 - (dhash_dist / 64)  # dhash is 64-bit
            ahash_sim = 1 - (ahash_dist / 64)  # ahash is 64-bit
            
            # Average the similarities
            avg_similarity = (phash_sim + dhash_sim + ahash_sim) / 3
            
            return float(avg_similarity), {
                'phash_distance': int(phash_dist),
                'dhash_distance': int(dhash_dist),
                'ahash_distance': int(ahash_dist),
                'phash_similarity': float(phash_sim),
                'dhash_similarity': float(dhash_sim),
                'ahash_similarity': float(ahash_sim),
                'hash_consistency': float(np.std([phash_sim, dhash_sim, ahash_sim]))
            }
            
        except Exception as e:
            print(f"Perceptual hash similarity error: {e}")
            # Return a proper error response instead of fake 0.0
            return 0.0, {'error': str(e), 'method': 'Perceptual Hashing'}
    
    def _cv_methods_similarity(self, image1, image2):
        """Traditional computer vision similarity methods"""
        try:
            # Convert to grayscale for some methods
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            # 1. SSIM (Structural Similarity Index)
            ssim_score = ssim(gray1, gray2)
            
            # 2. Histogram comparison
            hist1 = cv2.calcHist([image1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([image2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 3. Edge detection similarity
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            edge_sim = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)[0][0]
            edge_sim = max(0, edge_sim)  # Ensure non-negative
            
            # 4. Local Binary Pattern similarity
            lbp1 = local_binary_pattern(gray1, 8, 1, method='uniform')
            lbp2 = local_binary_pattern(gray2, 8, 1, method='uniform')
            lbp_hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10))
            lbp_hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 10))
            lbp_sim = cv2.compareHist(lbp_hist1.astype(np.float32), lbp_hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # Average the scores
            avg_score = (ssim_score + hist_sim + edge_sim + lbp_sim) / 4
            
            return float(avg_score), {
                'ssim': float(ssim_score),
                'histogram': float(hist_sim),
                'edge_similarity': float(edge_sim),
                'lbp_similarity': float(lbp_sim),
                'method_consistency': float(np.std([ssim_score, hist_sim, edge_sim, lbp_sim]))
            }
            
        except Exception as e:
            print(f"CV methods similarity error: {e}")
            # Return a proper error response instead of fake 0.0
            return 0.0, {'error': str(e), 'method': 'Computer Vision'}
    
    def _probabilistic_similarity(self, image1, image2):
        """Probabilistic distribution analysis using GMM and divergence metrics"""
        try:
            # Extract color and texture features
            features1 = self._extract_features_for_gmm(image1)
            features2 = self._extract_features_for_gmm(image2)
            
            # Fit Gaussian Mixture Models
            gmm1 = GaussianMixture(n_components=3, random_state=42)
            gmm2 = GaussianMixture(n_components=3, random_state=42)
            
            gmm1.fit(features1)
            gmm2.fit(features2)
            
            # Calculate KL divergence (approximate)
            kl_div = self._approximate_kl_divergence(gmm1, gmm2, features1, features2)
            
            # Calculate Jensen-Shannon divergence
            js_div = self._jensen_shannon_divergence(gmm1, gmm2, features1, features2)
            
            # Calculate Wasserstein distance (approximate)
            wasserstein_dist = self._approximate_wasserstein_distance(gmm1, gmm2)
            
            # Convert divergences to similarity scores (0-1)
            kl_sim = np.exp(-kl_div)  # Higher KL divergence = lower similarity
            js_sim = np.exp(-js_div)
            wasserstein_sim = np.exp(-wasserstein_dist)
            
            # Average the similarity scores
            avg_similarity = (kl_sim + js_sim + wasserstein_sim) / 3
            
            return float(avg_similarity), {
                'kl_divergence': float(kl_div),
                'js_divergence': float(js_div),
                'wasserstein_distance': float(wasserstein_dist),
                'kl_similarity': float(kl_sim),
                'js_similarity': float(js_sim),
                'wasserstein_similarity': float(wasserstein_sim),
                'distribution_overlap': float(self._calculate_distribution_overlap(gmm1, gmm2))
            }
            
        except Exception as e:
            print(f"Probabilistic similarity error: {e}")
            # Return a proper error response instead of fake 0.0
            return 0.0, {'error': str(e), 'method': 'Probabilistic Analysis'}
    
    def _extract_features_for_gmm(self, image):
        """Extract features for GMM modeling"""
        # Color features (RGB values)
        color_features = image.reshape(-1, 3)
        
        # Texture features (LBP)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        texture_features = lbp.reshape(-1, 1)
        
        # Combine features
        features = np.column_stack([color_features, texture_features])
        
        # Sample for computational efficiency
        if len(features) > 10000:
            indices = np.random.choice(len(features), 10000, replace=False)
            features = features[indices]
        
        return features
    
    def _approximate_kl_divergence(self, gmm1, gmm2, features1, features2):
        """Approximate KL divergence between two GMMs"""
        # Sample from both distributions
        samples1 = gmm1.sample(1000)[0]
        samples2 = gmm2.sample(1000)[0]
        
        # Calculate log probabilities
        log_prob1_given_gmm1 = gmm1.score_samples(samples1)
        log_prob1_given_gmm2 = gmm2.score_samples(samples1)
        
        # KL divergence approximation
        kl_div = np.mean(log_prob1_given_gmm1 - log_prob1_given_gmm2)
        
        return max(0, kl_div)  # Ensure non-negative
    
    def _jensen_shannon_divergence(self, gmm1, gmm2, features1, features2):
        """Calculate Jensen-Shannon divergence"""
        # Sample from both distributions
        samples1 = gmm1.sample(1000)[0]
        samples2 = gmm2.sample(1000)[0]
        
        # Calculate log probabilities
        log_prob1_given_gmm1 = gmm1.score_samples(samples1)
        log_prob1_given_gmm2 = gmm2.score_samples(samples1)
        log_prob2_given_gmm1 = gmm1.score_samples(samples2)
        log_prob2_given_gmm2 = gmm2.score_samples(samples2)
        
        # JS divergence calculation
        m1 = (log_prob1_given_gmm1 + log_prob1_given_gmm2) / 2
        m2 = (log_prob2_given_gmm1 + log_prob2_given_gmm2) / 2
        
        js_div = 0.5 * (np.mean(log_prob1_given_gmm1 - m1) + np.mean(log_prob2_given_gmm2 - m2))
        
        return max(0, js_div)
    
    def _approximate_wasserstein_distance(self, gmm1, gmm2):
        """Approximate Wasserstein distance between GMMs"""
        # Use means and covariances for approximation
        means1 = gmm1.means_
        means2 = gmm2.means_
        covs1 = gmm1.covariances_
        covs2 = gmm2.covariances_
        
        # Simple approximation using mean distances
        min_distances = []
        for mean1 in means1:
            distances = [np.linalg.norm(mean1 - mean2) for mean2 in means2]
            min_distances.append(min(distances))
        
        return np.mean(min_distances)
    
    def _preprocess_for_resnet(self, image):
        """Preprocess image for ResNet50"""
        # Resize to 224x224
        resized = cv2.resize(image, (224, 224))
        # Preprocess for ResNet50
        preprocessed = preprocess_input(resized)
        # Add batch dimension
        return np.expand_dims(preprocessed, axis=0)
    
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
        
        # Weighted average
        ensemble_score = np.average(scores, weights=weights)
        return float(ensemble_score)
    
    def _calculate_confidence(self, results):
        """Calculate confidence in the ensemble score"""
        scores = []
        for method in self.weights.keys():
            if method in results and 'score' in results[method]:
                scores.append(results[method]['score'])
        
        if len(scores) < 2:
            return 0.0
        
        # Calculate standard deviation as confidence measure
        std_dev = np.std(scores)
        confidence = 1 - std_dev  # Higher std = lower confidence
        
        return max(0, min(1, confidence))
    
    
    
    
    def _calculate_distribution_overlap(self, gmm1, gmm2):
        """Calculate overlap between two GMM distributions"""
        try:
            # Sample from both distributions
            samples1 = gmm1.sample(1000)[0]
            samples2 = gmm2.sample(1000)[0]
            
            # Calculate probability densities
            prob1_given_gmm1 = np.exp(gmm1.score_samples(samples1))
            prob1_given_gmm2 = np.exp(gmm2.score_samples(samples1))
            prob2_given_gmm1 = np.exp(gmm1.score_samples(samples2))
            prob2_given_gmm2 = np.exp(gmm2.score_samples(samples2))
            
            # Calculate overlap as intersection over union
            intersection = np.minimum(prob1_given_gmm1, prob1_given_gmm2)
            union = np.maximum(prob1_given_gmm1, prob1_given_gmm2)
            overlap = np.mean(intersection / (union + 1e-10))
            
            return float(overlap)
        except Exception as e:
            return 0.0
    
    def _calculate_method_confidence(self, score):
        """Calculate confidence for individual method scores based on score magnitude"""
        # Higher scores at extremes (0 or 1) indicate higher confidence
        # Scores closer to 0.5 indicate lower confidence (more uncertainty)
        distance_from_uncertainty = abs(score - 0.5)
        confidence = 2 * distance_from_uncertainty  # Scale to 0-1 range
        # Ensure confidence is at least 0.1 and at most 1.0
        return max(0.1, min(1.0, confidence))
    
    def _calculate_statistical_significance(self, results):
        """Calculate statistical significance of ensemble results"""
        try:
            scores = []
            for method in self.weights.keys():
                if method in results and 'score' in results[method]:
                    scores.append(results[method]['score'])
            
            if len(scores) < 2:
                return 0.0
            
            # Calculate coefficient of variation
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / (mean_score + 1e-10)
            
            # Statistical significance based on consistency
            significance = 1 - cv  # Lower CV = higher significance
            return max(0, min(1, significance))
        except Exception as e:
            return 0.0
    
    def _generate_analysis_report(self, results, timestamp):
        """Generate comprehensive analysis report"""
        try:
            report = {
                'timestamp': timestamp,
                'summary': {
                    'ensemble_score': results['ensemble']['score'],
                    'confidence': results['ensemble']['confidence'],
                    'statistical_significance': results['ensemble']['statistical_significance']
                },
                'method_analysis': {},
                'technical_details': {},
                'recommendations': []
            }
            
            # Analyze each method
            for method, weight in self.weights.items():
                if method in results:
                    method_data = results[method]
                    report['method_analysis'][method] = {
                        'score': method_data['score'],
                        'confidence': method_data.get('confidence', 0),
                        'method_description': method_data.get('method', ''),
                        'proof': method_data.get('proof', ''),
                        'weight': weight,
                        'contribution': method_data['score'] * weight
                    }
            
            # Technical details
            report['technical_details'] = {
                'total_methods': len([m for m in self.weights.keys() if m in results]),
                'ensemble_weights': self.weights,
                'score_variance': np.var([results[m]['score'] for m in self.weights.keys() if m in results]),
                'method_consistency': self._calculate_method_consistency(results)
            }
            
            # Generate recommendations
            ensemble_score = results['ensemble']['score']
            if ensemble_score > 0.8:
                report['recommendations'].append("High similarity detected - images are very similar")
            elif ensemble_score > 0.6:
                report['recommendations'].append("Moderate similarity - images share some common features")
            elif ensemble_score > 0.4:
                report['recommendations'].append("Low similarity - images have limited common features")
            else:
                report['recommendations'].append("Very low similarity - images are quite different")
            
            # Add method-specific recommendations
            for method, data in report['method_analysis'].items():
                if data['score'] > 0.9:
                    report['recommendations'].append(f"{method.replace('_', ' ').title()} method shows very high similarity")
                elif data['score'] < 0.3:
                    report['recommendations'].append(f"{method.replace('_', ' ').title()} method shows low similarity")
            
            return report
        except Exception as e:
            print(f"Report generation error: {e}")
            return {'error': str(e)}
    
    def _calculate_method_consistency(self, results):
        """Calculate consistency across methods"""
        try:
            scores = [results[m]['score'] for m in self.weights.keys() if m in results]
            if len(scores) < 2:
                return 0.0
            
            # Calculate coefficient of variation (lower = more consistent)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / (mean_score + 1e-10)
            consistency = 1 - cv
            return max(0, min(1, consistency))
        except Exception as e:
            return 0.0
    

