"""
Feature extraction modules using existing libraries
"""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import tensorflow as tf
from keras import Model
import warnings

warnings.filterwarnings('ignore')


class DeepLearningExtractor:
    """Deep learning feature extraction using pre-trained models"""
    
    def __init__(self):
        # Feature extraction model (no top layer)
        self.feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        # Classification model (with top layer for category detection)
        self.classification_model = ResNet50(weights='imagenet', include_top=True)
    
    def extract_features(self, image):
        """Extract features using ResNet50"""
        processed = self._preprocess_image(image)
        features = self.feature_model.predict(processed, verbose=0)
        return features
    
    def classify_image(self, image):
        """Classify image to detect object categories"""
        processed = self._preprocess_image(image)
        predictions = self.classification_model.predict(processed, verbose=0)
        # Get top 5 predictions with their probabilities
        decoded = decode_predictions(predictions, top=5)[0]
        return [(class_name, float(score)) for (_, class_name, score) in decoded]

    def gradcam_heatmap(self, image, class_index=None):
        """Generate Grad-CAM heatmap for the top predicted class or provided class_index.
        Returns a float32 heatmap in range [0,1] same HxW as input image (after resizing back).
        """
        try:
            # Prepare image
            resized = cv2.resize(image, (224, 224))
            x = np.expand_dims(preprocess_input(resized.astype(np.float32)), axis=0)

            # Predict class if not provided
            preds = self.classification_model.predict(x, verbose=0)
            if class_index is None:
                class_index = int(np.argmax(preds[0]))

            # Target layer for ResNet50
            last_conv_layer = self.classification_model.get_layer('conv5_block3_out')
            grad_model = Model(
                [self.classification_model.inputs],
                [last_conv_layer.output, self.classification_model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x)
                loss = predictions[:, class_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

            heatmap = np.maximum(heatmap.numpy(), 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            # Resize to input size
            heatmap = cv2.resize(heatmap.astype(np.float32), (image.shape[1], image.shape[0]))
            return heatmap.astype(np.float32)
        except Exception:
            # Fail gracefully; return None to skip Grad-CAM
            return None

    @staticmethod
    def _preprocess_image(image):
        """Preprocess image for ResNet50"""
        resized = cv2.resize(image, (224, 224))
        preprocessed = preprocess_input(resized)
        return np.expand_dims(preprocessed, axis=0)
    
    def compute_similarity(self, image1, image2):
        """Compute cosine similarity between features with category matching bonus"""
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        # Feature similarity
        feature_similarity = cosine_similarity(features1, features2)[0][0]
        
        # Category matching bonus
        categories1 = self.classify_image(image1)
        categories2 = self.classify_image(image2)
        
        # Check if top categories match
        top_cats1 = set([cat for cat, _ in categories1[:3]])
        top_cats2 = set([cat for cat, _ in categories2[:3]])
        category_overlap = len(top_cats1.intersection(top_cats2))
        
        # Add bonus based on category overlap (0-15% boost)
        category_bonus = category_overlap * 0.05  # 5% per matching category
        
        # Combine feature similarity with category bonus (capped at 1.0)
        final_similarity = min(1.0, feature_similarity + category_bonus)
        
        return float(final_similarity), {
            'features1': features1.flatten().tolist(),
            'features2': features2.flatten().tolist(),
            'feature_distance': float(np.linalg.norm(features1 - features2)),
            'feature_correlation': float(np.corrcoef(features1.flatten(), features2.flatten())[0, 1]),
            'feature_similarity': float(feature_similarity),
            'category_bonus': float(category_bonus),
            'categories1': categories1,
            'categories2': categories2,
            'category_overlap': int(category_overlap)
        }


class PerceptualHashExtractor:
    """Perceptual hashing using imagehash library"""

    @staticmethod
    def compute_similarity(image1, image2):
        """Compute perceptual hash similarity"""
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
        
        # Convert to similarity scores
        phash_sim = 1 - (phash_dist / 64)
        dhash_sim = 1 - (dhash_dist / 64)
        ahash_sim = 1 - (ahash_dist / 64)
        
        avg_similarity = (phash_sim + dhash_sim + ahash_sim) / 3
        
        return float(avg_similarity), {
            'phash_distance': int(phash_dist),
            'dhash_distance': int(dhash_dist),
            'ahash_distance': int(ahash_dist),
            'phash_similarity': float(phash_sim),
            'dhash_similarity': float(dhash_sim),
            'ahash_similarity': float(ahash_sim)
        }


class ComputerVisionExtractor:
    """Computer vision methods using OpenCV and scikit-image"""

    @staticmethod
    def compute_similarity(image1, image2):
        """Compute CV-based similarity"""
        # Resize images to same dimensions if they differ
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if h1 != h2 or w1 != w2:
            # Use the larger dimensions to preserve more detail
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            image1 = cv2.resize(image1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            image2 = cv2.resize(image2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        
        # SSIM - now both images have same dimensions
        ssim_score = ssim(gray1, gray2)
        
        # Histogram comparison (doesn't require same dimensions, but we keep consistent sizes)
        hist1 = cv2.calcHist([image1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([image2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Edge detection - now both images have same dimensions
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        edge_sim = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)[0][0]
        edge_sim = max(0, edge_sim)
        
        # LBP - now both images have same dimensions
        lbp1 = local_binary_pattern(gray1, 8, 1, method='uniform')
        lbp2 = local_binary_pattern(gray2, 8, 1, method='uniform')
        lbp_hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10))
        lbp_hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 10))
        lbp_sim = cv2.compareHist(lbp_hist1.astype(np.float32), lbp_hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        
        avg_score = (ssim_score + hist_sim + edge_sim + lbp_sim) / 4
        
        return float(avg_score), {
            'ssim': float(ssim_score),
            'histogram': float(hist_sim),
            'edge_similarity': float(edge_sim),
            'lbp_similarity': float(lbp_sim)
        }


class ProbabilisticExtractor:
    """Probabilistic analysis using scikit-learn GMM"""
    
    def compute_similarity(self, image1, image2):
        """Compute probabilistic similarity using GMM"""
        features1 = self._extract_features(image1)
        features2 = self._extract_features(image2)
        
        # Fit GMMs
        gmm1 = GaussianMixture(n_components=3, random_state=42)
        gmm2 = GaussianMixture(n_components=3, random_state=42)
        gmm1.fit(features1)
        gmm2.fit(features2)
        
        # Calculate divergences
        kl_div = self._kl_divergence(gmm1, gmm2)
        js_div = self._jensen_shannon_divergence(gmm1, gmm2)
        wasserstein_dist = self._wasserstein_distance(gmm1, gmm2)
        
        # Convert to similarities
        kl_sim = np.exp(-kl_div)
        js_sim = np.exp(-js_div)
        wasserstein_sim = np.exp(-wasserstein_dist)
        
        avg_similarity = (kl_sim + js_sim + wasserstein_sim) / 3
        
        return float(avg_similarity), {
            'kl_divergence': float(kl_div),
            'js_divergence': float(js_div),
            'wasserstein_distance': float(wasserstein_dist),
            'kl_similarity': float(kl_sim),
            'js_similarity': float(js_sim),
            'wasserstein_similarity': float(wasserstein_sim),
            'n_components': 3,
            'gmm1_means_shape': list(gmm1.means_.shape),
            'gmm2_means_shape': list(gmm2.means_.shape)
        }

    @staticmethod
    def _extract_features(image):
        """Extract features for GMM"""
        color_features = image.reshape(-1, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        texture_features = lbp.reshape(-1, 1)
        features = np.column_stack([color_features, texture_features])
        
        if len(features) > 10000:
            indices = np.random.choice(len(features), 10000, replace=False)
            features = features[indices]
        return features

    @staticmethod
    def _kl_divergence(gmm1, gmm2):
        """Calculate KL divergence using Monte Carlo estimation with more samples for accuracy"""
        # Use more samples for better accuracy
        samples = gmm1.sample(5000)[0]
        log_prob1 = gmm1.score_samples(samples)
        log_prob2 = gmm2.score_samples(samples)
        # KL divergence: E[log(p1/p2)] = E[log(p1) - log(p2)]
        kl_div = np.mean(log_prob1 - log_prob2)
        return max(0, kl_div)  # KL divergence is always non-negative

    @staticmethod
    def _jensen_shannon_divergence(gmm1, gmm2):
        """Calculate Jensen-Shannon divergence using more samples for accuracy"""
        # Use more samples for better accuracy
        samples1 = gmm1.sample(5000)[0]
        samples2 = gmm2.sample(5000)[0]
        
        log_prob1_gmm1 = gmm1.score_samples(samples1)
        log_prob1_gmm2 = gmm2.score_samples(samples1)
        log_prob2_gmm1 = gmm1.score_samples(samples2)
        log_prob2_gmm2 = gmm2.score_samples(samples2)
        
        # JS divergence: 0.5 * KL(P1||M) + 0.5 * KL(P2||M) where M = 0.5*(P1+P2)
        m1 = (log_prob1_gmm1 + log_prob1_gmm2) / 2
        m2 = (log_prob2_gmm1 + log_prob2_gmm2) / 2
        
        js_div = 0.5 * (np.mean(log_prob1_gmm1 - m1) + np.mean(log_prob2_gmm2 - m2))
        return max(0, js_div)  # JS divergence is always non-negative

    @staticmethod
    def _wasserstein_distance(gmm1, gmm2):
        """Calculate Wasserstein distance using optimal transport between GMM components"""
        means1 = gmm1.means_
        means2 = gmm2.means_
        weights1 = gmm1.weights_
        weights2 = gmm2.weights_
        
        # Calculate pairwise distances between all component means
        distances = np.zeros((len(means1), len(means2)))
        for i, mean1 in enumerate(means1):
            for j, mean2 in enumerate(means2):
                distances[i, j] = np.linalg.norm(mean1 - mean2)
        
        # Simple approximation: weighted average of minimum distances
        min_distances = []
        for i in range(len(means1)):
            min_dist = np.min(distances[i])
            min_distances.append(weights1[i] * min_dist)
        
        return np.sum(min_distances)
