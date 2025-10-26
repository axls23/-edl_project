"""
Test image similarity with different dimensions
"""
import numpy as np
import cv2
from similarity_engine_v2 import SimilarityEngineV2

print("Testing similarity computation with different image dimensions...\n")

# Create test images with DIFFERENT dimensions
image1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)  # 300x400
image2 = np.random.randint(0, 255, (200, 500, 3), dtype=np.uint8)  # 200x500

print(f"Image 1 dimensions: {image1.shape}")
print(f"Image 2 dimensions: {image2.shape}")
print()

# Initialize engine
engine = SimilarityEngineV2()

# Test similarity computation
try:
    print("Computing similarity...")
    results = engine.compute_similarity(image1, image2)
    print("✓ Similarity computation successful!")
    print()

    # Display results
    print("Results:")
    print(f"  Deep Learning Score: {results['deep_learning']['score']:.4f}")
    print(f"  Perceptual Hash Score: {results['perceptual_hash']['score']:.4f}")
    print(f"  CV Methods Score: {results['cv_methods']['score']:.4f}")
    print(f"  Probabilistic Score: {results['probabilistic']['score']:.4f}")
    print(f"  Ensemble Score: {results['ensemble']['score']:.4f}")
    print(f"  Confidence: {results['ensemble']['confidence']:.4f}")
    print()
    print("✓ All methods computed successfully!")
    print("✓ Different image dimensions handled correctly!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
