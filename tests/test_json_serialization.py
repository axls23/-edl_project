"""
Test JSON serialization of similarity results
"""
import json
import numpy as np
import cv2
from similarity_engine_v2 import SimilarityEngineV2

# Create a simple test
print("Testing JSON serialization of similarity results...")

# Create two simple test images
image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Initialize engine
engine = SimilarityEngineV2()

# Compute similarity
results = engine.compute_similarity(image1, image2)

# Try to serialize to JSON
try:
    json_str = json.dumps(results)
    print("✓ JSON serialization successful!")
    print(f"✓ Serialized {len(json_str)} characters")
    print("\nResults keys:", list(results.keys()))

    # Check each section
    for key in results.keys():
        if isinstance(results[key], dict):
            print(f"\n{key} keys:", list(results[key].keys()))

except Exception as e:
    print(f"✗ JSON serialization failed: {e}")
    import traceback
    traceback.print_exc()
