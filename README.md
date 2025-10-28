# Image Similarity Analyzer


A modular, efficient web application that computes image similarity using an ensemble of deep learning, computer vision, and probabilistic methods. Built with reusable components and existing libraries to minimize code duplication.

## Table of Contents

* [Image Similarity Analyzer](#image-similarity-analyzer)
  * [🚀 **Key Features**](#-key-features)
    * [**Modular Architecture**](#modular-architecture)
    * [**Advanced Similarity Analysis**](#advanced-similarity-analysis)
    * [**Comprehensive Visualizations**](#comprehensive-visualizations)
    * [**Modern User Interface**](#modern-user-interface)
  * [🏗️ **Modular Architecture**](#-modular-architecture)
    * [2. Perceptual Hashing (15% weight)](#2-perceptual-hashing-15-weight)
    * [3. Computer Vision (25% weight)](#3-computer-vision-25-weight)
    * [4. Probabilistic (10% weight)](#4-probabilistic-10-weight)
    * [Ensemble Approach](#ensemble-approach)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Technical Details](#technical-details)
    * [Similarity Methods with Theory](#similarity-methods-with-theory)
      * [1. Deep Learning (Weight: 50%)](#1-deep-learning-weight-50)
      * [2. Perceptual Hashing (Weight: 15%)](#2-perceptual-hashing-weight-15)
      * [3. Computer Vision Methods (Weight: 25%)](#3-computer-vision-methods-weight-25)
      * [4. Probabilistic Analysis (Weight: 10%)](#4-probabilistic-analysis-weight-10)
    * [Ensemble Scoring](#ensemble-scoring)
    * [Visualization Types](#visualization-types)
  * [📊 Quick Reference for Presentations](#-quick-reference-for-presentations)
    * [Summary Table of Methods](#summary-table-of-methods)
    * [Key Formulas for Presentations](#key-formulas-for-presentations)
    * [Why This Combination Works](#why-this-combination-works)
    * [Real-World Applications](#real-world-applications)
  * [Performance Considerations](#performance-considerations)
  * [📦 **Optimized Dependencies**](#-optimized-dependencies)
    * [**Core Framework**](#core-framework)
    * [**Machine Learning & Computer Vision**](#machine-learning--computer-vision)
    * [**Scientific Computing**](#scientific-computing)
    * [**Visualization**](#visualization)
    * [**Image Processing**](#image-processing)
  * [🔧 **Recent Improvements & Bug Fixes**](#-recent-improvements--bug-fixes)
    * [Version 2.0 Updates](#version-20-updates)
  * [🚀 **Real-World Performance**](#-real-world-performance)

## 🚀 **Key Features**

### **Modular Architecture**
- **Feature Extractors**: Reusable components for different similarity methods
- **Heatmap Generators**: Specialized visualization modules
- **Report Generator**: Comprehensive analysis reporting
- **Simplified Engine**: Clean, maintainable similarity computation

### **Advanced Similarity Analysis**
- **Deep Learning**: ResNet50 feature extraction + semantic category matching with bonus scoring
- **Perceptual Hashing**: Multi-hash comparison (pHash, dHash, aHash)
- **Computer Vision**: SSIM, histogram, edge detection, LBP analysis
- **Probabilistic**: GMM modeling with KL/JS divergence and Wasserstein distance metrics

### **Comprehensive Visualizations**
- Interactive similarity scores with confidence intervals
- Method-by-method breakdown and comparison
- Statistical analysis with uncertainty quantification
- Real-time progress tracking during analysis
- Thread-safe matplotlib rendering with Agg backend (fixes tkinter issues on Windows)

### **Modern User Interface**
- Polished dark theme by default (inspired by v0.app)
- Optional light mode via in-app toggle (persists across sessions)
- Drag-and-drop uploads with clear previews and actions
- Responsive layout, subtle glass surfaces, and smooth animations
- Tabbed interface for detailed analysis and SVG/PNG visualizations

## 🏗️ **Modular Architecture**

```
├── app.py                      # Flask application (uses SimilarityEngineV2)
├── similarity_engine_v2.py     # Modular similarity engine (ACTIVE)
├── visualizations.py           # Visualization generation with matplotlib (Agg backend)
├── utils.py                    # Helper utilities
├── run.py                      # Application entry point
├── test_setup.py               # Setup verification script
├── requirements.txt            # Optimized dependencies
├── modules/                    # Reusable components
│   ├── __init__.py
│   ├── feature_extractors.py   # Deep learning, CV, hash extractors with category detection
│   └── report_generator.py     # Comprehensive reporting
├── static/
│   ├── css/style.css          # Dark theme styling with design tokens + light mode
│   ├── js/app.js              # Frontend logic
│   └── uploads/               # Temporary storage
├── templates/
│   └── index.html             # Main interface
├── models/                     # Pre-trained model cache
└── test_images/                # Sample images for testing
```

**Note**: The application uses `similarity_engine_v2.py` for modular, maintainable similarity computation with enhanced semantic understanding through category detection.

### 1. Deep Learning (50% weight)

**What:** ResNet50 extracts 2048-dimensional feature vectors + ImageNet classification

**How:** Cosine similarity: cos(θ) = (A·B)/(||A||·||B||) + category matching bonus

**Why:** Captures semantic similarity (understands WHAT is in the image) and rewards same object types

### 2. Perceptual Hashing (15% weight)

**What:** Generates 64-bit fingerprints using DCT

**How:** Hamming distance measures bit differences

**Why:** Fast O(1) comparison, robust to minor changes, best for near-duplicate detection

### 3. Computer Vision (25% weight)

**What:** SSIM, Histogram, Canny Edges, LBP

**How:** Statistical comparison of structural features

**Why:** Captures low-level visual properties (colors, textures, edges)

### 4. Probabilistic (10% weight)

**What:** Gaussian Mixture Models with divergence metrics

**How:** KL/JS divergence + Wasserstein distance

**Why:** Statistical robustness, handles uncertainty

### Ensemble Approach
**Formula:** 0.50×DL + 0.15×PH + 0.25×CV + 0.10×Prob
**Confidence:** 1 - std_dev(scores)
**Result:** Robust similarity score with confidence interval



- **Existing library functions** instead of custom implementations
- **Modular design** for easy maintenance and extension

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/axls23/-edl_project.git
   cd edl_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
  Navigate to `http://localhost:5000`
  - Use the top-right toggle to switch between Dark/Light themes

## Usage

1. **Upload Images**: Drag and drop or click to select two images
2. **Analyze**: Click "Analyze Similarity" to start the computation
3. **View Results**: Explore different tabs for detailed analysis:
   - **Overview**: Main similarity score and method contributions
   - **Method Details**: Individual method scores and breakdowns
   - **Image Comparison**: Side-by-side comparison with difference maps
   - **Statistics**: Statistical analysis and confidence intervals
   - **Confidence**: Uncertainty quantification and reliability metrics

## Technical Details
### Similarity Methods with Theory

#### 1. Deep Learning (Weight: 50%)

**Methods Used:**
- **ResNet50 Feature Extraction**: Pre-trained ImageNet model (2048-dimensional embeddings)
- **ResNet50 Classification**: Object/scene category detection (1000 ImageNet classes)
- **Cosine Similarity**: Measures angle between feature vectors
- **Category Matching Bonus**: Rewards images containing same object types

**Theory:**
ResNet50 is a 50-layer deep convolutional neural network trained on ImageNet (14M images, 1000 categories). It performs two key functions:

1. **Feature Extraction** (include_top=False): Extracts high-dimensional feature vectors (2048 dimensions) from the final pooling layer that capture semantic content - shapes, textures, objects, and context.

2. **Classification** (include_top=True): Identifies what objects/scenes are present in the image (e.g., "electric_locomotive", "golden_retriever", "mountain").

The final similarity score combines feature similarity with a category matching bonus:
- Base score from cosine similarity of feature vectors
- +5% bonus for each matching category in top-3 predictions (max +15%)
- This ensures semantically similar images (e.g., two different dogs) score higher

**Formula:**
```
feature_similarity = cos(θ) = Σ(Ai × Bi) / (√Σ(Ai²) × √Σ(Bi²))
category_bonus = min(0.05 × category_overlap, 0.15)
final_score = min(1.0, feature_similarity + category_bonus)
```

**Advantages**: 
- Captures high-level semantic similarities
- Understands object types, not just visual features
- Robust to viewpoint and lighting changes
- Rewards "same category" matches (two trains, two cats, etc.)


---

#### 2. Perceptual Hashing (Weight: 15%)

**Methods Used:**
- **pHash**: Perceptual hash based on DCT (Discrete Cosine Transform)
- **dHash**: Difference hash comparing adjacent pixels
- **aHash**: Average hash comparing pixels to mean

**Theory:**
Perceptual hashing generates compact binary fingerprints (64-bit) of images. Unlike cryptographic hashes, similar images produce similar hashes. Hamming distance measures bit differences. pHash uses DCT to capture frequency information, dHash tracks gradients, aHash measures brightness distribution.

**Formula:**
```
similarity = 1 - (hamming_distance / 64)
hamming_distance = count of differing bits
```

**Advantages**: Fast O(1) comparison, robust to minor modifications, compression-resistant, excellent for near-duplicate detection

---

#### 3. Computer Vision Methods (Weight: 25%)

**Methods Used:**
- **SSIM**: Structural Similarity Index
- **Histogram Comparison**: Color distribution matching
- **Edge Detection**: Canny edge analysis
- **LBP**: Local Binary Pattern for texture

**Theory:**

**SSIM** combines luminance, contrast, and structure:
```
SSIM(x,y) = [l(x,y)^α × c(x,y)^β × s(x,y)^γ]
where:
l = luminance comparison = (2μxμy + C1)/(μx² + μy² + C1)
c = contrast comparison = (2σxσy + C2)/(σx² + σy² + C2)
s = structure comparison = (σxy + C3)/(σxσy + C3)
```

**Histogram** uses correlation coefficient between color distributions:
```
correlation = Σ(H1(i) × H2(i)) / √(Σ(H1(i)²) × Σ(H2(i)²))
```

**LBP** encodes local texture patterns as binary numbers, invariant to monotonic illumination changes.

**Advantages**: Captures low-level visual features, computationally efficient

---

#### 4. Probabilistic Analysis (Weight: 10%)

**Methods Used:**
- **GMM**: Gaussian Mixture Models for distribution modeling
- **KL Divergence**: Kullback-Leibler divergence
- **JS Divergence**: Jensen-Shannon divergence
- **Wasserstein Distance**: Earth mover's distance

**Theory:**

**GMM** models image features as mixture of Gaussians:
```
P(x) = Σ(wi × N(x|μi, Σi))
where wi are mixture weights, N is Gaussian distribution
```

**KL Divergence** measures distribution difference (non-symmetric):
```
DKL(P||Q) = Σ P(x) × log(P(x)/Q(x))
```

**JS Divergence** (symmetric version of KL):
```
DJS(P||Q) = 0.5 × DKL(P||M) + 0.5 × DKL(Q||M)
where M = 0.5(P + Q)
```

**Wasserstein Distance** measures optimal transport cost between distributions:
```
W(P,Q) = inf E[||X-Y||] over all joint distributions
```

**Advantages**: Statistical robustness, handles uncertainty, captures global color/texture distributions

---

### Ensemble Scoring

The final similarity score is computed as a weighted average with semantic understanding prioritized:

```python
final_score = (
    0.50 * deep_learning_score +      # Semantic + category matching
    0.25 * cv_methods_score +          # Structural features
    0.15 * perceptual_hash_score +     # Near-duplicate detection
    0.10 * probabilistic_score         # Statistical distributions
)
```

**Rationale for Updated Weights (v2.0):**
- **Deep Learning (50%)**: Highest weight due to semantic understanding + category detection
  - Understands WHAT objects are in images, not just how they look
  - Category matching bonus ensures semantically similar images score high
  - Example: Two different photos of trains score ~70-80% (not 50%)
  
- **CV Methods (25%)**: Strong weight for structural and pixel-level features
  - SSIM, histograms, edges, textures
  - Complements semantic understanding with visual analysis
  
- **Perceptual Hash (15%)**: Reduced weight, specialized for near-duplicates
  - Excellent for detecting exact/near copies
  - Less useful for "similar but different" images
  
- **Probabilistic (10%)**: Supporting role for statistical robustness
  - Handles edge cases and uncertainty
  - Provides statistical validation

**Key Improvement over v1.0:**
The original weights (35-20-25-20) gave equal importance to all methods, which under-valued semantic understanding. The new weights (50-25-15-10) prioritize "understanding what's in the image" over pixel-level matching, aligning better with human perception of similarity.

**Confidence Calculation:**
```
confidence = 1 - std_dev(method_scores)
```
Lower variance across methods indicates higher confidence in the result.

**Statistical Significance:**
```
significance = 1 - CV (coefficient of variation)
CV = std_dev / mean
```
Lower CV indicates more consistent results across methods.

### Visualization Types

1. **Overview**: Circular progress bar with method contributions (correct weights displayed)
2. **Method Comparison**: Detailed breakdown of individual methods
3. **Image Comparison**: Side-by-side with difference visualization
4. **Statistical Analysis**: Distribution plots and confidence intervals
5. **Confidence Analysis**: Uncertainty quantification and reliability

**Note**: All visualizations use matplotlib's Agg backend (non-interactive) to prevent tkinter threading issues on Windows/Flask applications.

## 📊 Quick Reference for Presentations

### Summary Table of Methods

| Method | Weight | Key Technique | Theory | Complexity |
|--------|--------|---------------|--------|------------|
| **Deep Learning** | 50% | ResNet50 Features + Category Detection | CNN feature extraction + ImageNet classification | O(n) per image |
| **CV Methods** | 25% | SSIM, Histogram, Edge, LBP | Structural + statistical pixel analysis | O(n) per method |
| **Perceptual Hash** | 15% | pHash, dHash, aHash | DCT-based fingerprinting + Hamming distance | O(1) comparison |
| **Probabilistic** | 10% | GMM + KL/JS/Wasserstein | Distribution modeling + divergence metrics | O(n×k) for k components |

### Key Formulas for Presentations

**1. Cosine Similarity (Deep Learning):**
```
cos(θ) = (A·B) / (||A|| × ||B||)
Range: [-1, 1], but typically [0, 1] for images
```

**2. Hamming Distance (Perceptual Hash):**
```
similarity = 1 - (bit_differences / 64)
Fast: O(1) comparison time
```

**3. SSIM (Computer Vision):**
```
SSIM = luminance × contrast × structure
Range: [-1, 1], perfect match = 1
```

**4. KL Divergence (Probabilistic):**
```
DKL(P||Q) = Σ P(x) log(P(x)/Q(x))
Non-symmetric: DKL(P||Q) ≠ DKL(Q||P)
```

**5. Ensemble Score:**
```
Final = 0.50×DL + 0.15×PH + 0.25×CV + 0.10×Prob
With semantic category bonus in DL component
```

### Why This Combination Works

1. **Complementary Approaches**: Each method captures different aspects
   - **Deep Learning**: Semantic content (what objects are in the image) + object categories
   - **CV Methods**: Low-level features (colors, edges, textures, structure)
   - **Perceptual Hash**: Overall structure and near-duplicate detection
   - **Probabilistic**: Statistical properties (distributions and patterns)

2. **Robust to Different Scenarios**:
   - **Identical images** → All methods score high (~95-100%)
   - **Semantically similar** (e.g., two trains) → Deep learning detects + category bonus (~70-85%)
   - **Slight modifications** → Perceptual hash still detects similarity (~80-90%)
   - **Different lighting** → SSIM and probabilistic handle well
   - **Different but related** → Category matching provides bonus

3. **Confidence Through Consensus**:
   - High agreement across methods → High confidence (>0.8)
   - Disagreement → Lower confidence, indicates uncertainty
   - Statistical significance measures consistency

4. **Human-Aligned Perception**:
   - Updated weights (50-25-15-10) better match human similarity judgments
   - Two photos of trains score ~75%, not ~50%
   - Semantic understanding prioritized over pixel matching

### Real-World Applications

- **Content Moderation**: Detect duplicate/similar content
- **Copyright Detection**: Find unauthorized image use
- **Image Search**: Find visually similar images
- **Quality Control**: Detect manufacturing defects
- **Medical Imaging**: Compare diagnostic scans

## Performance Considerations

- **Image Preprocessing**: Automatic resizing for large images (handles different dimensions)
- **Model Caching**: Pre-trained ResNet50 models loaded once at startup
- **Memory Management**: Efficient feature extraction with automatic cleanup
- **Progress Tracking**: Real-time feedback during computation
- **Thread Safety**: Matplotlib uses Agg backend to prevent tkinter conflicts in Flask
- **File Size Limits**: Configurable upload limits (default 16MB, easily adjustable in app.py)

## 📦 **Optimized Dependencies**

### **Core Framework**
- **Flask 3.0+**: Lightweight web framework
- **Werkzeug**: WSGI utilities

### **Machine Learning & Computer Vision**
- **TensorFlow 2.16+**: Deep learning models (ResNet50 for features + classification)
- **OpenCV**: Computer vision operations
- **scikit-learn**: Machine learning algorithms (GMM, cosine similarity)
- **scikit-image**: Image processing (SSIM, LBP)

### **Scientific Computing**
- **NumPy**: Numerical computing
- **SciPy**: Scientific algorithms (divergence metrics)
- **Pillow**: Image handling

### **Visualization**
- **Matplotlib**: Plotting and charts (configured with Agg backend for thread safety)

### **Image Processing**
- **imagehash**: Perceptual hashing (pHash, dHash, aHash)

## 🔧 **Recent Improvements & Bug Fixes**

### Version 2.0 Updates

**1. Enhanced Semantic Understanding**
- Added ResNet50 classification for object/scene category detection
- Implemented category matching bonus (up to +15% for matching categories)
- Increased deep learning weight from 35% to 50% for better semantic alignment

**2. Optimized Ensemble Weights**
- Rebalanced to prioritize semantic similarity over pixel matching
- New weights: 50% DL, 25% CV, 15% PH, 10% Prob (from 35-20-25-20)
- Better alignment with human perception of similarity

**3. Visualization Improvements**
- Fixed hardcoded weights in visualization generation
- Synchronized weight displays across all charts
- Implemented matplotlib Agg backend to prevent tkinter threading issues on Windows

**4. Technical Fixes**
- Resolved dimension mismatch issues in CV methods (auto-resizing)
- Fixed Python module caching problems
- Improved error handling for edge cases
- Enhanced resource cleanup (matplotlib figures, temporary files)

**5. Code Quality**
- Removed deprecated `similarity_engine.py` (monolithic version)
- Consolidated to modular `similarity_engine_v2.py`
- Eliminated hardcoded "magic numbers" in visualizations
- Improved documentation and inline comments

## 🚀 **Real-World Performance**

**Example Results (Two Similar Train Images):**
- **Deep Learning**: ~100% (detected as same category: electric_locomotive)
- **Perceptual Hash**: ~24% (different photos, different pixels)
- **CV Methods**: ~61.5% (similar colors/structure)
- **Probabilistic**: ~9.9% (different distributions)
- **Overall Similarity**: ~72.5% ✓ (correctly identified as "very similar")

**Why This Makes Sense:**
- Old system (35-20-25-20): Would score ~52% (seems too low)
- New system (50-25-15-10): Scores ~72% (aligns with human judgment)
- The semantic understanding (Deep Learning) dominates the final score

 **Image Processing**
- **imagehash**: Perceptual hashing
