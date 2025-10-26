# Image Similarity Analyzer

A modular, efficient web application that computes image similarity using an ensemble of deep learning, computer vision, and probabilistic methods. Built with reusable components and existing libraries to minimize code duplication.

## 🚀 **Key Features**

### **Modular Architecture**
- **Feature Extractors**: Reusable components for different similarity methods
- **Heatmap Generators**: Specialized visualization modules
- **Report Generator**: Comprehensive analysis reporting
- **Simplified Engine**: Clean, maintainable similarity computation

### **Advanced Similarity Analysis**
- **Deep Learning**: ResNet50 feature extraction with cosine similarity
- **Perceptual Hashing**: Multi-hash comparison (pHash, dHash, aHash)
- **Computer Vision**: SSIM, histogram, edge detection, LBP analysis
- **Probabilistic**: GMM modeling with KL/JS divergence metrics

### **Comprehensive Visualizations**
- Interactive similarity scores with confidence intervals
- Method-by-method breakdown and comparison
- Heatmap visualizations for each analysis method
- Statistical analysis with uncertainty quantification
- Real-time progress tracking during analysis

### **Modern User Interface**
- Drag-and-drop image upload with red-gold theme
- Responsive design with smooth animations
- Tabbed interface for detailed analysis
- Real-time preview and progress indicators

## 🏗️ **Modular Architecture**

```
├── app.py                      # Flask application (uses SimilarityEngineV2)
├── similarity_engine_v2.py     # Modular similarity engine (ACTIVE)
├── similarity_engine.py        # Original monolithic version (REFERENCE)
├── visualizations.py           # Visualization generation
├── utils.py                    # Helper utilities
├── run.py                      # Application entry point
├── test_setup.py               # Setup verification script
├── requirements.txt            # Optimized dependencies
├── modules/                    # Reusable components
│   ├── __init__.py
│   ├── feature_extractors.py   # Deep learning, CV, hash extractors
│   └── report_generator.py     # Comprehensive reporting
├── static/
│   ├── css/style.css          # Red-gold theme styling
│   ├── js/app.js              # Frontend logic
│   └── uploads/               # Temporary storage
├── templates/
│   └── index.html             # Main interface
└── models/                     # Pre-trained model cache
```

**Note**: The project includes both `similarity_engine.py` (original) and `similarity_engine_v2.py` (modular). The application uses the modular V2 version for better maintainability and code reusability.
1. Deep Learning (35% weight)
**What:** ResNet50 extracts 2048-dimensional feature vectors
**How:** Cosine similarity: cos(θ) = (A·B)/(||A||·||B||)
**Why:** Captures semantic similarity (understands WHAT is in the image)

### 2. Perceptual Hashing (20% weight)
**What:** Generates 64-bit fingerprints using DCT
**How:** Hamming distance measures bit differences
**Why:** Fast O(1) comparison, robust to minor changes

### 3. Computer Vision (25% weight)
**What:** SSIM, Histogram, Canny Edges, LBP
**How:** Statistical comparison of structural features
**Why:** Captures low-level visual properties (colors, textures, edges)

### 4. Probabilistic (20% weight)
**What:** Gaussian Mixture Models with divergence metrics
**How:** KL/JS divergence + Wasserstein distance
**Why:** Statistical robustness, handles uncertainty

### Ensemble Approach
**Formula:** 0.35×DL + 0.20×PH + 0.25×CV + 0.20×Prob
**Confidence:** 1 - std_dev(scores)
**Result:** Robust similarity score with confidence interval



- **Existing library functions** instead of custom implementations
- **Modular design** for easy maintenance and extension

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd edl_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

#### 1. Deep Learning (Weight: 35%)

**Methods Used:**
- **ResNet50**: Pre-trained ImageNet model for feature extraction
- **Cosine Similarity**: Measures angle between feature vectors

**Theory:**
ResNet50 is a 50-layer deep convolutional neural network trained on ImageNet (14M images). It extracts high-dimensional feature vectors (2048 dimensions) that capture semantic content. Cosine similarity measures the angle between vectors: 
cos(θ) = (A·B)/(||A||·||B||). Values close to 1 indicate similar semantic content.

**Formula:**
```
similarity = cos(θ) = Σ(Ai × Bi) / (√Σ(Ai²) × √Σ(Bi²))
```

**Advantages**: Captures high-level semantic similarities, robust to transformations

---

#### 2. Perceptual Hashing (Weight: 20%)

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

**Advantages**: Fast O(1) comparison, robust to minor modifications, compression-resistant

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

#### 4. Probabilistic Analysis (Weight: 20%)

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

**Advantages**: Statistical robustness, handles uncertainty, captures global distributions

---

### Ensemble Scoring

The final similarity score is computed as a weighted average:

```python
final_score = (
    0.35 * deep_learning_score +
    0.20 * perceptual_hash_score +
    0.25 * cv_methods_score +
    0.20 * probabilistic_score
)
```

**Rationale for Weights:**
- Deep Learning (35%): Highest weight due to semantic understanding capability
- CV Methods (25%): Strong weight for structural and pixel-level features
- Perceptual Hash (20%): Fast, reliable baseline for overall similarity
- Probabilistic (20%): Statistical robustness for edge cases

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

1. **Overview**: Circular progress bar with method contributions
2. **Method Comparison**: Detailed breakdown of individual methods
3. **Image Comparison**: Side-by-side with difference visualization
4. **Statistical Analysis**: Distribution plots and confidence intervals
5. **Confidence Analysis**: Uncertainty quantification and reliability

## 📊 Quick Reference for Presentations

### Summary Table of Methods

| Method | Weight | Key Technique | Theory | Complexity |
|--------|--------|---------------|--------|------------|
| **Deep Learning** | 35% | ResNet50 + Cosine Similarity | CNN feature extraction from ImageNet | O(n) per image |
| **Perceptual Hash** | 20% | pHash, dHash, aHash | DCT-based fingerprinting + Hamming distance | O(1) comparison |
| **CV Methods** | 25% | SSIM, Histogram, Edge, LBP | Structural + statistical pixel analysis | O(n) per method |
| **Probabilistic** | 20% | GMM + KL/JS/Wasserstein | Distribution modeling + divergence metrics | O(n×k) for k components |

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
Final = 0.35×DL + 0.20×PH + 0.25×CV + 0.20×Prob
```

### Why This Combination Works

1. **Complementary Approaches**: Each method captures different aspects
   - Deep Learning: Semantic content (what objects are in the image)
   - Perceptual Hash: Overall structure (layout and composition)
   - CV Methods: Low-level features (colors, edges, textures)
   - Probabilistic: Statistical properties (distributions and patterns)

2. **Robust to Different Scenarios**:
   - Identical images → All methods score high
   - Slight modifications → Perceptual hash detects
   - Different lighting → SSIM and probabilistic handle well
   - Semantic similarity → Deep learning excels

3. **Confidence Through Consensus**:
   - High agreement across methods → High confidence
   - Disagreement → Lower confidence, indicates uncertainty
   - Statistical significance measures consistency

### Real-World Applications

- **Content Moderation**: Detect duplicate/similar content
- **Copyright Detection**: Find unauthorized image use
- **Image Search**: Find visually similar images
- **Quality Control**: Detect manufacturing defects
- **Medical Imaging**: Compare diagnostic scans

## Performance Considerations

- **Image Preprocessing**: Automatic resizing for large images
- **Model Caching**: Pre-trained models loaded once
- **Memory Management**: Efficient feature extraction and cleanup
- **Progress Tracking**: Real-time feedback during computation

## 📦 **Optimized Dependencies**

### **Core Framework**
- **Flask 3.0+**: Lightweight web framework
- **Werkzeug**: WSGI utilities

### **Machine Learning & Computer Vision**
- **TensorFlow 2.16+**: Deep learning models
- **OpenCV**: Computer vision operations
- **scikit-learn**: Machine learning algorithms
- **scikit-image**: Image processing

### **Scientific Computing**
- **NumPy**: Numerical computing
- **SciPy**: Scientific algorithms
- **Pillow**: Image handling

### **Visualization**
- **Matplotlib**: Plotting and charts
- **Seaborn**: Statistical visualizations
- **Plotly.js**: Interactive charts

 **Image Processing**
- **imagehash**: Perceptual hashing




