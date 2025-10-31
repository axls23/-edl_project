// Image Similarity Analyzer - Frontend JavaScript

class ImageSimilarityApp {
    constructor() {
        this.image1 = null;
        this.image2 = null;
        this.results = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input change events
        document.getElementById('image1').addEventListener('change', (e) => this.handleImageUpload(e, 1));
        document.getElementById('image2').addEventListener('change', (e) => this.handleImageUpload(e, 2));

        // Drag and drop events
        this.setupDragAndDrop(1);
        this.setupDragAndDrop(2);
    }

    setupDragAndDrop(imageNumber) {
        const uploadZone = document.getElementById(`uploadZone${imageNumber}`);
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageUpload({ target: { files: [files[0]] } }, imageNumber);
            }
        });
    }

    handleImageUpload(event, imageNumber) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            alert('Image size must be less than 16MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.displayImagePreview(e.target.result, imageNumber);
            this.storeImage(file, imageNumber);
            this.updateAnalyzeButton();
        };
        reader.readAsDataURL(file);
    }

    displayImagePreview(imageSrc, imageNumber) {
        const uploadZone = document.getElementById(`uploadZone${imageNumber}`);
        const preview = document.getElementById(`preview${imageNumber}`);
        const img = document.getElementById(`img${imageNumber}`);

        uploadZone.classList.add('has-image');
        preview.style.display = 'flex';
        img.src = imageSrc;
    }

    storeImage(file, imageNumber) {
        if (imageNumber === 1) {
            this.image1 = file;
        } else {
            this.image2 = file;
        }
    }

    updateAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const canAnalyze = this.image1 && this.image2;
        analyzeBtn.disabled = !canAnalyze;
        analyzeBtn.classList.toggle('ready', !!canAnalyze);
    }

    async analyzeImages() {
        if (!this.image1 || !this.image2) {
            alert('Please upload both images before analyzing.');
            return;
        }
        addLoadingAnimation();
        this.showLoading();
        this.simulateProgress();

        try {
            const formData = new FormData();
            formData.append('image1', this.image1);
            formData.append('image2', this.image2);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.results = data;
            this.displayResults();
            
        } catch (error) {
            console.error('Analysis error:', error);
            alert(`Analysis failed: ${error.message}`);
            this.hideLoading();
        } finally {
            removeLoadingAnimation();
        }
    }

    showLoading() {
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
    }

    simulateProgress() {
        const progressFill = document.getElementById('progressFill');
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
            }
            progressFill.style.width = progress + '%';
        }, 200);
    }

    displayResults() {
        this.hideLoading();
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');

        this.displayMainResults();
        this.loadVisualizations();
        this.renderInsightsPanel();
        this.loadDiagnostics();
    }

    displayMainResults() {
        const mainResults = document.getElementById('mainResults');
        const ensemble = this.results.results.ensemble;
        const score = ensemble.score;
        const confidence = ensemble.confidence;

        mainResults.innerHTML = `
            <div class="similarity-score">
                <div class="score-circle" style="--score: ${score * 360}deg">
                    <div class="score-text">${(score * 100).toFixed(1)}%</div>
                </div>
                <h3>Overall Similarity</h3>
                <p class="confidence-text">Confidence: ${(confidence * 100).toFixed(1)}%</p>
            </div>
        `;
    }

    loadVisualizations() {
        const visualizations = this.results.visualizations;
        
        // Load each visualization into its respective tab
        if (visualizations.overview) {
            document.getElementById('overviewViz').innerHTML = 
                `<img src="${visualizations.overview}" alt="Overview Visualization">`;
        }

        if (visualizations.method_comparison) {
            document.getElementById('methodsViz').innerHTML = 
                `<img src="${visualizations.method_comparison}" alt="Method Comparison">`;
        }

        if (visualizations.side_by_side) {
            document.getElementById('comparisonViz').innerHTML = 
                `<img src="${visualizations.side_by_side}" alt="Image Comparison">`;
        }

        if (visualizations.detailed_breakdown) {
            document.getElementById('statisticsViz').innerHTML = 
                `<img src="${visualizations.detailed_breakdown}" alt="Statistical Breakdown">`;
        }

        if (visualizations.confidence) {
            document.getElementById('confidenceViz').innerHTML = 
                `<img src="${visualizations.confidence}" alt="Confidence Analysis">`;
        }
    }

    renderInsightsPanel() {
        const panel = document.getElementById('insightsPanel');
        if (!panel || !this.results) return;

        const ensemble = this.results.results?.ensemble || {};
        const report = this.results.results?.analysis_report || {};
        const dl = this.results.results?.deep_learning || {};
        const dld = dl.details || {};

        const score = ensemble.score ?? 0;
        const confidence = ensemble.confidence ?? 0;

        const level = score >= 0.8 ? 'High' : score >= 0.6 ? 'Medium' : score >= 0.4 ? 'Low' : 'Very Low';
        const confLevel = confidence >= 0.8 ? 'High' : confidence >= 0.5 ? 'Medium' : 'Low';

        const cats1 = (dld.categories1 || []).slice(0,3).map(([c,p]) => `${c} (${(p*100).toFixed(1)}%)`).join(', ') || '—';
        const cats2 = (dld.categories2 || []).slice(0,3).map(([c,p]) => `${c} (${(p*100).toFixed(1)}%)`).join(', ') || '—';
        const overlap = dld.category_overlap ?? 0;

        const recs = (report.recommendations || []).slice(0,2);

        const weightMap = { deep_learning: 0.50, cv_methods: 0.25, probabilistic: 0.10, perceptual_hash: 0.15 };
        const methods = ['deep_learning','cv_methods','probabilistic','perceptual_hash'];
        let dominantMethod = '—';
        let maxContribution = -1;
        methods.forEach(m => {
            const s = this.results.results?.[m]?.score;
            if (typeof s === 'number') {
                const contrib = s * (weightMap[m] || 0);
                if (contrib > maxContribution) { maxContribution = contrib; dominantMethod = m.replace('_',' '); }
            }
        });

        panel.innerHTML = `
            <div class="insights">
                <div class="badges">
                    <span class="badge">Similarity: ${level}</span>
                    <span class="badge">Confidence: ${confLevel}</span>
                </div>
                <div class="insight-row"><strong>Top categories (Image 1):</strong> ${cats1}</div>
                <div class="insight-row"><strong>Top categories (Image 2):</strong> ${cats2}</div>
                <div class="insight-row"><strong>Category overlap (top-3):</strong> ${overlap}</div>
                <div class="insight-row"><strong>Dominant method:</strong> ${dominantMethod}</div>
                <div class="insight-row"><strong>Recommendations:</strong> ${recs.map(r=>`<span class="recommendation">${r}</span>`).join(' ') || '—'}</div>
            </div>
        `;
    }

    loadDiagnostics() {
        const viz = this.results.visualizations || {};
        const diag = document.getElementById('diagnosticsViz');
        if (!diag) return;

        const blocks = [];
        if (viz.gradcam_overlays) {
            blocks.push(`<div class="diag-block"><h4>Grad-CAM (semantic focus)</h4><img src="${viz.gradcam_overlays}" alt="Grad-CAM"></div>`);
        }
        if (viz.ssim_heatmap) {
            blocks.push(`<div class="diag-block"><h4>SSIM Heatmap</h4><img src="${viz.ssim_heatmap}" alt="SSIM Heatmap"></div>`);
        }
        if (viz.edge_lbp_diffs) {
            blocks.push(`<div class="diag-block"><h4>Edge & LBP Differences</h4><img src="${viz.edge_lbp_diffs}" alt="Edge LBP Diffs"></div>`);
        }
        if (viz.method_attribution) {
            blocks.push(`<div class="diag-block"><h4>Method Attribution</h4><img src="${viz.method_attribution}" alt="Attribution"></div>`);
        }
        if (viz.hist_overlay) {
            blocks.push(`<div class="diag-block"><h4>RGB Histogram Overlay</h4><img src="${viz.hist_overlay}" alt="Histogram Overlay"></div>`);
        }
        if (viz.tsne_features) {
            blocks.push(`<div class="diag-block"><h4>Feature-space (t-SNE)</h4><img src="${viz.tsne_features}" alt="t-SNE"></div>`);
        }

        diag.innerHTML = blocks.join('');
    }

    resetAnalysis() {
        // Reset all state
        this.image1 = null;
        this.image2 = null;
        this.results = null;

        // Reset UI
        document.getElementById('uploadSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'none';

        // Reset upload zones
        this.resetUploadZone(1);
        this.resetUploadZone(2);

        // Reset analyze button
        this.updateAnalyzeButton();

        // Reset file inputs
        document.getElementById('image1').value = '';
        document.getElementById('image2').value = '';
    }

    resetUploadZone(imageNumber) {
        const uploadZone = document.getElementById(`uploadZone${imageNumber}`);
        const preview = document.getElementById(`preview${imageNumber}`);
        
        uploadZone.classList.remove('has-image');
        preview.style.display = 'none';
    }
}

// Tab functionality
function showTab(tabName) {
    // Hide all tab panels
    const panels = document.querySelectorAll('.tab-panel');
    panels.forEach(panel => panel.classList.remove('active'));

    // Remove active class from all tab buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(button => button.classList.remove('active'));

    // Show selected tab panel
    document.getElementById(tabName).classList.add('active');

    // Activate corresponding button
    event.target.classList.add('active');
}

// Global functions for HTML onclick handlers
function removeImage(imageNumber) {
    const app = window.imageSimilarityApp;
    if (imageNumber === 1) {
        app.image1 = null;
    } else {
        app.image2 = null;
    }
    
    app.resetUploadZone(imageNumber);
    app.updateAnalyzeButton();
}

function analyzeImages() {
    window.imageSimilarityApp.analyzeImages();
}

function resetAnalysis() {
    window.imageSimilarityApp.resetAnalysis();
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.imageSimilarityApp = new ImageSimilarityApp();
    initThemeToggle();
});

// Add some utility functions for enhanced UX
function addLoadingAnimation() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    analyzeBtn.disabled = true;
}

function removeLoadingAnimation() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    
    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
    analyzeBtn.disabled = false;
}

// Enhanced error handling
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff4444;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

// Theme toggle (dark/light)
function initThemeToggle() {
    const root = document.documentElement;
    const btn = document.getElementById('themeToggle');
    const saved = localStorage.getItem('theme') || 'dark';
    applyTheme(saved);

    if (btn) {
        btn.addEventListener('click', () => {
            const current = (localStorage.getItem('theme') || 'dark');
            const next = current === 'dark' ? 'light' : 'dark';
            applyTheme(next);
            localStorage.setItem('theme', next);
        });
    }

    function applyTheme(mode) {
        if (mode === 'light') {
            root.setAttribute('data-theme', 'light');
            if (btn) btn.textContent = '☀️ Light';
        } else {
            root.removeAttribute('data-theme');
            if (btn) btn.textContent = '🌙 Dark';
        }
    }
}


