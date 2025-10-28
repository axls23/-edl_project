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


