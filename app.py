from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from similarity_engine_v2 import SimilarityEngineV2
from visualizations import VisualizationGenerator
from utils import validate_image, preprocess_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
similarity_engine = SimilarityEngineV2()
viz_generator = VisualizationGenerator()

# Store for job results (in production, use Redis or database)
job_results = {}

@app.route('/')
def index():
    """Main page with drag-drop interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image uploads and trigger similarity computation"""
    image1_path = None
    image2_path = None

    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both images are required'}), 400
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        if image1.filename == '' or image2.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate images
        if not validate_image(image1) or not validate_image(image2):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save images temporarily
        filename1 = secure_filename(f"{job_id}_1_{image1.filename}")
        filename2 = secure_filename(f"{job_id}_2_{image2.filename}")
        
        image1_path = os.path.join('static', 'uploads', filename1)
        image2_path = os.path.join('static', 'uploads', filename2)
        
        image1.save(image1_path)
        image2.save(image2_path)
        
        # Process images
        processed_img1 = preprocess_image(image1_path)
        processed_img2 = preprocess_image(image2_path)
        
        # Compute similarity
        results = similarity_engine.compute_similarity(processed_img1, processed_img2)
        
        # Generate visualizations
        visualizations = viz_generator.generate_visualizations(
            processed_img1, processed_img2, results, image1_path, image2_path
        )
        
        # Cleanup matplotlib resources
        viz_generator.cleanup()
        
        # Store results
        job_results[job_id] = {
            'results': results,
            'visualizations': visualizations,
            'image1_path': image1_path,
            'image2_path': image2_path
        }
        
        # Clean up temporary files
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)

        return jsonify({
            'job_id': job_id,
            'results': results,
            'visualizations': visualizations
        })
        
    except Exception as e:
        # Clean up files on error
        if image1_path and os.path.exists(image1_path):
            os.remove(image1_path)
        if image2_path and os.path.exists(image2_path):
            os.remove(image2_path)

        # Log the full error for debugging
        import traceback
        app.logger.error(f"Error processing images: {str(e)}")
        app.logger.error(traceback.format_exc())

        return jsonify({'error': str(e)}), 500

@app.route('/results/<job_id>')
def get_results(job_id):
    """Get results for a specific job"""
    if job_id not in job_results:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_results[job_id])

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
