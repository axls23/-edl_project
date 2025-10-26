import os
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_IMAGE_SIZE = (1024, 1024)  # Resize large images

def validate_image(file):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False
    
    # Check file extension
    if '.' not in file.filename:
        return False
    
    extension = file.filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for similarity computation"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        height, width = image.shape[:2]
        if height > MAX_IMAGE_SIZE[0] or width > MAX_IMAGE_SIZE[1]:
            image = cv2.resize(image, MAX_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        return image
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / 255.0

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size for deep learning models"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def get_image_features(image):
    """Extract basic image features for analysis"""
    features = {}
    
    # Color statistics
    features['mean_color'] = np.mean(image, axis=(0, 1))
    features['std_color'] = np.std(image, axis=(0, 1))
    
    # Brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features['brightness'] = np.mean(gray)
    features['contrast'] = np.std(gray)
    
    # Image dimensions
    features['height'], features['width'] = image.shape[:2]
    features['aspect_ratio'] = features['width'] / features['height']
    
    return features

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval for similarity scores"""
    if len(scores) < 2:
        return scores[0], 0
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Simple confidence interval (can be improved with proper statistical methods)
    margin = 1.96 * std_score / np.sqrt(len(scores))  # 95% CI
    return mean_score, margin

def format_percentage(score):
    """Format similarity score as percentage"""
    return f"{score * 100:.1f}%"


