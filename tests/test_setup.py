#!/usr/bin/env python3
"""
Test script to verify the Image Similarity Analyzer setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import flask
        print("OK Flask imported successfully")
    except ImportError:
        print("FAIL Flask not found. Run: pip install flask")
        return False
    
    try:
        import numpy
        print("OK NumPy imported successfully")
    except ImportError:
        print("FAIL NumPy not found. Run: pip install numpy")
        return False
    
    try:
        import cv2
        print("OK OpenCV imported successfully")
    except ImportError:
        print("FAIL OpenCV not found. Run: pip install opencv-python")
        return False
    
    try:
        import matplotlib
        print("OK Matplotlib imported successfully")
    except ImportError:
        print("FAIL Matplotlib not found. Run: pip install matplotlib")
        return False
    
    try:
        import tensorflow
        print("OK TensorFlow imported successfully")
    except ImportError:
        print("FAIL TensorFlow not found. Run: pip install tensorflow")
        return False
    
    try:
        import imagehash
        print("OK ImageHash imported successfully")
    except ImportError:
        print("FAIL ImageHash not found. Run: pip install imagehash")
        return False
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'similarity_engine.py',
        'visualizations.py',
        'utils.py',
        'requirements.txt',
        'run.py',
        'README.md',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"OK {file_path}")
        else:
            print(f"FAIL {file_path} - Missing!")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_directories():
    """Test if all required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'static',
        'static/css',
        'static/js',
        'static/uploads',
        'templates',
        'models'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"OK {dir_path}/")
        else:
            print(f"FAIL {dir_path}/ - Missing!")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0

def main():
    """Run all tests"""
    print("Image Similarity Analyzer - Setup Test")
    print("=" * 50)
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test directories
    dirs_ok = test_directories()
    
    # Test imports (optional - packages might not be installed yet)
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"File Structure: {'PASS' if files_ok else 'FAIL'}")
    print(f"Directories: {'PASS' if dirs_ok else 'FAIL'}")
    print(f"Package Imports: {'PASS' if imports_ok else 'WARNING (packages not installed)'}")
    
    if files_ok and dirs_ok:
        print("\nSetup looks good! You can now:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the application: python run.py")
        print("3. Open browser to: http://localhost:5000")
    else:
        print("\nSetup incomplete. Please check missing files/directories.")
    
    return files_ok and dirs_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
