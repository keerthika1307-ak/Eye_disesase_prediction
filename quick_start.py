"""
Quick Start Setup Script for Eye Disease Prediction System
This script helps verify your setup and identifies any issues.
"""

import sys
import os
import importlib.util

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print_success(f"Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} is not compatible. Need Python 3.9+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{package_name} (version: {version})")
            return True
        except Exception as e:
            print_warning(f"{package_name} found but error importing: {e}")
            return False
    else:
        print_error(f"{package_name} not installed")
        return False

def check_required_packages():
    """Check all required packages"""
    print_header("Checking Required Packages")
    
    packages = [
        ('streamlit', 'streamlit'),
        ('tensorflow', 'tensorflow'),
        ('keras', 'keras'),
        ('PIL/Pillow', 'PIL'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('numpy', 'numpy'),
        ('google-generativeai', 'google.generativeai'),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    return all_installed

def check_directory_structure():
    """Check if directory structure is correct"""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'models',
        'utils',
        'sample_images',
        'sample_images/cnv',
        'sample_images/dme',
        'sample_images/drusen',
        'sample_images/normal'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print_success(f"Directory exists: {directory}/")
        else:
            print_error(f"Directory missing: {directory}/")
            all_exist = False
    
    return all_exist

def check_required_files():
    """Check if required files exist"""
    print_header("Checking Required Files")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'utils/__init__.py',
        'utils/model_utils.py',
        'utils/xai_utils.py',
        'utils/gemini_utils.py'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print_success(f"File exists: {file} ({size} bytes)")
        else:
            print_error(f"File missing: {file}")
            all_exist = False
    
    return all_exist

def check_model_file():
    """Check if model file exists"""
    print_header("Checking Model File")
    
    # Primary expected model path (user-specified)
    primary_path = 'model/fine_tuned_final_model.h5'
    # Backward-compatible fallback path
    fallback_path = 'models/fine_tuned_final_model.keras'

    existing_path = None
    if os.path.isfile(primary_path):
        existing_path = primary_path
    elif os.path.isfile(fallback_path):
        existing_path = fallback_path

    if existing_path:
        size = os.path.getsize(existing_path) / (1024 * 1024)  # Convert to MB
        print_success(f"Model file exists: {existing_path}")
        print(f"   Size: {size:.2f} MB")
        
        if size > 100:
            print_warning(f"Model size is large ({size:.2f} MB). May cause issues on free Streamlit Cloud")
        
        return True
    else:
        print_error("Model file not found.")
        print("   Please place your model at one of the following paths:")
        print(f"   - {primary_path}  (recommended)")
        print(f"   - {fallback_path} (legacy fallback)")
        return False

def check_sample_images():
    """Check sample images"""
    print_header("Checking Sample Images (Optional)")
    
    image_dirs = [
        'sample_images/cnv',
        'sample_images/dme',
        'sample_images/drusen',
        'sample_images/normal'
    ]
    
    for directory in image_dirs:
        if os.path.isdir(directory):
            files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                print_success(f"{directory}: {len(files)} image(s)")
            else:
                print_warning(f"{directory}: No images found (optional)")

def test_imports():
    """Test importing project modules"""
    print_header("Testing Project Imports")
    
    try:
        from utils import model_utils
        print_success("utils.model_utils imported successfully")
    except Exception as e:
        print_error(f"Failed to import utils.model_utils: {e}")
        return False
    
    try:
        from utils import xai_utils
        print_success("utils.xai_utils imported successfully")
    except Exception as e:
        print_error(f"Failed to import utils.xai_utils: {e}")
        return False
    
    try:
        from utils import gemini_utils
        print_success("utils.gemini_utils imported successfully")
    except Exception as e:
        print_error(f"Failed to import utils.gemini_utils: {e}")
        return False
    
    return True

def check_gemini_api():
    """Check Gemini API configuration"""
    print_header("Checking Gemini API Configuration")
    
    try:
        from utils import gemini_utils
        if hasattr(gemini_utils, 'GEMINI_API_KEY'):
            api_key = gemini_utils.GEMINI_API_KEY
            if api_key and len(api_key) > 10:
                masked_key = api_key[:10] + "..." + api_key[-4:]
                print_success(f"Gemini API key configured: {masked_key}")
                return True
            else:
                print_error("Gemini API key is not properly configured")
                return False
        else:
            print_error("Gemini API key not found in gemini_utils.py")
            return False
    except Exception as e:
        print_error(f"Error checking Gemini API: {e}")
        return False

def provide_recommendations():
    """Provide setup recommendations"""
    print_header("Recommendations")
    
    print("\n📋 Setup Recommendations:")
    print("1. Ensure all required packages are installed:")
    print("   pip install -r requirements.txt")
    print("\n2. Place your model file in models/fine_tuned_model.keras")
    print("\n3. (Optional) Add test images to sample_images/ subdirectories")
    print("\n4. Test locally before deploying:")
    print("   streamlit run app.py")
    print("\n5. For deployment, refer to SETUP_GUIDE.md and DEPLOYMENT_CHECKLIST.md")

def main():
    """Main function"""
    print("\n" + "🔬 Eye Disease Prediction System - Setup Verification".center(60))
    print("This script will check if your setup is ready\n")
    
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_required_packages(),
        "Directory Structure": check_directory_structure(),
        "Required Files": check_required_files(),
        "Model File": check_model_file(),
        "Project Imports": test_imports(),
        "Gemini API": check_gemini_api()
    }
    
    # Check sample images (informational only)
    check_sample_images()
    
    # Summary
    print_header("Setup Verification Summary")
    
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    if passed == total:
        print_success("\n🎉 Congratulations! Your setup is complete and ready!")
        print("\nNext steps:")
        print("1. Run locally: streamlit run app.py")
        print("2. Test all features")
        print("3. Follow DEPLOYMENT_CHECKLIST.md for deployment")
    else:
        print_error(f"\n⚠️  Setup incomplete: {total - passed} issue(s) found")
        print("\nPlease fix the issues above and run this script again.")
        provide_recommendations()
    
    print("\n" + "="*60)
    print("For detailed setup instructions, see SETUP_GUIDE.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()