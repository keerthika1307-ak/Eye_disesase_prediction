# 🚀 Complete Setup Guide

This guide will walk you through setting up the Eye Disease Prediction & Analysis System from scratch.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Directory Structure Setup](#directory-structure-setup)
3. [File Creation](#file-creation)
4. [Model Preparation](#model-preparation)
5. [Testing Locally](#testing-locally)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Python 3.9 or higher**: [Download Python](https://www.python.org/downloads/)
- **pip**: Comes with Python
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Text Editor**: VS Code, PyCharm, or any editor

### Check Installation
```bash
python --version  # Should show 3.9 or higher
pip --version     # Should show pip version
git --version     # Should show git version
```

## Directory Structure Setup

### Step 1: Create Project Directory
```bash
# Windows
mkdir eye-disease-prediction
cd eye-disease-prediction

# macOS/Linux
mkdir eye-disease-prediction
cd eye-disease-prediction
```

### Step 2: Create Subdirectories
```bash
# Windows
mkdir models
mkdir utils
mkdir sample_images
cd sample_images
mkdir cataract glaucoma diabetic_retinopathy normal
cd ..

# macOS/Linux
mkdir -p models utils sample_images/{cataract,glaucoma,diabetic_retinopathy,normal}
```

### Final Structure
```
eye-disease-prediction/
├── models/
├── utils/
└── sample_images/
    ├── cnv/
    ├── dme/
    ├── drusen/
    └── normal/
```

## File Creation

### Step 3: Create Python Files

#### Create utils/__init__.py
```bash
# Create empty file first
# Windows
type nul > utils\__init__.py

# macOS/Linux
touch utils/__init__.py
```

Then copy the content provided in the `utils/__init__.py` artifact.

#### Create utils/model_utils.py
```bash
# Windows
type nul > utils\model_utils.py

# macOS/Linux
touch utils/model_utils.py
```

Copy the content from `utils/model_utils.py` artifact.

#### Create utils/xai_utils.py
```bash
# Windows
type nul > utils\xai_utils.py

# macOS/Linux
touch utils/xai_utils.py
```

Copy the content from `utils/xai_utils.py` artifact.

#### Create utils/gemini_utils.py
```bash
# Windows
type nul > utils\gemini_utils.py

# macOS/Linux
touch utils/gemini_utils.py
```

Copy the content from `utils/gemini_utils.py` artifact.

#### Create app.py
```bash
# Windows
type nul > app.py

# macOS/Linux
touch app.py
```

Copy the content from `app.py` artifact.

#### Create requirements.txt
```bash
# Windows
type nul > requirements.txt

# macOS/Linux
touch requirements.txt
```

Copy the content from `requirements.txt` artifact.

#### Create .gitignore
```bash
# Windows
type nul > .gitignore

# macOS/Linux
touch .gitignore
```

Copy the content from `.gitignore` artifact.

#### Create README.md
```bash
# Windows
type nul > README.md

# macOS/Linux
touch README.md
```

Copy the content from `README.md` artifact.

## Model Preparation

### Step 4: Add Your Model

1. Locate your `fine_tuned_model.keras` file
2. Copy it to the `models/` directory:

```bash
# Windows
copy C:\path\to\your\fine_tuned_model.keras models\

# macOS/Linux
cp /path/to/your/fine_tuned_model.keras models/
```

### Step 5: Verify Model File
```bash
# Check if model exists
# Windows
dir models\fine_tuned_model.keras

# macOS/Linux
ls -lh models/fine_tuned_model.keras
```

## Installation

### Step 6: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 7: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will take several minutes as it downloads and installs all packages.

### Step 8: Verify Installation
```bash
pip list
```

Should show all installed packages including:
- streamlit
- tensorflow
- opencv-python
- matplotlib
- google-generativeai

## Testing Locally

### Step 9: Add Sample Images (Optional)
Add some test retinal images to the sample_images directories:


### Step 10: Run the Application
```bash
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`

### Step 11: Test Features

1. **Home Page**: Check if it loads correctly
2. **Upload & Predict**:
   - Upload a test image
   - Click "Analyze Image"
   - Verify prediction and visualizations
3. **Chat with AI Doctor**:
   - Ask a question
   - Verify Gemini AI response
4. **Other Pages**: Check all sections

## Deployment on Streamlit Cloud

### Step 12: Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Eye disease prediction system"
```

### Step 13: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New Repository"
3. Name it: `eye-disease-prediction`
4. Don't initialize with README (we already have one)
5. Click "Create repository"

### Step 14: Push to GitHub
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/eye-disease-prediction.git

# Push code
git branch -M main
git push -u origin main
```

### Step 15: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Fill in:
   - **Repository**: `YOUR_USERNAME/eye-disease-prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy"

### Step 16: Wait for Deployment
- First deployment takes 5-10 minutes
- You'll see logs in real-time
- Once complete, you'll get a public URL

### Step 17: Test Deployed App
Visit your app URL and test all features.

## Troubleshooting

### Issue 1: Model Not Loading
```
Error: Model file not found at models/fine_tuned_model.keras
```

**Solution:**
1. Check if model file exists: `ls models/` or `dir models\`
2. Verify file name is exactly `fine_tuned_model.keras`
3. Check file permissions

### Issue 2: Import Errors
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
1. Activate virtual environment
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python version: `python --version`

### Issue 3: Gemini API Error
```
Error: Invalid API key
```

**Solution:**
1. Check API key in `utils/gemini_utils.py`
2. Verify key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Check for rate limits

### Issue 4: Streamlit Not Starting
```
Command 'streamlit' not found
```

**Solution:**
1. Activate virtual environment
2. Reinstall streamlit: `pip install streamlit`
3. Try: `python -m streamlit run app.py`

### Issue 5: Memory Error on Streamlit Cloud
```
MemoryError or OOM killed
```

**Solution:**
1. Model might be too large
2. Consider model quantization
3. Use Streamlit Cloud paid tier for more memory

### Issue 6: Visualization Not Showing
```
Error in Grad-CAM generation
```

**Solution:**
1. Model architecture might not have conv layers
2. Update `last_conv_layer_name` in `xai_utils.py`
3. Check model summary: `model.summary()`

## Advanced Configuration

### Using Environment Variables for API Key

1. Create `.streamlit/secrets.toml`:
```bash
mkdir .streamlit
# Windows
type nul > .streamlit\secrets.toml
# macOS/Linux
touch .streamlit/secrets.toml
```

2. Add content:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

3. Update `utils/gemini_utils.py`:
```python
import streamlit as st
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "default_key")
```

4. Add to `.gitignore`:
```
.streamlit/secrets.toml
```

### Custom Model Configuration

If your model has different class names, update in `utils/model_utils.py`:
```python
CLASS_NAMES = ['YourClass1', 'YourClass2', 'YourClass3', 'YourClass4']
```

### Adjusting Image Size

If your model uses different input size, update in `utils/model_utils.py`:
```python
def preprocess_image(image, target_size=(256, 256)):  # Change from (224, 224)
    # ... rest of code
```

## Performance Optimization

### For Large Models
1. Use model quantization
2. Implement lazy loading
3. Cache predictions

### For Many Users
1. Use Streamlit Cloud paid tier
2. Implement request queuing
3. Add rate limiting

## Security Best Practices

1. **Never commit API keys**: Use environment variables
2. **Validate inputs**: Check file types and sizes
3. **Rate limiting**: Implement API request limits
4. **HTTPS only**: For deployed apps
5. **User authentication**: For production systems

## Next Steps

1. ✅ Test all features thoroughly
2. ✅ Add your own branding/logo
3. ✅ Customize color schemes
4. ✅ Add more sample images
5. ✅ Share with users
6. ✅ Collect feedback
7. ✅ Iterate and improve

## Getting Help

- **GitHub Issues**: Create an issue in your repository
- **Streamlit Forums**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Documentation**: 
  - [Streamlit Docs](https://docs.streamlit.io)
  - [TensorFlow Docs](https://www.tensorflow.org/api_docs)
  - [Gemini API Docs](https://ai.google.dev/docs)

## Conclusion

Congratulations! 🎉 You've successfully set up the Eye Disease Prediction & Analysis System. 

If you encounter any issues not covered here, check the main README.md or create a GitHub issue.

---

**Happy Coding! 👨‍💻👩‍💻**