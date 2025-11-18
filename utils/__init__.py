"""
Eye Disease Prediction Utilities Package

This package contains utility modules for:
- Model loading and prediction (model_utils.py)
- Explainable AI visualizations (xai_utils.py)
- Gemini AI integration (gemini_utils.py)
"""

import importlib
model_utils = importlib.import_module('.model_utils', __name__)
xai_utils = importlib.import_module('.xai_utils', __name__)
gemini_utils = importlib.import_module('.gemini_utils', __name__)

__version__ = '1.0.0'
__author__ = 'Eye Disease Prediction Team'

__all__ = [
    'model_utils',
    'xai_utils',
    'gemini_utils',
]
