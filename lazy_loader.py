"""
This module provides lazy loading for the prediction model to improve application startup time.
The model will be loaded only when needed, not at application startup.
"""
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model
_herbal_model = None
_preprocessing_info = None
_class_names = None

@lru_cache(maxsize=1)
def get_prediction_model():
    """
    Load and cache the prediction model and related data.
    Uses LRU cache to ensure the model is loaded only once.
    """
    global _herbal_model, _preprocessing_info, _class_names
    
    if _herbal_model is None:
        try:
            from predict import load_model_and_preprocessing
            import numpy as np
            
            logger.info("Loading prediction model...")
            _herbal_model, _preprocessing_info, _class_names = load_model_and_preprocessing()
            
            if _herbal_model is not None:
                logger.info("Prediction model loaded successfully")
            else:
                logger.warning("Failed to load prediction model")
                
        except Exception as e:
            logger.error(f"Error loading prediction model: {str(e)}")
            _herbal_model = None
            _preprocessing_info = None
            _class_names = None
    
    return _herbal_model, _preprocessing_info, _class_names

def clear_model_cache():
    """Clear the cached model and related data"""
    global _herbal_model, _preprocessing_info, _class_names
    _herbal_model = None
    _preprocessing_info = None
    _class_names = None
    get_prediction_model.cache_clear()
    logger.info("Cleared prediction model cache")
