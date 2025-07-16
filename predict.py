import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model_and_preprocessing():
    """Load the trained model and preprocessing information"""
    try:
        # Loading model
        logger.info("Loading model...")
        model = load_model("herbal_remedy_text_model.h5")
        logger.info("Model loaded successfully")
        
        # Loading preprocessing info
        logger.info("Loading preprocessing information...")
        with open('preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)
        logger.info("Preprocessing information loaded")
        
        # Loading class names
        logger.info("Loading class names...")
        class_names = np.load('class_names.npy', allow_pickle=True)
        logger.info(f"Loaded {len(class_names)} class names")
        
        return model, preprocessing_info, class_names
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading model or preprocessing information: {e}")
        return None, None, None

def predict_for_new_data(input_data, model, preprocessing_info, class_names):
    """Make predictions for new herbal remedy data"""
    try:
        logger.debug(f"Input data: {input_data}")
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Extract preprocessing info
        feature_columns = preprocessing_info['feature_columns']
        categorical_cols = preprocessing_info.get('categorical_cols', [])
        numerical_cols = preprocessing_info.get('numerical_cols', [])
        text_cols = preprocessing_info.get('text_cols', [])
        
        logger.debug(f"Feature columns: {feature_columns}")
        
        # Fill missing values
        input_df.fillna("Unknown", inplace=True)
        
        # Ensure all required columns exist
        for col in feature_columns:
            if col not in input_df.columns:
                logger.warning(f"Column '{col}' not found in input data. Adding with default value 'Unknown'")
                input_df[col] = "Unknown"
        
        # Process categorical features
        processed_features = pd.get_dummies(input_df[feature_columns])
        logger.debug(f"Processed features shape: {processed_features.shape}")
        
        # Make prediction
        logger.debug("Making prediction...")
        prediction = model.predict(processed_features)
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        
        results = []
        for i in top_indices:
            results.append({
                'predicted_class': class_names[i],
                'confidence': float(prediction[0][i])
            })
        
        logger.debug(f"Prediction results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

if __name__ == "__main__":
    model, preprocessing_info, class_names = load_model_and_preprocessing()
    
    if model is not None:
        print("\n--- Herbal Remedy Prediction System ---")
        print("Enter information about an herb to get predictions")
        
        example_input = {}
        
        for col in preprocessing_info['feature_columns']:
            value = input(f"Enter {col} (or press Enter to skip): ")
            if value: 
                example_input[col] = value
        
        try:
            # Make prediction
            predictions = predict_for_new_data(example_input, model, preprocessing_info, class_names)
            
            # Display results
            print("\nPrediction Results:")
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['predicted_class']} (Confidence: {pred['confidence']:.2%})")
        except Exception as e:
            print(f"Error making prediction: {e}")