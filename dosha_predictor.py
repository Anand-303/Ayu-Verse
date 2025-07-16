import os
import logging
import joblib
from typing import Dict, Any, Optional, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoshaPredictor:
    def __init__(self, model_path: str = None, encoders_path: str = None):
        """Initialize the Dosha predictor with paths to pre-trained model and encoders"""
        self.model_path = model_path or 'dosha_gb_model.joblib'
        self.encoders_path = encoders_path or 'dosha_encoders.joblib'
        self.model = None
        self.label_encoders = None
        self._is_loaded = False
        self.features = [
            'Body Frame', 'Type of Hair', 'Skin', 'Eating Habit',
            'Reaction under Adverse Situations', 'Sleep Pattern',
            'Body Temperature', 'Weather Conditions', 'Nature', 'Body Energy'
        ]
        self.target = 'Dosha'

    def load_model(self) -> bool:
        """Load the pre-trained model and encoders"""
        if self._is_loaded:
            return True
            
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.encoders_path):
                logger.error(f"Model files not found at {self.model_path} and {self.encoders_path}")
                return False
                
            start_time = time.time()
            self.model = joblib.load(self.model_path)
            self.label_encoders = joblib.load(self.encoders_path)
            self._is_loaded = True
            
            logger.info(f"Model and encoders loaded in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.label_encoders = None
            self._is_loaded = False
            return False

    def predict(self, input_data: Dict[str, Any], timeout_seconds: int = 10) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """
        Predict dosha for input data
        
        Args:
            input_data: Dictionary of input features
            timeout_seconds: Maximum time allowed for prediction
            
        Returns:
            Tuple of (predicted_dosha, confidence_scores)
        """
        if not self._is_loaded and not self.load_model():
            raise RuntimeError("Failed to load prediction model")
            
        start_time = time.time()
        
        try:
            # Prepare input features with default values
            features = {}
            for feature in self.features:
                features[feature] = input_data.get(feature, '')
            
            # Encode categorical features
            encoded_features = {}
            for feature, value in features.items():
                if feature in self.label_encoders:
                    try:
                        # Handle unseen categories by using the most frequent category
                        if value not in self.label_encoders[feature].classes_:
                            logger.warning(f"Unseen category '{value}' for feature '{feature}'. Using most frequent category.")
                            value = self.label_encoders[feature].classes_[0]
                        encoded_features[feature] = self.label_encoders[feature].transform([value])[0]
                    except Exception as e:
                        logger.error(f"Error encoding feature {feature}: {str(e)}")
                        encoded_features[feature] = 0  # Default to first category
            
            # Convert to model input format
            X = [encoded_features.get(f, 0) for f in self.features]
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("Prediction timed out")
            
            # Make prediction
            probas = self.model.predict_proba([X])[0]
            classes = self.model.classes_
            
            # Get confidence scores
            confidence_scores = {
                dosha: round(score * 100, 2)
                for dosha, score in zip(classes, probas)
            }
            
            # Get predicted dosha (class with highest probability)
            predicted_dosha = classes[probas.argmax()]
            
            logger.info(f"Prediction completed in {time.time() - start_time:.4f} seconds")
            return predicted_dosha, confidence_scores
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

if __name__ == "__main__":
    dataset_path = r"D:\Backup\ayurvedic_dosha_dataset (1).csv"
    predictor = DoshaPredictor()
    
    # Example prediction
    example_input = {
        'Body Frame': 'Medium',
        'Type of Hair': 'Normal',
        'Color of Hair': 'Black',
        'Skin': 'Soft,Sweating',
        'Complexion': 'Pinkish',
        'Body Weight': 'Normal',
        'Nails': 'Pinkish',
        'Size and Color of the Teeth': 'Medium,Yellowish',
        'Pace of Performing Work': 'Medium',
        'Mental Activity': 'Stable',
        'Memory': 'Good Memory',
        'Sleep Pattern': 'Moderate',
        'Weather Conditions': 'Dislike Heat',
        'Reaction under Adverse Situations': 'Calm',
        'Mood': 'Changes Quickly',
        'Eating Habit': 'Irregular Chewing',
        'Hunger': 'Irregular',
        'Body Temperature': 'Normal',
        'Joints': 'Healthy',
        'Nature': 'Forgiving,Grateful',
        'Body Energy': 'Medium',
        'Quality of Voice': 'Deep',
        'Dreams': 'Sky',
        'Social Relations': 'Ambivert',
        'Body Odor': 'Mild'
    }
    
    print("\nMaking example prediction...")
    dosha, confidence = predictor.predict(example_input)
    if dosha:
        print(f"\nPredicted Dosha: {dosha}")
        print("Confidence Scores:")
        for d, c in confidence.items():
            print(f"  {d}: {c}%")
