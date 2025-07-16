import logging
import time
from dosha_predictor import DoshaPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction():
    """Test the Dosha predictor with sample input"""
    try:
        logger.info("Testing Dosha predictor...")
        start_time = time.time()
        
        # Initialize predictor
        predictor = DoshaPredictor(
            model_path='dosha_gb_model.joblib',
            encoders_path='dosha_encoders.joblib'
        )
        
        # Test model loading
        logger.info("Loading model...")
        if not predictor.load_model():
            logger.error("Failed to load model")
            return False
            
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Test prediction
        test_input = {
            'Body Frame': 'Medium',
            'Type of Hair': 'Normal,Medium',
            'Skin': 'Normal',
            'Eating Habit': 'Regular',
            'Reaction under Adverse Situations': 'Calm',
            'Sleep Pattern': 'Normal',
            'Body Temperature': 'Normal',
            'Weather Conditions': 'Moderate',
            'Nature': 'Balanced',
            'Body Energy': 'Steady'
        }
        
        logger.info("Making prediction...")
        prediction_start = time.time()
        predicted_dosha, confidence_scores = predictor.predict(test_input)
        prediction_time = time.time() - prediction_start
        
        logger.info(f"Prediction completed in {prediction_time:.4f} seconds")
        logger.info(f"Predicted Dosha: {predicted_dosha}")
        logger.info(f"Confidence Scores: {confidence_scores}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        logger.info("✅ Test completed successfully")
    else:
        logger.error("❌ Test failed")
