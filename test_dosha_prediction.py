import os
import sys
import joblib
from dosha_predictor import DoshaPredictor

def test_dosha_prediction():
    try:
        print("Testing Dosha Prediction Model...")
        
        # Initialize the predictor
        dataset_path = r"D:\Backup\ayurvedic_dosha_dataset (1).csv"
        predictor = DoshaPredictor(dataset_path)
        
        # Check if model exists, if not, train a new one
        if not predictor.load_model():
            print("Model not found. Training a new model...")
            predictor.train(tune_hyperparams=False)
        
        # Create a test input
        test_input = {
            'Body Frame': 'Medium',
            'Type of Hair': 'Normal,Medium',
            'Skin': 'Normal,Soft',
            'Body Energy': 'Steady,Slow',
            'Eating Habit': 'Regular',
            'Reaction under Adverse Situations': 'Calm',
            'Sleep Pattern': 'Heavy,Long',
            'Body Temperature': 'Cool',
            'Weather Conditions': 'Cool',
            'Nature': 'Calm,Patient',
            # Add other features with default values to avoid missing features
            'Color of Hair': 'Black',
            'Complexion': 'Fair',
            'Body Weight': 'Medium',
            'Nails': 'Smooth',
            'Size and Color of the Teeth': 'Medium,White',
            'Pace of Performing Work': 'Steady',
            'Mental Activity': 'Steady',
            'Memory': 'Good',
            'Mood': 'Stable',
            'Hunger': 'Moderate',
            'Joints': 'Normal',
            'Quality of Voice': 'Soft',
            'Dreams': 'Pleasant',
            'Social Relations': 'Stable',
            'Body Odor': 'Mild'
        }
        
        print("\nMaking prediction with test input:")
        predicted_dosha, confidence_scores = predictor.predict(test_input)
        
        if predicted_dosha:
            print(f"\nPredicted Dosha: {predicted_dosha}")
            print("Confidence Scores:")
            for dosha, score in confidence_scores.items():
                print(f"  - {dosha}: {score}%")
            print("\nTest completed successfully!")
            return True
        else:
            print("\nError: Failed to get prediction")
            return False
            
    except Exception as e:
        print(f"\nError during test: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dosha_prediction()
