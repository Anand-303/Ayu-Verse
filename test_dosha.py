import mysql.connector
from mysql_config import MYSQL_CONFIG

def check_dosha_table():
    """Check if the user_dosha table exists and has the correct structure"""
    try:
        # Create a database connection
        config = MYSQL_CONFIG.copy()
        config['database'] = 'ayuverse_db'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor(dictionary=True)
        
        # Check if the table exists
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'ayuverse_db' 
            AND TABLE_NAME = 'user_dosha'
        """)
        
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("Error: user_dosha table does not exist.")
            return False
        
        # Check the table structure
        cursor.execute("DESCRIBE user_dosha")
        columns = cursor.fetchall()
        
        required_columns = {
            'id', 'user_id', 'dosha_type', 'confidence_scores', 'test_date'
        }
        
        existing_columns = {col['Field'] for col in columns}
        
        if not required_columns.issubset(existing_columns):
            print("Error: user_dosha table is missing required columns.")
            print(f"Required: {required_columns}")
            print(f"Found: {existing_columns}")
            return False
            
        print("✓ user_dosha table exists and has the correct structure.")
        return True
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def test_dosha_prediction():
    """Test the dosha prediction functionality"""
    try:
        from dosha_predictor import DoshaPredictor
        import os
        import json
        
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ayurvedic_dosha_dataset (1).csv')
        predictor = DoshaPredictor(dataset_path)
        
        # Test with sample data
        test_data = {
            'Body Frame': 'Medium',
            'Type of Hair': 'Normal',
            'Color of Hair': 'Black',
            'Skin': 'Soft,Sweating',
            'Complexion': 'Pinkish',
            'Body Weight': 'Normal',
            'Nails': 'Pinkish',
            'Size and Color of the Teeth': 'Medium,White',
            'Pace of Performing Work': 'Medium',
            'Mental Activity': 'Stable',
            'Memory': 'Good Memory',
            'Sleep Pattern': 'Moderate',
            'Weather Conditions': 'Dislike Heat',
            'Reaction under Adverse Situations': 'Calm',
            'Mood': 'Changes Quickly',
            'Eating Habit': 'Proper Chewing',
            'Hunger': 'Regular',
            'Body Temperature': 'Normal',
            'Joints': 'Healthy',
            'Nature': 'Forgiving,Grateful',
            'Body Energy': 'Medium',
            'Quality of Voice': 'Clear',
            'Dreams': 'Water',
            'Social Relations': 'Ambivert',
            'Body Odor': 'Mild'
        }
        
        print("\nTesting dosha prediction...")
        
        # Train the model if it doesn't exist
        if not predictor.load_model():
            print("Training dosha prediction model...")
            predictor.train()
        
        # Make a prediction
        predicted_dosha, confidence_scores = predictor.predict(test_data)
        
        print(f"Predicted Dosha: {predicted_dosha}")
        print("Confidence Scores:")
        for dosha, score in confidence_scores.items():
            print(f"  {dosha}: {score}%")
            
        return True
        
    except Exception as e:
        print(f"Error testing dosha prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Dosha Functionality")
    print("==========================")
    
    # Check database table
    print("\n[1/2] Checking database setup...")
    db_ok = check_dosha_table()
    
    # Test prediction
    print("\n[2/2] Testing dosha prediction...")
    prediction_ok = test_dosha_prediction()
    
    # Print summary
    print("\nTest Summary")
    print("============")
    print(f"Database Setup: {'✓' if db_ok else '✗'}")
    print(f"Prediction Test: {'✓' if prediction_ok else '✗'}")
    
    if db_ok and prediction_ok:
        print("\n✅ All tests passed! The dosha test functionality should work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
