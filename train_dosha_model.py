import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoders = {}
        self.features = [
            'Body Frame', 'Type of Hair', 'Skin', 'Eating Habit',
            'Reaction under Adverse Situations', 'Sleep Pattern',
            'Body Temperature', 'Weather Conditions', 'Nature', 'Body Energy'
        ]
        self.target = 'Dosha'
        self.model_path = 'dosha_gb_model.joblib'
        self.encoders_path = 'dosha_encoders.joblib'

    def load_data(self):
        """Load and preprocess the dataset"""
        logger.info("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        return df

    def preprocess_data(self, df):
        """Preprocess the data and encode categorical variables"""
        logger.info("Preprocessing data...")
        df_processed = df.copy()
        
        # Initialize label encoders for each categorical column
        for column in self.features + [self.target]:
            self.label_encoders[column] = LabelEncoder()
            # Convert to string and handle NaN values
            df_processed[column] = df_processed[column].astype(str)
            # Fit the encoder
            self.label_encoders[column].fit(df_processed[column])
            # Transform the column
            df_processed[column] = self.label_encoders[column].transform(df_processed[column])
            
        return df_processed

    def train(self):
        """Train the model"""
        try:
            # Load and preprocess data
            df = self.load_data()
            df_processed = self.preprocess_data(df)
            
            # Prepare features and target
            X = df_processed[self.features]
            y = df_processed[self.target]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize and train the model
            logger.info("Training model...")
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8,
                max_features='sqrt',
                verbose=1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            logger.info(f"Training accuracy: {train_accuracy:.2%}")
            logger.info(f"Test accuracy: {test_accuracy:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def save_model(self):
        """Save the trained model and encoders"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.label_encoders, self.encoders_path)
            logger.info(f"Model saved to {self.model_path}")
            logger.info(f"Encoders saved to {self.encoders_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

def train_and_save_model():
    """Train and save the Dosha prediction model"""
    try:
        logger.info("Starting model training...")
        
        # Initialize the trainer with dataset path
        dataset_path = r"D:\Backup\ayurvedic_dosha_dataset (1).csv"
        trainer = ModelTrainer(dataset_path)
        
        # Train the model
        if trainer.train():
            logger.info("Model trained successfully")
            
            # Save the model and encoders
            if trainer.save_model():
                logger.info("Model and encoders saved successfully")
                return True
            
        logger.error("Model training or saving failed")
        return False
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model()
