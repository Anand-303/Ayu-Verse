import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash
from mysql_config import MYSQL_CONFIG

def create_mysql_database():
    # First connect without database selected
    config = MYSQL_CONFIG.copy()
    if 'database' in config:
        del config['database']
    
    try:
        # Connect to MySQL server
        print("Connecting to MySQL server...")
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor(buffered=True)
        
        # Create database if it doesn't exist
        print("Creating database if it doesn't exist...")
        cursor.execute("DROP DATABASE IF EXISTS ayuverse_db")
        cursor.execute("CREATE DATABASE ayuverse_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute("USE ayuverse_db")
        
        # Create users table
        print("Creating users table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chat_sessions table
        print("Creating chat_sessions table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(100) NOT NULL,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users(email) ON DELETE CASCADE
            )
        ''')
        
        # Create chat_history table
        print("Creating chat_history table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                message TEXT NOT NULL,
                is_bot BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        ''')

        # Create remedies table
        print("Creating remedies table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS remedies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                condition_name VARCHAR(100) NOT NULL,
                symptoms TEXT,
                herbs TEXT,
                recommendations TEXT,
                precautions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add test users
        print("\nAdding test users...")
        test_users = [
            ('Anand Singh', 'anand@gmail.com', generate_password_hash('test123')),
            ('Niviti User', 'niviti@gmail.com', generate_password_hash('123'))
        ]
        for user in test_users:
            cursor.execute('''
                INSERT IGNORE INTO users (name, email, password)
                VALUES (%s, %s, %s)
            ''', user)
        
        # Add test remedies
        print("\nAdding test remedies...")
        test_remedies = [
            {
                'condition': 'Digestive Issues',
                'symptoms': 'Bloating, Gas, Indigestion, Stomach pain',
                'herbs': 'Triphala, Ginger, Cumin, Fennel',
                'recommendations': 'Take herbs after meals, drink warm water',
                'precautions': 'Avoid cold drinks and heavy meals'
            },
            {
                'condition': 'Stress and Anxiety',
                'symptoms': 'Restlessness, Insomnia, Mental tension, Worry',
                'herbs': 'Ashwagandha, Brahmi, Jatamansi, Holy Basil',
                'recommendations': 'Practice meditation, take herbs before bed',
                'precautions': 'Consult doctor if symptoms persist'
            },
            {
                'condition': 'Headache',
                'symptoms': 'Head pain, Tension, Migraine',
                'herbs': 'Brahmi, Shankhpushpi, Jatamansi',
                'recommendations': 'Rest in quiet place, apply herbal oil',
                'precautions': 'Avoid bright lights and loud noises'
            },
            {
                'condition': 'Joint Pain',
                'symptoms': 'Stiffness, Inflammation, Reduced mobility',
                'herbs': 'Turmeric, Guggulu, Ginger, Boswellia',
                'recommendations': 'Apply warm oil, gentle exercise',
                'precautions': 'Avoid cold exposure'
            },
            {
                'condition': 'Respiratory Issues',
                'symptoms': 'Cough, Cold, Congestion, Breathing difficulty',
                'herbs': 'Tulsi, Ginger, Mulethi, Pippali',
                'recommendations': 'Steam inhalation, herbal tea',
                'precautions': 'Stay warm and hydrated'
            }
        ]
        
        for remedy in test_remedies:
            cursor.execute('''
                INSERT IGNORE INTO remedies 
                (condition_name, symptoms, herbs, recommendations, precautions)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                remedy['condition'],
                remedy['symptoms'],
                remedy['herbs'],
                remedy['recommendations'],
                remedy['precautions']
            ))
        
        cnx.commit()
        print("Database setup completed successfully!")
        
    except Error as err:
        print(f"Error: {err}")
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()

if __name__ == "__main__":
    create_mysql_database() 