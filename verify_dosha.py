import mysql.connector
import sys

def check_database():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="bokaro#1",
            database="ayuverse_db"
        )
        cursor = conn.cursor(dictionary=True)
        
        # Check if user_dosha table exists
        cursor.execute("SHOW TABLES LIKE 'user_dosha'")
        if not cursor.fetchone():
            print("Creating user_dosha table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_dosha (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    dosha_type ENUM('Vata', 'Pitta', 'Kapha') NOT NULL,
                    confidence_scores JSON,
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE KEY unique_user (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            conn.commit()
            print("✓ Created user_dosha table")
        else:
            print("✓ user_dosha table already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False

def check_python_packages():
    try:
        import pandas
        import sklearn
        import joblib
        print("✓ Required Python packages are installed")
        return True
    except ImportError as e:
        print(f"Missing package: {e.name}")
        return False

def main():
    print("Verifying Dosha Test Setup")
    print("=========================")
    
    print("\n[1/2] Checking database...")
    if not check_database():
        print("❌ Database check failed")
        return
    
    print("\n[2/2] Checking Python packages...")
    if not check_python_packages():
        print("❌ Some required packages are missing")
        return
    
    print("\n✅ Setup verification complete! The dosha test should work correctly.")
    print("You can now run the Flask application and access the dosha test through the web interface.")

if __name__ == "__main__":
    main()
