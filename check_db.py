import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from mysql_config import MYSQL_CONFIG

def get_db_connection():
    """Get a database connection"""
    config = MYSQL_CONFIG.copy()
    config['database'] = 'ayuverse_db'
    return mysql.connector.connect(**config)

def test_user_insertion(cursor, cnx):
    print("\nTesting user insertion...")
    try:
        cursor.execute('''
            INSERT IGNORE INTO users (name, email, password)
            VALUES (%s, %s, %s)
        ''', ('Test User', 'test@example.com', generate_password_hash('test123')))
        cnx.commit()
        print("Test user insertion successful")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def test_chat_session(cursor, cnx):
    print("\nTesting chat session creation...")
    try:
        cursor.execute('''
            INSERT INTO chat_sessions (user_email, title)
            VALUES (%s, %s)
        ''', ('test@example.com', 'Test Chat'))
        session_id = cursor.lastrowid
        cnx.commit()
        print("Chat session created successfully")
        return session_id
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def test_chat_message(cursor, cnx, session_id):
    print("\nTesting chat message insertion...")
    if session_id:
        try:
            cursor.execute('''
                INSERT INTO chat_history (session_id, message, is_bot)
                VALUES (%s, %s, %s)
            ''', (session_id, 'Hello, how can I help you?', True))
            cnx.commit()
            print("Chat message inserted successfully")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

def check_user_auth(cursor, email, password):
    print(f"\nTesting authentication for user {email}...")
    try:
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        if user:
            print(f"User found: {user[1]}")  # user[1] is the name field
            print(f"Stored password hash: {user[3]}")  # user[3] is the password hash
            if check_password_hash(user[3], password):
                print("Password is correct!")
            else:
                print("Password is incorrect!")
        else:
            print("User not found")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def main():
    cnx = None
    cursor = None
    try:
        cnx = get_db_connection()
        cursor = cnx.cursor()

        # Test user insertion
        test_user_insertion(cursor, cnx)

        # Test chat functionality
        session_id = test_chat_session(cursor, cnx)
        test_chat_message(cursor, cnx, session_id)

        print("\nChecking remedies in database:\n")
        cursor.execute('SELECT * FROM remedies')
        remedies = cursor.fetchall()
        
        print(f"Found {len(remedies)} remedies:\n")
        for remedy in remedies:
            print(f"Condition: {remedy[1]}")
            print(f"Symptoms: {remedy[2]}")
            print(f"Herbs: {remedy[3]}")
            print("-" * 50 + "\n")

        # Check specific user
        test_email = "anand@gmail.com"
        print(f"Checking user {test_email} in database...")
        cursor.execute('SELECT * FROM users WHERE email = %s', (test_email,))
        user = cursor.fetchone()
        if user:
            print(f"User found: {user[1]}")  # user[1] is the name field
        else:
            print("User not found")

        # Test authentication
        check_user_auth(cursor, test_email, "test123")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()

if __name__ == "__main__":
    main() 