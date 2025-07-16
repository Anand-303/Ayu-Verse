from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
from flask_wtf.csrf import CSRFProtect, generate_csrf
import json
from pathlib import Path
import nltk
import mysql.connector.pooling
from contextlib import contextmanager
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import os
from fuzzywuzzy import fuzz
from collections import defaultdict
from dataset_processor import HerbalDataset
from disease_variations import disease_variations
from dosha_predictor import DoshaPredictor
import joblib
import time

# Initialize the dataset processor
dataset = HerbalDataset('herbal_remedies_dataset.csv')

# Create a reverse mapping of all variations to their base conditions
variation_to_condition = {}
for base_condition, variations in disease_variations.items():
    for variation in variations:
        variation_to_condition[variation.lower()] = base_condition.lower()

# Initialize Dosha Predictor - Will be loaded on first use
DOSHA_PREDICTOR = None

def get_dosha_predictor():
    """Get the Dosha predictor instance, initializing it if needed"""
    global DOSHA_PREDICTOR
    if DOSHA_PREDICTOR is None:
        try:
            logger.info("Initializing Dosha Predictor...")
            start_time = time.time()
            
            # Initialize with paths to pre-trained model files
            DOSHA_PREDICTOR = DoshaPredictor(
                model_path='dosha_gb_model.joblib',
                encoders_path='dosha_encoders.joblib'
            )
            
            # Try to load the pre-trained model
            if not DOSHA_PREDICTOR.load_model():
                logger.error("Failed to load pre-trained model")
                return None
                
            logger.info(f"Dosha Predictor initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing Dosha Predictor: {str(e)}")
            return None
            
    return DOSHA_PREDICTOR
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import Error
import os
import logging
from mysql_config import MYSQL_CONFIG
# Model loading is now handled by lazy_loader
import traceback
from flask_session import Session
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
ADMIN_EMAIL = "Ayuverse16@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = ADMIN_EMAIL
SMTP_PASSWORD = os.getenv('AYUVERSE_EMAIL_PASSWORD')
SMTP_USE_TLS = True

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = 'a_random_secret_key_here'  # Change this to a secure secret key

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Add CSRF token to all templates
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf())

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Process a chat message
def process_message(message):
    """Process a chat message and generate a response"""
    try:
        # Extract symptoms and conditions from the message
        conditions = extract_symptoms_conditions(message)
        
        # Get recommendations based on conditions
        response = get_recommendations(conditions, message)
        
        if not response:
            response = "I'm sorry, I couldn't understand your symptoms. Could you please describe them more clearly?"
            
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return "I'm having trouble processing your request. Please try again later."

# Configure session
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session')

@app.before_request
def before_request():
    # Skip session check for static files and login page
    if request.endpoint in ('static', 'login', 'register'):
        return
        
    # Allow access to the landing page if not logged in
    if request.endpoint == 'index' and 'email' not in session:
        return
        
    # Check if user is logged in for protected routes
    if 'email' not in session:
        # Store the intended URL to redirect back after login
        if request.endpoint != 'login':
            session['next_url'] = request.url
        return redirect(url_for('login'))
        
    # Set session as permanent
    session.permanent = True
    logger.debug('Session before request: %s', dict(session))

@app.after_request
def after_request(response):
    logger.debug('Session after request: %s', dict(session))
    return response

# Database connection pool
_db_pool = None

def init_db_pool():
    """Initialize the database connection pool"""
    global _db_pool
    if _db_pool is None:
        try:
            config = MYSQL_CONFIG.copy()
            config['database'] = 'ayuverse_db'
            
            # Create connection pool
            _db_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="ayuverse_pool",
                pool_size=5,  # Adjust based on your needs
                **config
            )
            logger.info("Database connection pool initialized")
            
            # Initialize database schema if needed
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_dosha (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        dosha_type ENUM('Vata', 'Pitta', 'Kapha') NOT NULL,
                        confidence_scores JSON,
                        test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE KEY unique_user_dosha (user_id)
                    )
                ''')
                conn.commit()
                cursor.close()
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

@contextmanager
def get_db():
    """Get a database connection from the pool with context management
    
    Example usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            result = cursor.fetchall()
    """
    conn = None
    try:
        if _db_pool is None:
            init_db_pool()
        
        conn = _db_pool.get_connection()
        conn.ping(reconnect=True, attempts=3, delay=5)
        yield conn
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

# Initialize NLTK components as None, will be loaded on first use
lemmatizer = None
stop_words = None

def init_nltk():
    """Initialize NLTK components on first use"""
    global lemmatizer, stop_words
    
    if lemmatizer is None or stop_words is None:
        try:
            # Download NLTK data if not found
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
                nltk.data.find('corpora/omw-1.4')
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize components
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            print(f"Warning: Failed to initialize NLTK components: {e}")
            # Fallback to basic initialization
            lemmatizer = WordNetLemmatizer()
            stop_words = set()

# Herbal database
herbal_db = {
    "Ashwagandha": {
        "Properties": "Adaptogenic herb that helps reduce stress and anxiety",
        "Benefits": "Boosts immunity, improves sleep, reduces inflammation",
        "Usage": "Available as powder, capsules, or liquid extract"
    },
    "Turmeric": {
        "Properties": "Anti-inflammatory and antioxidant properties",
        "Benefits": "Reduces inflammation, supports joint health, boosts immunity",
        "Usage": "Can be used in cooking, as supplements, or golden milk"
    },
    "Brahmi": {
        "Properties": "Brain tonic and memory enhancer",
        "Benefits": "Improves memory, reduces anxiety, supports brain health",
        "Usage": "Available as powder, tablets, or liquid extract"
    },
    "Shatavari": {
        "Properties": "Rejuvenating herb for reproductive health",
        "Benefits": "Balances hormones, supports immune system, improves vitality",
        "Usage": "Can be taken as powder, tablets, or liquid extract"
    },
    "Triphala": {
        "Properties": "Combination of three fruits with detoxifying properties",
        "Benefits": "Improves digestion, cleanses colon, supports eye health",
        "Usage": "Usually taken as powder or tablets before bed"
    }
}

# Create symptom and condition indices for faster lookup
symptom_index = defaultdict(list)
condition_index = defaultdict(list)
herb_properties_index = defaultdict(list)

def initialize_indices():
    """Initialize search indices for faster lookup"""
    for herb, info in herbal_db.items():
        # Index symptoms
        for symptom in info.get('treats_symptoms', []):
            symptom_index[symptom.lower()].append(herb)
        
        # Index conditions
        for condition in info.get('treats_conditions', []):
            condition_index[condition.lower()].append(herb)
        
        # Index properties
        for prop in info.get('properties', []):
            herb_properties_index[prop.lower()].append(herb)

initialize_indices()

def preprocess_text(text):
    """Preprocess text for better matching"""
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def fuzzy_match(query, choices, threshold=80):
    """Find fuzzy matches in a list of choices"""
    matches = []
    query = query.lower()
    for choice in choices:
        ratio = fuzz.ratio(query, choice.lower())
        if ratio >= threshold:
            matches.append((choice, ratio))
    return sorted(matches, key=lambda x: x[1], reverse=True)

def identify_doshas(symptoms):
    """Identify potential dosha imbalances based on symptoms"""
    dosha_patterns = {
        'vata': [
            'anxiety', 'stress', 'insomnia', 'dry', 'cold', 'constipation', 
            'joint pain', 'nervousness', 'restlessness', 'irregular digestion'
        ],
        'pitta': [
            'inflammation', 'anger', 'acid', 'burning', 'fever', 'rash',
            'irritation', 'hot', 'sharp pain', 'excessive hunger'
        ],
        'kapha': [
            'congestion', 'weight', 'lethargy', 'depression', 'slow',
            'cold', 'mucus', 'heaviness', 'drowsiness', 'water retention'
        ]
    }
    
    identified_doshas = []
    symptoms_text = ' '.join(symptoms).lower()
    
    for dosha, patterns in dosha_patterns.items():
        if any(pattern in symptoms_text for pattern in patterns):
            identified_doshas.append(dosha)
    
    return identified_doshas

def extract_symptoms_conditions(message):
    """Extract symptoms and conditions from message using dataset"""
    # Preprocess message
    words = preprocess_text(message.lower())
    
    # Search for conditions in the dataset
    conditions = []
    
    # Check each word in the message
    for word in words:
        # Check for known variations first
        if word in variation_to_condition:
            base_condition = variation_to_condition[word]
            conditions.append(base_condition)
            continue
            
        # Search for exact matches from dataset
        matches = dataset.search_conditions(word)
        if matches:
            conditions.extend(matches)
            continue
            
        # Try fuzzy matching for variations
        for condition in dataset.medical_conditions:
            # Check if the word is a substring of the condition
            if word in condition:
                conditions.append(condition)
                continue
                
            # Check if the condition is a substring of the word
            if condition in word:
                conditions.append(condition)
                continue
                
            # Check for common variations
            variations = [
                condition.replace(' ', ''),  # Remove spaces
                condition.replace('-', ''),  # Remove hyphens
                condition.replace('_', ''),  # Remove underscores
                condition.replace('and', ''),  # Remove 'and'
                condition.replace('with', ''),  # Remove 'with'
                condition.replace('for', '')  # Remove 'for'
            ]
            
            if any(word in var for var in variations):
                conditions.append(condition)
                continue
    
    # Remove duplicates while preserving order
    conditions = list(dict.fromkeys(conditions))
    
    # If no conditions found, try splitting the message into phrases
    if not conditions:
        phrases = message.split(',')
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase:
                # Check phrase against known variations
                if phrase in variation_to_condition:
                    base_condition = variation_to_condition[phrase]
                    conditions.append(base_condition)
                    continue
                    
                # Check phrase against dataset
                matches = dataset.search_conditions(phrase)
                if matches:
                    conditions.extend(matches)
    
    return conditions

def get_recommendations(symptoms, conditions):
    """Get herb recommendations based on symptoms and conditions"""
    recommendations = []
    
    symptom_herbs = {
        'stress': ['Ashwagandha', 'Brahmi'],
        'anxiety': ['Brahmi', 'Ashwagandha'],
        'pain': ['Turmeric'],
        'fatigue': ['Ashwagandha', 'Shatavari'],
        'insomnia': ['Ashwagandha', 'Brahmi'],
        'digestion': ['Triphala']
    }
    
    condition_herbs = {
        'diabetes': ['Turmeric', 'Triphala'],
        'arthritis': ['Turmeric'],
        'hypertension': ['Brahmi'],
        'asthma': ['Turmeric']
    }
    
    # Add recommendations based on symptoms
    for symptom in symptoms:
        if symptom in symptom_herbs:
            recommendations.extend(symptom_herbs[symptom])
    
    # Add recommendations based on conditions
    for condition in conditions:
        if condition in condition_herbs:
            recommendations.extend(condition_herbs[condition])
    
    # Remove duplicates while preserving order
    recommendations = list(dict.fromkeys(recommendations))
    
    if not recommendations:
        return None
    
    response = "Based on your query, here are some recommended herbs:\n\n"
    for herb in recommendations:
        response += f"- {herb}: {herbal_db[herb]['Benefits']}\n"
    
    return response

def save_chat_message(user_email, user_message, bot_response, session_id=None):
    try:
        with get_db() as db:
            c = db.cursor()
            
            # If no session_id provided, create a new session with first message as title
            if not session_id:
                title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                c.execute('''
                    INSERT INTO chat_sessions (user_email, title)
                    VALUES (%s, %s)
                ''', (user_email, title))
                session_id = c.lastrowid
            
            # Save user message
            c.execute('''
                INSERT INTO chat_history (session_id, message, is_bot)
                VALUES (%s, %s, %s)
            ''', (session_id, user_message, False))
            
            # Save bot response
            c.execute('''
                INSERT INTO chat_history (session_id, message, is_bot)
                VALUES (%s, %s, %s)
            ''', (session_id, bot_response, True))
            
            db.commit()
            return session_id
            
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")
        if 'db' in locals() and db.is_connected():
            db.rollback()
        raise
        db.close()

def get_user_chat_history(user_email):
    sessions = []
    
    try:
        with get_db() as cnx:
            cursor = cnx.cursor(dictionary=True)
            
            # Get all chat sessions for the user
            cursor.execute('''
                SELECT id, title, created_at
                FROM chat_sessions
                WHERE user_email = %s
                ORDER BY created_at DESC
            ''', (user_email,))
            
            for row in cursor.fetchall():
                # Get the last message for each session
                cursor.execute('''
                    SELECT message, is_bot, created_at
                    FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', (row['id'],))
                last_message = cursor.fetchone()
                
                sessions.append({
                    'id': row['id'],
                    'title': row['title'],
                    'created_at': row['created_at'],
                    'last_message_at': row['created_at'],
                    'last_message': last_message['message'] if last_message else None
                })
    except Exception as e:
        logger.error(f"Error fetching user chat history: {e}")
        # Return empty list on error
        return []
    
    return sessions

def get_session_messages(session_id):
    messages = []
    
    try:
        with get_db() as cnx:
            cursor = cnx.cursor(dictionary=True)
            
            cursor.execute('''
                SELECT message, is_bot, created_at
                FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at ASC
            ''', (session_id,))
            
            for row in cursor.fetchall():
                messages.append({
                    'message': row['message'],
                    'is_bot': row['is_bot'],
                    'created_at': row['created_at']
                })
    except Exception as e:
        logger.error(f"Error fetching session messages: {e}")
        # Return empty list on error
        return []
    
    return messages

@app.route('/')
def index():
    """Landing page route - should always show index.html first"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Home page route - requires login"""
    if 'email' not in session:
        # If not logged in, redirect to login
        return redirect(url_for('login'))
    
    # Get user's dosha information if available
    dosha_info = None
    
    try:
        with get_db() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('''
                SELECT ud.dosha_type, ud.confidence_scores, ud.test_date 
                FROM user_dosha ud 
                WHERE ud.user_id = %s 
                ORDER BY ud.test_date DESC 
                LIMIT 1
            ''', (session['user_id'],))
            dosha_result = cursor.fetchone()
            
            if dosha_result:
                dosha_info = {
                    'dosha_type': dosha_result['dosha_type'],
                    'confidence_scores': json.loads(dosha_result['confidence_scores']) if dosha_result['confidence_scores'] else None,
                    'test_date': dosha_result['test_date']
                }
    except Exception as e:
        logger.error(f"Error fetching dosha info: {str(e)}")
    
    return render_template('home.html', 
                         username=session.get('user_name', 'User'),
                         dosha_info=dosha_info)

@app.route('/about')
def about():
    """About Us page route"""
    return render_template('about.html')

@app.route('/support')
def support():
    """Support page route"""
    return render_template('support.html')

@app.route('/submit_support', methods=['POST'])
def submit_support():
    """Handle support request submission"""
    try:
        data = request.json
        user_email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        if not all([user_email, subject, message]):
            return jsonify({
                "success": False,
                "message": "All fields are required"
            }), 400

        # Create email
        msg = MIMEMultipart()
        msg['From'] = ADMIN_EMAIL
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Support Request: {subject}"

        # Email body
        body = f"""
        New Support Request
        
        User Email: {user_email}
        Subject: {subject}
        
        Message:
        {message}
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
                return jsonify({
                    "success": True,
                    "message": "Support request submitted successfully"
                })
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP Authentication failed")
            return jsonify({
                "success": False,
                "message": "Failed to authenticate with email server"
            }), 500
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Error sending email: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Error processing support request: {str(e)}")
        return jsonify({
            "success": False,
            "message": "An unexpected error occurred"
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    # If user is already logged in, redirect to home
    if 'email' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
            
        db = get_db()
        cursor = db.cursor(dictionary=True)
        
        try:
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password'], password):
                # Set session variables
                session['user_id'] = user['id']
                session['email'] = user['email']
                session['user_name'] = user['name']
                session.permanent = True
                
                logger.info(f"User {user['email']} logged in successfully")
                flash('Login successful!', 'success')
                
                # Check for next_url in session or request args
                next_url = session.pop('next_url', None) or request.args.get('next')
                if next_url and next_url.startswith('/'):
                    return redirect(next_url)
                return redirect(url_for('home'))
                
            else:
                flash('Invalid email or password', 'error')
                
        except Error as e:
            logger.error(f"Database error during login: {e}")
            flash('An error occurred. Please try again.', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        logger.debug("Received registration POST request")
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        logger.debug(f"Registration attempt for email: {email}")
        
        # Validate required fields
        if not all([name, email, password, confirm_password]):
            logger.warning("Registration failed: Missing required fields")
            return render_template('register.html', error="All fields are required")
        
        if password != confirm_password:
            logger.warning("Registration failed: Passwords do not match")
            return render_template('register.html', error="Passwords do not match")
        
        conn = None
        cursor = None
        try:
            logger.debug("Attempting database connection")
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
            if cursor.fetchone():
                logger.warning(f"Registration failed: Email already exists: {email}")
                return render_template('register.html', error="Email already registered")
            
            # Create new user
            logger.debug("Creating new user")
            hashed_password = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                (name, email, hashed_password)
            )
            conn.commit()
            
            logger.info(f"New user registered successfully: {email}")
            return redirect(url_for('login'))
            
        except Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error during registration: {e}")
            return render_template('register.html', error=f"An error occurred: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            logger.debug("Database connection closed")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route - redirects to landing page"""
    session.clear()
    return redirect(url_for('index'))  # Redirect to landing page after logout

@app.route('/chat_history')
def chat_history():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    cnx = get_db()
    cursor = cnx.cursor(dictionary=True)
    
    # Get all chat sessions for the user with their last messages
    cursor.execute('''
        SELECT 
            cs.id,
            cs.title,
            cs.created_at,
            ch.created_at as last_message_at,
            ch.message as last_message
        FROM chat_sessions cs
        LEFT JOIN (
            SELECT 
                session_id,
                message,
                created_at,
                ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at DESC) as rn
            FROM chat_history
        ) ch ON cs.id = ch.session_id AND ch.rn = 1
        WHERE cs.user_email = %s
        ORDER BY COALESCE(ch.created_at, cs.created_at) DESC
    ''', (session['user_email'],))
    
    chat_sessions = []
    for row in cursor.fetchall():
        chat_sessions.append({
            'id': row['id'],
            'title': row['title'],
            'created_at': row['created_at'],
            'last_message_at': row['last_message_at'] or row['created_at'],
            'last_message': row['last_message'] or 'No messages yet'
        })
    
    cursor.close()
    cnx.close()
    
    return render_template('chat_history.html', 
                         chat_sessions=chat_sessions,
                         user_name=session.get('user_name'))

@app.route('/chat')
def chat():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if 'email' not in session:
        logger.warning("Unauthorized chat attempt - no session")
        return jsonify({'error': 'Please log in first'}), 401
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data received'}), 400
            
        user_message = data.get('message', '').strip()
        logger.debug(f"Received message: {user_message}")
        
        if not user_message:
            logger.warning("Empty message received")
            return jsonify({'error': 'Message cannot be empty'}), 400
            
        # Load model only when needed
        herbal_model, preprocessing_info, class_names = get_prediction_model()
        if not herbal_model:
            logger.error("Failed to load prediction model")
            return jsonify({'error': 'Chat service is temporarily unavailable. Please try again later.'}), 503

        # Process the message using the process_message function
        response = process_message(user_message)
        
        # Save the chat message to database
        logger.debug("Saving chat message to database")
        save_chat_message(session['email'], user_message, response)
        
        logger.debug("Successfully processed chat message")
        return jsonify({'response': response})
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500
    return response

def generate_basic_response(user_message, symptoms, conditions, recommendations):
    """Generate a basic response without using the ML model"""
    response = "Based on your symptoms, here are my recommendations:\n\n"
    
    if recommendations:
        response += "🌿 Recommended Herbs:\n"
        for herb in set(recommendations):
            if herb in herbal_db:
                info = herbal_db[herb]
                response += f"\n• {herb}:\n"
                response += f"  - Properties: {info['Properties']}\n"
                response += f"  - Benefits: {info['Benefits']}\n"
                response += f"  - Usage: {info['Usage']}\n"
    else:
        # Check for general keywords in the message
        message_lower = user_message.lower()
        if any(word in message_lower for word in ['stress', 'anxiety', 'tension']):
            response += "For stress and anxiety, I recommend:\n"
            response += "• Ashwagandha: Helps reduce stress and anxiety\n"
            response += "• Holy Basil (Tulsi): Calming and adaptogenic properties\n"
            response += "• Brahmi: Supports mental clarity and relaxation\n"
        elif any(word in message_lower for word in ['sleep', 'insomnia']):
            response += "For sleep issues, consider:\n"
            response += "• Ashwagandha: Promotes restful sleep\n"
            response += "• Jatamansi: Natural sleep aid\n"
            response += "• Shankhpushpi: Calming herb that supports sleep\n"
        elif any(word in message_lower for word in ['digestion', 'stomach', 'gut']):
            response += "For digestive health, try:\n"
            response += "• Triphala: Supports digestive health\n"
            response += "• Ginger: Aids digestion and reduces bloating\n"
            response += "• Cumin: Helps with digestion and gas\n"
        else:
            response += "I couldn't find specific herb recommendations for your symptoms. "
            response += "Please provide more details about your symptoms or consult with an Ayurvedic practitioner.\n"
            response += "\nYou can ask me about:\n"
            response += "• Specific conditions (e.g., digestive issues, stress, headaches)\n"
            response += "• Specific symptoms (e.g., bloating, anxiety, pain)\n"
            response += "• Specific herbs (e.g., Ashwagandha, Triphala, Brahmi)\n"
    
    return response

def get_remedy_by_symptoms(cursor, symptoms):
    # Search for remedies based on symptoms
    search_terms = symptoms.lower().split()
    query = '''
        SELECT condition_name, symptoms, herbs, recommendations, precautions 
        FROM remedies 
        WHERE LOWER(symptoms) REGEXP %s
    '''
    
    # Create a regex pattern that matches any of the search terms
    pattern = '|'.join(search_terms)
    cursor.execute(query, (pattern,))
    
    remedies = cursor.fetchall()
    if remedies:
        response = []
        for remedy in remedies:
            response.append(f"For {remedy[0]}:\n")
            response.append(f"Recommended herbs: {remedy[2]}\n")
            response.append("Treatment recommendations:\n")
            response.append(f"{remedy[3]}\n")
            if remedy[4]:  # If there are precautions
                response.append(f"Precautions: {remedy[4]}\n")
            response.append("---\n")
        return '\n'.join(response)
    return None

def find_best_match(message, conditions, threshold=70):
    """Find the best matching condition using fuzzy matching with variations"""
    message = message.lower().strip()
    best_match = None
    highest_score = threshold
    
    # First check for direct matches in variations
    for base_condition, variations in disease_variations.items():
        # Check if any variation is in the message
        for variation in variations:
            if variation.lower() in message:
                # Find the closest match in our actual conditions
                for cond in conditions:
                    if pd.notna(cond) and base_condition.lower() in cond.lower():
                        return cond, 100
    
    # Then check for direct matches in conditions
    for condition in conditions:
        if pd.isna(condition):
            continue
            
        condition_lower = condition.lower()
        
        # Check for direct match
        if condition_lower in message or message in condition_lower:
            return condition, 100
        
        # Check for partial match
        if condition_lower.replace(' ', '') in message.replace(' ', '') or \
           any(word in condition_lower.split() for word in message.split() if len(word) > 3):
            return condition, 95
    
    # Then try fuzzy matching
    for condition in conditions:
        if pd.isna(condition):
            continue
            
        condition_lower = condition.lower()
        
        # Use token sort ratio for better handling of word order
        score = fuzz.token_sort_ratio(message, condition_lower)
        
        # Boost score if there's a partial match
        if any(word in condition_lower for word in message.split() if len(word) > 3):
            score = min(100, score + 15)  # Boost score but cap at 100
            
        if score > highest_score:
            highest_score = score
            best_match = condition
    
    # If we found a good match, return it
    if best_match:
        return best_match, highest_score
    
    # As a last resort, check for any word in the condition
    message_words = set(word for word in message.split() if len(word) > 3)
    for condition in conditions:
        if pd.isna(condition):
            continue
            
        condition_words = set(condition.lower().split())
        if message_words & condition_words:  # If there's any word overlap
            return condition, 85  # Medium confidence
    
    return best_match, highest_score

def get_condition_details(condition):
    """Get details for a specific condition from the dataset"""
    try:
        # Get the column name for conditions
        condition_col = 'Medical Condition Treated'
        
        # First try exact match
        condition_lower = condition.lower()
        for cond in dataset.df[condition_col].unique():
            if pd.notna(cond) and cond.lower() == condition_lower:
                return dataset.df[dataset.df[condition_col].str.lower() == condition_lower].iloc[0].to_dict()
        
        # Check for variations in disease_variations
        for base_condition, variations in disease_variations.items():
            if condition_lower in [v.lower() for v in variations]:
                # Found a matching variation, now find the base condition in the dataset
                for cond in dataset.df[condition_col].unique():
                    if pd.notna(cond) and base_condition.lower() in cond.lower():
                        return dataset.df[dataset.df[condition_col].str.lower() == cond.lower()].iloc[0].to_dict()
        
        # Try fuzzy match if exact match not found
        best_match = None
        highest_score = 70  # Minimum threshold
        
        for cond in dataset.df[condition_col].unique():
            if pd.isna(cond):
                continue
                
            # Check direct match first
            if condition_lower in cond.lower():
                return dataset.df[dataset.df[condition_col].str.lower() == cond.lower()].iloc[0].to_dict()
                
            # Try fuzzy matching
            score = fuzz.token_sort_ratio(condition_lower, cond.lower())
            if score > highest_score:
                highest_score = score
                best_match = cond
        
        if best_match:
            return dataset.df[dataset.df[condition_col].str.lower() == best_match.lower()].iloc[0].to_dict()
            
    except Exception as e:
        logger.error(f"Error getting condition details for {condition}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

def format_condition_response(condition_details):
    """Format a detailed response for a specific condition"""
    if not condition_details:
        return "I'm sorry, I couldn't find detailed information about this condition."
    
    response = f"🌿 **{condition_details['Condition']}**\n\n"
    
    if 'Herbs' in condition_details and pd.notna(condition_details['Herbs']):
        response += f"**Recommended Herbs:** {condition_details['Herbs']}\n\n"
    
    if 'Recommendations' in condition_details and pd.notna(condition_details['Recommendations']):
        response += f"**Recommendations:**\n{condition_details['Recommendations']}\n\n"
    
    if 'Precautions' in condition_details and pd.notna(condition_details['Precautions']):
        response += f"**Precautions:**\n{condition_details['Precautions']}\n\n"
    
    response += "⚠️ *Note: This is general information. Please consult with an Ayurvedic practitioner for personalized advice.*"
    
    return response

def process_message(user_message):
    try:
        # Convert message to lowercase for matching
        message = user_message.lower()
        
        # Check for greetings
        if any(greeting in message for greeting in ['hi', 'hello', 'hey', 'namaste']):
            session.pop('clarification_asked', None)
            return "Namaste! 🙏 I'm your Ayurvedic health assistant. How can I help you today?"
            
        # Check for thanks/bye
        if any(word in message for word in ['thank', 'thanks', 'bye', 'goodbye']):
            return "You're welcome! Feel free to ask if you have any more questions. Wishing you good health! 🙏"

        # Get all unique conditions from the dataset
        all_conditions = dataset.df['Medical Condition Treated'].unique().tolist()
        
        # Find the best matching condition
        matched_condition, confidence = find_best_match(message, all_conditions)
        
        if confidence > 70:  # Threshold for considering it a match
            condition_details = get_condition_details(matched_condition)
            if condition_details:
                session.pop('clarification_asked', None)
                return format_condition_response(condition_details)
        
        # Check for specific herb mentions
        for herb in herbal_db.keys():
            if herb.lower() in message:
                session.pop('clarification_asked', None)
                return format_herb_info(herb)
                
        # Handle common symptoms
        symptom_responses = {
            'headache': "For headaches, try applying a paste of sandalwood or lavender oil on your forehead. Peppermint tea may also help.",
            'fever': "For fever, drink tulsi (holy basil) tea and stay hydrated. Consult a doctor if fever persists for more than 2 days.",
            'cough': "For cough, try taking 1 tsp of honey with a pinch of turmeric powder before bed.",
            'cold': "For cold, drink ginger tea with honey and black pepper. Steam inhalation with eucalyptus oil may help."
        }
        
        for symptom, response in symptom_responses.items():
            if symptom in message:
                return f"{response} Would you like more specific information?"

        # If no specific condition is matched, provide a general response
        if not session.get('clarification_asked'):
            session['clarification_asked'] = True
            return ("I notice you're experiencing some health concerns. To help you better, could you please describe:\n\n"
                    "1. Your specific symptoms\n"
                    "2. How long you've had them\n"
                    "3. Any other relevant details\n\n"
                    "For example: \"I've had a persistent cough for 3 days with chest congestion.\"")
        else:
            # If clarification has already been asked, provide a different response
            return ("I'm still having trouble understanding. You can ask me about:\n\n"
                   "• Specific symptoms (e.g., headache, fever, indigestion)\n"
                   "• Herbal remedies for common conditions\n"
                   "• Information about specific herbs\n\n"
                   "Or try rephrasing your question.")

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())
        return ("I apologize, but I encountered an error processing your request. "
                "Please try rephrasing your question or ask about a different topic.")

def generate_chat_response(user_message):
    """Generate a chat response"""
    try:
        user_message_lower = user_message.lower()
        
        # Check for basic greetings
        greeting_words = ['hi', 'hello', 'namaste', 'hey', 'hola', 'greetings']
        if any(word == user_message_lower for word in greeting_words):
            return format_greeting()
        
        # Check for specific herb queries
        for herb in herbal_db.keys():
            if herb.lower() in user_message_lower:
                return format_herb_info(herb)
        
        # Extract symptoms and conditions
        symptoms, conditions = extract_symptoms_conditions(user_message)
        
        # If we have symptoms/conditions, try to get ML-based recommendations
        if symptoms or conditions:
            # Load the model only when needed
            herbal_model, preprocessing_info, class_names = get_prediction_model()
            
            # If model is available, try to get ML-based recommendations
            if herbal_model is not None and preprocessing_info is not None and class_names is not None:
                try:
                    from predict import predict_for_new_data
                    
                    # Prepare input for the model
                    input_data = {
                        'symptoms': ' '.join(symptoms),
                        'conditions': ' '.join(conditions)
                    }
                    
                    # Make prediction
                    prediction = predict_for_new_data(
                        input_data, 
                        herbal_model, 
                        preprocessing_info, 
                        class_names
                    )
                    
                    # If we got a good prediction, format it
                    if prediction and 'predicted_herb' in prediction:
                        herb = prediction['predicted_herb']
                        confidence = prediction.get('confidence', 0) * 100
                        
                        # Get additional information about the herb
                        herb_info = herbal_db.get(herb, {})
                        
                        response = f"Based on your symptoms, I recommend {herb} (confidence: {confidence:.1f}%).\n\n"
                        
                        if 'description' in herb_info:
                            response += f"{herb_info['description']}\n\n"
                            
                        if 'benefits' in herb_info:
                            response += f"Benefits: {', '.join(herb_info['benefits'])}\n"
                            
                        if 'precautions' in herb_info:
                            response += f"Precautions: {', '.join(herb_info['precautions'])}"
                            
                        return response
                        
                except Exception as e:
                    logger.error(f"Error generating ML-based response: {str(e)}")
            
            # Fall back to rule-based recommendations if ML fails or isn't available
            recommendations = get_recommendations(symptoms, conditions)
            if recommendations:
                return recommendations
        
        # If no specific query was identified, provide a helpful response
        return format_help_message()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error processing your request. Please try again later."

def format_recommendations(recommendations):
    """Format herb recommendations"""
    if not recommendations:
        return format_help_message()
        
    response = ["🌿 Based on your symptoms, here are some recommended herbs:\n"]
    
    for rec in recommendations[:3]:  # Show top 3 recommendations
        herb = rec['herb']
        response.extend([
            f"• {herb}",
            f"  - {rec.get('description', '').split('.')[0]}",
            f"  - Usage: {rec.get('usage', '')}",
            ""
        ])
    
    response.append("\nWould you like more details about any of these herbs?")
    return "\n".join(response)

def format_greeting():
    """Format a greeting message"""
    greetings = [
        "Namaste! 🙏 I'm your Ayurvedic health assistant. How can I help you today?",
        "Welcome! I'm here to guide you through Ayurvedic wellness. What would you like to know?",
        "Greetings! I'm your Ayurvedic guide. How may I assist you on your wellness journey?"
    ]
    return greetings[0]  # Using first greeting for consistency

def format_herb_info(herb):
    """Format detailed information about a specific herb"""
    try:
        herb_info = herbal_db.get(herb, {})
        if not herb_info:
            return f"I'm sorry, I don't have detailed information about {herb}. Would you like to know about other similar herbs?"

        response = f"🌿 {herb} Information:\n\n"

        # Basic Information
        if 'Scientific Name' in herb_info:
            response += f"Scientific Name: {herb_info['Scientific Name']}\n\n"

        # Properties and Benefits
        if 'Properties' in herb_info:
            response += "Properties:\n"
            response += f"{herb_info['Properties']}\n\n"

        if 'Benefits' in herb_info:
            response += "Benefits:\n"
            response += f"{herb_info['Benefits']}\n\n"

        # Usage and Dosage
        if 'Usage' in herb_info:
            response += "Usage & Dosage:\n"
            response += f"{herb_info['Usage']}\n\n"

        # Part Used
        if 'Part Used' in herb_info:
            response += "Part Used:\n"
            response += f"{herb_info['Part Used']}\n\n"

        # Side Effects and Precautions
        if 'Side Effects' in herb_info:
            response += "⚠️ Side Effects & Precautions:\n"
            response += f"{herb_info['Side Effects']}\n\n"

        # Additional Information
        response += "📝 Additional Notes:\n"
        response += "• Start with a small dose and gradually increase as needed\n"
        response += "• Best taken under the guidance of an Ayurvedic practitioner\n"
        response += "• Store in a cool, dry place away from direct sunlight\n"
        response += "• Quality of herbs can vary, choose reliable sources\n\n"

        response += "Would you like to know more about:\n"
        response += "1. Specific health conditions this herb treats?\n"
        response += "2. How to prepare this herb?\n"
        response += "3. Similar herbs with comparable benefits?"

        return response

    except Exception as e:
        logger.error(f"Error formatting herb info: {e}")
        return f"I apologize, but I encountered an error while retrieving information about {herb}. Please try again."

def format_digestive_response(user_message):
    """Format response for digestive health queries"""
    # Find herbs that treat digestive issues from the database
    digestive_herbs = []
    digestive_symptoms = ['digestion', 'digestive', 'stomach', 'gut', 'bloating', 'gas', 'constipation']
    
    for herb, info in herbal_db.items():
        if any(symptom.lower() in digestive_symptoms for symptom in info.get('treats_symptoms', [])):
            digestive_herbs.append((herb, info))
    
    # Sort by effectiveness score if available
    digestive_herbs.sort(key=lambda x: x[1].get('effectiveness', 0), reverse=True)
    
    response = ["🌿 Ayurvedic Remedies for Digestive Health\n"]
    
    # Take top 4 most effective herbs
    for herb, info in digestive_herbs[:4]:
        response.extend([
            f"• {herb} ({info.get('scientific_name', '')})",
            f"  - {info.get('description', '').split('.')[0]}",
            f"  - Usage: {info.get('usage', '')}"
        ])
        
        if info.get('effectiveness'):
            response.append(f"  - Effectiveness score: {info['effectiveness']}/10")
            
        if info.get('precautions'):
            response.append(f"  - Precaution: {info['precautions'][0]}")
            
        response.append("")  # Add blank line between herbs
    
    response.append("📝 Recommended Daily Practices:")
    practices = [
        "• Drink warm water throughout the day",
        "• Eat mindfully and at regular times",
        "• Include fresh ginger tea in your routine",
        "• Avoid eating late at night",
        "• Chew your food thoroughly"
    ]
    response.extend(practices)
    
    response.append("\nWould you like specific details about any of these herbs or additional digestive health tips?")
    return "\n".join(response)

def format_help_message():
    """Format a help message"""
    return """I can help you with:
- Information about Ayurvedic herbs
- Common health concerns and remedies
- Understanding your dosha type
- General wellness advice

Feel free to ask about specific herbs or health concerns!"""

@app.route('/diet-plans')
def diet_plans():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('diet_plans.html', username=session['user_name'])

@app.route('/yoga-poses')
def yoga_poses():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('yoga_poses.html', username=session['user_name'])

@app.route('/doctor-consultation')
def doctor_consultation():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('doctor_consultation.html', username=session['user_name'])

@app.route('/dosha-test')
def dosha_test():
    """Display the dosha test questionnaire"""
    if 'email' not in session:
        flash('Please log in to take the dosha test.', 'info')
        return redirect(url_for('login'))
    
    # Ensure model is loaded
    predictor = get_dosha_predictor()
    if predictor is None:
        flash('Dosha test is not properly initialized. Please try again.', 'error')
        return redirect(url_for('home'))
    
    # Check if user has already taken the test
    try:
        with get_db() as db:
            cursor = db.cursor(dictionary=True)
            cursor.execute('SELECT * FROM user_dosha WHERE user_id = %s', (session['user_id'],))
            existing_result = cursor.fetchone()
            
            if existing_result:
                flash('You have already taken the dosha test.', 'info')
                return redirect(url_for('dosha_results'))
    except Exception as e:
        logger.error(f"Error checking for existing dosha test: {e}")
        flash('An error occurred while checking your test status. Please try again.', 'error')
        return redirect(url_for('home'))
    
    # Define the questions and options
    questions = [
        {
            'question': 'What best describes your body frame?',
            'options': [
                'Thin and light',
                'Medium build',
                'Heavy and solid'
            ]
        },
        {
            'question': 'How would you describe your skin?',
            'options': [
                'Dry, rough, thin, cool',
                'Sensitive, fair, warm, prone to rashes',
                'Oily, smooth, thick, cool'
            ]
        },
        {
            'question': 'How is your hair?',
            'options': [
                'Dry, curly, thin, brittle',
                'Straight, fine, oily, premature graying',
                'Thick, wavy, oily, lustrous'
            ]
        },
        {
            'question': 'What is your typical energy level like?',
            'options': [
                'Variable, bursts of energy',
                'Moderate, steady energy',
                'Slow to start but strong endurance'
            ]
        },
        {
            'question': 'How is your digestion?',
            'options': [
                'Irregular, prone to gas and bloating',
                'Strong, can digest most foods, prone to heartburn',
                'Slow but steady, heavy after meals'
            ]
        },
        {
            'question': 'How do you handle stress?',
            'options': [
                'Worry, anxiety, nervousness',
                'Anger, irritability, frustration',
                'Withdrawal, avoidance, holding on'
            ]
        },
        {
            'question': 'What is your typical sleep pattern?',
            'options': [
                'Light sleeper, difficulty falling asleep',
                'Moderate sleeper, wakes up easily',
                'Heavy sleeper, difficult to wake up'
            ]
        },
        {
            'question': 'What is your typical body temperature?',
            'options': [
                'Cold hands and feet, prefer warmth',
                'Warm, prefer cooler temperatures',
                'Normal, prefer warm and dry'
            ]
        },
        {
            'question': 'How do you handle weather changes?',
            'options': [
                'Dislike cold and wind, prefer warm',
                'Dislike heat, prefer cool',
                'Dislike damp, prefer warmth and dry'
            ]
        },
        {
            'question': 'What best describes your personality?',
            'options': [
                'Enthusiastic, creative, talkative',
                'Intense, focused, goal-oriented',
                'Calm, steady, supportive'
            ]
        }
    ]
    
    return render_template('dosha_test.html', 
                         username=session['user_name'],
                         questions=questions,
                         total_questions=len(questions))

@app.route('/init-dosha-test')
def init_dosha_test():
    """Initialize the dosha test by loading the model"""
    if 'email' not in session:
        return jsonify({'error': 'Please log in to take the dosha test.'}), 401
    
    try:
        # This will trigger lazy loading of the predictor if not already loaded
        predictor = get_dosha_predictor()
        if predictor is None:
            raise Exception("Failed to initialize the dosha predictor")
            
        return jsonify({
            'success': True,
            'redirect': url_for('dosha_test')
        })
    except Exception as e:
        logger.error(f"Error initializing dosha test: {str(e)}")
        return jsonify({
            'error': 'Failed to initialize the dosha test. Please try again later.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/dosha-results')
def dosha_results():
    """Display the user's dosha test results"""
    if 'email' not in session:
        flash('Please log in to view your dosha test results.', 'info')
        return redirect(url_for('login'))
    
    try:
        with get_db() as db:
            cursor = db.cursor(dictionary=True)
            
            # Get the latest test result for the user
            cursor.execute('''
                SELECT ud.*, u.name 
                FROM user_dosha ud
                JOIN users u ON ud.user_id = u.id
                WHERE ud.user_id = %s
                ORDER BY ud.test_date DESC
                LIMIT 1
            ''', (session['user_id'],))
            
            result = cursor.fetchone()
            
            if not result:
                flash('You have not taken the dosha test yet.', 'info')
                return redirect(url_for('dosha_test'))
            
            # Parse confidence scores
            try:
                confidence_scores = json.loads(result['confidence_scores'])
            except (json.JSONDecodeError, TypeError):
                confidence_scores = {}
            
            # Get recommendations based on dosha type
            cursor.execute('''
                SELECT * FROM dosha_recommendations 
                WHERE dosha_type = %s
            ''', (result['dosha_type'],))
            
            recommendations = cursor.fetchone()
            
            # If no recommendations found in database, use default recommendations
            if not recommendations:
                recommendations = {
                    'diet': 'Eat warm, cooked foods. Avoid cold and raw foods.',
                    'lifestyle': 'Maintain a regular routine. Get plenty of rest.',
                    'yoga': 'Practice gentle yoga and meditation.',
                    'herbs': 'Consider herbs like Ashwagandha and Brahmi.'
                }
            
            # Define dosha descriptions
            dosha_descriptions = {
                'Vata': {
                    'description': 'Vata represents air and space elements. Vata types are creative, energetic, and lively when in balance. When out of balance, they may experience anxiety, insomnia, and digestive issues.',
                    'balancing_tips': [
                        'Follow a regular daily routine',
                        'Eat warm, cooked, and slightly oily foods',
                        'Stay warm and avoid cold weather',
                        'Engage in calming activities like yoga and meditation',
                        'Get plenty of rest and avoid overexertion'
                    ]
                },
                'Pitta': {
                    'description': 'Pitta represents fire and water elements. Pitta types are intelligent, hard-working, and goal-oriented when in balance. When out of balance, they may experience anger, inflammation, and digestive issues.',
                    'balancing_tips': [
                        'Avoid excessive heat and steam',
                        'Eat cooling, non-spicy foods',
                        'Engage in calming activities and avoid excessive competition',
                        'Spend time in nature',
                        'Practice moderation in work and play'
                    ]
                },
                'Kapha': {
                    'description': 'Kapha represents earth and water elements. Kapha types are strong, loyal, and calm when in balance. When out of balance, they may experience weight gain, congestion, and sluggishness.',
                    'balancing_tips': [
                        'Engage in regular exercise',
                        'Eat light, dry, and warm foods',
                        'Try new experiences and vary your routine',
                        'Stay active and avoid excessive sleep',
                        'Use warming spices like ginger and black pepper'
                    ]
                }
            }
            
            # Get the description for the user's dosha type
            dosha_info = dosha_descriptions.get(result['dosha_type'], {
                'description': 'No description available for this dosha type.',
                'balancing_tips': []
            })
            
            # Format the test date
            test_date = result['test_date'].strftime('%B %d, %Y') if result['test_date'] else 'Recently'
            
            return render_template('dosha_results.html',
                                username=session['user_name'],
                                dosha_type=result['dosha_type'],
                                confidence_scores=confidence_scores,
                                test_date=test_date,
                                dosha_description=dosha_info['description'],
                                balancing_tips=dosha_info['balancing_tips'],
                                recommendations=recommendations)
    
    except Exception as e:
        logger.error(f"Error fetching dosha results: {str(e)}")
        flash('An error occurred while fetching your dosha results. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/privacy-policy')
def privacy_policy():
    """Display the privacy policy page"""
    return render_template('privacy_policy.html', username=session.get('user_name'))

@app.route('/process-dosha-test', methods=['POST'])
def process_dosha_test():
    """Process the dosha test form and display results using ML model"""
    start_time = time.time()
    
    if 'email' not in session:
        return jsonify({'error': 'Please log in to take the dosha test.'}), 401

    try:
        # Log request details
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        
        # Get form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            logger.info(f"Received JSON data: {data}")
        else:
            # Try form data if not JSON
            data = request.form.to_dict()
            logger.info(f"Received form data: {data}")
            
        if not data:
            logger.error("No data received in request")
            return jsonify({
                'success': False,
                'error': 'No data received',
                'status': 'error',
                'content_type': request.content_type,
                'form_data': bool(request.form),
                'json_data': bool(request.get_json(silent=True))
            }), 400
            
        # Debug: Log the raw request data
        logger.info(f"Raw request data: {data}")
        if 'answers' in data and isinstance(data['answers'], dict):
            logger.info(f"Answers data: {data['answers']}")
            logger.info(f"Answer keys: {list(data['answers'].keys())}")
            logger.info(f"Answer values: {list(data['answers'].values())}")
            
        # Check for timeout after getting JSON data
        if time.time() - start_time > 4:  # 4 seconds max for receiving data
            raise TimeoutError("Request took too long to process")
            
        if not data or 'answers' not in data or not isinstance(data['answers'], dict):
            return jsonify({
                'success': False,
                'error': 'Invalid form data format',
                'status': 'error'
            }), 400
            
        # Check for timeout during processing
        if time.time() - start_time > 4:  # 4 seconds max processing time
            raise TimeoutError("Processing took too long")
        
        # Get the predictor instance
        predictor = get_dosha_predictor()
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Prediction service is not available. Please try again later.',
                'status': 'error'
            }), 500
            
        # Map form answers to model features with expected values and text matching patterns
        question_mapping = {
            'q0': {
                'feature': 'Body Frame',
                'mapping': {
                    'Thin and light': 'Thin and Lean',
                    'Medium build': 'Medium',
                    'Heavy and solid': 'Well Built'
                },
                'text_mapping': [
                    (['thin', 'light', 'lean'], 'Thin and Lean'),
                    (['medium', 'average', 'moderate', 'build'], 'Medium'),
                    (['heavy', 'solid', 'large', 'stocky', 'built'], 'Well Built')
                ]
            },
            'q1': {
                'feature': 'Skin',
                'mapping': {
                    'Dry, rough, thin, cool': 'Dry,Rough',
                    'Sensitive, fair, warm, prone to rashes': 'Sensitive,Red',
                    'Oily, smooth, thick, cool': 'Oily,Smooth'
                },
                'text_mapping': [
                    (['dry', 'rough', 'flaky', 'thin', 'cool'], 'Dry,Rough'),
                    (['sensitive', 'fair', 'rash', 'red', 'warm'], 'Sensitive,Red'),
                    (['oily', 'smooth', 'greasy', 'thick'], 'Oily,Smooth')
                ]
            },
            'q2': {
                'feature': 'Type of Hair',
                'mapping': {
                    'Dry, curly, thin, brittle': 'Dry,Thin',
                    'Straight, fine, oily, premature graying': 'Oily,Thick',
                    'Thick, wavy, oily, lustrous': 'Normal,Medium'
                },
                'text_mapping': [
                    (['dry', 'curly', 'thin', 'brittle'], 'Dry,Thin'),
                    (['straight', 'fine', 'oily', 'graying'], 'Oily,Thick'),
                    (['thick', 'wavy', 'lustrous', 'normal', 'medium'], 'Normal,Medium')
                ]
            },
            'q3': {
                'feature': 'Body Energy',
                'mapping': {
                    'Variable, bursts of energy': 'High,Variable',
                    'Moderate, steady energy': 'Intense,Sharp',
                    'Slow to start but strong endurance': 'Steady,Slow'
                },
                'text_mapping': [
                    (['variable', 'bursts', 'erratic', 'high'], 'High,Variable'),
                    (['moderate', 'steady', 'intense', 'sharp'], 'Intense,Sharp'),
                    (['slow', 'endurance', 'consistent'], 'Steady,Slow')
                ]
            },
            'q4': {
                'feature': 'Eating Habit',
                'mapping': {
                    'Irregular, prone to gas and bloating': 'Irregular,Variable',
                    'Strong, can digest most foods, prone to heartburn': 'Intense,Sharp',
                    'Slow but steady, heavy after meals': 'Slow,Steady'
                },
                'text_mapping': [
                    (['irregular', 'gas', 'bloating', 'variable'], 'Irregular,Variable'),
                    (['strong', 'digest', 'heartburn', 'intense'], 'Intense,Sharp'),
                    (['slow', 'steady', 'heavy', 'meals'], 'Slow,Steady')
                ]
            },
            'q5': {
                'feature': 'Reaction under Adverse Situations',
                'mapping': {
                    'Worry, anxiety, nervousness': 'Anxiety,Worry',
                    'Anger, irritability, frustration': 'Anger,Irritability',
                    'Withdrawal, avoidance, holding on': 'Calm,Steady'
                },
                'text_mapping': [
                    (['worry', 'anxiety', 'nervousness', 'nervous'], 'Anxiety,Worry'),
                    (['anger', 'irritability', 'frustration', 'irritable'], 'Anger,Irritability'),
                    (['withdrawal', 'avoidance', 'holding on', 'calm', 'steady'], 'Calm,Steady')
                ]
            },
            'q6': {
                'feature': 'Sleep Pattern',
                'mapping': {
                    'Light sleeper, difficulty falling asleep': 'Light,Disturbed',
                    'Moderate sleeper, wakes up easily': 'Moderate,Interrupted',
                    'Heavy sleeper, difficult to wake up': 'Deep,Long'
                },
                'text_mapping': [
                    (['light sleeper', 'difficulty falling asleep', 'trouble sleeping'], 'Light,Disturbed'),
                    (['moderate sleeper', 'wakes up easily', 'interrupted'], 'Moderate,Interrupted'),
                    (['heavy sleeper', 'difficult to wake up', 'deep sleeper'], 'Deep,Long')
                ]
            },
            'q7': {
                'feature': 'Body Temperature',
                'mapping': {
                    'Cold hands and feet, prefer warmth': 'Cold',
                    'Warm, prefer cooler temperatures': 'Warm',
                    'Normal, prefer warm and dry': 'Cool'
                },
                'text_mapping': [
                    (['cold', 'hands', 'feet', 'warmth'], 'Cold'),
                    (['warm', 'prefer cool', 'cooler temperatures'], 'Warm'),
                    (['normal', 'prefer warm', 'dry'], 'Cool')
                ]
            },
            'q8': {
                'feature': 'Weather Conditions',
                'mapping': {
                    'Dislike cold and wind, prefer warm': 'Cold,Dry',
                    'Dislike heat, prefer cool': 'Hot,Humid',
                    'Dislike damp, prefer warmth and dry': 'Cool,Moist'
                },
                'text_mapping': [
                    (['dislike cold', 'wind', 'prefer warm', 'winter'], 'Cold,Dry'),
                    (['dislike heat', 'prefer cool', 'summer'], 'Hot,Humid'),
                    (['dislike damp', 'prefer warmth', 'dry', 'spring', 'autumn'], 'Cool,Moist')
                ]
            },
            'q9': {
                'feature': 'Nature',
                'mapping': {
                    'Enthusiastic, creative, talkative': 'Creative,Enthusiastic',
                    'Intense, focused, goal-oriented': 'Intense,Goal-oriented',
                    'Calm, steady, supportive': 'Calm,Steady'
                },
                'text_mapping': [
                    (['enthusiastic', 'creative', 'talkative', 'expressive'], 'Creative,Enthusiastic'),
                    (['intense', 'focused', 'goal-oriented', 'driven'], 'Intense,Goal-oriented'),
                    (['calm', 'steady', 'supportive', 'peaceful', 'relaxed'], 'Calm,Steady')
                ]
            },
            # Alternative keys mapping to the same features (for backward compatibility)
            'body_frame': {'feature': 'Body Frame', 'mapping': {}, 'text_mapping': []},
            'skin': {'feature': 'Skin', 'mapping': {}, 'text_mapping': []},
            'hair': {'feature': 'Type of Hair', 'mapping': {}, 'text_mapping': []},
            'energy': {'feature': 'Body Energy', 'mapping': {}, 'text_mapping': []},
            'digestion': {'feature': 'Eating Habit', 'mapping': {}, 'text_mapping': []},
            'stress': {'feature': 'Reaction under Adverse Situations', 'mapping': {}, 'text_mapping': []},
            'sleep': {'feature': 'Sleep Pattern', 'mapping': {}, 'text_mapping': []},
            'temperature': {'feature': 'Body Temperature', 'mapping': {}, 'text_mapping': []},
            'weather': {'feature': 'Weather Conditions', 'mapping': {}, 'text_mapping': []},
            'personality': {'feature': 'Nature', 'mapping': {}, 'text_mapping': []}
        }
        
        # Prepare input for the model with validation
        input_data = {}
        missing_fields = []
        
        try:
            # Get the answers from the form data
            answers = {}
            
            # Try different possible answer locations in the request
            if 'answers' in data and isinstance(data['answers'], dict):
                answers = data['answers']
                logger.info(f"Found answers in data['answers']: {answers}")
            else:
                # If no 'answers' key, use the entire data dict
                answers = {k: v for k, v in data.items() if k != 'csrf_token'}
                logger.info(f"Using entire data dict as answers: {answers}")
            
            logger.info(f"All available answers: {answers}")
            logger.info(f"Answer keys: {list(answers.keys())}")
            
            # Process each expected question
            for q_key, q_info in question_mapping.items():
                feature_name = q_info['feature']
                
                # Skip if we've already processed this feature
                if feature_name in input_data:
                    continue
                
                # Get the answer for this question
                answer = answers.get(q_key, '').strip()
                
                # Log the processing
                logger.info(f"Processing question {q_key} ({feature_name}): {answer}")
                
                # If answer is empty, try alternative keys
                if not answer and q_key in ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']:
                    # Try alternative keys for this question
                    alt_keys = {
                        'q0': ['body_frame'],
                        'q1': ['skin'],
                        'q2': ['hair'],
                        'q3': ['energy'],
                        'q4': ['digestion'],
                        'q5': ['stress'],
                        'q6': ['sleep'],
                        'q7': ['temperature'],
                        'q8': ['weather'],
                        'q9': ['personality']
                    }
                    for alt_key in alt_keys.get(q_key, []):
                        if alt_key in answers:
                            answer = str(answers[alt_key]).strip()
                            if answer:
                                logger.info(f"Found answer in alternative key {alt_key}: {answer}")
                                break
                
                # If answer is still empty, mark as missing
                if not answer:
                    missing_fields.append(feature_name)
                    logger.warning(f"Missing or empty answer for {feature_name} ({q_key})")
                    continue
                
                # Map the answer to the expected format if we have a mapping
                if q_key in question_mapping and question_mapping[q_key]['mapping']:
                    q_info = question_mapping[q_key]
                    mapped_value = None
                    
                    # First try exact match with mapped values
                    if answer in q_info['mapping']:
                        mapped_value = q_info['mapping'][answer]
                        logger.info(f"Exact match: Mapped '{answer}' to '{mapped_value}' for {feature_name}")
                    else:
                        # Try to match based on text patterns
                        answer_lower = answer.lower()
                        for patterns, value in q_info.get('text_mapping', []):
                            if any(pattern in answer_lower for pattern in patterns):
                                mapped_value = value
                                logger.info(f"Pattern match: Mapped '{answer}' to '{mapped_value}' for {feature_name}")
                                break
                    
                    if mapped_value is not None:
                        input_data[feature_name] = mapped_value
                    else:
                        # If no match found, try to use the answer as is
                        input_data[feature_name] = answer
                        logger.warning(f"Could not map answer '{answer}' for {feature_name}. Using as is.")
                else:
                    # For alternative keys without direct mapping, use the answer as is
                    input_data[feature_name] = answer
            
            # Log the prepared input data
            logger.info(f"Prepared input data for prediction: {input_data}")
            
            # If any required fields are missing, raise an error
            if missing_fields:
                raise ValueError(f"Missing or invalid answers for fields: {', '.join(missing_fields)}")
            
        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error processing your answers. Please try again.',
                'status': 'error',
                'details': str(e) if app.debug else None
            }), 400
        
        # Make prediction using the ML model with timeout
        try:
            logger.info("Starting prediction...")
            prediction_start = time.time()
                
            # Get the predictor instance and make prediction with timeout (10 seconds)
            predictor = get_dosha_predictor()
            if predictor is None:
                raise ValueError("Failed to initialize Dosha predictor")
                    
            predicted_dosha, confidence_scores = predictor.predict(input_data, timeout_seconds=10)
            
            if not predicted_dosha or not confidence_scores:
                raise ValueError("No valid prediction returned from model")
                
            logger.info(f"Prediction completed in {time.time() - prediction_start:.2f} seconds")
            
            # Ensure confidence scores are in the correct format (0-100%)
            formatted_scores = {}
            for dosha in ['Vata', 'Pitta', 'Kapha']:
                score = confidence_scores.get(dosha, 0)
                # Ensure score is a number between 0 and 100
                try:
                    score = float(score)
                    if score < 0 or score > 100:
                        logger.warning(f"Score out of range for {dosha}: {score}")
                        score = max(0, min(100, score))  # Clamp to 0-100 range
                    formatted_scores[dosha] = round(score, 1)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid score for {dosha}: {score}")
                    formatted_scores[dosha] = 0.0
            
            # Sort scores by value in descending order
            confidence_scores = dict(sorted(
                formatted_scores.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
            logger.info(f"Prediction successful - Dosha: {predicted_dosha}, Scores: {confidence_scores}")
            
        except TimeoutError as te:
            logger.error(f"Prediction timed out: {str(te)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Prediction timed out. Please try again.',
                'status': 'error'
            }), 504  # Gateway Timeout
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'An error occurred during prediction.',
                'status': 'error',
                'details': str(e) if app.debug else None
            }), 500
        
        # Save results to database with timeout
        try:
            with get_db() as db:
                cursor = db.cursor(dictionary=True)
                # Delete any previous test results for this user
                cursor.execute('DELETE FROM user_dosha WHERE user_id = %s', (session['user_id'],))
                
                # Insert new test result with current timestamp
                cursor.execute('''
                    INSERT INTO user_dosha (user_id, dosha_type, confidence_scores, test_date)
                    VALUES (%s, %s, %s, NOW())
                ''', (
                    session['user_id'],
                    predicted_dosha,
                    json.dumps(confidence_scores)
                ))
                db.commit()
                
                logger.info(f"Dosha test completed in {time.time() - start_time:.2f}s for user {session['user_id']}")
                
        except Exception as db_error:
            db.rollback()
            logger.error(f"Database error in process_dosha_test: {str(db_error)}")
            # Continue with the response even if DB save fails
        
        # Return success response with results
        return jsonify({
            'success': True,
            'redirect': url_for('dosha_results'),
            'dosha': predicted_dosha,
            'scores': confidence_scores,
            'processing_time': f"{time.time() - start_time:.2f}s"
        })

    except TimeoutError as te:
        logger.error(f"Dosha test processing timeout: {str(te)}")
        return jsonify({
            'success': False,
            'error': 'Test processing took too long. Please try again.',
            'status': 'timeout',
            'details': str(te) if app.debug else None
        }), 408
        
    except Exception as e:
        logger.error(f"Unexpected error in process_dosha_test: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred while processing your test. Please try again.',
            'status': 'error',
            'details': str(e) if app.debug else None
        }), 500

if __name__ == '__main__':
    # The predictor will be initialized on first use via get_dosha_predictor()
    app.run(debug=True)