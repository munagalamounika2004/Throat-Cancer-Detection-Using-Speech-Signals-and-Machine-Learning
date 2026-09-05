import os
import numpy as np
import librosa
import tensorflow as tf
import sqlite3
import json
import uuid
import secrets
import subprocess
import glob
from datetime import datetime, timedelta 
from io import BytesIO
import sys
import tempfile 
import shutil
import time 
import threading
# Imports required for Flask
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    jsonify,
    send_from_directory,
    abort
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import zipfile
from collections import defaultdict

# Imports required for PDF generation with ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from functools import wraps

# Fix for SQLite3 datetime deprecation warning
sqlite3.register_adapter(datetime, lambda val: val.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda val: datetime.fromisoformat(val.decode()))

# -------------------------------
# Utility Functions
# -------------------------------
def admin_required(f):
    """Decorator to restrict access to admin users only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'admin':
            flash("You do not have permission to view this page.", "danger")
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def get_db_connection():
    """Establishes and returns a SQLite database connection."""
    conn = sqlite3.connect('voice_analysis.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and creates a default admin user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users Table - Updated with age and gender
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Submissions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            result TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            admin_override TEXT,
            user_notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # User Activities Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_type TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Notifications Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Data Labeling Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            labeled_by INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (labeled_by) REFERENCES users (id)
        )
    """)
    
    # User Feedback Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_reviewed INTEGER DEFAULT 0,
            reviewed_by INTEGER,
            reviewed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (reviewed_by) REFERENCES users (id)
        )
    """)
    
    # Security Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event_type TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()

    # Migration check for 'is_active'
    try:
        conn.execute("SELECT is_active FROM users LIMIT 1")
    except sqlite3.OperationalError:
        print("Database migration: Adding 'is_active' column to 'users' table.")
        conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        conn.commit()
    
    # Migration check for 'age' and 'gender'
    try:
        conn.execute("SELECT age FROM users LIMIT 1")
    except sqlite3.OperationalError:
        print("Database migration: Adding 'age' column to 'users' table.")
        conn.execute("ALTER TABLE users ADD COLUMN age INTEGER")
        conn.commit()
    
    try:
        conn.execute("SELECT gender FROM users LIMIT 1")
    except sqlite3.OperationalError:
        print("Database migration: Adding 'gender' column to 'users' table.")
        conn.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        conn.commit()
    
    # Migration check for 'notifications' table
    try:
        conn.execute("SELECT * FROM notifications LIMIT 1")
    except sqlite3.OperationalError:
        print("Database migration: 'notifications' table already exists.")
    
    # Create default admin user if one doesn't exist
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    admin_password = os.environ.get('ADMIN_PASSWORD', 'adminpass')
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
    
    admin_user = conn.execute("SELECT * FROM users WHERE username = ?", (admin_username,)).fetchone()
    
    if not admin_user:
        hashed_password = generate_password_hash(admin_password)
        conn.execute("INSERT INTO users (username, password, email, role, is_active) VALUES (?, ?, ?, ?, ?)",
                     (admin_username, hashed_password, admin_email, 'admin', 1))
        conn.commit()
        print(f"Default admin user '{admin_username}' created.")
    
    conn.close()

# -------------------------------
# Helper Functions
# -------------------------------
def log_user_activity(user_id, activity_type, description):
    """Logs a user activity to the database."""
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO user_activities (user_id, activity_type, description) VALUES (?, ?, ?)",
        (user_id, activity_type, description)
    )
    conn.commit()
    conn.close()

def add_notification(user_id, type, title, message):
    """Adds a notification for a user."""
    conn = get_db_connection()
    try:
        now_utc = datetime.utcnow()  # Use UTC time
        conn.execute(
            "INSERT INTO notifications (user_id, type, title, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, type, title, message, now_utc)
        )
        conn.commit()
    except Exception as e:
        print(f"Error adding notification: {e}")
    finally:
        conn.close()

def check_test_reminders():
    """Checks for users who haven't taken a test in a while and adds reminder notifications."""
    conn = get_db_connection()
    
    # Get all users
    users = conn.execute("SELECT id FROM users WHERE role = 'user'").fetchall()
    
    for user in users:
        user_id = user['id']
        
        # Get the last submission date for this user
        last_submission = conn.execute(
            "SELECT created_at FROM submissions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
        
        # If the user has no submissions or the last submission was more than 7 days ago
        if not last_submission:
            # User has never taken a test
            # Check if we already sent a welcome notification
            welcome_notification = conn.execute(
                "SELECT id FROM notifications WHERE user_id = ? AND title = ?",
                (user_id, "Welcome to Voice Health AI")
            ).fetchone()
            
            if not welcome_notification:
                add_notification(
                    user_id,
                    "info",
                    "Welcome to Voice Health AI",
                    "Thank you for joining our platform. Start by taking your first voice test."
                )
        else:
            # Check if the last submission was more than 7 days ago
            last_date = datetime.fromisoformat(last_submission['created_at'])
            if (datetime.now() - last_date).days > 7:
                # Check if we already sent a reminder in the last 3 days
                recent_reminder = conn.execute(
                    "SELECT id FROM notifications WHERE user_id = ? AND title = ? AND created_at > ?",
                    (user_id, "Test Reminder", (datetime.now() - timedelta(days=3)).isoformat())
                ).fetchone()
                
                if not recent_reminder:
                    add_notification(
                        user_id,
                        "warning",
                        "Test Reminder",
                        "It's been a while since your last voice test. Consider taking another test to monitor your voice health."
                    )
    
    conn.close()

def reminder_scheduler():
    """Runs in the background and checks for test reminders every 24 hours."""
    while True:
        time.sleep(86400)  # Sleep for 24 hours
        check_test_reminders()

# -------------------------------
# Suppress TensorFlow warnings
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
app.secret_key = secrets.token_hex(24) 
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# --- FOLDER SETUP ---
UPLOAD_FOLDER = "dataset/uploads" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("ml_models", exist_ok=True)
os.makedirs("ml_models_up", exist_ok=True)
os.makedirs("dataset/uploads/healthy_voices", exist_ok=True)
os.makedirs("dataset/uploads/diseased_voices", exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

# Initialize DB
with app.app_context():
    init_db()

# Start the scheduler when the app starts
scheduler_thread = threading.Thread(target=reminder_scheduler, daemon=True)
scheduler_thread.start()

# -------------------------------
# User Model for Flask-Login
# -------------------------------
class User(UserMixin):
    def __init__(self, id, username, email, role, is_active=1, age=None, gender=None):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.is_active_status = is_active 
        self.age = age
        self.gender = gender
    
    def is_active(self):
        return self.is_active_status == 1

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_data = conn.execute("SELECT id, username, email, role, is_active, age, gender FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(
            user_data['id'], 
            user_data['username'], 
            user_data['email'], 
            user_data['role'], 
            user_data['is_active'],
            user_data['age'],
            user_data['gender']
        )
    return None

# -------------------------------
# ML Model Setup (Stub functions for dependency)
# -------------------------------
PRIMARY_MODEL_PATH = "ml_models/final_cnn_bilstm.h5"
UPLOAD_MODEL_PATH = "ml_models_up/updated_cnn_bilstm.h5" 
CLASSES = ["healthy_voices", "diseased_voices"]
SAMPLE_RATE = 16000
MAX_LEN = 160
N_MELS = 64

# Dummy feature stats for initial run
FEATURE_MEAN = np.zeros(N_MELS)
FEATURE_STD = np.ones(N_MELS)

try:
    if os.path.exists("ml_models/feature_mean.npy") and os.path.exists("ml_models/feature_std.npy"):
        FEATURE_MEAN = np.load("ml_models/feature_mean.npy")
        FEATURE_STD = np.load("ml_models/feature_std.npy")
except Exception:
    pass

def get_current_model():
    """Dynamically loads the active model (Uploads model preferred) and its stats."""
    
    if os.path.exists(UPLOAD_MODEL_PATH):
        try:
            mean = np.load("ml_models_up/feature_mean.npy") if os.path.exists("ml_models_up/feature_mean.npy") else FEATURE_MEAN
            std = np.load("ml_models_up/feature_std.npy") if os.path.exists("ml_models_up/feature_std.npy") else FEATURE_STD
            model = tf.keras.models.load_model(UPLOAD_MODEL_PATH, compile=False)
            return model, mean, std, "Uploaded Model"
        except Exception as e:
            print(f"Error loading uploaded model/stats: {e}. Falling back to primary.")

    if os.path.exists(PRIMARY_MODEL_PATH):
        try:
            model = tf.keras.models.load_model(PRIMARY_MODEL_PATH, compile=False)
            return model, FEATURE_MEAN, FEATURE_STD, "Primary Model"
        except Exception as e:
            print(f"Error loading primary model: {e}")

    return None, FEATURE_MEAN, FEATURE_STD, "No Model Found"

def extract_features(file_path):
    """Extracts Mel-spectrogram features for prediction."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y)
    y = y / (np.max(np.abs(y)) + 1e-9) 
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).T

    if mel_db.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mel_db.shape[0]
        mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:MAX_LEN, :]
    return mel_db

def predict_voice(file_path):
    """Performs prediction using the currently active model."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return "File not found", 0.0, "N/A"
        
        model, feat_mean, feat_std, model_source = get_current_model()
        
        if model is None:
            return "No Model", 0.0, "N/A"
            
        feat = extract_features(file_path)
        feat = np.expand_dims(feat, axis=0)
        
        feat_norm = (feat - feat_mean) / (feat_std + 1e-9)
        
        pred = model.predict(feat_norm)[0]
        class_idx = int(np.argmax(pred))
        confidence = float(pred[class_idx])
        
        result_class = CLASSES[class_idx].replace('_voices', '').capitalize()
        
        return result_class, confidence, model_source
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error: {str(e)}", 0.0, "N/A"


@app.route('/api/unread-feedback-count')
@admin_required
def api_unread_feedback_count():
    """Returns the count of unread feedback."""
    conn = get_db_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE is_reviewed = 0"
    ).fetchone()[0]
    conn.close()
    return jsonify({"count": count})

def get_model_metrics(model_path):
    """Loads and returns model metrics from the specified path."""
    metrics_file = os.path.join(os.path.dirname(model_path), "model_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

# -------------------------------
# PDF GENERATION FUNCTION
# -------------------------------
def generate_report_pdf(submission):
    """Generates a PDF report for a submission."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            title=f"Voice Analysis Report - Submission {submission['id']}")
    styles = getSampleStyleSheet()
    Story = []

    # Title
    Story.append(Paragraph("Voice Analysis Report", styles['Title']))
    Story.append(Spacer(1, 0.25 * inch))

    # General Info
    data = [
        ['Submission ID:', str(submission['id'])],
        ['User:', submission.get('username', current_user.username)],
        ['Date of Submission:', submission['created_at'].split('.')[0] if isinstance(submission['created_at'], str) else str(submission['created_at'])],
        ['File Path:', os.path.basename(submission['file_path'])],
    ]
    t = Table(data, colWidths=[2 * inch, 5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ]))
    Story.append(t)
    Story.append(Spacer(1, 0.25 * inch))

    # Prediction Result
    Story.append(Paragraph("<h2>Model Prediction Result</h2>", styles['Heading2']))
    
    result_style = ParagraphStyle('Result', parent=styles['Normal'], fontSize=16, alignment=TA_CENTER)
    
    result_class = submission['result']
    confidence = submission['confidence']

    result_color = colors.green if result_class == 'Healthy' else colors.red
    
    prediction_data = [
        ['Result:', Paragraph(result_class, result_style)],
        ['Confidence:', f"{confidence * 100:.2f}%"]
    ]
    t_result = Table(prediction_data, colWidths=[2 * inch, 5 * inch])
    t_result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),
    ]))
    Story.append(t_result)
    Story.append(Spacer(1, 0.25 * inch))

    # Admin Override/User Notes
    if submission.get('admin_override') or submission.get('user_notes'):
        Story.append(Paragraph("<h2>Additional Notes</h2>", styles['Heading2']))
        if submission.get('admin_override'):
            Story.append(Paragraph(f"<b>Admin Override:</b> {submission['admin_override']}", styles['Normal']))
        if submission.get('user_notes'):
            Story.append(Paragraph(f"<b>User Notes:</b> {submission['user_notes']}", styles['Normal']))
        Story.append(Spacer(1, 0.25 * inch))

    doc.build(Story)
    buffer.seek(0)
    return buffer

# -------------------------------
# Core Application Routes
# -------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        
        # Check if username or password is empty
        if not username or not password:
            flash('Please enter both username and password.', 'warning')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        user_data = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data['id'], user_data['username'], user_data['email'], user_data['role'], user_data['is_active'])
            
            if user.is_active_status != 1:
                # Log failed login attempt due to inactive account
                conn.execute("""
                    INSERT INTO security_logs (user_id, event_type, ip_address, user_agent, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_data['id'], 'login_failure_inactive', ip_address, user_agent, 'Account is inactive'))
                conn.commit()
                
                flash('Your account has been blocked by an administrator. Please contact support.', 'danger')
                return redirect(url_for('login'))
            
            # Log successful login
            conn.execute("""
                INSERT INTO security_logs (user_id, event_type, ip_address, user_agent)
                VALUES (?, ?, ?, ?)
            """, (user_data['id'], 'login_success', ip_address, user_agent))
            conn.commit()
            
            # Log user activity
            log_user_activity(user_data['id'], 'login', 'Logged in to account')
            
            login_user(user, remember=True)
            flash(f'Welcome back, {user.username}!', 'success')
            
            # Check for test reminders
            check_test_reminders()
            
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        else:
            # Log failed login attempt
            user_id = user_data['id'] if user_data else None
            conn.execute("""
                INSERT INTO security_logs (user_id, event_type, ip_address, user_agent, details)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, 'login_failure', ip_address, user_agent, 'Invalid username or password'))
            conn.commit()
            
            flash('Invalid username or password. Please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    # Log user activity
    log_user_activity(current_user.id, 'logout', 'Logged out of account')
    
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        gender = request.form.get('gender')

        if not (username and email and password):
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO users (username, password, email, role, age, gender) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, hashed_password, email, 'user', age, gender))
            user_id = cursor.lastrowid
            conn.commit()
            
            # Log user activity
            log_user_activity(user_id, 'registration', 'Created a new account')
            
            # Add welcome notification
            add_notification(
                user_id,
                "info",
                "Welcome to Voice Health AI",
                "Thank you for joining our platform. Start by taking your first voice test."
            )
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.', 'danger')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return render_template('index.html')  # Render the landing page for non-authenticated users

# -------------------------------
# User Routes 
# -------------------------------
@app.route('/user_dashboard')
@login_required
def user_dashboard():
    conn = get_db_connection()
    submissions = conn.execute("SELECT * FROM submissions WHERE user_id = ? ORDER BY created_at DESC", 
                               (current_user.id,)).fetchall()
    
    total_submissions = len(submissions)
    healthy_count = conn.execute("SELECT COUNT(id) FROM submissions WHERE user_id = ? AND result = 'Healthy'",
                                (current_user.id,)).fetchone()[0]
    diseased_count = total_submissions - healthy_count
    
    last_submission = None
    if submissions:
        # Convert the Row object to a dictionary
        last_submission = dict(submissions[0])
    
    # Calculate min values for goals
    goal1_progress = min(total_submissions, 5)
    goal2_target = total_submissions if total_submissions > 0 else 1
    
    # Get user activities
    activities = conn.execute("""
        SELECT activity_type, description, created_at 
        FROM user_activities 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    """, (current_user.id,)).fetchall()
    
    # Get user goals
    goals = [
        {
            "id": 1,
            "title": "Complete 5 Voice Tests",
            "description": "Complete 5 voice tests to establish a baseline for your voice health.",
            "status": "active" if total_submissions < 5 else "completed",
            "progress": goal1_progress,
            "target": 5,
            "action": "Take Test",
            "actionLink": "voice-test"
        },
        {
            "id": 2,
            "title": "Improve Voice Health",
            "description": "Achieve more healthy test results than diseased ones.",
            "status": "completed" if healthy_count > (total_submissions - healthy_count) else "active",
            "progress": healthy_count,
            "target": goal2_target,
            "action": "Health Tips",
            "actionLink": "health-tips"
        },
        {
            "id": 3,
            "title": "Complete Profile",
            "description": "Fill in all your profile information including username, email, age, and gender.",
            "status": "completed",
            "progress": 4,
            "target": 4,
            "action": "View Profile",
            "actionLink": "profile"
        }
    ]
    
    # Get user notifications
    notifications = conn.execute("""
        SELECT id, type, title, message, created_at 
        FROM notifications 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    """, (current_user.id,)).fetchall()
    
    # Format notifications
    formatted_notifications = []
    for notification in notifications:
        # Calculate time ago
        created_at = datetime.fromisoformat(notification['created_at'])
        now = datetime.now()
        diff = now - created_at
        
        if diff.days > 0:
            time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            time_ago = "Just now"
        
        formatted_notifications.append({
            "id": notification['id'],
            "type": notification['type'],
            "title": notification['title'],
            "text": notification['message'],
            "time": time_ago
        })
    
    # Get user recommendations
    recommendations = []
    
    # Basic recommendations for all users
    recommendations.append({
        "id": 1,
        "icon": "fas fa-microphone-alt",
        "title": "Improve Recording Quality",
        "text": "Record in a quiet environment to get more accurate results.",
        "action": "Learn More"
    })
    
    recommendations.append({
        "id": 2,
        "icon": "fas fa-calendar-check",
        "title": "Regular Testing Schedule",
        "text": "Test your voice weekly to track changes over time.",
        "action": "Set Reminder"
    })
    
    # Add specific recommendation based on last test result
    if last_submission and last_submission['result'] == 'Diseased':
        recommendations.append({
            "id": 3,
            "icon": "fas fa-user-md",
            "title": "Professional Consultation",
            "text": "Based on your test results, consider consulting a voice specialist.",
            "action": "Find Specialists"
        })
    
    # Add recommendation based on test frequency
    if total_submissions > 0:
        # Calculate average time between tests
        if len(submissions) > 1:
            first_test = datetime.fromisoformat(submissions[-1]['created_at'])
            last_test = datetime.fromisoformat(submissions[0]['created_at'])
            days_diff = (last_test - first_test).days
            avg_days_between_tests = days_diff / (len(submissions) - 1)
            
            if avg_days_between_tests > 14:
                recommendations.append({
                    "id": 4,
                    "icon": "fas fa-clock",
                    "title": "Test More Frequently",
                    "text": "Consider testing your voice more frequently to better track changes.",
                    "action": "Set Reminder"
                })
    
    # Get user settings
    settings = [
        {
            "id": "emailNotifications",
            "name": "Email Notifications",
            "description": "Receive email updates about your voice health",
            "enabled": True
        },
        {
            "id": "testReminders",
            "name": "Test Reminders",
            "description": "Get reminders to take regular voice tests",
            "enabled": True
        },
        {
            "id": "healthTips",
            "name": "Health Tips",
            "description": "Receive personalized voice health tips",
            "enabled": True
        },
        {
            "id": "dataSharing",
            "name": "Data Sharing",
            "description": "Share anonymous data to improve our AI models",
            "enabled": False
        }
    ]
    
    conn.close()
    
    return render_template('user_dashboard.html', 
                            submissions=submissions,
                            total_submissions=total_submissions,
                            healthy_count=healthy_count,
                            diseased_count=diseased_count,
                            last_submission=last_submission,
                            activities=activities,
                            goals=goals,
                            notifications=formatted_notifications,
                            recommendations=recommendations,
                            settings=settings)

@app.route('/user_predict_voice', methods=['POST'])
@login_required
def user_predict_voice():
    """Handles AJAX file upload for prediction and returns a JSON response."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        user_upload_dir = os.path.join(UPLOAD_FOLDER, str(current_user.id))
        os.makedirs(user_upload_dir, exist_ok=True)
        
        file_path = os.path.join(user_upload_dir, filename)
        
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{uuid.uuid4().hex[:4]}{ext}"
            file_path = os.path.join(user_upload_dir, filename)
            
        file.save(file_path)

        result_class, confidence, model_source = predict_voice(file_path)
        
        if result_class == "Error!" or result_class == "No Model":
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": f"Prediction failed: {result_class}. Please try a valid .wav file."}), 500

        conn = get_db_connection()
        cursor = conn.execute(
            "INSERT INTO submissions (user_id, file_path, result, confidence) VALUES (?, ?, ?, ?)",
            (current_user.id, file_path, result_class, confidence)
        )
        submission_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Log user activity
        log_user_activity(
            current_user.id, 
            "voice_test", 
            f"Completed voice test: {result_class} with {confidence*100:.2f}% confidence"
        )
        
        # Add notification
        add_notification(
            current_user.id,
            "success",
            "Voice Test Completed",
            f"Your voice test result is: {result_class} with {confidence*100:.2f}% confidence"
        )

        return jsonify({
            "success": True, 
            "result_class": result_class, 
            "confidence": f"{confidence * 100:.2f}",
            "submission_id": submission_id
        })
    
    return jsonify({"success": False, "message": "File upload failed."}), 500

@app.route('/serve_uploaded_file/<path:filename>')
@login_required
def serve_uploaded_file(filename):
    """
    Serves an audio file from a user's upload directory.
    The 'filename' parameter from the URL is expected to be 'user_id/filename.wav'.
    """
    # Construct the full path from the base UPLOAD_FOLDER and the provided filename.
    # e.g., if UPLOAD_FOLDER is "dataset/uploads" and filename is "123/some_file.wav",
    # the resulting file_path will be "dataset/uploads/123/some_file.wav".
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # --- Security Check ---
    # 1. Ensure the user is only trying to access files from their own subdirectory.
    #    The filename from the URL should start with the current user's ID.
    if not filename.startswith(f"{current_user.id}{os.sep}"):
        print(f"SECURITY ALERT: User {current_user.id} tried to access file '{filename}' outside their directory.")
        abort(403)  # Forbidden

    # 2. Prevent directory traversal attacks (e.g., trying to access '../../etc/passwd').
    if not os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_FOLDER)):
        print(f"SECURITY ALERT: Path traversal attempt for '{file_path}'.")
        abort(403)  # Forbidden

    # --- File Existence Check ---
    if not os.path.exists(file_path):
        print(f"File not found at path: {file_path}")
        abort(404)  # Not Found

    # --- Serve the File ---
    try:
        # send_from_directory is the safest way to serve files.
        # We need to split the full path into the directory and the actual filename.
        directory, actual_filename = os.path.split(file_path)
        
        return send_from_directory(
            directory,
            actual_filename,
            mimetype='audio/wav',
            as_attachment=False  # This makes the browser play the file instead of downloading it.
        )
    except Exception as e:
        print(f"Error serving audio file '{file_path}': {e}")
        abort(500) # Internal Server Error

@app.route('/download_user_report/<int:sub_id>')
@login_required
def download_user_report(sub_id): 
    conn = get_db_connection()
    # Query only necessary fields for PDF generation
    submission = conn.execute("SELECT id, user_id, file_path, result, confidence, created_at, admin_override, user_notes FROM submissions WHERE id = ? AND user_id = ?", (sub_id, current_user.id)).fetchone()
    conn.close()

    if not submission:
        flash("Submission not found or you are not authorized.", "danger")
        return redirect(url_for('user_dashboard'))

    try:
        # Pass the dictionary/Row object to the generator
        buffer = generate_report_pdf(dict(submission))
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'voice_analysis_report_{sub_id}.pdf'
        )
    except Exception as e:
        # IMPORTANT: Log the full error to help debugging
        print(f"FATAL ERROR during PDF generation for user {current_user.id}: {e}")
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for('user_dashboard'))

@app.route('/user/update_notes/<int:sub_id>', methods=['POST'])
@login_required
def update_notes(sub_id):
    """Updates user notes for a submission via AJAX."""
    notes = request.form.get('notes', '') 
    
    conn = get_db_connection()
    submission = conn.execute("SELECT * FROM submissions WHERE id = ? AND user_id = ?", (sub_id, current_user.id)).fetchone()
    
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized."}), 404

    conn.execute("UPDATE submissions SET user_notes=? WHERE id=? AND user_id=?", (notes, sub_id, current_user.id))
    conn.commit()
    conn.close()
    
    # Log user activity
    log_user_activity(current_user.id, 'notes_update', f'Updated notes for submission #{sub_id}')
    
    return jsonify({"success": True, "message": "Notes updated successfully!", "notes": notes})

@app.route('/user/delete_notes/<int:sub_id>', methods=['POST'])
@login_required
def delete_user_notes(sub_id):
    """Deletes user notes for a submission via AJAX."""
    conn = get_db_connection()
    submission = conn.execute("SELECT * FROM submissions WHERE id = ? AND user_id = ?", (sub_id, current_user.id)).fetchone()
    
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized."}), 404

    conn.execute("UPDATE submissions SET user_notes=NULL WHERE id=? AND user_id=?", (sub_id, current_user.id))
    conn.commit()
    conn.close()
    
    # Log user activity
    log_user_activity(current_user.id, 'notes_delete', f'Deleted notes for submission #{sub_id}')
    
    return jsonify({"success": True, "message": "Notes deleted successfully!"})

@app.route('/user/delete_submission/<int:sub_id>', methods=['POST'])
@login_required
def delete_user_submission(sub_id):
    """Deletes an entire submission (record and file) via AJAX."""
    conn = get_db_connection()
    submission = conn.execute("SELECT * FROM submissions WHERE id = ? AND user_id = ?", (sub_id, current_user.id)).fetchone()
    
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized."}), 404

    file_path = submission['file_path']
    
    try:
        # 1. Delete file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # 2. Delete record from database
        conn.execute("DELETE FROM submissions WHERE id=? AND user_id=?", (sub_id, current_user.id))
        conn.commit()
        conn.close()
        
        # Log user activity
        log_user_activity(current_user.id, 'submission_delete', f'Deleted submission #{sub_id}')
        
        return jsonify({"success": True, "message": f"Submission {sub_id} and associated file deleted successfully!"})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Error deleting submission: {str(e)}"}), 500

# -------------------------------
# New User Dashboard Routes
# -------------------------------
@app.route('/user/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Updates user profile information."""
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    age = data.get('age')
    gender = data.get('gender')
    
    # Validate required fields
    if not username or not email:
        return jsonify({"success": False, "message": "Username and email are required."})
    
    conn = get_db_connection()
    
    # Check if username or email already exists for another user
    existing_user = conn.execute(
        "SELECT id FROM users WHERE (username = ? OR email = ?) AND id != ?",
        (username, email, current_user.id)
    ).fetchone()
    
    if existing_user:
        conn.close()
        return jsonify({"success": False, "message": "Username or email already exists."})
    
    try:
        # Update user profile
        conn.execute(
            "UPDATE users SET username = ?, email = ?, age = ?, gender = ? WHERE id = ?",
            (username, email, age, gender, current_user.id)
        )
        conn.commit()
        conn.close()
        
        # Log user activity
        log_user_activity(
            current_user.id, 
            "profile_update", 
            "Updated profile information"
        )
        
        # Add notification
        add_notification(
            current_user.id,
            "success",
            "Profile Updated",
            "Your profile information has been successfully updated"
        )
        
        return jsonify({"success": True, "message": "Profile updated successfully!"})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Error updating profile: {str(e)}"})

@app.route('/user/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')
    
    # Validate required fields
    if not current_password or not new_password or not confirm_password:
        return jsonify({"success": False, "message": "All password fields are required."})
    
    # Check if new password and confirmation match
    if new_password != confirm_password:
        return jsonify({"success": False, "message": "New password and confirmation do not match."})
    
    conn = get_db_connection()
    user_data = conn.execute("SELECT password FROM users WHERE id = ?", (current_user.id,)).fetchone()
    
    # Verify current password
    if not check_password_hash(user_data['password'], current_password):
        conn.close()
        return jsonify({"success": False, "message": "Current password is incorrect."})
    
    try:
        # Update password
        hashed_password = generate_password_hash(new_password)
        conn.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (hashed_password, current_user.id)
        )
        conn.commit()
        conn.close()
        
        # Log user activity
        log_user_activity(
            current_user.id, 
            "password_change", 
            "Changed account password"
        )
        
        # Add notification
        add_notification(
            current_user.id,
            "success",
            "Password Changed",
            "Your password has been successfully changed"
        )
        
        return jsonify({"success": True, "message": "Password changed successfully!"})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Error changing password: {str(e)}"})

@app.route('/user/activities')
@login_required
def user_activities():
    """Returns user activities."""
    conn = get_db_connection()
    activities = conn.execute(
        "SELECT id, activity_type, description, created_at FROM user_activities WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
        (current_user.id,)
    ).fetchall()
    
    # Format the activities for the frontend
    formatted_activities = []
    for activity in activities:
        # Calculate time ago using UTC
        created_at = datetime.fromisoformat(activity['created_at'])
        now = datetime.utcnow()  # Use UTC for comparison
        diff = now - created_at
        
        if diff.days > 0:
            time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            time_ago = "Just now"
        
        formatted_activities.append({
            'id': activity['id'],
            'type': activity['activity_type'],
            'text': activity['description'],
            'time': time_ago,
            'created_at': activity['created_at']  # Keep original timestamp for formatting
        })
    
    conn.close()
    return jsonify({"success": True, "activities": formatted_activities})

@app.route('/user/notifications')
@login_required
def user_notifications():
    """Returns user notifications."""
    conn = get_db_connection()
    
    # Get user notifications from the database
    notifications_data = conn.execute(
        "SELECT id, type, title, message, created_at FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
        (current_user.id,)
    ).fetchall()
    
    # Format the notifications for the frontend
    notifications = []
    for notification in notifications_data:
        # Calculate time ago using UTC time
        created_at = datetime.fromisoformat(notification['created_at'])
        now = datetime.utcnow()  # Use UTC for comparison
        diff = now - created_at
        
        if diff.days > 0:
            time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            time_ago = "Just now"
        
        notifications.append({
            "id": notification['id'],
            "type": notification['type'],
            "title": notification['title'],
            "text": notification['message'],
            "time": time_ago
        })
    
    conn.close()
    return jsonify({"success": True, "notifications": notifications})

@app.route('/user/mark_notification_read/<int:notification_id>', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    """Marks a notification as read."""
    conn = get_db_connection()
    
    # Check if the notification belongs to the current user
    notification = conn.execute(
        "SELECT id FROM notifications WHERE id = ? AND user_id = ?",
        (notification_id, current_user.id)
    ).fetchone()
    
    if not notification:
        conn.close()
        return jsonify({"success": False, "message": "Notification not found or you are not authorized."})
    
    try:
        conn.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ?",
            (notification_id,)
        )
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Notification marked as read."})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Error marking notification as read: {str(e)}"})

@app.route('/user/recommendations')
@login_required
def user_recommendations():
    """Returns personalized recommendations for the user."""
    conn = get_db_connection()
    
    # Get user's last submission
    last_submission = conn.execute(
        "SELECT result, confidence FROM submissions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (current_user.id,)
    ).fetchone()
    
    # Get all submissions
    all_submissions = conn.execute(
        "SELECT created_at FROM submissions WHERE user_id = ? ORDER BY created_at",
        (current_user.id,)
    ).fetchall()
    
    # Generate recommendations based on user's history
    recommendations = []
    
    # Basic recommendations for all users
    recommendations.append({
        "id": 1,
        "icon": "fas fa-microphone-alt",
        "title": "Improve Recording Quality",
        "text": "Record in a quiet environment to get more accurate results.",
        "action": "Learn More"
    })
    
    recommendations.append({
        "id": 2,
        "icon": "fas fa-calendar-check",
        "title": "Regular Testing Schedule",
        "text": "Test your voice weekly to track changes over time.",
        "action": "Set Reminder"
    })
    
    # Add specific recommendation based on last test result
    if last_submission:
        if last_submission['result'] == 'Diseased':
            recommendations.append({
                "id": 3,
                "icon": "fas fa-user-md",
                "title": "Professional Consultation",
                "text": "Based on your test results, consider consulting a voice specialist.",
                "action": "Find Specialists"
            })
    
    # Add recommendation based on test frequency
    if len(all_submissions) > 1:
        # Calculate average time between tests
        first_test = datetime.fromisoformat(all_submissions[0]['created_at'])
        last_test = datetime.fromisoformat(all_submissions[-1]['created_at'])
        days_diff = (last_test - first_test).days
        avg_days_between_tests = days_diff / (len(all_submissions) - 1)
        
        if avg_days_between_tests > 14:
            recommendations.append({
                "id": 4,
                "icon": "fas fa-clock",
                "title": "Test More Frequently",
                "text": "Consider testing your voice more frequently to better track changes.",
                "action": "Set Reminder"
            })
    
    # Add recommendation based on test results consistency
    if len(all_submissions) >= 3:
        # Get last 3 results
        last_results = conn.execute(
            "SELECT result FROM submissions WHERE user_id = ? ORDER BY created_at DESC LIMIT 3",
            (current_user.id,)
        ).fetchall()
        
        # Count healthy and diseased results
        healthy_count = sum(1 for r in last_results if r['result'] == 'Healthy')
        diseased_count = sum(1 for r in last_results if r['result'] == 'Diseased')
        
        # If all recent results are the same, add a recommendation
        if healthy_count == 3:
            recommendations.append({
                "id": 5,
                "icon": "fas fa-trophy",
                "title": "Great Job!",
                "text": "Your recent voice tests have all been healthy. Keep up the good work!",
                "action": "View Tips"
            })
        elif diseased_count == 3:
            recommendations.append({
                "id": 6,
                "icon": "fas fa-exclamation-triangle",
                "title": "Consistent Issues",
                "text": "Your recent tests all show voice issues. Please consult a specialist.",
                "action": "Find Specialists"
            })
    
    conn.close()
    return jsonify({"success": True, "recommendations": recommendations})

@app.route('/user/goals')
@login_required
def user_goals():
    """Returns user goals and progress."""
    conn = get_db_connection()
    
    # Get user statistics
    total_submissions = conn.execute(
        "SELECT COUNT(id) FROM submissions WHERE user_id = ?",
        (current_user.id,)
    ).fetchone()[0]
    
    healthy_count = conn.execute(
        "SELECT COUNT(id) FROM submissions WHERE user_id = ? AND result = 'Healthy'",
        (current_user.id,)
    ).fetchone()[0]
    
    # Check if user has completed their profile
    user_data = conn.execute(
        "SELECT username, email, age, gender FROM users WHERE id = ?",
        (current_user.id,)
    ).fetchone()
    
    profile_complete = 0
    if user_data['username']: profile_complete += 1
    if user_data['email']: profile_complete += 1
    if user_data['age']: profile_complete += 1
    if user_data['gender']: profile_complete += 1
    
    # Define goals
    goals = [
        {
            "id": 1,
            "title": "Complete 5 Voice Tests",
            "description": "Complete 5 voice tests to establish a baseline for your voice health.",
            "status": "active" if total_submissions < 5 else "completed",
            "progress": min(total_submissions, 5),
            "target": 5,
            "action": "Take Test",
            "actionLink": "voice-test"
        },
        {
            "id": 2,
            "title": "Improve Voice Health",
            "description": "Achieve more healthy test results than diseased ones.",
            "status": "completed" if healthy_count > (total_submissions - healthy_count) else "active",
            "progress": healthy_count,
            "target": total_submissions if total_submissions > 0 else 1,
            "action": "Health Tips",
            "actionLink": "health-tips"
        },
        {
            "id": 3,
            "title": "Complete Profile",
            "description": "Fill in all your profile information including username, email, age, and gender.",
            "status": "completed" if profile_complete == 4 else "active",
            "progress": profile_complete,
            "target": 4,
            "action": "View Profile",
            "actionLink": "profile"
        }
    ]
    
    # Add a goal for consistent testing if user has enough submissions
    if total_submissions >= 5:
        # Calculate testing consistency
        first_test = conn.execute(
            "SELECT created_at FROM submissions WHERE user_id = ? ORDER BY created_at ASC LIMIT 1",
            (current_user.id,)
        ).fetchone()
        
        if first_test:
            first_date = datetime.fromisoformat(first_test['created_at'])
            days_since_first = (datetime.now() - first_date).days
            
            # Calculate ideal number of tests (1 per week)
            ideal_tests = days_since_first / 7
            
            if total_submissions >= ideal_tests * 0.8:  # If user has done at least 80% of ideal tests
                consistency_status = "completed"
            else:
                consistency_status = "active"
            
            goals.append({
                "id": 4,
                "title": "Consistent Testing",
                "description": "Maintain a regular testing schedule of at least one test per week.",
                "status": consistency_status,
                "progress": min(total_submissions, int(ideal_tests)),
                "target": int(ideal_tests),
                "action": "Take Test",
                "actionLink": "voice-test"
            })
    
    conn.close()
    return jsonify({"success": True, "goals": goals})

@app.route('/user/settings')
@login_required
def user_settings():
    """Returns user settings."""
    conn = get_db_connection()
    
    # Get user settings (placeholder implementation)
    settings = [
        {
            "id": "emailNotifications",
            "name": "Email Notifications",
            "description": "Receive email updates about your voice health",
            "enabled": True
        },
        {
            "id": "testReminders",
            "name": "Test Reminders",
            "description": "Get reminders to take regular voice tests",
            "enabled": True
        },
        {
            "id": "healthTips",
            "name": "Health Tips",
            "description": "Receive personalized voice health tips",
            "enabled": True
        },
        {
            "id": "dataSharing",
            "name": "Data Sharing",
            "description": "Share anonymous data to improve our AI models",
            "enabled": False
        }
    ]
    
    conn.close()
    return jsonify({"success": True, "settings": settings})

@app.route('/user/update_settings', methods=['POST'])
@login_required
def update_settings():
    """Updates user settings."""
    data = request.get_json()
    
    # In a real implementation, you would save these settings to the database
    # For now, we'll just return a success response
    
    return jsonify({
        "success": True, 
        "message": "Settings updated successfully!"
    })

@app.route('/user/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    content = data.get('content', '')
    
    if not content.strip():
        return jsonify({"success": False, "message": "Feedback content is required."})
    
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO feedback (user_id, content) VALUES (?, ?)",
            (current_user.id, content)
        )
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log user activity
        log_user_activity(current_user.id, 'feedback', 'Submitted feedback to admin')
        
        return jsonify({"success": True, "message": "Feedback submitted successfully!", "feedback_id": feedback_id})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Error submitting feedback: {str(e)}"})
# -------------------------------
# Admin Routes
# -------------------------------
@app.route('/admin_dashboard')
@login_required
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    total_users = conn.execute("SELECT COUNT(id) FROM users WHERE role='user'").fetchone()[0]
    total_submissions = conn.execute("SELECT COUNT(id) FROM submissions").fetchone()[0]
    
    # Get unread feedback count
    unread_feedback = conn.execute("SELECT COUNT(id) FROM feedback WHERE is_reviewed = 0").fetchone()[0]
    
    # Get prediction counts for charts
    healthy_count = conn.execute("SELECT COUNT(id) FROM submissions WHERE result = 'Healthy'").fetchone()[0]
    diseased_count = conn.execute("SELECT COUNT(id) FROM submissions WHERE result = 'Diseased'").fetchone()[0]
    
    # Get submissions and convert to dictionaries
    submissions_result = conn.execute("""
        SELECT s.id, u.username, s.file_path, s.result, s.confidence, s.created_at, s.admin_override, s.user_notes
        FROM submissions s LEFT JOIN users u ON s.user_id = u.id
        ORDER BY s.created_at DESC
    """).fetchall()
    submissions = [dict(sub) for sub in submissions_result]

    metrics_up = get_model_metrics(UPLOAD_MODEL_PATH)
    metrics_primary = get_model_metrics(PRIMARY_MODEL_PATH)
    
    active_model, _, _, model_source = get_current_model()
    
    if metrics_up:
        metrics_to_display = metrics_up
        metrics_source = "Uploaded Model"
    elif metrics_primary:
        metrics_to_display = metrics_primary
        metrics_source = "Primary Model"
    else:
        metrics_to_display = None
        metrics_source = "N/A"
        
    # Get users and convert to dictionaries
    users_result = conn.execute("SELECT id, username, email, role, is_active, age, gender FROM users WHERE role='user' ORDER BY created_at DESC").fetchall()
    users = [dict(user) for user in users_result]
    
    # Get age and gender distribution for analytics
    age_distribution = conn.execute("""
        SELECT 
            CASE 
                WHEN age < 18 THEN 'Under 18'
                WHEN age BETWEEN 18 AND 24 THEN '18-24'
                WHEN age BETWEEN 25 AND 34 THEN '25-34'
                WHEN age BETWEEN 35 AND 44 THEN '35-44'
                WHEN age BETWEEN 45 AND 54 THEN '45-54'
                WHEN age BETWEEN 55 AND 64 THEN '55-64'
                ELSE '65+'
            END as age_group,
            COUNT(*) as count
        FROM users 
        WHERE age IS NOT NULL AND role='user'
        GROUP BY age_group
        ORDER BY MIN(age)
    """).fetchall()
    
    gender_distribution = conn.execute("""
        SELECT gender, COUNT(*) as count
        FROM users 
        WHERE gender IS NOT NULL AND role='user'
        GROUP BY gender
    """).fetchall()
    
    # Format age and gender data for charts, with defaults for empty data
    if age_distribution:
        age_labels = [item['age_group'] for item in age_distribution]
        age_counts = [item['count'] for item in age_distribution]
    else:
        age_labels = ['No Data']
        age_counts = [0]
    
    if gender_distribution:
        gender_labels = [item['gender'] for item in gender_distribution]
        gender_counts = [item['count'] for item in gender_distribution]
    else:
        gender_labels = ['No Data']
        gender_counts = [0]
    
    # Get submission trends - remove the 30-day filter and get all data
    submission_trends = conn.execute("""
        SELECT DATE(created_at) as date, COUNT(id) as count
        FROM submissions
        GROUP BY DATE(created_at)
        ORDER BY date
    """).fetchall()
    
    if submission_trends:
        trend_dates = [item['date'] for item in submission_trends]
        trend_counts = [item['count'] for item in submission_trends]
    else:
        trend_dates = ['No Data']
        trend_counts = [0]
    
    # Get data for labeling section
    all_wav_files = []
    for root, dirs, files in os.walk('dataset'):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                all_wav_files.append(file_path)
    
    # Get labeled files to exclude them
    labeled_files_result = conn.execute("SELECT file_path FROM voice_labels").fetchall()
    labeled_paths = {row['file_path'] for row in labeled_files_result}
    
    # Filter out already labeled files
    unlabeled_files = [path for path in all_wav_files if path not in labeled_paths]
    
    # Get labeled files for display and convert to dictionaries
    labeled_files_result = conn.execute("""
        SELECT vl.*, u.username 
        FROM voice_labels vl 
        JOIN users u ON vl.labeled_by = u.id 
        ORDER BY vl.created_at DESC
    """).fetchall()
    labeled_files = [dict(label) for label in labeled_files_result]
    
    # Get feedback data and convert to dictionaries
    feedback_result = conn.execute("""
        SELECT f.*, u.username 
        FROM feedback f 
        LEFT JOIN users u ON f.user_id = u.id 
        ORDER BY f.created_at DESC
    """).fetchall()
    feedback_list = [dict(feedback) for feedback in feedback_result]
    
    # Get security logs and convert to dictionaries
    logs_result = conn.execute("""
        SELECT sl.*, u.username 
        FROM security_logs sl 
        LEFT JOIN users u ON sl.user_id = u.id 
        ORDER BY sl.created_at DESC 
        LIMIT 100
    """).fetchall()
    logs = [dict(log) for log in logs_result]
    
    # Get active users with last activity - FIXED VERSION
    active_users_result = conn.execute("""
        SELECT u.username, COUNT(s.id) as submission_count, MAX(s.created_at) as last_activity
        FROM users u
        LEFT JOIN submissions s ON u.id = s.user_id
        WHERE u.role = 'user'
        GROUP BY u.id
        ORDER BY submission_count DESC
        LIMIT 10
    """).fetchall()
    
    # Format the last_activity timestamp and convert to dictionaries
    active_users = []
    for row in active_users_result:
        last_activity = row['last_activity']
        
        # Check if last_activity is already a string or if it's a datetime object
        if last_activity:
            if isinstance(last_activity, str):
                # It's already a string, so we can use it directly
                formatted_activity = last_activity
            else:
                # It's a datetime object, so format it
                formatted_activity = last_activity.strftime('%Y-%m-%d %H:%M')
        else:
            # No activity, so use N/A
            formatted_activity = 'N/A'
        
        user_dict = {
            'username': row['username'],
            'submission_count': row['submission_count'],
            'last_activity': formatted_activity
        }
        active_users.append(user_dict)
    
    conn.close()

    if active_model:
        del active_model

    # Get the section parameter from the URL
    section = request.args.get('section', 'dashboard')

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        total_submissions=total_submissions,
        submissions=submissions,
        metrics=metrics_to_display,
        metrics_source=metrics_source,
        users=users,
        unread_feedback=unread_feedback,
        age_distribution=age_distribution,
        gender_distribution=gender_distribution,
        unlabeled_files=unlabeled_files,
        labeled_files=labeled_files,
        feedback_list=feedback_list,
        logs=logs,
        active_users=active_users,
        # New data for charts
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        age_labels=age_labels,
        age_counts=age_counts,
        gender_labels=gender_labels,
        gender_counts=gender_counts,
        trend_dates=trend_dates,
        trend_counts=trend_counts,
        section=section
    )


@app.route('/user/feedback')
@login_required
def get_user_feedback():
    conn = get_db_connection()
    try:
        feedback = conn.execute("""
            SELECT id, content, created_at, is_reviewed 
            FROM feedback 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        """, (current_user.id,)).fetchall()
        
        # Format feedback for the frontend
        formatted_feedback = []
        for item in feedback:
            # Calculate time ago
            created_at = datetime.fromisoformat(item['created_at'])
            now = datetime.now()
            diff = now - created_at
            
            if diff.days > 0:
                time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                time_ago = "Just now"
            
            formatted_feedback.append({
                "id": item['id'],
                "content": item['content'],
                "time": time_ago,
                "is_reviewed": bool(item['is_reviewed'])
            })
        
        return jsonify({"success": True, "feedback": formatted_feedback})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error retrieving feedback: {str(e)}"})
    finally:
        conn.close()

@app.route('/admin/download_report/<int:sub_id>')
@admin_required
def admin_download_report(sub_id):
    """Generates and returns a PDF report for a submission."""
    conn = get_db_connection()
    submission = conn.execute("""
        SELECT s.*, u.username 
        FROM submissions s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = ?
    """, (sub_id,)).fetchone()
    conn.close()

    if not submission:
        flash("Submission not found.", "danger")
        return redirect(url_for('admin_dashboard', section='submissions'))

    try:
        # Pass the dictionary/Row object to the generator
        buffer = generate_report_pdf(dict(submission))
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'voice_analysis_report_{sub_id}.pdf'
        )
    except Exception as e:
        print(f"FATAL ERROR during PDF generation for admin: {e}")
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for('admin_dashboard', section='submissions'))

@app.route('/admin/download_submission_audio/<int:sub_id>')
@admin_required
def download_submission_audio(sub_id):
    """Serves an audio file for a specific submission ID."""
    conn = get_db_connection()
    
    # Get the submission from the database
    submission = conn.execute(
        "SELECT file_path, user_id FROM submissions WHERE id = ?", 
        (sub_id,)
    ).fetchone()
    
    conn.close()

    if not submission:
        abort(404)  # Not Found

    file_path = submission['file_path']

    # Check if the file actually exists on the server
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found at {file_path}")
        abort(404)  # Not Found

    # Serve the file
    try:
        return send_file(
            file_path,
            mimetype='audio/wav',  # Or 'audio/mpeg' for .mp3 files
            as_attachment=False,
            download_name=os.path.basename(file_path)
        )
    except Exception as e:
        print(f"Error serving file {file_path}: {e}")
        abort(500)

@app.route('/admin/override/<int:sub_id>', methods=['POST'])
@login_required
@admin_required
def admin_override(sub_id):
    override = request.form.get('admin_override')
    
    conn = get_db_connection()
    submission = conn.execute("SELECT * FROM submissions WHERE id = ?", (sub_id,)).fetchone()
    
    if not submission:
        conn.close()
        flash("Submission not found.", "danger")
        return redirect(url_for('admin_dashboard', section='submissions'))

    conn.execute("UPDATE submissions SET admin_override=? WHERE id=?", (override, sub_id))
    conn.commit()
    conn.close()
    flash(f"Admin override saved for Submission #{sub_id}.", "success")
    return redirect(url_for('admin_dashboard', section='submissions'))

@app.route('/admin/toggle_user_status/<int:user_id>/<int:status>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_user_status(user_id, status):
    if user_id == current_user.id:
        flash("You cannot change the status of your own admin account.", "danger")
        return redirect(url_for('admin_dashboard', section='users'))

    if status not in [0, 1]:
        flash("Invalid status code.", "danger")
        return redirect(url_for('admin_dashboard', section='users'))

    conn = get_db_connection()
    user_data = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    
    if not user_data or user_data['role'] == 'admin':
        conn.close()
        flash("User not found or is an admin account.", "danger")
        return redirect(url_for('admin_dashboard', section='users'))

    try:
        conn.execute("UPDATE users SET is_active=? WHERE id=?", (status, user_id))
        conn.commit()
        
        action = "Unblocked" if status == 1 else "Blocked"
        flash(f"User ID {user_id} successfully {action}.", "success")
    except Exception as e:
        flash(f"Error toggling user status: {e}", "danger")
    finally:
        conn.close()
        
    return redirect(url_for('admin_dashboard', section='users'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash("You cannot delete your own admin account.", "danger")
        return redirect(url_for('admin_dashboard', section='users'))
        
    conn = get_db_connection()
    
    user_data = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not user_data or user_data['role'] == 'admin':
        conn.close()
        flash("User not found or is an admin account and cannot be deleted here.", "danger")
        return redirect(url_for('admin_dashboard', section='users'))
        
    conn.execute("DELETE FROM submissions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    flash(f"User ID {user_id} and all their submissions have been successfully deleted.", "success")
    return redirect(url_for('admin_dashboard', section='users'))

@app.route('/admin/upload_training_data', methods=['POST'])
@login_required
@admin_required
def upload_training_data():
    """Handles admin upload of training data files."""
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('admin_dashboard', section='training'))
    
    file = request.files['file']
    voice_type = request.form.get('voice_type')
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('admin_dashboard', section='training'))
    
    if file and voice_type:
        filename = secure_filename(file.filename)
        
        # Determine the target directory based on voice type
        if voice_type == 'healthy':
            target_dir = os.path.join(UPLOAD_FOLDER, "healthy_voices")
        elif voice_type == 'diseased':
            target_dir = os.path.join(UPLOAD_FOLDER, "diseased_voices")
        else:
            flash('Invalid voice type specified', 'danger')
            return redirect(url_for('admin_dashboard', section='training'))
        
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, filename)
        
        # Check if file already exists and append UUID if needed
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{uuid.uuid4().hex[:4]}{ext}"
            file_path = os.path.join(target_dir, filename)
        
        file.save(file_path)
        flash(f'File successfully uploaded to {voice_type} directory.', 'success')
    else:
        flash('File or voice type missing', 'danger')
    
    return redirect(url_for('admin_dashboard', section='training'))

@app.route('/admin/retrain_model', methods=['POST'])
@login_required
@admin_required
def retrain_model():
    """Initiates model retraining process."""
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, "train_on_uploads.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            flash("Model retraining completed successfully!", "success")
            return jsonify({
                "success": True,
                "message": "Model retraining completed successfully!"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Training failed with error: {result.stderr}"
            })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Training process timed out after 30 minutes."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during training: {str(e)}"
        })

@app.route('/admin/test_uploaded_voice', methods=['POST'])
@login_required
@admin_required
def test_uploaded_voice():
    """Tests the active model with an uploaded voice file."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file:
        # Use a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file path within the temporary directory
            file_path = os.path.join(temp_dir, secure_filename(file.filename))
            
            # Save the uploaded file to the temporary file
            file.save(file_path)
            
            # Get prediction
            result_class, confidence, model_source = predict_voice(file_path)
            
            if result_class == "Error!" or result_class == "No Model":
                return jsonify({
                    "success": False,
                    "message": f"Prediction failed: {result_class}. Please try a valid .wav file."
                }), 500
            
            return jsonify({
                "success": True,
                "result_class": result_class,
                "confidence": f"{confidence * 100:.2f}",
                "model_source": model_source
            })
    
    return jsonify({"success": False, "message": "File upload failed."}), 500

@app.route('/admin/labeling', methods=['POST'])
@login_required
@admin_required
def admin_labeling():
    """Handles data labeling submissions."""
    file_path = request.form.get('file_path')
    manual_file_path = request.form.get('manual_file_path')
    label = request.form.get('label')
    
    # Determine which file path to use
    if manual_file_path:
        actual_file_path = manual_file_path
    elif file_path:
        actual_file_path = file_path
    else:
        flash('No file path provided', 'danger')
        return redirect(url_for('admin_dashboard', section='labeling'))
    
    if not label:
        flash('No label provided', 'danger')
        return redirect(url_for('admin_dashboard', section='labeling'))
    
    conn = get_db_connection()
    try:
        # Check if file is already labeled
        existing_label = conn.execute("SELECT * FROM voice_labels WHERE file_path = ?", (actual_file_path,)).fetchone()
        
        if existing_label:
            # Update existing label
            conn.execute("UPDATE voice_labels SET label = ?, labeled_by = ? WHERE file_path = ?",
                         (label, current_user.id, actual_file_path))
            flash('Label updated successfully!', 'success')
        else:
            # Insert new label
            conn.execute("INSERT INTO voice_labels (file_path, label, labeled_by) VALUES (?, ?, ?)",
                         (actual_file_path, label, current_user.id))
            flash('Label saved successfully!', 'success')
        
        conn.commit()
    except sqlite3.IntegrityError:
        flash('Error saving label to database.', 'danger')
    finally:
        conn.close()
    
    return redirect(url_for('admin_dashboard', section='labeling'))

@app.route('/admin/upload_labeling_file', methods=['POST'])
@login_required
@admin_required
def upload_labeling_file():
    """Handles file upload specifically for labeling purposes."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        labeling_dir = "dataset/labeling_uploads"
        os.makedirs(labeling_dir, exist_ok=True)
        
        file_path = os.path.join(labeling_dir, filename)
        
        # Check if file already exists and append UUID if needed
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{uuid.uuid4().hex[:4]}{ext}"
            file_path = os.path.join(labeling_dir, filename)
        
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully for labeling.",
            "file_path": file_path
        })
    
    return jsonify({"success": False, "message": "File upload failed."}), 500

@app.route('/admin/feedback', methods=['POST'])
@login_required
@admin_required
def admin_feedback():
    """Handles admin actions on feedback (mark as reviewed or delete)."""
    feedback_id = request.form.get('feedback_id')
    action = request.form.get('action')
    
    if not feedback_id or not action:
        return jsonify({"success": False, "message": "Missing feedback ID or action"})
    
    conn = get_db_connection()
    
    if action == 'review':
        try:
            conn.execute(
                "UPDATE feedback SET is_reviewed = 1, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
                (current_user.id, datetime.now(), feedback_id)
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "Feedback marked as reviewed"})
        except Exception as e:
            conn.close()
            return jsonify({"success": False, "message": f"Error updating feedback: {str(e)}"})
    
    elif action == 'delete':
        try:
            conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "Feedback deleted successfully"})
        except Exception as e:
            conn.close()
            return jsonify({"success": False, "message": f"Error deleting feedback: {str(e)}"})
    
    else:
        conn.close()
        return jsonify({"success": False, "message": "Invalid action"})

# --- NEW: Route to check dataset status ---
@app.route('/admin/check_dataset_status')
@admin_required
def check_dataset_status():
    """Checks if the dataset ZIP is ready for download."""
    # Since we are creating the ZIP in memory, it's always "ready"
    return jsonify({'ready': True})

# --- MODIFIED: Fully implement the dataset download route ---
@app.route('/admin/download_dataset')
@login_required
@admin_required
def download_dataset():
    """Creates and sends a ZIP file of the labeled dataset in memory."""
    try:
        # Create a BytesIO buffer to hold the ZIP file in memory
        zip_buffer = BytesIO()
        
        # Create metadata
        metadata = {
            'description': 'Voice Health Dataset',
            'created_at': datetime.now().isoformat(),
            'structure': {
                'healthy_voices': 'Contains voice samples labeled as healthy.',
                'diseased_voices': 'Contains voice samples labeled as diseased.',
                'metadata.json': 'This file contains information about the dataset.'
            }
        }

        # Create the ZIP file in memory
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add healthy voice files
            healthy_path = os.path.join(os.getcwd(), 'dataset', 'uploads', 'healthy_voices')
            if os.path.exists(healthy_path):
                for foldername, subfolders, filenames in os.walk(healthy_path):
                    for filename in filenames:
                        if filename.endswith('.wav'):
                            file_path = os.path.join(foldername, filename)
                            arcname = os.path.join('healthy_voices', filename)
                            zipf.write(file_path, arcname)

            # Add diseased voice files
            diseased_path = os.path.join(os.getcwd(), 'dataset', 'uploads', 'diseased_voices')
            if os.path.exists(diseased_path):
                for foldername, subfolders, filenames in os.walk(diseased_path):
                    for filename in filenames:
                        if filename.endswith('.wav'):
                            file_path = os.path.join(foldername, filename)
                            arcname = os.path.join('diseased_voices', filename)
                            zipf.write(file_path, arcname)
            
            # Add metadata.json to the zip
            zipf.writestr('metadata.json', json.dumps(metadata, indent=4))

        # Rewind the buffer to the beginning
        zip_buffer.seek(0)

        # Send the file to the user from memory
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'voice_dataset_{datetime.now().strftime("%Y-%m-%d")}.zip'
        )

    except Exception as e:
        print(f"Error creating dataset zip: {e}")
        flash(f'Error creating dataset ZIP: {e}', 'danger')
        return redirect(url_for('admin_dashboard', section='dataset-download'))


@app.route('/admin/check_test_reminders')
@login_required
@admin_required
def admin_check_test_reminders():
    """Admin route to check for test reminders and send notifications."""
    check_test_reminders()
    return jsonify({"success": True, "message": "Test reminders checked and notifications sent if needed."})



@app.route('/user/feedback')
@login_required
def user_feedback():
    try:
        conn = get_db_connection()
        feedback = conn.execute(
            "SELECT f.id, f.content, f.created_at, f.is_reviewed, "
            "CASE WHEN f.is_reviewed = 1 THEN 'Reviewed' ELSE 'Pending' END as status "
            "FROM feedback f "
            "WHERE f.user_id = ? "
            "ORDER BY f.created_at DESC",
            (current_user.id,)
        ).fetchall()
        
        formatted_feedback = []
        for item in feedback:
            created_at = datetime.fromisoformat(item['created_at'])
            now = datetime.utcnow()
            diff = now - created_at
            
            if diff.days > 0:
                time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                time_ago = "Just now"
            
            formatted_feedback.append({
                'id': item['id'],
                'content': item['content'],
                'time': time_ago,
                'is_reviewed': item['is_reviewed'] == 1,
                'status': item['status']
            })
        
        conn.close()
        return jsonify({"success": True, "feedback": formatted_feedback})
    except Exception as e:
        print(f"Error fetching user feedback: {e}")
        return jsonify({"success": False, "message": "Error fetching feedback"})
@app.route('/serve_submission_audio/<int:sub_id>')
@login_required
def serve_submission_audio(sub_id):
    """Serves an audio file for a specific submission ID."""
    conn = get_db_connection()
    
    # Get the submission from the database
    submission = conn.execute(
        "SELECT file_path, user_id FROM submissions WHERE id = ?", 
        (sub_id,)
    ).fetchone()
    
    conn.close()

    # Check if submission exists and belongs to the current user
    if not submission or submission['user_id'] != current_user.id:
        abort(403)  # Forbidden

    file_path = submission['file_path']

    # Check if the file actually exists on the server
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found at {file_path}")
        abort(404)  # Not Found

    # Serve the file
    try:
        return send_file(
            file_path,
            mimetype='audio/wav',  # Or 'audio/mpeg' for .mp3 files
            as_attachment=False,
            download_name=os.path.basename(file_path)
        )
    except Exception as e:
        print(f"Error serving file {file_path}: {e}")
        abort(500)
if __name__ == '__main__':
    app.run(debug=True)