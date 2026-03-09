"""
Configuration settings for AI Face Recognition and Monitoring System
"""
import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Flask settings
SECRET_KEY = 'face-rec-mon-secret-key-2026'
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Database
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'database.db')

# Face encodings storage
ENCODINGS_DIR = os.path.join(BASE_DIR, 'encodings')
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, 'face_encodings.pkl')

# Image storage
REGISTERED_FACES_DIR = os.path.join(BASE_DIR, 'static', 'registered_faces')
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, 'static', 'unknown_faces')

# Reports
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Face recognition settings
RECOGNITION_TOLERANCE = 0.4  # Lower = more strict. 0.4 = 60% confidence threshold
MODEL = 'hog'  # 'hog' (CPU, faster) or 'cnn' (GPU, more accurate) — used for live feed
TRAIN_MODEL = 'hog'  # 'hog' for CPU, 'cnn' only if you have CUDA GPU
FRAME_RESIZE_FACTOR = 0.5  # Resize frames for processing (was 0.25, higher = more accurate)
NUM_JITTERS = 3  # Re-sample face encoding N times and average (higher = slower but better)
NUM_ENCODINGS_PER_PERSON = 5  # Store top N diverse encodings per person

# Unknown face throttle settings
UNKNOWN_COOLDOWN_SECONDS = 300  # Seconds between saves for the same spatial bucket (5 min)
UNKNOWN_BUCKET_SIZE = 80        # Spatial bucket granularity in px (higher = less sensitive to movement)
UNKNOWN_MIN_FACE_SIZE = 60      # Min face bbox dimension in px (skip tiny/far-away detections)

# Attendance settings
LATE_THRESHOLD_HOUR = 9   # Hour after which attendance is "Late"
LATE_THRESHOLD_MINUTE = 30

# Create directories if they don't exist
for directory in [DATABASE_DIR, ENCODINGS_DIR, REGISTERED_FACES_DIR,
                  UNKNOWN_FACES_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)
