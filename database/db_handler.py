"""
Database handler for the AI Face Recognition and Monitoring System.
Manages SQLite operations for users, attendance, unknown faces, and admin.
"""
import sqlite3
import hashlib
from datetime import datetime, date


class DatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize all database tables and seed default admin."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                roll_no TEXT,
                department TEXT,
                semester TEXT,
                photo_path TEXT,
                photo_dir TEXT,
                trained INTEGER DEFAULT 0,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Migrate existing DB: add columns if they don't exist yet
        existing_cols = [r[1] for r in cursor.execute('PRAGMA table_info(users)').fetchall()]
        if 'trained' not in existing_cols:
            cursor.execute('ALTER TABLE users ADD COLUMN trained INTEGER DEFAULT 0')
        if 'photo_dir' not in existing_cols:
            cursor.execute('ALTER TABLE users ADD COLUMN photo_dir TEXT')
        if 'roll_no' not in existing_cols:
            cursor.execute('ALTER TABLE users ADD COLUMN roll_no TEXT')
        if 'department' not in existing_cols:
            cursor.execute('ALTER TABLE users ADD COLUMN department TEXT')
        if 'semester' not in existing_cols:
            cursor.execute('ALTER TABLE users ADD COLUMN semester TEXT')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                check_in_time TEXT NOT NULL,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unknown_faces (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_status TEXT DEFAULT 'Unreviewed'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin (
                admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                email TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Seed default admin if the table is empty
        cursor.execute('SELECT COUNT(*) FROM admin')
        if cursor.fetchone()[0] == 0:
            default_pass_hash = hashlib.sha256(b'admin123').hexdigest()
            cursor.execute(
                'INSERT INTO admin (username, password) VALUES (?, ?)',
                ('admin', default_pass_hash)
            )

        conn.commit()
        conn.close()

    # ── Admin operations ──────────────────────────────────────────

    def verify_admin(self, username, password):
        """Verify admin login credentials."""
        hashed = hashlib.sha256(password.encode()).hexdigest()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM admin WHERE username=? AND password=?',
            (username, hashed)
        )
        admin = cursor.fetchone()
        conn.close()
        return admin is not None

    # ── User operations ───────────────────────────────────────────

    def add_user(self, name, email, photo_path, photo_dir=None, trained=0,
                 roll_no=None, department=None, semester=None):
        """Add a new registered user."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, email, photo_path, photo_dir, trained, roll_no, department, semester) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (name, email, photo_path, photo_dir, trained, roll_no, department, semester)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id

    def get_all_users(self):
        """Get all registered users."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users ORDER BY registered_date DESC')
        users = cursor.fetchall()
        conn.close()
        return users

    def get_untrained_users(self):
        """Get users whose faces have not yet been trained into the model."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE trained=0 ORDER BY registered_date DESC')
        users = cursor.fetchall()
        conn.close()
        return users

    def mark_user_trained(self, user_id):
        """Mark a user as trained."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET trained=1 WHERE user_id=?', (user_id,))
        conn.commit()
        conn.close()

    def get_user_by_name(self, name):
        """Get user by name."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE name=?', (name,))
        user = cursor.fetchone()
        conn.close()
        return user

    def get_user_by_id(self, user_id):
        """Get user by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        return user

    def get_user_count(self):
        """Get total number of registered users."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_user(self, user_id):
        """Delete a user by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE user_id=?', (user_id,))
        # Reset autoincrement so IDs fill gaps
        cursor.execute(
            "UPDATE sqlite_sequence SET seq = "
            "(SELECT COALESCE(MAX(user_id), 0) FROM users) "
            "WHERE name = 'users'"
        )
        conn.commit()
        conn.close()

    def update_user(self, user_id, roll_no=None, department=None, semester=None,
                    name=None, email=None):
        """Update editable student fields for a user."""
        conn = self.get_connection()
        cursor = conn.cursor()
        fields = []
        values = []
        if roll_no is not None:
            fields.append('roll_no=?')
            values.append(roll_no)
        if department is not None:
            fields.append('department=?')
            values.append(department)
        if semester is not None:
            fields.append('semester=?')
            values.append(semester)
        if name is not None:
            fields.append('name=?')
            values.append(name)
        if email is not None:
            fields.append('email=?')
            values.append(email)
        if not fields:
            conn.close()
            return False
        values.append(user_id)
        cursor.execute(
            f'UPDATE users SET {", ".join(fields)} WHERE user_id=?',
            tuple(values)
        )
        conn.commit()
        conn.close()
        return True

    # ── Attendance operations ─────────────────────────────────────

    def mark_attendance(self, name, user_id=None):
        """Mark attendance for a recognized person (once per day)."""
        today = date.today().strftime('%Y-%m-%d')
        now = datetime.now().strftime('%H:%M:%S')

        conn = self.get_connection()
        cursor = conn.cursor()

        # Prevent duplicate entries for the same day
        cursor.execute(
            'SELECT * FROM attendance WHERE name=? AND date=?',
            (name, today)
        )
        if cursor.fetchone() is not None:
            conn.close()
            return False  # Already marked

        # Determine status (Present / Late)
        hour = datetime.now().hour
        minute = datetime.now().minute
        from config import LATE_THRESHOLD_HOUR, LATE_THRESHOLD_MINUTE
        if hour > LATE_THRESHOLD_HOUR or (
            hour == LATE_THRESHOLD_HOUR and minute > LATE_THRESHOLD_MINUTE
        ):
            status = 'Late'
        else:
            status = 'Present'

        cursor.execute(
            '''INSERT INTO attendance (user_id, name, date, check_in_time, status)
               VALUES (?, ?, ?, ?, ?)''',
            (user_id, name, today, now, status)
        )
        conn.commit()
        conn.close()
        return True

    def get_attendance(self, date_filter=None):
        """Get attendance records, optionally filtered by date."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if date_filter:
            cursor.execute(
                'SELECT * FROM attendance WHERE date=? ORDER BY check_in_time',
                (date_filter,)
            )
        else:
            cursor.execute(
                'SELECT * FROM attendance ORDER BY date DESC, check_in_time DESC'
            )
        records = cursor.fetchall()
        conn.close()
        return records

    def get_today_attendance_count(self):
        """Get count of today's attendance."""
        today = date.today().strftime('%Y-%m-%d')
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) FROM attendance WHERE date=?', (today,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_recent_attendance(self, limit=10):
        """Get most recent attendance records."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM attendance ORDER BY date DESC, check_in_time DESC LIMIT ?',
            (limit,)
        )
        records = cursor.fetchall()
        conn.close()
        return records

    # ── Unknown faces operations ──────────────────────────────────

    def log_unknown_face(self, image_path):
        """Log an unknown face detection event."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO unknown_faces (image_path) VALUES (?)',
            (image_path,)
        )
        conn.commit()
        conn.close()

    def get_unknown_faces(self):
        """Get all unknown face records."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM unknown_faces ORDER BY timestamp DESC'
        )
        records = cursor.fetchall()
        conn.close()
        return records

    def get_unknown_faces_count(self):
        """Get count of unreviewed unknown face alerts."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM unknown_faces WHERE alert_status='Unreviewed'"
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def mark_alert_reviewed(self, log_id):
        """Mark an unknown face alert as reviewed."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE unknown_faces SET alert_status='Reviewed' WHERE log_id=?",
            (log_id,)
        )
        conn.commit()
        conn.close()

    def delete_alert(self, log_id):
        """Delete an alert record and return its image_path so the caller
        can remove the file from disk."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT image_path FROM unknown_faces WHERE log_id=?', (log_id,)
        )
        row = cursor.fetchone()
        image_path = row['image_path'] if row else None
        cursor.execute('DELETE FROM unknown_faces WHERE log_id=?', (log_id,))
        conn.commit()
        conn.close()
        return image_path

    def bulk_delete_alerts(self, log_ids):
        """Delete multiple alert records. Returns list of image_paths."""
        if not log_ids:
            return []
        conn = self.get_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in log_ids)
        cursor.execute(
            f'SELECT image_path FROM unknown_faces WHERE log_id IN ({placeholders})',
            log_ids
        )
        image_paths = [row['image_path'] for row in cursor.fetchall() if row['image_path']]
        cursor.execute(
            f'DELETE FROM unknown_faces WHERE log_id IN ({placeholders})',
            log_ids
        )
        conn.commit()
        conn.close()
        return image_paths
