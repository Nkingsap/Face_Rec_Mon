"""
AI Face Recognition and Monitoring System
Main Flask Application
"""
import os
import base64
import json
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   Response, session, flash, send_file, jsonify)
from werkzeug.utils import secure_filename
from config import (SECRET_KEY, DATABASE_PATH, ENCODINGS_FILE,
                    REGISTERED_FACES_DIR, UNKNOWN_FACES_DIR,
                    REPORTS_DIR, RECOGNITION_TOLERANCE, MODEL,
                    TRAIN_MODEL, NUM_JITTERS, NUM_ENCODINGS_PER_PERSON,
                    DEBUG, HOST, PORT)
from database.db_handler import DatabaseHandler
from face_rec_engine import FaceRecognitionEngine, FACE_REC_AVAILABLE
from attendance_manager import AttendanceManager

# ── Initialize Flask App ──────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ── Initialize Components ────────────────────────────────────────
db = DatabaseHandler(DATABASE_PATH)
engine = FaceRecognitionEngine(ENCODINGS_FILE, RECOGNITION_TOLERANCE, MODEL)
att_mgr = AttendanceManager(db, REPORTS_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator to require admin login."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Login page (redirects to dashboard if already logged in)."""
    if 'admin' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    """Handle admin login."""
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    if not username or not password:
        flash('Please enter both username and password.', 'error')
        return redirect(url_for('index'))

    if db.verify_admin(username, password):
        session['admin'] = username
        flash(f'Welcome back, {username}!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid username or password.', 'error')
        return redirect(url_for('index'))


@app.route('/logout')
def logout():
    """Handle logout."""
    session.pop('admin', None)
    engine.stop_camera()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with statistics and recent activity."""
    stats = att_mgr.get_today_stats()
    recent = db.get_recent_attendance(limit=10)
    return render_template('dashboard.html',
                           stats=stats,
                           recent=recent,
                           face_rec_available=FACE_REC_AVAILABLE)


@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    """Register a new face."""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        roll_no = request.form.get('roll_no', '').strip()
        department = request.form.get('department', '').strip()
        semester = request.form.get('semester', '').strip()

        if not name:
            flash('Name is required.', 'error')
            return redirect(url_for('register'))

        # Check if photo was uploaded
        if 'photo' not in request.files:
            flash('Please upload a photo.', 'error')
            return redirect(url_for('register'))

        file = request.files['photo']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('register'))

        if file and allowed_file(file.filename):
            # Save the photo
            filename = secure_filename(f"{name.replace(' ', '_')}_{file.filename}")
            filepath = os.path.join(REGISTERED_FACES_DIR, filename)
            file.save(filepath)

            # Add user to database
            user_id = db.add_user(name, email, f"registered_faces/{filename}",
                                 roll_no=roll_no, department=department,
                                 semester=semester)

            # Register face encoding
            if FACE_REC_AVAILABLE:
                success = engine.register_face(filepath, name, user_id)
                if success:
                    flash(f'Successfully registered {name}!', 'success')
                else:
                    flash(f'User saved but no face detected in the photo. '
                          f'Please try with a clearer photo.', 'warning')
            else:
                flash(f'User {name} saved. Face recognition library not '
                      f'installed — encoding not created.', 'warning')

            return redirect(url_for('register'))
        else:
            flash('Invalid file type. Use JPG or PNG.', 'error')
            return redirect(url_for('register'))

    users = db.get_all_users()
    return render_template('register.html', users=users)


@app.route('/attendance')
@login_required
def attendance():
    """View attendance records."""
    date_filter = request.args.get('date', None)
    records = att_mgr.get_attendance(date_filter)
    summary = att_mgr.get_attendance_summary(date_filter)
    return render_template('attendance.html',
                           records=records,
                           summary=summary,
                           date_filter=date_filter or '')


@app.route('/live_feed')
@login_required
def live_feed():
    """Dedicated live feed page with face recognition."""
    return render_template('live_feed.html',
                           face_rec_available=FACE_REC_AVAILABLE)


@app.route('/alerts')
@login_required
def alerts():
    """View unknown face alerts."""
    unknown = db.get_unknown_faces()
    return render_template('alerts.html', alerts=unknown)


@app.route('/mark_reviewed/<int:log_id>')
@login_required
def mark_reviewed(log_id):
    """Mark an alert as reviewed."""
    db.mark_alert_reviewed(log_id)
    flash('Alert marked as reviewed.', 'success')
    return redirect(url_for('alerts'))


@app.route('/delete_alert/<int:log_id>', methods=['POST'])
@login_required
def delete_alert(log_id):
    """Delete an alert and its image file from disk."""
    image_path = db.delete_alert(log_id)
    if image_path:
        full_path = os.path.join('static', image_path)
        if os.path.exists(full_path):
            os.remove(full_path)
    flash('Alert deleted.', 'info')
    return redirect(url_for('alerts'))


@app.route('/bulk_delete_alerts', methods=['POST'])
@login_required
def bulk_delete_alerts():
    """Delete multiple alerts and their image files."""
    data = request.get_json()
    if not data or 'ids' not in data:
        return jsonify({'success': False, 'error': 'No IDs provided'}), 400

    log_ids = [int(i) for i in data['ids']]
    image_paths = db.bulk_delete_alerts(log_ids)

    # Remove image files from disk
    deleted_files = 0
    for img_path in image_paths:
        full_path = os.path.join('static', img_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            deleted_files += 1

    return jsonify({
        'success': True,
        'message': f'Deleted {len(log_ids)} alert(s) and {deleted_files} image(s).'
    })


@app.route('/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    """Permanently delete a registered user, their photos, and their face encoding."""
    import shutil

    # Fetch user info BEFORE deleting from DB
    user = db.get_user_by_id(user_id)
    user_name = user['name'] if user else None
    photo_dir = user['photo_dir'] if user else None
    photo_path = user['photo_path'] if user else None

    # 1. Remove from database
    db.delete_user(user_id)

    # 2. Remove the user's face encoding(s) from the in-memory lists and save
    if user_name and user_name in engine.known_names:
        indices_to_remove = [
            i for i, n in enumerate(engine.known_names) if n == user_name
        ]
        for i in sorted(indices_to_remove, reverse=True):
            engine.known_encodings.pop(i)
            engine.known_names.pop(i)
            engine.known_ids.pop(i)
        engine.save_encodings()
        print(f"[INFO] Removed encoding(s) for '{user_name}' and saved updated pickle.")
    else:
        # Still reload to stay in sync
        engine.load_encodings()

    # 3. Delete the photo folder/file from disk permanently
    static_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    deleted_something = False
    if photo_dir:
        # photo_dir is stored relative to static/  e.g. "registered_faces/John"
        full_dir = os.path.join(static_base, photo_dir)
        if os.path.isdir(full_dir):
            import shutil
            shutil.rmtree(full_dir)
            deleted_something = True
            print(f"[INFO] Deleted photo folder: {full_dir}")
    if not deleted_something and photo_path:
        # photo_path is stored relative to static/  e.g. "registered_faces/John/img.jpg"
        full_file = os.path.join(static_base, photo_path)
        if os.path.isfile(full_file):
            os.remove(full_file)
            print(f"[INFO] Deleted photo file: {full_file}")

    flash(f'User "{user_name or user_id}" permanently deleted.', 'info')
    return redirect(url_for('register'))


@app.route('/video_feed')
@login_required
def video_feed():
    """MJPEG video stream with live face recognition."""
    if not FACE_REC_AVAILABLE:
        return "Face recognition not available", 503
    return Response(
        engine.generate_frames(db, UNKNOWN_FACES_DIR),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/camera_stream')
@login_required
def camera_stream():
    """MJPEG stream for the registration camera preview (no recognition)."""
    return Response(
        engine.generate_preview_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture_frame')
@login_required
def capture_frame():
    """Capture a single frame from the camera and return as JPEG base64."""
    frame_b64 = engine.capture_single_frame()
    if frame_b64 is None:
        return jsonify({'success': False, 'error': 'Could not capture frame'}), 500
    return jsonify({'success': True, 'frame': frame_b64})


@app.route('/register_camera', methods=['POST'])
@login_required
def register_camera():
    """Save 50 captured frames to disk and add user as UNTRAINED. No model training yet."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data received'}), 400

    name   = data.get('name', '').strip()
    email  = data.get('email', '').strip()
    roll_no = data.get('roll_no', '').strip()
    department = data.get('department', '').strip()
    semester = data.get('semester', '').strip()
    frames = data.get('frames', [])

    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    if len(frames) == 0:
        return jsonify({'success': False, 'error': 'No frames captured'}), 400

    import cv2
    safe_name  = name.replace(' ', '_')
    person_dir = os.path.join(REGISTERED_FACES_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)

    saved      = 0
    preview_path = None

    for i, frame_b64 in enumerate(frames):
        try:
            img_bytes = base64.b64decode(frame_b64.split(',')[-1])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            filename = f"{safe_name}_{i:03d}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
            if preview_path is None:
                preview_path = f"registered_faces/{safe_name}/{filename}"
        except Exception as e:
            print(f"[WARN] Frame {i} save failed: {e}")

    if saved == 0:
        return jsonify({'success': False, 'error': 'Failed to save any frames.'}), 500

    # Add user to DB as UNTRAINED (trained=0)
    photo_dir_rel = f"registered_faces/{safe_name}"
    db.add_user(name, email, preview_path,
                photo_dir=photo_dir_rel, trained=0,
                roll_no=roll_no, department=department,
                semester=semester)

    return jsonify({
        'success': True,
        'message': f'{name} added with {saved} photos. '
                   f'Go to the Train Model section and click "Train" to activate recognition.',
        'saved': saved
    })


@app.route('/untrained_users')
@login_required
def untrained_users():
    """Return untrained users as JSON for the train panel."""
    users = db.get_untrained_users()
    data  = [{'user_id': u['user_id'],
               'name':    u['name'],
               'email':   u['email'] or '',
               'roll_no': u['roll_no'] or '',
               'department': u['department'] or '',
               'semester': u['semester'] or '',
               'photo_path': u['photo_path'] or '',
               'registered_date': u['registered_date']}
             for u in users]
    return jsonify({'users': data})


@app.route('/edit_user/<int:user_id>', methods=['POST'])
@login_required
def edit_user(user_id):
    """Update student info (roll_no, department, semester)."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data received'}), 400
    success = db.update_user(
        user_id,
        roll_no=data.get('roll_no'),
        department=data.get('department'),
        semester=data.get('semester'),
        name=data.get('name'),
        email=data.get('email'),
    )
    if success:
        return jsonify({'success': True, 'message': 'Student info updated.'})
    return jsonify({'success': False, 'error': 'Nothing to update.'}), 400


@app.route('/train_model', methods=['POST'])
@login_required
def train_model():
    """Encode faces for all untrained users and update the recognition model."""
    print("\n" + "=" * 55)
    print("[TRAIN] Training started")
    print("=" * 55)

    if not FACE_REC_AVAILABLE:
        print("[TRAIN] ERROR: face_recognition library not available")
        return jsonify({'success': False, 'error': 'Face recognition not available'}), 503

    try:
        import cv2
        import face_recognition as fr
        import time

        train_detect_model = TRAIN_MODEL
        print(f"[TRAIN] Detection model: {train_detect_model}, "
              f"num_jitters: {NUM_JITTERS}, "
              f"encodings_per_person: {NUM_ENCODINGS_PER_PERSON}")

        untrained = db.get_untrained_users()
        if not untrained:
            print("[TRAIN] No untrained users found.")
            return jsonify({'success': False, 'error': 'No untrained users found.'}), 400

        print(f"[TRAIN] Found {len(untrained)} untrained user(s)")

        trained_names   = []
        failed_names    = []
        total_start = time.time()

        for user in untrained:
            user_id   = user['user_id']
            name      = user['name']
            safe_name = name.replace(' ', '_')
            photo_dir = os.path.join(REGISTERED_FACES_DIR, safe_name)

            print(f"\n[TRAIN] --- Processing: {name} (ID: {user_id}) ---")
            encodings_found = []
            user_start = time.time()

            if os.path.isdir(photo_dir):
                image_files = [f for f in sorted(os.listdir(photo_dir))
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"[TRAIN] Found {len(image_files)} images in {photo_dir}")

                for i, fname in enumerate(image_files):
                    fpath = os.path.join(photo_dir, fname)
                    try:
                        img = cv2.imread(fpath)
                        if img is None:
                            print(f"[TRAIN]   [{i+1}/{len(image_files)}] {fname} - SKIP (unreadable)")
                            continue
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        locs = fr.face_locations(rgb, model=train_detect_model)
                        if not locs:
                            print(f"[TRAIN]   [{i+1}/{len(image_files)}] {fname} - no face detected")
                            continue
                        encs = fr.face_encodings(rgb, locs,
                                                 num_jitters=NUM_JITTERS)
                        if encs:
                            encodings_found.append(encs[0])
                            if (i + 1) % 10 == 0 or i == 0:
                                print(f"[TRAIN]   [{i+1}/{len(image_files)}] {fname} - OK "
                                      f"(total: {len(encodings_found)})")
                    except Exception as e:
                        print(f"[TRAIN]   [{i+1}/{len(image_files)}] {fname} - ERROR: {e}")
            elif user['photo_path']:
                fpath = os.path.join('static', user['photo_path'])
                print(f"[TRAIN] Single photo fallback: {fpath}")
                try:
                    img = cv2.imread(fpath)
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        locs = fr.face_locations(rgb, model=train_detect_model)
                        encs = fr.face_encodings(rgb, locs,
                                                 num_jitters=NUM_JITTERS)
                        if encs:
                            encodings_found.extend(encs)
                            print(f"[TRAIN] Single photo encoded OK")
                        else:
                            print(f"[TRAIN] Single photo - no face detected")
                    else:
                        print(f"[TRAIN] Single photo - could not read file")
                except Exception as e:
                    print(f"[TRAIN] Single photo encoding failed: {e}")
            else:
                print(f"[TRAIN] No photo_dir or photo_path for {name}")

            elapsed = time.time() - user_start
            print(f"[TRAIN] {name}: {len(encodings_found)} encodings in {elapsed:.1f}s")

            if not encodings_found:
                print(f"[TRAIN] SKIPPING {name} - no faces detected")
                failed_names.append(name)
                continue

            # Store top N most diverse encodings
            n_keep = min(NUM_ENCODINGS_PER_PERSON, len(encodings_found))
            if len(encodings_found) <= n_keep:
                selected = encodings_found
            else:
                selected = [encodings_found[0]]
                remaining = list(range(1, len(encodings_found)))
                while len(selected) < n_keep and remaining:
                    best_idx = None
                    best_min_dist = -1
                    for idx in remaining:
                        min_dist = min(
                            np.linalg.norm(
                                encodings_found[idx] - s
                            ) for s in selected
                        )
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    selected.append(encodings_found[best_idx])
                    remaining.remove(best_idx)

            for enc in selected:
                engine.known_encodings.append(enc)
                engine.known_names.append(name)
                engine.known_ids.append(user_id)

            db.mark_user_trained(user_id)
            trained_names.append(
                f"{name} ({len(encodings_found)} samples → {len(selected)} encodings)"
            )
            print(f"[TRAIN] ✓ {name}: stored {len(selected)} encodings "
                  f"(from {len(encodings_found)} samples)")

        total_elapsed = time.time() - total_start
        print(f"\n[TRAIN] Complete in {total_elapsed:.1f}s")
        print(f"[TRAIN] Trained: {trained_names}")
        print(f"[TRAIN] Failed:  {failed_names}")

        if not trained_names:
            return jsonify({
                'success': False,
                'error': f'No faces could be detected for: {", ".join(failed_names)}. '
                         f'Please re-register with a clearer face visible.'
            }), 400

        engine.save_encodings()
        print(f"[TRAIN] Saved. Total encodings: {len(engine.known_encodings)}")
        print("=" * 55 + "\n")

        msg = f'Model trained for: {", ".join(trained_names)}.'
        if failed_names:
            msg += f' Failed (no face detected): {", ".join(failed_names)}.'

        return jsonify({'success': True, 'message': msg,
                        'trained': trained_names, 'failed': failed_names})

    except Exception as e:
        import traceback
        print(f"\n[TRAIN] FATAL ERROR: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Training crashed: {str(e)}'
        }), 500


@app.route('/stop_feed')
@login_required
def stop_feed():
    """Stop the live camera feed."""
    engine.stop_camera()
    flash('Camera feed stopped.', 'info')
    return redirect(url_for('dashboard'))


@app.route('/export_csv')
@login_required
def export_csv():
    """Export attendance to CSV and download."""
    date_filter = request.args.get('date', None)
    filepath, filename = att_mgr.export_to_csv(date_filter)
    return send_file(filepath, as_attachment=True, download_name=filename)


# ── Error Handlers ────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    flash('Page not found.', 'error')
    return redirect(url_for('dashboard'))


@app.errorhandler(500)
def internal_error(e):
    flash('An internal error occurred.', 'error')
    return redirect(url_for('dashboard'))


# ── Run ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  AI Face Recognition and Monitoring System")
    print("=" * 55)
    print(f"  Face Recognition: {'Available' if FACE_REC_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  Server: http://{HOST}:{PORT}")
    print(f"  Default Login: admin / admin123")
    print("=" * 55)
    app.run(debug=DEBUG, host=HOST, port=PORT)
