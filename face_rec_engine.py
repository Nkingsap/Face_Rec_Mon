"""
Face Recognition Engine for the AI Face Recognition and Monitoring System.
Handles face detection, encoding, registration, and real-time recognition.
"""
import os
import pickle
import cv2
import numpy as np
from config import (FRAME_RESIZE_FACTOR, NUM_JITTERS,
                    UNKNOWN_COOLDOWN_SECONDS, UNKNOWN_BUCKET_SIZE,
                    UNKNOWN_MIN_FACE_SIZE, RECOGNITION_TOLERANCE)
from datetime import datetime

# Try to import face_recognition; provide fallback if unavailable
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[WARNING] face_recognition library not installed. "
          "Face recognition features will be disabled.")
    print("Install with: pip install face-recognition dlib cmake")


class FaceRecognitionEngine:
    """Core engine for face detection and recognition."""

    def __init__(self, encodings_path, tolerance=0.6, model='hog'):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.model = model  # 'hog' for CPU, 'cnn' for GPU
        self.resize_factor = FRAME_RESIZE_FACTOR
        self.scale_back = int(1.0 / FRAME_RESIZE_FACTOR)
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.is_running = False
        self.camera = None
        self.load_encodings()

    def load_encodings(self):
        """Load saved face encodings from pickle file."""
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_names = data.get('names', [])
                self.known_ids = data.get('ids', [])
                print(f"[INFO] Loaded {len(self.known_encodings)} face encodings.")
            except Exception as e:
                print(f"[ERROR] Failed to load encodings: {e}")
                self.known_encodings = []
                self.known_names = []
                self.known_ids = []
        else:
            print("[INFO] No existing encodings file found. Starting fresh.")

    def save_encodings(self):
        """Save face encodings to pickle file."""
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names,
            'ids': self.known_ids
        }
        with open(self.encodings_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Saved {len(self.known_encodings)} face encodings.")

    def register_face(self, image_path, name, user_id=None):
        """
        Register a new face from an image file.
        Returns True if successful, False if no face detected.
        """
        if not FACE_REC_AVAILABLE:
            print("[ERROR] face_recognition not available.")
            return False

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(
                image, num_jitters=NUM_JITTERS
            )

            if len(encodings) == 0:
                print(f"[WARNING] No face detected in {image_path}")
                return False

            # Use the first detected face
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            self.known_ids.append(user_id)
            self.save_encodings()
            print(f"[INFO] Registered face for: {name}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to register face: {e}")
            return False

    def recognize_faces(self, frame):
        """
        Detect and recognize faces in a video frame.
        Returns list of dicts with 'name', 'location', 'confidence'.
        """
        if not FACE_REC_AVAILABLE:
            return []

        # Resize for faster processing (configurable factor)
        fx = self.resize_factor
        small_frame = cv2.resize(frame, (0, 0), fx=fx, fy=fx)

        # Histogram equalization for better lighting handling
        yuv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        preprocessed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        rgb_frame = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            confidence = 0.0
            user_id = None

            if len(self.known_encodings) > 0:
                # Compare against all stored encodings
                face_distances = face_recognition.face_distance(
                    self.known_encodings, encoding
                )

                # Find the best match (lowest distance)
                best_match_idx = np.argmin(face_distances)
                best_distance = face_distances[best_match_idx]

                # Only accept if within tolerance
                if best_distance <= self.tolerance:
                    name = self.known_names[best_match_idx]
                    confidence = round(1.0 - best_distance, 2)
                    if best_match_idx < len(self.known_ids):
                        user_id = self.known_ids[best_match_idx]

            # Scale back coordinates based on resize factor
            sb = self.scale_back
            top, right, bottom, left = location
            top *= sb
            right *= sb
            bottom *= sb
            left *= sb

            results.append({
                'name': name,
                'user_id': user_id,
                'location': (top, right, bottom, left),
                'confidence': confidence,
                'encoding': encoding  # raw encoding for dedup
            })

        return results

    def draw_results(self, frame, results):
        """Draw bounding boxes and names on the frame."""
        for res in results:
            top, right, bottom, left = res['location']
            name = res['name']
            confidence = res['confidence']

            # Color: green for recognized, red for unknown
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 200, 0)  # Green

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label background
            label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (left, bottom),
                (left + label_size[0] + 10, bottom + label_size[1] + 16),
                color, cv2.FILLED
            )
            cv2.putText(
                frame, label,
                (left + 5, bottom + label_size[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame

    def generate_frames(self, db_handler=None, unknown_faces_dir=None):
        """
        Generator that yields MJPEG frames from the camera.
        Performs face recognition on each frame and optionally
        marks attendance and logs unknown faces.
        Supports multiple faces simultaneously.
        """
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            print("[ERROR] Could not open camera.")
            return

        self.is_running = True
        frame_count = 0
        # Cache last recognition results so boxes are drawn on EVERY frame
        cached_results = []
        # ── Unknown-face throttle state ──────────────────
        # Spatial cooldown: bucket key → last-saved datetime
        unknown_cooldown = {}
        # Encoding ring-buffer for deduplication (last N saved encodings)
        _RING_SIZE = 20
        recent_unknown_encs = []
        # Global rate cap: timestamps of saves in the last minute
        save_timestamps = []
        _MAX_SAVES_PER_MIN = 3

        try:
            while self.is_running:
                success, frame = self.camera.read()
                if not success:
                    break

                # Run recognition every 3rd frame for performance;
                # reuse cached_results on the other frames so boxes always show
                if frame_count % 3 == 0:
                    cached_results = self.recognize_faces(frame)

                    for res in cached_results:
                        if res['name'] != "Unknown" and db_handler:
                            # Mark attendance for recognised faces
                            db_handler.mark_attendance(
                                res['name'], res.get('user_id')
                            )
                        elif res['name'] == "Unknown" and db_handler and unknown_faces_dir:
                            top, right, bottom, left = res['location']

                            # ── Filter 1: minimum face size ─────────
                            face_w = right - left
                            face_h = bottom - top
                            if face_w < UNKNOWN_MIN_FACE_SIZE or face_h < UNKNOWN_MIN_FACE_SIZE:
                                continue  # too small / too far away

                            # ── Filter 2: spatial-bucket cooldown ───
                            bucket = UNKNOWN_BUCKET_SIZE
                            cx = ((left + right) // 2) // bucket
                            cy = ((top + bottom) // 2) // bucket
                            cooldown_key = f"{cx}_{cy}"
                            now = datetime.now()
                            last_saved = unknown_cooldown.get(cooldown_key)
                            if last_saved is not None and (now - last_saved).total_seconds() < UNKNOWN_COOLDOWN_SECONDS:
                                continue  # same spot, still in cooldown

                            # ── Filter 3: encoding deduplication ────
                            encoding = res.get('encoding')
                            if encoding is not None and len(recent_unknown_encs) > 0:
                                dists = face_recognition.face_distance(
                                    recent_unknown_encs, encoding
                                )
                                if np.min(dists) <= RECOGNITION_TOLERANCE:
                                    # Same unknown person already saved recently
                                    unknown_cooldown[cooldown_key] = now
                                    continue

                            # ── Filter 4: global rate cap ───────────
                            save_timestamps = [
                                t for t in save_timestamps
                                if (now - t).total_seconds() < 60
                            ]
                            if len(save_timestamps) >= _MAX_SAVES_PER_MIN:
                                continue  # hit per-minute cap

                            # ── All filters passed — save ───────────
                            unknown_cooldown[cooldown_key] = now
                            save_timestamps.append(now)
                            if encoding is not None:
                                recent_unknown_encs.append(encoding)
                                if len(recent_unknown_encs) > _RING_SIZE:
                                    recent_unknown_encs.pop(0)

                            timestamp = now.strftime('%Y%m%d_%H%M%S_%f')[:19]
                            filename = f"unknown_{timestamp}.jpg"
                            os.makedirs(unknown_faces_dir, exist_ok=True)
                            filepath = os.path.join(unknown_faces_dir, filename)
                            face_img = frame[max(0, top - 20):bottom + 20,
                                              max(0, left - 20):right + 20]
                            if face_img.size > 0:
                                cv2.imwrite(filepath, face_img)
                                db_handler.log_unknown_face(
                                    f"unknown_faces/{filename}"
                                )
                                print(f"[UNKNOWN] Saved {filename}")

                # Always draw cached results so boxes are visible on every frame
                frame = self.draw_results(frame, cached_results)
                frame_count += 1

                # Add timestamp overlay
                timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(
                    frame, timestamp_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

        finally:
            self.stop_camera()

    def generate_preview_frames(self):
        """
        Simple MJPEG stream for registration preview.
        Draws a face-detection overlay but does NOT run recognition.
        """
        import cv2 as _cv2
        cam = _cv2.VideoCapture(0)
        if not cam.isOpened():
            return
        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    break
                # Draw a green guide oval so the user knows where to position their face
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                _cv2.ellipse(frame, (cx, cy), (w // 5, h // 3), 0, 0, 360, (0, 220, 100), 2)
                _cv2.putText(frame, 'Position face inside oval', (cx - 130, cy + h // 3 + 25),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1)
                ret, buf = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        finally:
            cam.release()

    def capture_single_frame(self):
        """
        Open the camera, grab one frame, return it as a base64 JPEG data-URL.
        Returns None on failure.
        """
        import cv2 as _cv2
        import base64
        cam = _cv2.VideoCapture(0)
        if not cam.isOpened():
            return None
        # Warm-up: skip first few frames so exposure adjusts
        for _ in range(3):
            cam.read()
        ok, frame = cam.read()
        cam.release()
        if not ok or frame is None:
            return None
        ret, buf = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'

    def stop_camera(self):
        """Release camera resources."""
        self.is_running = False
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        print("[INFO] Camera released.")
