<div align="center">

# 🤖 Face Recognition & Monitoring System

Real-time face recognition and attendance tracking built with Python and Flask.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## What it does

- 🎥 Live camera feed with face detection and recognition
- 📸 Register faces by capturing 50 photos automatically
- 🗓️ Logs attendance with timestamps (no duplicates)
- 🚨 Saves and alerts on unknown faces
- 📁 Export attendance to CSV
- 🔐 Admin login with hashed passwords

---
### 1. Install System Dependencies (Linux)

```bash
sudo apt install cmake g++ libopenblas-dev liblapack-dev libx11-dev
```

### 2. Clone & install

```bash
git clone https://github.com/Nkingsap/Face_Rec_Mon.git
cd Face_Rec_Mon
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

> ⚠️ Use `localhost`, not your IP address — browsers block camera access on plain HTTP.

---

## How to use

### Register a face
1. Go to **Register Face** in the sidebar
2. Enter name and email
3. Click **Open Camera** → **Start Capture** (captures 50 photos automatically)

### Train the model
After registering, the person won't be recognized yet.

1. Scroll down to **Train Model** on the same page
2. Click **Train Model** and wait ~30–60 seconds
3. Done — the person will now be recognized ✅

### Attendance & Alerts
- **Attendance** tab — filter by date/name, export to CSV
- **Alerts** tab — review unknown face snapshots, mark reviewed or delete.

---

## Default login

| Username | Password |
|----------|----------|
| `admin`  | `admin123` |

> ⚠️ Change this after your first login.

---

## Project structure

```
Face_Rec_Mon/
│
├── app.py                    # Main Flask app — all routes (login, register,
│                             #   attendance, alerts, video feed, train model)
├── face_rec_engine.py        # Core recognition engine — loads encodings,
│                             #   detects faces, draws overlays, streams MJPEG
├── attendance_manager.py     # Marks attendance in DB, prevents duplicate entries
├── config.py                 # Central config — file paths, tolerance, model type
├── requirements.txt          # Python dependencies
│
├── database/
│   ├── db_handler.py         # All SQLite queries (users, attendance, alerts, admin)
│   └── database.db           # SQLite DB — auto-created on first run
│
├── encodings/
│   └── face_encodings.pkl    # Saved face vectors — auto-created after training
│
├── static/
│   ├── css/
│   │   └── style.css         # App stylesheet
│   ├── registered_faces/     # Captured photos used for training (50 per person)
│   └── unknown_faces/        # Snapshots of unrecognized faces (alert images)
│
├── templates/
│   ├── base.html             # Shared layout with sidebar navigation
│   ├── login.html            # Admin login page
│   ├── dashboard.html        # Overview — stats, recent attendance, unknown alerts
│   ├── register.html         # Face registration + Train Model panel
│   ├── attendance.html       # Attendance records with date/name filter & CSV export
│   └── alerts.html           # Unknown face alerts with lightbox viewer
│
└── reports/                  # CSV exports saved here
```

---

## Notes

- **dlib** takes 5–15 min to compile on first install
- For better accuracy (GPU required), switch model to `cnn` in `config.py`

---

<div align="center">
Who is Hero?
Nking is Hero
</div>
