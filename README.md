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

## Setup

### 1. Install system dependencies (Linux)

```bash
sudo apt install cmake g++ libopenblas-dev liblapack-dev libx11-dev
```

### 2. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/Face_Rec_Mon.git
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
- **Alerts** tab — review unknown face snapshots, mark reviewed or delete

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
├── app.py                 # Flask app & routes
├── face_rec_engine.py     # Face recognition core
├── attendance_manager.py  # Attendance logging
├── config.py              # Settings (paths, tolerances)
├── database/
│   └── db_handler.py      # SQLite operations
├── static/                # CSS, JS, face photos
├── templates/             # HTML pages
└── requirements.txt
```

---

## Notes

- **dlib** takes 5–15 min to compile on first install
- For better accuracy (GPU required), switch model to `cnn` in `config.py`

---

<div align="center">
Made with ❤️ using Python, Flask, OpenCV & dlib
</div>
