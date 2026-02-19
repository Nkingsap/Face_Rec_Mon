<div align="center">

# 🤖 AI Face Recognition & Monitoring System

A real-time face recognition and attendance monitoring system built with Python and Flask.  
Automatically detects and identifies faces from a live camera feed, logs attendance, and alerts on unknown visitors.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Screenshots](#-screenshots)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Default Credentials](#-default-credentials)
- [Utility Scripts](#-utility-scripts)
- [Known Limitations](#-known-limitations)
- [License](#-license)

---

## ✨ Features

- 🎥 **Live camera feed** with real-time face detection and recognition overlay
- 📸 **Multi-photo registration** — captures 50 frames automatically via browser camera
- 🧠 **Two-step training workflow** — capture photos first, then train the model separately for higher accuracy
- 📊 **Averaged face encodings** — 50 samples per person → one robust 128-dimensional vector
- 🗓️ **Automatic attendance logging** with timestamps and duplicate-prevention
- 🚨 **Alert system** — unknown faces are captured, stored, and shown with filter/lightbox UI
- 🖼️ **Alert lightbox** — view full-size unknown face photos with download and delete options
- 📁 **CSV export** of attendance reports
- 🔐 **Admin authentication** with hashed passwords
- 🗄️ **SQLite database** — zero configuration, single file

---

## 📸 Screenshots

> Screenshots will appear here once the app is running.

| Dashboard | Register Face | Alerts |
|:---------:|:-------------:|:------:|
| *(add screenshot)* | *(add screenshot)* | *(add screenshot)* |

---

## ⚙️ Requirements

| Dependency | Version |
|---|---|
| Python | 3.10 or higher (tested on 3.14) |
| Flask | ≥ 2.3.0 |
| OpenCV | ≥ 4.8.0 |
| face-recognition | ≥ 1.3.0 |
| dlib | ≥ 19.24.0 |
| NumPy | ≥ 1.24.0 |
| Pandas | ≥ 2.0.0 |
| CMake | ≥ 3.27.0 |

> ⚠️ **dlib** requires `cmake` and a C++ compiler (e.g. `g++`).  
> On Ubuntu/Debian: `sudo apt install cmake g++ libopenblas-dev liblapack-dev`

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Face_Rec_Mon.git
cd Face_Rec_Mon
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install system dependencies (Linux)

```bash
sudo apt update
sudo apt install cmake g++ libopenblas-dev liblapack-dev libx11-dev
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> 💡 **Python 3.12+ users:** If you get a `pkg_resources` error from `face_recognition_models`, the project includes a compatibility patch that applies automatically on startup.

### 5. Run the application

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

> ⚠️ **Use `localhost` not your IP address** for the live camera capture to work.  
> Browsers block camera access on plain HTTP (`http://192.168.x.x`) — this is a browser security restriction.  
> The server-side recognition feed works from any address.

---

## 🚀 Usage

### Registering a New Face

1. Log in with the default admin credentials (see below)
2. Go to **Register Face** in the sidebar
3. Enter the person's **Name** and **Email**
4. Click **Open Camera** → wait for the live feed to appear
5. Click **Start Capture** — the system automatically captures **50 photos**
6. Watch the progress bar fill to 100% and the thumbnail strip appear
7. A success message confirms the photos were saved

### Training the Model

After capturing, the person is marked as **Untrained** — they won't be recognised yet.

1. Scroll down to the **Train Model** panel (same page)
2. You'll see all untrained users listed
3. Click **Train Model**
4. Wait ~30–60 seconds while it processes all 50 photos per person
5. A result message shows who was trained successfully
6. The person will now be recognised by the live camera feed ✅

### Viewing Attendance

- Go to **Attendance** in the sidebar
- Filter by date or search by name
- Export to **CSV** using the download button

### Managing Alerts (Unknown Faces)

- Go to **Alerts** in the sidebar
- Use filter buttons: **All / Unreviewed / Reviewed**
- Click any photo to open the **lightbox** — view full size, download, or delete
- Use **Mark Reviewed** on cards you've checked
- Use **Delete** to permanently remove an alert and its image file

---

## 📁 Project Structure

```
Face_Rec_Mon/
│
├── app.py                    # Main Flask application & all routes
├── face_rec_engine.py        # Face recognition core (encoding, detection, streaming)
├── attendance_manager.py     # Attendance logging logic
├── config.py                 # App configuration (paths, tolerances, etc.)
├── inspect_encodings.py      # Utility: view face_encodings.pkl contents
├── requirements.txt
│
├── database/
│   ├── db_handler.py         # All database operations (SQLite)
│   └── database.db           # SQLite database (auto-created on first run)
│
├── encodings/
│   └── face_encodings.pkl    # Serialized face encodings (auto-created after training)
│
├── static/
│   ├── css/style.css         # Custom stylesheet
│   ├── js/main.js            # Client-side JavaScript
│   ├── registered_faces/     # Captured face photos (50 per person)
│   └── unknown_faces/        # Captured unknown face snapshots
│
├── templates/
│   ├── base.html             # Base layout with sidebar navigation
│   ├── login.html            # Admin login page
│   ├── dashboard.html        # Main dashboard
│   ├── register.html         # Face registration + Train Model panel
│   ├── attendance.html       # Attendance records
│   ├── alerts.html           # Unknown face alerts with lightbox
│   └── reports.html          # Report generation
│
├── reports/                  # Generated CSV reports
├── chapters/                 # Project documentation chapters (college report)
└── documentation.txt         # Full project report (college format)
```

---

## 🔬 How It Works

### Face Recognition Pipeline

```
Live Camera → Capture Frame → Detect Face Locations (HOG)
    → Extract 128-dim Encoding → Compare with Known Encodings
    → Match (Euclidean distance < 0.6) → Log Attendance
    → No Match → Capture & Store as Unknown Face Alert
```

### Two-Step Registration Workflow

```
Step 1 — Capture:
  Browser getUserMedia → 50 frames at 350ms intervals
  → Base64 JPEG frames → POST /register_camera
  → Saved to static/registered_faces/<name>/
  → User added to DB with trained=0

Step 2 — Train:
  POST /train_model
  → Read all saved photos for each untrained user
  → Extract face encoding from each photo
  → Average all encodings → single 128-dim vector
  → Append to engine.known_encodings
  → Save to face_encodings.pkl
  → Mark user as trained=1 in DB
```

### Database Schema

| Table | Purpose |
|---|---|
| `users` | Registered users (`trained`, `photo_dir` columns included) |
| `attendance` | Daily attendance records |
| `unknown_faces` | Unknown face alert log |
| `admin` | Admin credentials (SHA-256 hashed passwords) |

---

## 🔑 Default Credentials

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `admin123` |

> ⚠️ Change these immediately after your first login in a production environment.

---

## 🛠️ Utility Scripts

### Inspect face encodings

View the contents of `face_encodings.pkl` in human-readable form:

```bash
python inspect_encodings.py
```

Output shows each stored person's name, user ID, first 8 encoding values, and vector norm.

---

## ⚠️ Known Limitations

| Limitation | Notes |
|---|---|
| Camera access requires `localhost` or HTTPS | Browsers block `getUserMedia` on plain HTTP + IP. Use `localhost:5000` for registration. |
| Single camera only | Only one webcam (index `0`) is currently used for the live recognition feed |
| dlib build time | First install can take 5–15 minutes to compile dlib from source |
| HOG model | The default `hog` model is CPU-only. Switch to `cnn` in `config.py` for better accuracy (requires NVIDIA GPU + CUDA) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ using Python, Flask, OpenCV & dlib
</div>
