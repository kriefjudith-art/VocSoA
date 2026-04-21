# 🐥 Baby Pitch Agency Experiment (v2.0 - Precision Edition)

This is a developmental psychology experiment for 18-month-old infants investigating Sense of Agency (SoA) through vocal pitch control and gaze-contingent reveals.

## 👁 High-Precision Gaze Tracking
The project now supports three advanced gaze tracking engines to maximize accuracy for infant research.

### Option 1: MediaPipe 3D (Recommended)
Best for handling infant head movements and 3D facial geometry.
- **Run**: `python realtime_calibrated_gaze.py`
- **Features**: 3D Eye-Socket vectors, Head Pose (Pitch/Yaw/Roll), and 3D Translation.
- **Mapping**: Ridge Polynomial Regression (Degree 2).

### Option 2: WebGazer (Web-Native)
Best for quick, browser-only deployment without local Python scripts.
- **Access**: `http://localhost:5001/precision`
- **Features**: 3D Face Mesh integration + JavaScript-native Ridge Regression.
- **Smoothing**: Asymmetric EMA filter (heavy vertical stabilization).

### Option 3: Owlet (Dlib-Based)
Best for high-contrast, sub-pixel pupil localization.
- **Run**: `python owlet_precision_v2.py`
- **Features**: Owlet pupil ratios + Dlib 3D Head Pose + solvePnP.

---

## 🛠 Setup & Installation

### 1. Prerequisites
- Python 3.8+ and `ffmpeg` (for video conversion).
- MediaPipe Model: `face_landmarker.task` (Included in root).

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./venv/bin/pip install dlib cmake joblib scikit-learn  # Precision dependencies
```

---

## 🚀 Running the Experiment

### 1. Calibration (Stealth HUD)
All trackers now use a **Stealth Calibration** protocol designed for infants:
- **One-Tap Burst**: Press **Spacebar** once to capture 15 high-quality frames while the baby fixates on the sun.
- **Stealth HUD**: A tiny status circle in the bottom-right corner (Green = Tracking, Red = Lost).
- **Looming Stimulus**: Pulsing sun (☀️) on a dark background to prevent pupil constriction.

### 2. Main Experiment Loop
1. Start server: `python server.py`
2. Access interface: `http://localhost:5001/`
3. Enter Participant ID and Group.
4. Complete Calibration -> Baseline -> Exploration -> Test.

---

## 📊 Data Analysis
Recorded data is saved to `data/` with clean `.csv` extensions.
- **Metrics**: Vocal duration, distance traveled, ROI reveal percentages, and 15-dimensional gaze feature vectors.
- **Offline Validation**: Use `python test_gaze_pipeline.py` to overlay gaze tracking on recorded videos for verification.

---

## 📝 Configuration Note
- **C1**: Full control.
- **C2**: Partial control (vocal triggers playback).
- **C3**: No control (full yoked playback).

Documentation updated for Precision Phase (v2.0).
