# 🐥 Baby Pitch Agency Experiment (v15_04)

This project is a developmental psychology experiment designed for 18-month-old infants. It investigates Sense of Agency (SoA) by allowing infants to control a hot air balloon on screen using their vocal pitch.

## 🌟 Key Features
- **3-Phase Procedure**: Baseline, Exploration (Eraser mechanism), and Test (Eye-tracking reveal).
- **Yoked System**: Automatically records and plays back trajectories across participants for control conditions (C1, C2, C3).
- **Real-time Gaze Tracking**: Uses Mediapipe Face Mesh to detect infant focus on screen regions (ROIs).
- **Automated Data Management**: A Python Flask backend silently saves CSV logs, JSON trajectories, and video recordings to local folders.

---

## 🛠 Setup & Installation

### 1. Prerequisites
- Python 3.8 or higher.
- A modern web browser (Chrome or Firefox recommended for MediaRecorder support).
- A webcam and microphone.

### 2. Environment Setup
It is recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it (MacOS/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Folder Structure
Ensure the following folders exist (the server will create them automatically if missing):
- `data/`: Stores CSV logs and video files.
- `trajectories/`: Stores JSON trajectory files for yoked conditions.
- `VocSoA/sounds/`: Place your experiment audio files (e.g., `cat_20.mp3`) here.

---

## 🚀 Running the Experiment

### 1. Start the Local Server
The server handles file saving and serves the experiment interface.

```bash
python server.py
```
By default, the server runs on `http://localhost:5001`.

### 2. Access the Interface
1. Open your browser and navigate to `http://localhost:5001`.
2. Enter the **Participant Number** and select the **Group Assignment**.
3. Toggle "Test eyetracking setup" if you need to calibrate the webcam.
4. Click **Start Experiment**.

### 3. Experiment Flow
- **Baseline**: 5 vocalizations move the balloon before the trial starts.
- **Exploration**: 30 trials (3 blocks of 10). The balloon reveals animals behind clouds when pitch is detected.
- **Test Phase**: Looking at a cloud ROI reveals the animal using gaze-contingency.

---

## 📊 Data Analysis

Once you have collected data, use the provided analysis tools:

### Option A: Python Script
Run a quick summary of all files in the `data/` folder:
```bash
python analysis.py
```

### Option B: Jupyter Notebook
For professional visualizations and condition comparisons:
1. Launch Jupyter: `jupyter notebook`
2. Open `experiment_analysis.ipynb`.
3. Run all cells to generate Seaborn plots and detailed metrics.

---

## 📝 Configuration Note
- **Conditions**: 
    - **C1**: Full control (where/when).
    - **C2**: Partial control (vocal triggers playback of a yoked trajectory).
    - **C3**: No control (full yoked playback).
- **Yoking Pool**: The first participant (ID: 1) is automatically assigned C1 for all blocks to seed the yoked trajectory pool.

---

## 👥 Contributors
Built for research in developmental psychology. Documentation and environment formalization by Gemini CLI.
