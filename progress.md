# Project Progress: Baby Pitch Agency Experiment

## Phase 1: Backend Setup & UI Initialization ✅ (Complete)
- [x] Implement Group Assignment (G1, G2, G3) and Participant Number in UI.
- [x] Hide webcam preview by default; add "Test eyetracking setup" toggle.
- [x] Create Flask `server.py` to handle silent file saving/loading (trajectories, media, CSVs).
- [x] Create folders `trajectories/`, `VocSoA/sounds/`, `data/`.
- [x] Note: Real-time pitch detection robustness will be iteratively improved in Phase 3.
- [x] Note: Bird replacement with balloon will be done in Phase 3.

## Phase 2: Core State Machine & Trial Loop ✅ (Complete)
- [x] Implement the overarching state machine: Baseline -> Exploration -> Test.
- [x] Configure the 30-trial structure (3 blocks of 10 trials).
- [x] Implement Group assignment logic (G1, G2, G3) mapping to conditions (C1, C2, C3).

## Phase 3: Baseline & Pitch Physics ✅ (Complete)
- [x] Baseline Phase: Free movement, transitions after 5 vocalizations.
- [x] Replaced bird with a procedural Hot Air Balloon.
- [x] Improved pitch mapping (instant) with 100ms movement inertia.

## Phase 4: Exploration Environment ✅ (Complete)
- [x] Draw 4 staggered clouds and define 4 ROIs.
- [x] Implement the "Eraser" mechanism using alpha transparency.
- [x] Implement animal sound system (Cat, Dog, Cow, Duck).
- [x] Balloon movement: Forward (right) -> Return (left) defines the trial.

## Phase 5: Yoked System (C1, C2, C3) ✅ (Complete)
- [x] Implement trajectory recording during C1.
- [x] Automatic JSON saving to `trajectories/` via the local backend.
- [x] Implement C2 (Partial Control) and C3 (Yoked Only) playback logic.

## Phase 6: Mediapipe Eye Tracking ✅ (Complete)
- [x] Integrated Mediapipe Face Mesh for real-time gaze estimation.
- [x] Mapping gaze to ROIs for cloud reveal in the Test Phase.
- [x] Added "Test eyetracking setup" toggle for calibration.

## Phase 7: Data Logging & Output ✅ (Complete)
- [x] Comprehensive CSV logging including gaze, pitch, and yoked data.
- [x] Automatic upload of experiment and webcam video files to `data/`.
- [x] Updated Python analysis script to parse new metrics.
