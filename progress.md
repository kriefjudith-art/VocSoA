# Project Progress: Baby Pitch Agency Experiment

## Phase 1: Backend Setup & UI Initialization ✅ (Complete)
- [x] Implement Group Assignment (G1, G2, G3) and Participant Number in UI.
- [x] Create Flask `server.py` to handle silent file saving/loading.

## Phase 2: Core State Machine & Trial Loop ✅ (Complete)
- [x] Overarching state machine: Calibration -> Baseline -> Exploration -> Test.
- [x] Configure 30-trial structure and condition mapping.

## Phase 3: Baseline & Pitch Physics ✅ (Complete)
- [x] Replaced bird with procedural Hot Air Balloon.
- [x] Improved pitch mapping with YIN algorithm and movement inertia.

## Phase 4: Exploration Environment ✅ (Complete)
- [x] Implement "Eraser" mechanism for cloud reveal via balloon/vocal pitch.
- [x] Animal sound system (Cat, Dog, Cow, Duck).

## Phase 5: Yoked System (C1, C2, C3) ✅ (Complete)
- [x] Trajectory recording and JSON playback logic for control conditions.

## Phase 6: Precision Gaze Tracking & Calibration ✅ (Complete)
- [x] **Consensus-Based Calibration**: Implemented "One-Tap Burst" 15-frame capture.
- [x] **Precision 3D Vectors**: Shifted from 2D eyelid ratios to 3D Eye-Socket centroids.
- [x] **Asymmetric Smoothing**: Heavily stabilized vertical jitter (Alpha Y < Alpha X).
- [x] **Multi-Engine Support**: 
    - [x] MediaPipe 3D (Python/JS): High head-pose robustness.
    - [x] WebGazer (JS): Smooth browser-native tracking.
    - [x] Owlet (Python): Sub-pixel pupil localization using Dlib.

## Phase 7: Data Integrity & Analysis ✅ (Complete)
- [x] Fixed file extension bug in backend (e.g., `.csv_G1` -> `.csv`).
- [x] Added `test_gaze_pipeline.py` for offline validation of eye tracking.
- [x] Updated analysis script to handle expanded 15-feature datasets.
