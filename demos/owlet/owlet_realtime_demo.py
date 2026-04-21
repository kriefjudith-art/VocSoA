"""
🐥 Baby Pitch Agency - Owlet Real-Time Demo
==========================================

This script adapts the 'Owlet' eye-tracking logic for our real-time experiment.
It uses Dlib-based pupil localization and a Random Forest Regressor for gaze mapping.

1. CALIBRATION PHASE:
   - 9-point grid sequence.
   - "One-Tap Burst" 15-frame capture.
   - Stealth HUD in bottom-right.

2. INFERENCE PHASE:
   - Real-time red dot with Asymmetric Smoothing.
"""

import cv2
import numpy as np
import time
import random
import os
import sys
import math
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Path setup for Owlet
OWLET_DIR = os.path.join(os.path.dirname(__file__), 'owlet_repo')
sys.path.append(OWLET_DIR)
from eyetracker.gaze_tracking import GazeTracking

# --- CONFIG ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 
TARGET_POSITIONS = [
    (0.5, 0.5), (0.1, 0.1), (0.5, 0.5), (0.5, 0.1), 
    (0.5, 0.5), (0.9, 0.1), (0.5, 0.5), (0.9, 0.5), 
    (0.5, 0.5), (0.9, 0.9), (0.5, 0.5), (0.5, 0.9), 
    (0.5, 0.5), (0.1, 0.9), (0.5, 0.5), (0.1, 0.5)
]
REPS = 1 
STIM_DURATION = 1.2 

class OwletGazeSystem:
    def __init__(self):
        cwd = OWLET_DIR
        # Initialize Owlet GazeTracking with default baby-ish parameters
        # mean=2.7, max=4, min=1, ratio=1
        self.gaze = GazeTracking(2.7, 4, 1, 1, cwd)
        
        self.raw_data = {pos: [] for pos in TARGET_POSITIONS}
        self.trained_model = None
        
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])

    def get_features(self, frame):
        """Extracts 9 features from Owlet's Dlib-based logic"""
        self.gaze.refresh(frame)
        
        if not self.gaze.pupils_located:
            return None, False

        # 1-4: XY Position components
        xavg, yavg, yleft, yright = self.gaze.xy_gaze_position()
        
        # 5-6: Horizontal gaze
        pupil_left, pupil_right = self.gaze.horizontal_gaze()
        
        # 7: Eye ratio (blinking/opening)
        eye_ratio = self.gaze.eye_ratio()
        
        # 8-9: Eye areas
        areas = self.gaze.get_LR_eye_area()
        if areas:
            l_area, r_area = areas
        else:
            l_area, r_area = 0, 0

        # Features: [xavg, yavg, yleft, yright, p_left, p_right, ratio, l_area, r_area]
        features = [xavg, yavg, yleft, yright, pupil_left, pupil_right, eye_ratio, l_area, r_area]
        
        # Check for NaNs or Nones
        if any(f is None or np.isnan(f) for f in features):
            return None, False
            
        return features, True

    def calibrate(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Owlet_UI", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Owlet_UI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        FRAMES_REQUIRED_PER_POINT = 15 

        for pos in TARGET_POSITIONS:
            frames_collected = 0
            capturing_burst = False
            
            while frames_collected < FRAMES_REQUIRED_PER_POINT:
                ret, frame = cap.read()
                if not ret: break
                
                feat, is_tracked = self.get_features(frame)
                ui = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 15, dtype=np.uint8)
                px, py = int(pos[0]*SCREEN_WIDTH), int(pos[1]*SCREEN_HEIGHT)
                pulse = int(60 * (1 + 0.3 * np.sin(time.time() * 6)))
                
                cv2.circle(ui, (px, py), pulse, (0, 200, 255), -1) 
                
                if not is_tracked:
                    cv2.rectangle(ui, (SCREEN_WIDTH-15, SCREEN_HEIGHT-15), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 255), -1)
                else:
                    cv2.rectangle(ui, (SCREEN_WIDTH-15, SCREEN_HEIGHT-15), (SCREEN_WIDTH, SCREEN_HEIGHT), (30, 30, 30), -1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): 
                    capturing_burst = True 
                elif key == ord('q'): 
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if capturing_burst:
                    if is_tracked and feat:
                        self.raw_data[pos].append(feat)
                        frames_collected += 1
                        cv2.circle(ui, (px, py), pulse, (200, 240, 255), -1) 
                
                cv2.imshow("Owlet_UI", ui)
                
            print('\a') 

        cap.release()
        cv2.destroyAllWindows()
        self.train()

    def train(self):
        print(">>> ANALYZING DATA AND TRAINING OWLET MODEL...")
        
        X, y = [], []
        for pos, feats in self.raw_data.items():
            for feat in feats:
                X.append(feat)
                y.append([pos[0]*SCREEN_WIDTH, pos[1]*SCREEN_HEIGHT])

        if len(X) >= 20: 
            self.model_pipeline.fit(np.array(X), np.array(y))
            self.trained_model = self.model_pipeline
            print(f">>> SUCCESS: Model trained on {len(X)} frames.")
            self.run_inference()
        else:
            print(f">>> ERROR: Not enough data total ({len(X)} frames).")

    def run_inference(self):
        if not self.trained_model: return
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("LiveGaze_Owlet", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("LiveGaze_Owlet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        smooth_gx, smooth_gy = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
        ALPHA_X = 0.20 
        ALPHA_Y = 0.10 

        print(">>> STARTING LIVE OWLET GAZE TRACKING...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            feat, is_tracked = self.get_features(frame)
            screen = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 35, dtype=np.uint8)
            
            if is_tracked and feat:
                prediction = self.trained_model.predict([feat])[0]
                tx, ty = prediction[0], prediction[1]
                
                smooth_gx += ALPHA_X * (tx - smooth_gx)
                smooth_gy += ALPHA_Y * (ty - smooth_gy)
                
                gx, gy = int(smooth_gx), int(smooth_gy)
                gx = max(0, min(SCREEN_WIDTH, gx))
                gy = max(0, min(SCREEN_HEIGHT, gy))
                
                cv2.circle(screen, (gx, gy), 20, (0, 0, 255), -1)
                cv2.circle(screen, (gx, gy), 30, (255, 255, 255), 2)
                cv2.putText(screen, "OWLET GAZE", (gx + 50, gy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("LiveGaze_Owlet", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys = OwletGazeSystem()
    sys.calibrate()
