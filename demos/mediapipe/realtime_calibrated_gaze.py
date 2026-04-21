"""
🐥 Baby Pitch Agency - Real-Time Calibrated Gaze Tracker (Full ML Edition)
==========================================================================

1. CALIBRATION PHASE: 
   - 9-point grid sequence (Corners + Midpoints + Center Anchors).
   - "One-Tap Burst" 15-frame capture per point.
   - Stealth HUD for tracking status.
   
2. ANALYSIS PHASE:
   - Random Forest Regressor (n=100, depth=10).
   - 15 Features: 3D Gaze Vectors, 3D Head Pose/Translation, Scale (IOD), and Eyelid State (EAR).

3. INFERENCE PHASE:
   - Real-time prediction with Asymmetric EMA Smoothing.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os
import sys
import math
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- CONFIG ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../face_landmarker.task')
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 
TARGET_POSITIONS = [
    (0.5, 0.5), (0.1, 0.1), (0.5, 0.5), (0.5, 0.1), 
    (0.5, 0.5), (0.9, 0.1), (0.5, 0.5), (0.9, 0.5), 
    (0.5, 0.5), (0.9, 0.9), (0.5, 0.5), (0.5, 0.9), 
    (0.5, 0.5), (0.1, 0.9), (0.5, 0.5), (0.1, 0.5)
]
REPS = 1 
STIM_DURATION = 1.2 

class GazeSystem:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.raw_data = {pos: [] for pos in TARGET_POSITIONS}
        self.trained_model = None
        
        # ML PIPELINE: Random Forest is robust to non-linear infant head movements
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])

    def get_features(self, frame):
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return None, False 

        lm = res.face_landmarks[0]
        
        def dist_3d(i, j):
            return math.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2 + (lm[i].z - lm[j].z)**2)

        # 1. 3D Gaze Vectors (6 features)
        def get_gaze_v(iris_idx, inner_idx, outer_idx):
            iris = lm[iris_idx]
            inner, outer = lm[inner_idx], lm[outer_idx]
            cx, cy, cz = (inner.x + outer.x)/2, (inner.y + outer.y)/2, (inner.z + outer.z)/2
            return [iris.x - cx, iris.y - cy, iris.z - cz]
        
        lv = get_gaze_v(468, 133, 33)
        rv = get_gaze_v(473, 362, 263)
        
        # 2. Head Pose (6 features)
        mat = res.facial_transformation_matrixes[0]
        rotation_mat = mat[:3, :3]
        decomp = cv2.RQDecomp3x3(rotation_mat)
        pitch, yaw, roll = decomp[0]
        trans_x, trans_y, trans_z = mat[0, 3], mat[1, 3], mat[2, 3]
        
        # 3. Scale Proxy: Inter-ocular Distance (1 feature)
        iod = dist_3d(33, 263)
        
        # 4. Eyelid State: Eye Aspect Ratio (2 features)
        l_ear = (dist_3d(159, 145) + dist_3d(158, 153)) / (2.0 * dist_3d(33, 133))
        r_ear = (dist_3d(386, 374) + dist_3d(385, 380)) / (2.0 * dist_3d(362, 263))
        
        # Total: 15 Features
        features = [*lv, *rv, pitch, yaw, roll, trans_x, trans_y, trans_z, iod, l_ear, r_ear]
        return features, True 

    def calibrate(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Baby_UI", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Baby_UI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
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
                
                cv2.imshow("Baby_UI", ui)
                
            print('\a') 

        cap.release()
        cv2.destroyAllWindows()
        self.train()

    def train(self):
        print(">>> ANALYZING DATA AND TRAINING MODEL...")
        
        import pandas as pd
        export_rows = []
        for pos, feats in self.raw_data.items():
            for f in feats:
                export_rows.append([pos[0], pos[1], *f])
        
        cols = ['Target_X', 'Target_Y', 'LVx', 'LVy', 'LVz', 'RVx', 'RVy', 'RVz', 'Pitch', 'Yaw', 'Roll', 'TX', 'TY', 'TZ', 'IOD', 'L_EAR', 'R_EAR']
        df = pd.DataFrame(export_rows, columns=cols)
        df.to_csv('calibration_data_raw.csv', index=False)

        X, y = [], []
        for pos, feats in self.raw_data.items():
            for feat in feats:
                X.append(feat)
                y.append([pos[0]*SCREEN_WIDTH, pos[1]*SCREEN_HEIGHT])

        if len(X) >= 20: 
            print(f">>> TRAINING RANDOM FOREST MODEL ON {len(X)} SAMPLES...")
            self.model_pipeline.fit(np.array(X), np.array(y))
            self.trained_model = self.model_pipeline
            print(">>> SUCCESS: Model trained.")
            self.run_inference()
        else:
            print(f">>> ERROR: Not enough data total ({len(X)} frames).")

    def run_inference(self):
        if not self.trained_model: return
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("LiveGaze", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("LiveGaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        smooth_gx, smooth_gy = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
        ALPHA_X = 0.20 
        ALPHA_Y = 0.10 

        print(">>> STARTING LIVE GAZE TRACKING (Press 'q' to exit)...")
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
                cv2.putText(screen, "CALIBRATED GAZE", (gx + 50, gy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("LiveGaze", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: MediaPipe model '{MODEL_PATH}' not found.")
    else:
        sys = GazeSystem()
        sys.calibrate()
