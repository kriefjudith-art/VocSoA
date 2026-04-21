"""
🐥 Baby Pitch Agency - Owlet Precision Demo (V2)
==============================================

Combines Owlet's pupil localization with 3D Head Pose and Ridge Regression.

1. CALIBRATION: 9-point grid with center anchors. One-tap burst.
2. FEATURES: Pupil ratios + Dlib-based 3D Head Pose (15 total).
3. MODEL: Ridge Regression with Asymmetric EMA Smoothing.
"""

import cv2
import numpy as np
import time
import random
import os
import sys
import math
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Path setup for Owlet
sys.path.append(os.path.abspath("owlet_repo"))
from eyetracker.gaze_tracking import GazeTracking

# --- CONFIG ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 
TARGET_POSITIONS = [
    (0.5, 0.5), (0.05, 0.05), (0.5, 0.5), (0.5, 0.05), 
    (0.5, 0.5), (0.95, 0.05), (0.5, 0.5), (0.95, 0.5), 
    (0.5, 0.5), (0.95, 0.95), (0.5, 0.5), (0.5, 0.95), 
    (0.5, 0.5), (0.05, 0.95), (0.5, 0.5), (0.05, 0.5)
]
REPS = 1 
STIM_DURATION = 1.2 

class OwletPrecisionSystem:
    def __init__(self):
        cwd = os.path.abspath("owlet_repo")
        self.gaze = GazeTracking(2.7, 4, 1, 1, cwd)
        self.raw_data = {pos: [] for pos in TARGET_POSITIONS}
        self.trained_model = None
        
        # 3D Face Model for Head Pose (approximate generic model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Stable Ridge Pipeline
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=10.0))
        ])

    def get_head_pose(self, landmarks, frame_shape):
        """Extracts 3D Rotation (Pitch, Yaw, Roll) and Translation using solvePnP"""
        h, w = frame_shape[:2]
        # Map 68-landmarks to our 6-point 3D model
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye corner
            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye corner
            (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
        ], dtype="double")

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        success, rot_vec, trans_vec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)
        
        # Convert rotation vector to Euler angles
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        decomp = cv2.RQDecomp3x3(rot_mat)
        euler = decomp[0]
        return [*euler, *trans_vec.flatten()]

    def get_features(self, frame):
        self.gaze.refresh(frame)
        if not self.gaze.pupils_located or not self.gaze.face:
            return None, False

        # 1. Gaze Ratios (4 features)
        xavg, yavg, yleft, yright = self.gaze.xy_gaze_position()
        
        # 2. Pupil positions (2 features)
        pl, pr = self.gaze.horizontal_gaze()
        
        # 3. Eye metrics (3 features)
        ratio = self.gaze.eye_ratio()
        areas = self.gaze.get_LR_eye_area() or (0,0)
        
        # 4. Head Pose (6 features)
        pose = self.get_head_pose(self.gaze.landmarks, frame.shape)
        
        # Total 15 features
        features = [xavg, yavg, yleft, yright, pl, pr, ratio, *areas, *pose]
        if any(f is None or np.isnan(f) for f in features):
            return None, False
        return features, True

    def calibrate(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Owlet_Precision", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Owlet_Precision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for pos in TARGET_POSITIONS:
            collected = 0
            burst = False
            while collected < 15:
                ret, frame = cap.read()
                if not ret: break
                
                feat, is_tracked = self.get_features(frame)
                ui = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 10, dtype=np.uint8)
                px, py = int(pos[0]*SCREEN_WIDTH), int(pos[1]*SCREEN_HEIGHT)
                s = int(60 * (1 + 0.2 * np.sin(time.time() * 6)))
                
                cv2.circle(ui, (px, py), s, (0, 200, 255), -1) 
                # Stealth HUD
                color = (30,30,30) if is_tracked else (0,0,255)
                cv2.rectangle(ui, (SCREEN_WIDTH-20, SCREEN_HEIGHT-20), (SCREEN_WIDTH, SCREEN_HEIGHT), color, -1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): burst = True
                elif key == ord('q'): return cap.release()

                if burst and is_tracked and feat:
                    self.raw_data[pos].append(feat)
                    collected += 1
                    cv2.circle(ui, (px, py), s, (200, 240, 255), -1) 

                cv2.imshow("Owlet_Precision", ui)
            print('\a') # Beep

        cap.release()
        cv2.destroyAllWindows()
        self.train()

    def train(self):
        X, y = [], []
        for pos, feats in self.raw_data.items():
            for f in feats:
                X.append(f)
                y.append([pos[0], pos[1]])
        
        if len(X) > 20:
            self.model_pipeline.fit(np.array(X), np.array(y))
            self.trained_model = self.model_pipeline
            self.run_inference()

    def run_inference(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Live_Owlet", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Live_Owlet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        sx, sy = 0.5, 0.5
        ALX, ALY = 0.15, 0.08 # Asymmetric Smoothing
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            feat, tracked = self.get_features(frame)
            ui = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 30, dtype=np.uint8)
            
            if tracked:
                pred = self.trained_model.predict([feat])[0]
                sx += ALX * (pred[0] - sx)
                sy += ALY * (pred[1] - sy)
                gx, gy = int(sx * SCREEN_WIDTH), int(sy * SCREEN_HEIGHT)
                cv2.circle(ui, (gx, gy), 25, (0, 255, 0), -1)
                cv2.circle(ui, (gx, gy), 35, (255, 255, 255), 2)
            
            cv2.imshow("Live_Owlet", ui)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()

if __name__ == "__main__":
    sys = OwletPrecisionSystem()
    sys.calibrate()
