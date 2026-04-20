"""
🐥 Baby Pitch Agency - Gaze Pipeline Tester
===========================================

This script tests the real-time eyetracking pipeline used in the 'Baby Pitch Agency' experiment.
It uses MediaPipe Face Mesh to detect landmarks and applies the same heuristic-based gaze
estimation logic found in the experimental frontend (running_version.html).

PREREQUISITES:
--------------
1. Ensure you have Python 3.8+ installed.
2. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # MacOS/Linux
   # venv\Scripts\activate   # Windows

3. Install dependencies:
   pip install opencv-python mediapipe numpy

RUNNING THE TEST:
-----------------
python test_gaze_pipeline.py

OUTPUT:
-------
The script will process the video at 'video/FatherReco_13-03-2026_12m_5437_HS_R1303.wmv'
and save the result to 'video/gaze_pipeline_test_output.mp4'.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys

def process_video(input_path, output_path):
    # Initialize MediaPipe Face Mesh with refined landmarks (required for Iris detection)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open Video
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties for output
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Read one frame to get actual dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    actual_height, actual_width = first_frame.shape[:2]
    
    # Use avc1 codec (H264) for better compatibility on Mac
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))
    
    if not out.isOpened():
        print(f"Warning: 'avc1' failed. Trying 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        return

    # Temporal Smoothing (Matching JS GAZE_SMOOTHING = 0.12)
    smoothed_gaze_x = actual_width / 2
    smoothed_gaze_y = actual_height / 2
    GAZE_SMOOTHING = 0.12

    print(f"Starting Pipeline (Binocular + Smoothing)...")
    print(f"Processing: {input_path}")
    print(f"Resolution: {actual_width}x{actual_height} | FPS: {fps}")

    frame_count = 0
    # Process the first frame we already read
    def process_frame(frame, current_smoothed_x, current_smoothed_y):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # --- BINOCULAR GAZE ESTIMATION ---
                # Left Eye: Iris 468, Inner 133, Outer 33, Top 159, Bottom 145
                # Right Eye: Iris 473, Inner 362, Outer 263, Top 386, Bottom 374
                
                # 1. Left Eye Ratios
                lx_ratio = (landmarks[468].x - landmarks[33].x) / (landmarks[133].x - landmarks[33].x)
                ly_ratio = (landmarks[468].y - landmarks[159].y) / (landmarks[145].y - landmarks[159].y)

                # 2. Right Eye Ratios
                rx_ratio = (landmarks[473].x - landmarks[263].x) / (landmarks[362].x - landmarks[263].x)
                ry_ratio = (landmarks[473].y - landmarks[386].y) / (landmarks[374].y - landmarks[386].y)

                # 3. Average Ratios (Binocular Fusion)
                avg_x_ratio = (lx_ratio + rx_ratio) / 2
                avg_y_ratio = (ly_ratio + ry_ratio) / 2

                # 4. Map to Screen (with experimental offsets)
                raw_gaze_x = ((avg_x_ratio - 0.35) / 0.3) * actual_width
                raw_gaze_y = ((avg_y_ratio - 0.4) / 0.2) * actual_height

                # 5. Temporal Smoothing (Exponential Moving Average)
                current_smoothed_x += GAZE_SMOOTHING * (raw_gaze_x - current_smoothed_x)
                current_smoothed_y += GAZE_SMOOTHING * (raw_gaze_y - current_smoothed_y)
                
                gaze_x = int(max(0, min(actual_width, current_smoothed_x)))
                gaze_y = int(max(0, min(actual_height, current_smoothed_y)))

                # --- VISUALIZATION ---
                # Subtle Mesh
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * actual_width), int(lm.y * actual_height)), 1, (0, 150, 0), -1)

                # Eye Highlights (Both Eyes)
                for idx in [468, 133, 33, 159, 145, 473, 362, 263, 386, 374]:
                    pt = landmarks[idx]
                    cv2.circle(frame, (int(pt.x * actual_width), int(pt.y * actual_height)), 5, (255, 0, 0), -1)

                # Smoothed Gaze Target
                cv2.circle(frame, (gaze_x, gaze_y), 25, (0, 0, 255), 3)
                cv2.circle(frame, (gaze_x, gaze_y), 8, (0, 0, 255), -1)
                
                cv2.putText(frame, f"SMOOTHED GAZE", (gaze_x + 30, gaze_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        return frame, current_smoothed_x, current_smoothed_y

    frame, smoothed_gaze_x, smoothed_gaze_y = process_frame(first_frame, smoothed_gaze_x, smoothed_gaze_y)
    out.write(frame)
    frame_count += 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, smoothed_gaze_x, smoothed_gaze_y = process_frame(frame, smoothed_gaze_x, smoothed_gaze_y)
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            sys.stdout.write(f"\rProcessed {frame_count} frames...")
            sys.stdout.flush()

    cap.release()
    out.release()
    print(f"\nSuccess! Processed video saved to: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "video/sample_input.mp4"
    OUTPUT_VIDEO = "video/gaze_pipeline_test_output.mp4"
    
    # Ensure video directory exists
    if not os.path.exists('video'):
        os.makedirs('video')
        
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
