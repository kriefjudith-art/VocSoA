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

    print(f"Starting Pipeline...")
    print(f"Processing: {input_path}")
    print(f"Resolution: {actual_width}x{actual_height} | FPS: {fps}")

    frame_count = 0
    # Process the first frame we already read
    def process_frame(frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # --- EXACT GAZE ESTIMATION LOGIC FROM EXPERIMENT ---
                li = landmarks[468] # Iris
                lin = landmarks[133] # Inner
                lo = landmarks[33]  # Outer
                lt = landmarks[159] # Top
                lb = landmarks[145] # Bottom

                x_ratio = (li.x - lo.x) / (lin.x - lo.x)
                y_ratio = (li.y - lt.y) / (lb.y - lt.y)

                gaze_x = int(((x_ratio - 0.35) / 0.3) * actual_width)
                gaze_y = int(((y_ratio - 0.4) / 0.2) * actual_height)

                gaze_x = max(0, min(actual_width, gaze_x))
                gaze_y = max(0, min(actual_height, gaze_y))

                # --- VISUALIZATION ---
                # 1. Draw Mesh Landmarks (Slightly larger Green dots)
                for lm in landmarks:
                    px = int(lm.x * actual_width)
                    py = int(lm.y * actual_height)
                    cv2.circle(frame, (px, py), 2, (0, 200, 0), -1)

                # 2. Highlight Gaze Source Points (Much larger Blue dots)
                for idx in [468, 133, 33, 159, 145]:
                    pt = landmarks[idx]
                    cv2.circle(frame, (int(pt.x * actual_width), int(pt.y * actual_height)), 8, (255, 0, 0), -1)

                cv2.circle(frame, (gaze_x, gaze_y), 20, (0, 0, 255), 3)
                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                
                cv2.putText(frame, f"GAZE: ({gaze_x}, {gaze_y})", (gaze_x + 25, gaze_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"X-Ratio: {x_ratio:.2f} | Y-Ratio: {y_ratio:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    frame = process_frame(first_frame)
    out.write(frame)
    frame_count += 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
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
