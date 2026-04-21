import cv2
import mediapipe as mp
import numpy as np
import os

def process_baby_video(input_path, output_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for processing (mirror effect usually used in webcams)
        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # --- GAZE ESTIMATION (Matching JS Logic) ---
                # Left Eye Iris: 468. Top: 159, Bottom: 145, Inner: 133, Outer: 33
                left_iris = landmarks[468]
                left_inner = landmarks[133]
                left_outer = landmarks[33]
                left_top = landmarks[159]
                left_bottom = landmarks[145]

                # Ratios
                x_ratio = (left_iris.x - left_outer.x) / (left_inner.x - left_outer.x)
                y_ratio = (left_iris.y - left_top.y) / (left_bottom.y - left_top.y)

                # Map to screen coordinates (1:1 with frame for visualization)
                # Sensitivity factors from JS: (ratio - offset) / range
                # Here we show it relative to the frame size
                gaze_x = int(((x_ratio - 0.35) / 0.3) * width)
                gaze_y = int(((y_ratio - 0.4) / 0.2) * height)

                # Clamp values to frame
                gaze_x = max(0, min(width, gaze_x))
                gaze_y = max(0, min(height, gaze_y))

                # --- DRAWING ---
                # 1. Draw all landmarks in green
                for lm in landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

                # 2. Highlight the Iris and Eyelids in blue
                for idx in [468, 133, 33, 159, 145]:
                    px = int(landmarks[idx].x * width)
                    py = int(landmarks[idx].y * height)
                    cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)

                # 3. Draw the PREDICTED GAZE DOT in RED
                cv2.circle(frame, (gaze_x, gaze_y), 15, (0, 0, 255), -1)
                cv2.putText(frame, "PREDICTED GAZE", (gaze_x + 20, gaze_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write frame
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Finished! Processed video saved to: {output_path}")

if __name__ == "__main__":
    input_vid = "video/FatherReco_13-03-2026_12m_5437_HS_R1303.wmv"
    output_vid = "video/processed_baby_video.mp4"
    
    if not os.path.exists('video'):
        os.makedirs('video')
        
    process_baby_video(input_vid, output_vid)
