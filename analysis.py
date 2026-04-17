import pandas as pd
import numpy as np
import os
import sys

def analyze_experiment(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"--- Comprehensive Analysis for {os.path.basename(file_path)} ---")

    # 1. Pitch Metrics
    vocal_df = df[(df['is_vocalizing'] == True) | (df['is_vocalizing'] == "True")]
    vocal_df = vocal_df[vocal_df['pitch_hz'] > 0]
    
    if not vocal_df.empty:
        pitches = vocal_df['pitch_hz']
        print("\nPitch Distribution (Hz):")
        print(f"  Mean: {pitches.mean():.2f} | SD: {pitches.std():.2f}")
        print(f"  Min: {pitches.min():.2f} | Max: {pitches.max():.2f}")
        print(f"  Median: {pitches.median():.2f}")
    
    # 2. Timing
    total_duration = df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]
    vocal_count = len(vocal_df)
    total_count = len(df)
    vocal_percent = (vocal_count / total_count) * 100
    print(f"\nTiming:")
    print(f"  Total Trial Duration: {total_duration/1000:.2f}s")
    print(f"  Vocalization Engagement: {vocal_percent:.1f}%")

    # 3. Distance Traveled
    df['dx'] = df['balloon_x'].diff().fillna(0)
    df['dy'] = df['balloon_y'].diff().fillna(0)
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    total_dist = df['dist'].sum()
    print(f"\nMovement:")
    print(f"  Total Distance Traveled: {total_dist:.2f} pixels")

    # 4. Gaze & Cloud Visits
    # Filter for valid ROIs (0-3)
    gaze_events = df[df['gaze_roi'].notnull()]
    if not gaze_events.empty:
        unique_clouds = gaze_events['gaze_roi'].unique()
        print(f"\nExploration:")
        print(f"  Number of unique clouds visited: {len(unique_clouds)}")
        
        # Latency to first visit
        first_visit_time = gaze_events['timestamp_ms'].iloc[0] - df['timestamp_ms'].iloc[0]
        print(f"  Latency to first cloud visit: {first_visit_time/1000:.2f}s")
    else:
        print("\nNo cloud visits detected via gaze.")

    # 5. Condition Info
    condition = df['condition'].iloc[0]
    phase = df['phase'].iloc[0]
    print(f"\nContext:")
    print(f"  Current Phase: {phase}")
    print(f"  Condition: {condition}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no file provided, look in data/
        data_dir = 'data'
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                for f in files:
                    analyze_experiment(f)
                    print("-" * 40)
            else:
                print("No CSV files found in data/")
        else:
            print("Usage: python analysis.py <path_to_csv>")
    else:
        analyze_experiment(sys.argv[1])
