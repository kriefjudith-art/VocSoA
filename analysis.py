import pandas as pd
import numpy as np
import os
import sys

def analyze_experiment(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"--- Analysis for {os.path.basename(file_path)} ---")

    # 1. Pitch Utterances
    # Define an utterance as a continuous block of is_vocalizing == 1
    df['vocal_change'] = df['is_vocalizing'].diff().fillna(0)
    utterance_starts = df[df['vocal_change'] == 1].index
    utterance_ends = df[df['vocal_change'] == -1].index

    # Handle cases where it starts/ends vocalizing
    if df['is_vocalizing'].iloc[0] == 1:
        utterance_starts = utterance_starts.insert(0, 0)
    if df['is_vocalizing'].iloc[-1] == 1:
        utterance_ends = utterance_ends.append(pd.Index([len(df)-1]))

    num_utterances = len(utterance_starts)
    print(f"Number of pitch utterances: {num_utterances}")

    # 2. Pitch Distribution
    # Only consider samples where is_vocalizing is True and pitch > 0
    vocal_df = df[(df['is_vocalizing'] == 1) & (df['pitch_hz'] > 0)]
    
    if not vocal_df.empty:
        pitches = vocal_df['pitch_hz']
        stats = {
            "Mean": pitches.mean(),
            "SD": pitches.std(),
            "Min": pitches.min(),
            "Max": pitches.max(),
            "25th Percentile": pitches.quantile(0.25),
            "50th (Median)": pitches.quantile(0.50),
            "75th Percentile": pitches.quantile(0.75)
        }
        print("\nPitch Distribution (Hz):")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}")
    else:
        print("\nNo vocalization data found for pitch distribution.")

    # 3. Total Timing: Vocalization vs Silence
    # Assuming timestamp_ms is accurate
    total_duration_ms = df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]
    
    # Calculate duration of each sample (diff between timestamps)
    df['duration'] = df['timestamp_ms'].diff().fillna(0)
    vocal_duration = df[df['is_vocalizing'] == 1]['duration'].sum()
    silence_duration = df[df['is_vocalizing'] == 0]['duration'].sum()

    print(f"\nTiming:")
    print(f"  Total Duration: {total_duration_ms/1000:.2f}s")
    print(f"  Total Vocalization: {vocal_duration/1000:.2f}s ({(vocal_duration/total_duration_ms)*100:.1f}%)")
    print(f"  Total Silence: {silence_duration/1000:.2f}s ({(silence_duration/total_duration_ms)*100:.1f}%)")

    # 4. Balloons Popped & Latency
    # ballon_popped_id is "False" until a pop occurs
    pop_indices = df[df['ballon_popped_id'] != "False"].index
    popped_balloons = df.loc[pop_indices, 'ballon_popped_id'].unique()
    
    print(f"\nBalloons:")
    print(f"  Number of balloons popped: {len(popped_balloons)}")
    
    if len(pop_indices) > 0:
        print("  Pop Latencies (from start of experiment):")
        for idx in pop_indices:
            bid = df.loc[idx, 'ballon_popped_id']
            time = df.loc[idx, 'timestamp_ms']
            print(f"    Balloon {bid}: {time/1000:.2f}s")
    else:
        print("  No balloons were popped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <path_to_csv>")
    else:
        analyze_experiment(sys.argv[1])
