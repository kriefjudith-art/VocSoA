import pandas as pd
import numpy as np
import os
import sys

def compute_trial_metrics(df):
    """
    Computes all advanced metrics for a single trial dataframe.
    """
    if df.empty:
        return None

    # 1. Basic Info
    p_id = df['Participant_number'].iloc[0]
    group = df['Assigned_group'].iloc[0]
    trial = df['Trial_number'].iloc[0]
    block = df['Block_number'].iloc[0]
    cond = df['Actual_condition'].iloc[0]
    phase = df['phase'].iloc[0]

    # 2. Pitch Stats
    vocal_df = df[df['is_vocalizing'].astype(str).str.lower() == 'true']
    pitches = vocal_df['pitch_hz'].replace(0, np.nan).dropna()
    
    pitch_mean = pitches.mean() if not pitches.empty else 0
    pitch_sd = pitches.std() if not pitches.empty else 0
    
    # 3. Timing
    total_time = df['Trial_elapsed_ms'].max()
    vocal_time = len(vocal_df) * (total_time / len(df)) if len(df) > 0 else 0
    
    # 4. Movement (Distance)
    df['dx'] = df['balloon_x'].diff().fillna(0)
    df['dy'] = df['balloon_y'].diff().fillna(0)
    total_dist = np.sqrt(df['dx']**2 + df['dy']**2).sum()

    # 5. Cloud Exploration & Reveals
    # Clouds visited (at least one frame in ROI)
    visited_rois = df[df['Gaze_ROI'] != "null"]['Gaze_ROI'].unique()
    num_visited = len(visited_rois)
    
    # Reveal stats
    # Get max reveal percentage reached for each cloud id
    reveal_stats = df[df['Cloud_reveal_id'] != "null"].groupby('Cloud_reveal_id')['Reveal_percentage'].max()
    num_revealed = (reveal_stats >= 50).sum()
    avg_erased_pct = reveal_stats.mean() if not reveal_stats.empty else 0
    
    # Latency to first visit
    gaze_events = df[df['Gaze_ROI'] != "null"]
    latency_visit = gaze_events['Trial_elapsed_ms'].min() if not gaze_events.empty else total_time
    
    # Time spent exploring (total time inside any ROI)
    exploration_time = len(gaze_events) * (total_time / len(df)) if len(df) > 0 else 0

    return {
        'P_ID': p_id,
        'Group': group,
        'Trial': trial,
        'Block': block,
        'Condition': cond,
        'Phase': phase,
        'Pitch_Mean': pitch_mean,
        'Pitch_SD': pitch_sd,
        'Total_Time_Sec': total_time / 1000,
        'Vocal_Time_Sec': vocal_time / 1000,
        'Distance_Pixels': total_dist,
        'Clouds_Visited': num_visited,
        'Clouds_Revealed': num_revealed,
        'Exploration_Time_Sec': exploration_time / 1000,
        'First_Visit_Latency_Sec': latency_visit / 1000,
        'Avg_Erased_Pct': avg_erased_pct
    }

def analyze_all_data(data_dir='data'):
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV logs found.")
        return

    results = []
    for file in csv_files:
        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(path)
            metrics = compute_trial_metrics(df)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    summary_df = pd.DataFrame(results)
    print("\n--- EXPERIMENT SUMMARY ---")
    print(summary_df.to_string(index=False))
    
    # Save to master summary
    summary_df.to_csv('experiment_summary_report.csv', index=False)
    print("\nMaster report saved to 'experiment_summary_report.csv'")

if __name__ == "__main__":
    analyze_all_data()
