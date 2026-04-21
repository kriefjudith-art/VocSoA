import os
import json
from flask import Flask, request, send_from_directory, jsonify
from datetime import datetime

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAJECTORIES_DIR = os.path.join(BASE_DIR, 'trajectories')
SOUNDS_DIR = os.path.join(BASE_DIR, 'VocSoA', 'sounds')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)
os.makedirs(SOUNDS_DIR, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'running_version.html')

@app.route('/webgazer')
def webgazer():
    return send_from_directory(BASE_DIR, 'webgazer_experiment.html')

@app.route('/webgazer_v2')
def webgazer_v2():
    return send_from_directory(BASE_DIR, 'webgazer_v2.html')

@app.route('/webgazer_v3')
def webgazer_v3():
    return send_from_directory(BASE_DIR, 'webgazer_v3.html')

@app.route('/precision')
def precision():
    return send_from_directory(BASE_DIR, 'precision_js_tracker.html')

@app.route('/VocSoA/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory(SOUNDS_DIR, filename)

@app.route('/api/save_trajectory', methods=['POST'])
def save_trajectory():
    data = request.json
    p_id = data.get('participant_id')
    trial_n = data.get('trial_number')
    
    filename = f"participant_{p_id}_trial_{trial_n}.json"
    filepath = os.path.join(TRAJECTORIES_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return jsonify({"status": "success", "message": f"Saved {filename}"})

@app.route('/api/get_yoked_trajectory', methods=['GET'])
def get_yoked_trajectory():
    # Example logic: Find the most recent C1 trajectory from a different participant
    # If p_id and trial_n are provided, we can be more specific
    p_id = request.args.get('participant_id')
    trial_n = request.args.get('trial_number')
    
    # List all available trajectories
    files = [f for f in os.listdir(TRAJECTORIES_DIR) if f.endswith('.json')]
    if not files:
        return jsonify({"status": "error", "message": "No trajectories available"}), 404
    
    # Filter out current participant if possible
    other_files = [f for f in files if f"participant_{p_id}_" not in f]
    target_files = other_files if other_files else files
    
    # Find matching trial number if possible, else pick the first
    matching_trial = [f for f in target_files if f"_trial_{trial_n}.json" in f]
    selected_file = matching_trial[0] if matching_trial else target_files[0]
    
    filepath = os.path.join(TRAJECTORIES_DIR, selected_file)
    with open(filepath, 'r') as f:
        traj_data = json.load(f)
        
    return jsonify({
        "status": "success", 
        "data": traj_data, 
        "source_p_id": traj_data.get('participant_id'),
        "source_trial_n": traj_data.get('trial_number')
    })

@app.route('/api/save_data', methods=['POST'])
def save_data():
    # Used for CSV logs, videos (WebM), and audio (WAV)
    p_id = request.form.get('participant_id')
    file_type = request.form.get('file_type', 'data') # e.g., 'csv_G1', 'webm_webcam_G2'
    trial_n = request.form.get('trial_number', 'full')
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
        
    file = request.files['file']
    
    # Extract extension (e.g., 'csv_G1' -> 'csv')
    extension = file_type.split('_')[0] if file_type else 'data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Construct filename: participant_001_trial_1_20260420_120000_csv_G1.csv
    filename = f"participant_{p_id}_trial_{trial_n}_{timestamp}_{file_type}.{extension}"
    filepath = os.path.join(DATA_DIR, filename)
    
    file.save(filepath)
    return jsonify({"status": "success", "message": f"Saved {filename}"})

if __name__ == '__main__':
    print(f"Server starting on http://localhost:5001")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Trajectories Dir: {TRAJECTORIES_DIR}")
    app.run(debug=True, port=5001)
