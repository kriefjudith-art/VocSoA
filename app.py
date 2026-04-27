import os
import json
import re
from flask import Flask, request, send_from_directory, jsonify
from datetime import datetime

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAJECTORIES_DIR = os.path.join(BASE_DIR, 'trajectories')
SOUNDS_DIR = os.path.join(BASE_DIR, 'assets', 'sounds')
VOCSOA_SOUNDS_DIR = os.path.join(BASE_DIR, 'assets', 'VocSoA', 'sounds')
IMAGES_DIR = os.path.join(BASE_DIR, 'assets', 'images')

# Ensure directories exist
for d in [DATA_DIR, TRAJECTORIES_DIR, SOUNDS_DIR, VOCSOA_SOUNDS_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

def get_next_p_id():
    """Scans data/ to find the highest participant ID and returns next."""
    files = os.listdir(DATA_DIR)
    max_id = 0
    for f in files:
        match = re.search(r'participant_(\d+)', f)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1

def determine_group(p_id):
    """
    P1: Seed (C1-C1-C1)
    P2, P4, ...: G1 (C1-C2-C1)
    P3, P5, ...: G2 (C2-C1-C2)
    """
    if p_id == 1:
        return "SEED"
    return "G1" if p_id % 2 == 0 else "G2"

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'running_version.html')

@app.route('/api/init_participant', methods=['GET'])
def init_participant():
    p_id = get_next_p_id()
    group = determine_group(p_id)
    
    # Check for required trajectories (P1 C1 trials)
    files = [f for f in os.listdir(TRAJECTORIES_DIR) if f.endswith('.json')]
    p1_c1_count = len([f for f in files if "participant_1_" in f])
    
    return jsonify({
        "participant_id": p_id,
        "group": group,
        "yoked_available": p1_c1_count >= 10 # Assuming 10 trials per block
    })

@app.route('/api/save_trajectory', methods=['POST'])
def save_trajectory():
    data = request.json
    p_id = data.get('participant_id')
    trial_n = data.get('trial_number')
    
    # Filename: participant_001_trial_1.json
    filename = f"participant_{p_id}_trial_{trial_n}.json"
    filepath = os.path.join(TRAJECTORIES_DIR, filename)
    
    # Add server-side timestamp
    data['recorded_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return jsonify({"status": "success", "message": f"Saved {filename}"})

@app.route('/api/get_yoked_trajectory', methods=['GET'])
def get_yoked_trajectory():
    trial_n = request.args.get('trial_number')
    
    # Logic: Always pull from Participant 1 (Seed)
    filename = f"participant_1_trial_{trial_n}.json"
    filepath = os.path.join(TRAJECTORIES_DIR, filename)
    
    if not os.path.exists(filepath):
        # Fallback to any available trial if specific number missing
        files = [f for f in os.listdir(TRAJECTORIES_DIR) if f.endswith('.json') and "participant_1_" in f]
        if not files:
            return jsonify({"status": "error", "message": "No yoked data found"}), 404
        filepath = os.path.join(TRAJECTORIES_DIR, files[0])

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
    p_id = request.form.get('participant_id')
    file_type = request.form.get('file_type', 'data')
    trial_n = request.form.get('trial_number', 'full')
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
        
    file = request.files['file']
    extension = file_type.split('_')[0] if file_type else 'data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"participant_{p_id}_trial_{trial_n}_{timestamp}_{file_type}.{extension}"
    filepath = os.path.join(DATA_DIR, filename)
    
    file.save(filepath)
    return jsonify({"status": "success", "message": f"Saved {filename}"})

# Asset routes
@app.route('/webgazer')
def webgazer():
    return send_from_directory(os.path.join(BASE_DIR, 'demos', 'web'), 'webgazer_experiment.html')

@app.route('/precision')
def precision():
    return send_from_directory(os.path.join(BASE_DIR, 'demos', 'web'), 'precision_js_tracker.html')

@app.route('/sounds/<path:filename>')
def serve_general_sounds(filename):
    return send_from_directory(SOUNDS_DIR, filename)

@app.route('/VocSoA/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory(VOCSOA_SOUNDS_DIR, filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    print(f"Server starting on http://localhost:5001")
    app.run(debug=True, port=5001)
