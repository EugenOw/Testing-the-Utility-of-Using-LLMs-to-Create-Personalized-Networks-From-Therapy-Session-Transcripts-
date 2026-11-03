import os
import json
import shutil

# Paths to input files
EVALUATOR_ASSIGNMENT_FILE = 'evaluator_assignment.json'
SESSION_EVALUATION_FILE = 'session_evaluation.json'

# Network source directories
OUR_NETWORKS_DIR = 'finalized_networks_our'
BASELINE_NETWORKS_DIR = 'finalized_networks_baseline'

# Output base directory
OUTPUT_DIR = 'Evaluator'

# Log file for missing files
MISSING_LOG = 'missing_networks.txt'

# Load JSON data
with open(EVALUATOR_ASSIGNMENT_FILE, 'r') as f:
    evaluator_assignments = json.load(f)

with open(SESSION_EVALUATION_FILE, 'r') as f:
    session_evaluations = json.load(f)

# Convert session_evaluation list into a dictionary for quick access
session_eval_dict = {item['Session']: item for item in session_evaluations}

def get_network_path(session_id, source):
    patient_id = session_id.split('_')[0]  # e.g., P11 from P11_S01
    filename = f"{session_id}_network.html"
    if source == "Network A":
        return os.path.join(BASELINE_NETWORKS_DIR, patient_id, filename)
    elif source == "Network B":
        return os.path.join(OUR_NETWORKS_DIR, patient_id, filename)
    else:
        raise ValueError(f"Unknown network source: {source}")

# Open log file
missing_log_entries = []

# Main logic
for assignment in evaluator_assignments:
    evaluator = assignment['Evaluator']
    session = assignment['Session']

    if session not in session_eval_dict:
        missing_log_entries.append(f"{session}: session not found in session_evaluation.json")
        continue

    eval_info = session_eval_dict[session]
    net_a_source = eval_info['Baseline']
    net_b_source = eval_info['Majority Vote']

    net_a_path = get_network_path(session, net_a_source)
    net_b_path = get_network_path(session, net_b_source)

    output_session_dir = os.path.join(OUTPUT_DIR, evaluator, session)
    os.makedirs(output_session_dir, exist_ok=True)

    # Track missing files
    missing = False

    if os.path.exists(net_a_path):
        shutil.copy(net_a_path, os.path.join(output_session_dir, "Network_A.html"))
    else:
        missing_log_entries.append(f"{session}: Missing Network_A from {net_a_path}")
        missing = True

    if os.path.exists(net_b_path):
        shutil.copy(net_b_path, os.path.join(output_session_dir, "Network_B.html"))
    else:
        missing_log_entries.append(f"{session}: Missing Network_B from {net_b_path}")
        missing = True

    if not missing:
        print(f"Prepared {session} for {evaluator}")

# Write missing entries to log
if missing_log_entries:
    with open(MISSING_LOG, 'w') as f:
        for entry in missing_log_entries:
            f.write(entry + "\n")
    print(f"Missing networks logged in {MISSING_LOG}")
else:
    print("All network files were found and processed.")
