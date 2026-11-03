import json
import random

with open("data/session_id_pre_processed_modified_pbt_annotation_of_mini_session.json") as f:
    data = json.load(f)



already_used_session = ["P11_S03", "P11_S04", "P11_S06", "P11_S09", "P11_S12", "P11_S14", "P11_S15",
                        "P12_S01", "P12_S04", "P12_S06", "P12_S08", "P12_S10", "P12_S11", "P12_S12", "P12_S13", "P12_S14", "P12_S15",
                        "P13_S02", "P13_S04", "P13_S10", "P13_S13" ,"P13_S15", "P13_S16"
                        "P14_S03", "P14_S04", "P14_S02", "P14_S11", "P14_S12" ,"P14_S14",
                        "P15_S03", "P15_S02", "P15_S05" ,"P15_S06",
                        "P16_S01", "P16_S06", "P16_S09", "P16_S10"]


def select_random_sessions(data, num_sessions, blacklist, output_path):
    """
    Randomly selects unique sessions from the JSON output, excluding blacklisted session_ids,
    and saves the result to a new JSON file.

    Args:
        data (dict): Original JSON output.
        num_sessions (int): Number of session_ids to select.
        blacklist (list): List of session_ids to exclude.
        output_path (str): File path to save the selected session output.
    """
    # Flatten the structure to map session_id -> list of process objects
    session_map = {}

    for session_key, session_blocks in data.items():
        for block_index, block in enumerate(session_blocks):
            session_id = f"{session_key}_S{block_index + 1:02}"
            if session_id not in blacklist:
                if session_id not in session_map:
                    session_map[session_id] = []
                session_map[session_id].extend(block)

    # Get available session_ids (excluding blacklist)
    available_session_ids = list(session_map.keys())

    if num_sessions > len(available_session_ids):
        raise ValueError("Not enough sessions available after applying blacklist.")

    # Randomly select session_ids
    selected_session_ids = random.sample(available_session_ids, num_sessions)

    # Build the filtered output
    selected_data = {}
    for session_id in selected_session_ids:
        key, block_id = session_id.split("_S")
        if key not in selected_data:
            selected_data[key] = []
        # Add a new block to maintain format
        block = [p for p in session_map[session_id]]
        selected_data[key].append(block)

    # Write to output file
    with open(output_path, "w") as f:
        json.dump(selected_data, f, indent=4)

    print(f"Saved {num_sessions} selected sessions to: {output_path}")

# Print selected session IDs
    print("Selected session_ids:")
    for sid in selected_session_ids:
        print(f" - {sid}")

select_random_sessions(data, num_sessions=43, blacklist=already_used_session, output_path="data/Third_Phase_Clustering_2.json")