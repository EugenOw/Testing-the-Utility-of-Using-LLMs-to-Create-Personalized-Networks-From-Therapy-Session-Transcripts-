import json
import os
import re

def load_master_dimensions(master_file_path):
    with open(master_file_path, 'r') as f:
        return json.load(f)

def build_process_dimension_map(master_data):
    process_map = {}
    for patient_sessions in master_data.values():
        for session in patient_sessions:
            for process_entry in session:
                session_id = process_entry.get("session_id")
                process_text = process_entry.get("process", "").strip()
                dimensions = process_entry.get("dimensions", [])
                if session_id and process_text:
                    key = (session_id, process_text)
                    if isinstance(dimensions, list):
                        process_map[key] = dimensions
                    else:
                        process_map[key] = [dimensions]
    return process_map

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    classified_processes = None
    if isinstance(data.get("clusters"), dict) and "classified_processes" in data["clusters"]:
        classified_processes = data["clusters"]["classified_processes"]
    elif "classified_processes" in data:
        classified_processes = data["classified_processes"]

    if classified_processes:
        transformed_clusters = []
        for theme_key, theme_data in classified_processes.items():
            cluster = {
                "name": theme_key,
                "title": theme_data.get("Title", theme_key),
                "Processes": theme_data.get("Processes", [])
            }
            transformed_clusters.append(cluster)
        return {"clusters": transformed_clusters}

    return {"clusters": []}  # fallback if neither format matched



def transform_with_dimensions(cluster_data, session_id, process_map):
    result = []
    for cluster in cluster_data.get("clusters", []):
        print(cluster)
        theme_key = cluster.get("name", "")
        print("Theme KEy", theme_key)
        title = cluster.get("title", theme_key)
        dimension_set = set()
        process_count = 0

        for process in cluster.get("Processes", []):
            process_text = process.get("Process", "").strip()
            key = (session_id, process_text)
            dimensions = process_map.get(key)

            if dimensions:
                dimension_set.update(dimensions)
                process_count += 1
            else:
                print(f"[Warning] No dimensions found for: {key}")

        if dimension_set:
            result.append({
                "input_processes": [theme_key],
                "dimensions": sorted(dimension_set),
                "num_processes": process_count,
                "title": title
            })
        else:
            print(f"[Info] No valid dimensions collected for theme: {theme_key}")



    return result


def normalize_session_id(filename_base):
    cleaned = filename_base.replace("cluster_", "").lower()
    match = re.match(r"p(\d+)_s(\d+)", cleaned)
    if match:
        patient_num = int(match.group(1))
        session_num = int(match.group(2))
        return f"P{patient_num:02d}_S{session_num:02d}"
    return filename_base.upper()  # fallback if pattern fails

def main(input_folder, output_folder, master_file_path):
    os.makedirs(output_folder, exist_ok=True)
    master_data = load_master_dimensions(master_file_path)
    process_map = build_process_dimension_map(master_data)

    print(f"Loaded {len(process_map)} process-to-dimensions mappings.")

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]

            session_id = normalize_session_id(base_filename)
            print(f"\nProcessing file: {input_file_path} as session {session_id}")

            cluster_data = read_sentences_from_file(input_file_path)
            print(cluster_data)
            output_data = transform_with_dimensions(cluster_data, session_id, process_map)

            output_file_path = os.path.join(output_folder, session_id + "_dimensions.json")
            with open(output_file_path, 'w') as out_file:
                json.dump(output_data, out_file, indent=2)

            print(f"Saved transformed data to {output_file_path}")


if __name__ == '__main__':
    main("output/P13", "process_dimensions", "session_id_pre_processed_modified_pbt_annotation_of_mini_session.json")
