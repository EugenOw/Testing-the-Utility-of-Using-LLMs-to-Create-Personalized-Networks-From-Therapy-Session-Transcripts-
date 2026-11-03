import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_processes_allow_duplicates(data):
    process_entries = []
    skipped = 0
    for key, entries in data.items():
        for group in entries:
            for item in group:
                proc = item.get("process")
                if not proc:
                    skipped += 1
                    continue
                dimensions = item.get("dimensions", [])
                process_entries.append((proc, dimensions))
    print(f"Total processes collected: {len(process_entries)}, Skipped: {skipped}")
    return process_entries

def collect_new_processes_set(data):
    new_processes = set()
    for entries in data.values():
        for group in entries:
            for item in group:
                proc = item.get("process")
                if proc:
                    new_processes.add(proc)
    print(f"Total unique processes in new data: {len(new_processes)}")
    return new_processes

def compare_processes_all(old_data, new_data):
    old_entries = collect_processes_allow_duplicates(old_data)
    new_processes_set = collect_new_processes_set(new_data)

    result = []
    for proc, dims in old_entries:
        result.append({
            "utterance": proc,
            "process": proc in new_processes_set,
            "dimensions": dims if proc in new_processes_set else []
        })
    print(f"Final output will have {len(result)} entries.")
    return result

def main(old_file_path, new_file_path, output_path):
    old_data = load_json(old_file_path)
    new_data = load_json(new_file_path)

    result = compare_processes_all(old_data, new_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
old = "modified_pbt_annotation_of_mini_session (1).json"
new = "pre_processed_modified_pbt_annotation_of_mini_session.json"
main(old, new, "data/processes.json")
