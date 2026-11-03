import json
import os
from collections import defaultdict, Counter
import random
import re

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_all_json_files(directory_path):
    json_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            full_path = os.path.join(directory_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_data.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading {filename}: {e}")

    return json_data



#gpt = read_sentences_from_file("chatgpt/P12_S13_gpt.json")


def majority_vote_from_json_list_old(json_list):
    total_files = len(json_list)
    aggregated = defaultdict(lambda: defaultdict(list))
    connection_votes = defaultdict(list)  # Store 1/0 votes for each pair_label

    for json_data in json_list:
        print(json_data)
        for rel in json_data.get("relationships", []):
            pair_label = rel["pair_label"]

            # Track connection values explicitly
            connection_value = rel.get("connection")
            if isinstance(connection_value, list) and connection_value:
                connection_votes[pair_label].append(connection_value[0])  # Should be 0 or 1

            # Handle both str and list cases for relationship_type
            relationship_type = rel.get("relationship_type")
            if isinstance(relationship_type, list):
                aggregated[pair_label]["relationship_type"].extend(relationship_type)
            elif isinstance(relationship_type, str):
                aggregated[pair_label]["relationship_type"].append(relationship_type)

            # Handle both str and list cases for strength_of_relationship
            strength = rel.get("strength_of_relationship")
            if isinstance(strength, list):
                aggregated[pair_label]["strength_of_relationship"].extend(strength)
            elif isinstance(strength, str):
                aggregated[pair_label]["strength_of_relationship"].append(strength)

            if rel.get("explanation"):
                aggregated[pair_label]["explanations"].append(rel["explanation"])

            aggregated[pair_label]["input_processes"] = rel.get("input_processes", [])

    final_output = []
    analysis_output = []

    for pair_label, values in aggregated.items():
        connections = connection_votes.get(pair_label, [])
        connected_votes = sum(1 for c in connections if c == 1)
        disconnected_votes = sum(1 for c in connections if c == 0)
        total_votes = connected_votes + disconnected_votes
        connected = connected_votes > total_votes / 2 if total_votes > 0 else False
        if connected:
            connected_value = [1]
        else:
            connected_value = [0]

        # Count relationship types and strength votes
        rel_type_counter = Counter(values["relationship_type"])
        strength_counter = Counter(values["strength_of_relationship"])

        # Fill missing keys with zero
        for k in ["excitatory", "inhibitory"]:
            if k not in rel_type_counter:
                rel_type_counter[k] = 0
        for k in ["strong", "moderate", "weak"]:
            if k not in strength_counter:
                strength_counter[k] = 0

        relationship_type = (
            rel_type_counter.most_common(1)[0][0]
            if rel_type_counter else None
        )
        if relationship_type == 'excitatory':
            relationship_type = ['excitatory']
        elif relationship_type == 'inhibitory':
            relationship_type = ['inhibitory']
        strength = (
            strength_counter.most_common(1)[0][0]
            if strength_counter else None
        )
        if strength == 'strong':
            strength = ['strong']
        elif strength == 'weak':
            strength = ['weak']
        elif strength == 'moderate':
            strength = ['moderate']
        explanation = (
            random.choice(values["explanations"]) if values["explanations"] else ""
        )
        input_processes = values.get("input_processes", [])

        final_output.append({
            "input_processes": input_processes,
            "connection": connected_value,
            "relationship_type": relationship_type,
            "strength_of_relationship": strength,
            "explanation": explanation,
            "pair_label": pair_label
        })

        analysis_output.append({
            "input_processes": input_processes,
            "pair_label": pair_label,
            "total_votes": total_votes,
            "connected_votes": connected_votes,
            "disconnected_votes": disconnected_votes,
            "connection": connected_value,
            "relationship_type_votes": dict(rel_type_counter),
            "strength_of_relationship_votes": dict(strength_counter),
            "most_common_relationship_type": relationship_type,
            "most_common_strength_of_relationship": strength,
            "sample_explanation": explanation,
            "total_files": total_files
        })

    return final_output, analysis_output



from collections import defaultdict, Counter
import random

def majority_vote_from_json_list(json_list):
    total_files = len(json_list)
    aggregated = defaultdict(lambda: defaultdict(list))
    connection_votes = defaultdict(list)  # Store 1/0 votes for each pair_label

    for json_data in json_list:
        for rel in json_data.get("relationships", []):
            pair_label = rel["pair_label"]

            # Track connection values explicitly
            connection_value = rel.get("connection")
            if isinstance(connection_value, list) and connection_value:
                connection_votes[pair_label].append(connection_value[0])  # Should be 0 or 1

            # Handle both str and list cases for relationship_type
            relationship_type = rel.get("relationship_type")
            if isinstance(relationship_type, list):
                aggregated[pair_label]["relationship_type"].extend(relationship_type)
            elif isinstance(relationship_type, str):
                aggregated[pair_label]["relationship_type"].append(relationship_type)

            # Handle both str and list cases for strength_of_relationship
            strength = rel.get("strength_of_relationship")
            if isinstance(strength, list):
                aggregated[pair_label]["strength_of_relationship"].extend(strength)
            elif isinstance(strength, str):
                aggregated[pair_label]["strength_of_relationship"].append(strength)

            if rel.get("explanation"):
                aggregated[pair_label]["explanations"].append(rel["explanation"])

            aggregated[pair_label]["input_processes"] = rel.get("input_processes", [])

    final_output = []
    analysis_output = []

    for pair_label, values in aggregated.items():
        connections = connection_votes.get(pair_label, [])
        connected_votes = sum(1 for c in connections if c == 1)
        disconnected_votes = sum(1 for c in connections if c == 0)
        total_votes = connected_votes + disconnected_votes
        connected = connected_votes > total_votes / 2 if total_votes > 0 else False
        connected_value = [1] if connected else [0]

        # Count relationship types and strength votes
        rel_type_counter = Counter(values["relationship_type"])
        strength_counter = Counter(values["strength_of_relationship"])

        # Fill missing keys with zero
        for k in ["excitatory", "inhibitory"]:
            if k not in rel_type_counter:
                rel_type_counter[k] = 0
        for k in ["strong", "moderate", "weak"]:
            if k not in strength_counter:
                strength_counter[k] = 0

        # Determine majority or override rule
        relationship_type = rel_type_counter.most_common(1)[0][0] if rel_type_counter else None
        relationship_type = [relationship_type] if relationship_type else None

        # Override logic: if "strong" appears even once, it's strong
        if "strong" in strength_counter and strength_counter["strong"] > 0:
            strength = ['strong']
        elif "moderate" in strength_counter and strength_counter["moderate"] > 0:
            strength = ['moderate']
        elif "weak" in strength_counter and strength_counter["weak"] > 0:
            strength = ['weak']
        else:
            strength = None

        explanation = (
            random.choice(values["explanations"]) if values["explanations"] else ""
        )
        input_processes = values.get("input_processes", [])

        final_output.append({
            "input_processes": input_processes,
            "connection": connected_value,
            "relationship_type": relationship_type,
            "strength_of_relationship": strength,
            "explanation": explanation,
            "pair_label": pair_label
        })

        analysis_output.append({
            "input_processes": input_processes,
            "pair_label": pair_label,
            "total_votes": total_votes,
            "connected_votes": connected_votes,
            "disconnected_votes": disconnected_votes,
            "connection": connected_value,
            "relationship_type_votes": dict(rel_type_counter),
            "strength_of_relationship_votes": dict(strength_counter),
            "most_common_relationship_type": relationship_type,
            "most_common_strength_of_relationship": strength,
            "sample_explanation": explanation,
            "total_files": total_files
        })

    return final_output, analysis_output


def majority_vote_with_stats_final(json_list):
    aggregated = defaultdict(lambda: defaultdict(list))
    per_pair_stats = {}
    global_relationship_counter = Counter()
    global_strength_counter = Counter()
    agreement_scores = []

    for json_data in json_list:
        for rel in json_data.get("relationships", []):
            pair_label = rel["pair_label"]
            aggregated[pair_label]["relationship_type"].extend(rel.get("relationship_type", []))
            aggregated[pair_label]["strength_of_relationship"].extend(rel.get("strength_of_relationship", []))
            if rel.get("explanation"):
                aggregated[pair_label]["explanations"].append(rel["explanation"])
            aggregated[pair_label]["input_processes"] = rel.get("input_processes", [])
            if "count" not in aggregated[pair_label]:
                aggregated[pair_label]["count"] = 0
            aggregated[pair_label]["count"] += 1

    final_output = {}
    max_pair_label = 0

    for pair_label, values in aggregated.items():
        relationship_counts = Counter(values["relationship_type"])
        strength_counts = Counter(values["strength_of_relationship"])
        num_votes = values.get("count", len(values["relationship_type"]))
        max_pair_label = max(max_pair_label, pair_label)

        # Majority voting
        relationship_type, rel_votes = relationship_counts.most_common(1)[0] if relationship_counts else (None, 0)
        strength, str_votes = strength_counts.most_common(1)[0] if strength_counts else (None, 0)
        explanation = random.choice(values["explanations"]) if values["explanations"] else ""
        input_processes = values.get("input_processes", [])

        final_output[pair_label] = {
            "input_processes": input_processes,
            "relationship_type": relationship_type,
            "strength_of_relationship": strength,
            "explanation": explanation,
            "pair_label": pair_label,
            "votes": {
                "relationship_type": dict(relationship_counts),
                "strength_of_relationship": dict(strength_counts),
                "num_jsons_with_connection": num_votes
            }
        }

        # Stats
        agreement_scores.append(rel_votes / num_votes if num_votes else 0)
        global_relationship_counter.update(relationship_counts)
        global_strength_counter.update(strength_counts)

        per_pair_stats[pair_label] = {
            "connection_found": True,
            "num_votes": num_votes,
            "relationship_type_counts": dict(relationship_counts),
            "strength_of_relationship_counts": dict(strength_counts),
            "agreement_rate": round(rel_votes / num_votes, 2) if num_votes else 0
        }

    # Detect missing labels (no connection found)
    missing_count = 0
    for lbl in range(max_pair_label + 1):  # if label range is numeric
        if lbl not in final_output:
            per_pair_stats[lbl] = {
                "connection_found": False,
                "num_votes": 0,
                "relationship_type_counts": {},
                "strength_of_relationship_counts": {},
                "agreement_rate": 0
            }
            missing_count += 1

    # Global summary
    total_labels = max_pair_label + 1
    global_summary = {
        "total_pair_labels": total_labels,
        "connection_found": total_labels - missing_count,
        "no_connection": missing_count,
        "relationship_type_counts": dict(global_relationship_counter),
        "strength_of_relationship_counts": dict(global_strength_counter),
        "average_agreement_rate": round(sum(agreement_scores) / len(agreement_scores), 3) if agreement_scores else 0
    }

    return final_output, per_pair_stats, global_summary



BASE_INPUT_DIR = "output_data/llm_output_temp"
BASE_OUTPUT_DIR = "output_data/finalized_linkage"

def is_valid_session_folder(name):
    return re.match(r"^P\d+_S\d+$", name) is not None

for participant_folder in os.listdir(BASE_INPUT_DIR):
    participant_path = os.path.join(BASE_INPUT_DIR, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    for session_folder in os.listdir(participant_path):
        if not is_valid_session_folder(session_folder):
            continue

        session_path = os.path.join(participant_path, session_folder)
        print(f"Processing: {session_path}")

        llm_file = load_all_json_files(session_path)
        if not llm_file:
            print(f"No valid JSON files found in {session_path}")
            continue

        final_json, analyze_json = majority_vote_from_json_list(llm_file)

        # Ensure the output directory exists
        output_session_dir = os.path.join(BASE_OUTPUT_DIR, participant_folder)
        os.makedirs(output_session_dir, exist_ok=True)

        final_json_path = os.path.join(output_session_dir, f"{session_folder}_final.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump({"relationships": final_json}, f, ensure_ascii=False, indent=2)
            print(f"Saved final output to: {final_json_path}")

        analyze_json_path = os.path.join(output_session_dir, f"{session_folder}_analyse.json")
        with open(analyze_json_path, "w", encoding="utf-8") as f:
            json.dump(analyze_json, f, ensure_ascii=False, indent=2)
            print(f"Saved analysis output to: {analyze_json_path}")


#is_valid_session_folder("P16")







#------------------------------------------------------------------------------------------------------

# TODO: final file and analyze is still wrong it need to start with {
#   "relationships": .... und es endted mit ], also nicht mal da richtig
#llm_file = load_all_json_files("output/pbt_examples/pbt_examples_llm/P13_S08")
#llm_file = load_all_json_files("output_data/llm_output_temp/P11/P11_S02")

#final_json, analyze_json = majority_vote_from_json_list(llm_file)
#final_json_dict = f"output/pbt_examples/P13_S08_final_llm.json"
#final_json_dict = f"output_data/finalized_linkage/P11/P11_S02_final.json"
#with open(final_json_dict, "w", encoding="utf-8") as f:
#    json.dump({"relationships": final_json}, f, ensure_ascii=False, indent=2)
#    print(f"Saved {len(final_json)} failed outputs to: {final_json_dict}")

#analyze_json_dict =  f"output_data/finalized_linkage/P11/P11_S02_analyse.json"
#with open(analyze_json_dict, "w", encoding="utf-8") as f:
#    json.dump(analyze_json, f, ensure_ascii=False, indent=2)
#    print(f"Saved {len(analyze_json)} failed outputs to: {analyze_json_dict}")




# Next Step comparing the final output with the output from chatgpt 40 mini and make therefore the final decision




