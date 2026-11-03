import os
import json
import xml.etree.ElementTree as ET
from collections import Counter


def extract_speaker_and_text(ttml_file):
    tree = ET.parse(ttml_file)
    root = tree.getroot()

    namespaces = {
        "tt": "http://www.w3.org/ns/ttml",
        "ttm": "http://www.w3.org/ns/ttml#metadata"
    }

    dialogues = []
    for p in root.findall(".//tt:body/tt:div/tt:p", namespaces):
        speaker = p.get("{http://www.w3.org/ns/ttml#metadata}agent")
        text = "".join(p.itertext()).strip()
        if speaker and text:
            dialogues.append({"speaker": speaker, "text": text})

    return dialogues


def calculate_semantic_cluster_frequencies(session):
    return Counter(process["semantic_cluster"] for process in session)


def add_tokens_to_utterances(dialogues, session, cluster_counts, token_start="[process dimension=",
                             token_middle=", freq=", token_end=r"[/process]"):
    labeled_clusters = set()
    for process in session:
        process_text = process["process"]
        cluster = process["semantic_cluster"]
        dimension = process["dimensions"]
        frequency = cluster_counts[cluster]
        if cluster not in labeled_clusters:
            for dialogue in dialogues:
                if process_text in dialogue["text"]:
                    dialogue["text"] = dialogue["text"].replace(
                        process_text,
                        f"{token_start}{dimension}{token_middle}{frequency}]{process_text}{token_end}"
                    )
            labeled_clusters.add(cluster)
    return dialogues


def process_ttml_with_labels(ttml_file, session):
    dialogues = extract_speaker_and_text(ttml_file)
    cluster_counts = calculate_semantic_cluster_frequencies(session)
    return add_tokens_to_utterances(dialogues, session, cluster_counts)


def save_dialogues_to_text(dialogues, session_label, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{session_label}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for dialogue in dialogues:
            f.write(f"{dialogue['speaker']}: {dialogue['text']}\n")


def process_all_ttml_in_folder(ttml_folder, session_lookup, output_folder):
    for file in os.listdir(ttml_folder):
        if file.endswith(".ttml"):
            filepath = os.path.join(ttml_folder, file)
            session_label = os.path.splitext(file)[0]  # e.g., P01_S01_working_phase
            session = session_lookup.get(session_label, [])
            print(f"Processing: {session_label}")
            dialogues = process_ttml_with_labels(filepath, session)
            save_dialogues_to_text(dialogues, session_label, output_folder)


ttml_folder = "slurm_clustering_clinicly/mini_sessions"
output_folder = "output_sessions_txt"



# Load the full session data just once
with open("slurm_clustering_clinicly/data/session_id_pre_processed_modified_pbt_annotation_of_mini_session.json", "r", encoding="utf-8") as f:
    all_sessions = json.load(f)

# Build lookup by session label
session_lookup = {}
for patient_id, sessions in all_sessions.items():
    for session in sessions:
        if session:
            process_id = session[0]["process_id"][:-2]
            patient_num = patient_id[1:].zfill(2)
            session_num = process_id.split('_')[1].zfill(2)
            session_label = f"P{patient_num}_S{session_num}_working_phase"
            session_lookup[session_label] = session

process_all_ttml_in_folder(ttml_folder, session_lookup, output_folder)

