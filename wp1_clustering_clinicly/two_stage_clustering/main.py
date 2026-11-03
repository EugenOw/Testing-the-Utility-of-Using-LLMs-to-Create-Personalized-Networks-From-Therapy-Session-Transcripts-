import time
from vllm import LLM, SamplingParams
from label_generation import prompt_construct_generate_labels
from given_label_classification import given_label_classification
import json
import os
import ast
import re
import ast
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
from sentence_transformers import util


def load_transcripts(transcript_path):
    with open(transcript_path, 'r') as file:
        data = json.load(file)
    return data

# Contains the Few Shot Examples we provide the model as prompt. These should not be used as Theme Lables for Clustering
black_list = ["Self confidence tied to competence in work role", "Professional Networking and Collaboration", "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others",
              "Emotional regulation and self-compassion", "Evaluating thoughts to identify any cognitive distortions", "Breathing and Behavioral Patterns" "Assessing impact of reframed/alternate thought after identifying cognitive distortion",
              "Therapy and Reflection", "Decision Making and Consequences", "Recognizing the role of thoughts and behaviors in emotional experiences",  "Reflection and Understanding", "Emotional Intimacy and Relationships",
              "Difficulty with focus and concentration due to trauma and ADHD-like symptoms", "Tension between value of achievement and family obligations", "Exploring the Impact of Surprises on Relationships", "Self confidence tied to competence in work role.", "Professional Networking and Collaboration.", "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others.",
              "Emotional regulation and self-compassion.", "Evaluating thoughts to identify any cognitive distortions.", "Breathing and Behavioral Patterns." "Assessing impact of reframed/alternate thought after identifying cognitive distortion.",
              "Therapy and Reflection.", "Decision Making and Consequences.", "Recognizing the role of thoughts and behaviors in emotional experiences.",  "Reflection and Understanding.", "Emotional Intimacy and Relationships.",
              "Difficulty with focus and concentration due to trauma and ADHD-like symptoms.", "Tension between value of achievement and family obligations.", "Exploring the Impact of Surprises on Relationships." ]

# Helper functions for semantic themes clustering


# --- Semantic merge config ---
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEMANTIC_SIM_THRESHOLD = 0.75      # cosine similarity threshold (0..1)
SEMANTIC_MIN_COMMUNITY = 1         # min items per community

SEMANTIC_TXT_PATH = Path(__file__).resolve().parent / "semantic_groups_all.txt"
SEMANTIC_JSON_BASE = Path(__file__).resolve().parent / "merged_json"

# Initialize embedding model ONCE (separate from your vLLM)
_semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

def normalize_session_in_filename(name: str) -> str:
    """
    e.g. cluster_p11_s1.json -> cluster_p11_s01.json (or S1 -> S01).
    Leaves already padded forms unchanged.
    """
    m = re.match(r"^(.*[_-][sS])(\d+)(\.json)$", name)
    if not m:
        return name
    prefix, digits, suffix = m.groups()
    pad = digits if len(digits) >= 2 else digits.zfill(2)
    return f"{prefix}{pad}{suffix}"

def merge_from_texts(texts: List[str], labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Build a theme-shaped dict from plain strings so downstream stays the same.
    Picks a random representative per cluster and merges 'Processes' (empty here).
    """
    themes = [(f"Label {i+1}", {"Theme": t, "Processes": []}) for i, t in enumerate(texts)]
    return merge_by_semantics_random(themes, labels, None)


def dedup_list(items: List[Any]) -> List[Any]:
    """Deduplicate while being robust to dicts/lists."""
    seen = set()
    out = []
    for x in items:
        if isinstance(x, (dict, list)):
            key = json.dumps(x, sort_keys=True, ensure_ascii=False)
        else:
            key = f"{type(x).__name__}:{x}"
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def encode(texts: List[str]):
    # returns L2-normalized torch.Tensor for cosine similarity
    with torch.no_grad():
        emb = _semantic_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True  # important for cosine thresholding
        )
    return emb  # shape: [N, D] torch.Tensor


def cluster(emb) -> np.ndarray:
    """
    Fast clustering: builds communities via cosine similarity thresholding.
    Returns an integer label per item (0..K-1). Items not in any community
    get their own singleton cluster so downstream stays identical.
    """
    # Handle trivial cases
    n = emb.shape[0] if hasattr(emb, "shape") else len(emb)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0], dtype=int)

    # Find communities (list of lists of indices)
    communities = util.community_detection(
        embeddings=emb,
        min_community_size=SEMANTIC_MIN_COMMUNITY,
        threshold=SEMANTIC_SIM_THRESHOLD,
    )

    labels = np.full(n, -1, dtype=int)
    next_lab = 0

    # Assign labels to discovered communities
    for comm in communities:
        for idx in comm:
            labels[idx] = next_lab
        next_lab += 1

    # Any points not assigned -> make them their own singleton clusters
    for idx in range(n):
        if labels[idx] == -1:
            labels[idx] = next_lab
            next_lab += 1

    return labels


def coerce_to_label_list(classified_labels_any: Any) -> List[str]:
    # dict with 'labels'
    if isinstance(classified_labels_any, dict) and "labels" in classified_labels_any:
        items = classified_labels_any["labels"]
        return [x.strip() for x in items if isinstance(x, str) and x.strip()]

    # already a list
    if isinstance(classified_labels_any, list):
        return [x.strip() for x in classified_labels_any if isinstance(x, str) and x.strip()]

    # maybe a JSON-encoded string
    if isinstance(classified_labels_any, str):
        try:
            parsed = json.loads(classified_labels_any)
            if isinstance(parsed, list):
                return [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
            if isinstance(parsed, dict) and "labels" in parsed:
                items = parsed["labels"]
                return [x.strip() for x in items if isinstance(x, str) and x.strip()]
        except json.JSONDecodeError:
            # treat it as a single label if it's a non-empty string
            s = classified_labels_any.strip()
            return [s] if s else []

    return []


def merge_by_semantics_random(themes: List[Tuple[str, Dict[str, Any]]], labels: np.ndarray, emb: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    themes: list of (theme_key, payload). payload expected to have "Theme", "Processes".
    Randomly choose a representative theme text within each semantic cluster.
    Merge processes; MergedFrom = list of theme texts used.
    Renumber output keys as "Theme 1", "Theme 2", ...
    """
    from collections import defaultdict
    by_label = defaultdict(list)  # lab -> list of indices
    for idx, lab in enumerate(labels):
        by_label[int(lab)].append(idx)

    merged = {}
    for new_idx, lab in enumerate(sorted(by_label.keys()), start=1):
        idxs = by_label[lab]
        # Random representative
        rep_idx = random.choice(idxs)
        _, rep_payload = themes[rep_idx]
        rep_text = (rep_payload.get("Theme") or "").strip()

        # Merge processes
        merged_proc = []
        merged_from_texts = []
        for j in idxs:
            _, p = themes[j]
            t_text = (p.get("Theme") or "").strip()
            if t_text:
                merged_from_texts.append(t_text)
            procs = p.get("Processes") or []
            if isinstance(procs, list):
                merged_proc.extend(procs)

        merged_entry = dict(rep_payload)  # start from representative payload
        merged_entry["Theme"] = rep_text  # enforce clean text
        merged_entry["Processes"] = dedup_list(merged_proc)
        merged_entry["MergedFrom"] = dedup_list(merged_from_texts)

        merged[f"Theme {new_idx}"] = merged_entry

    return merged

def append_semantic_report_txt(source_name: str, texts: List[str], labels: np.ndarray) -> None:
    from collections import defaultdict
    by_label = defaultdict(list)
    for text, lab in zip(texts, labels):
        by_label[int(lab)].append(text.strip())

    lines = []
    lines.append("=" * 80)
    lines.append(f"Semantic groups for {source_name}")
    lines.append("Method = fast community detection (cosine)\n"
             f"Threshold = {SEMANTIC_SIM_THRESHOLD}, MinSize = {SEMANTIC_MIN_COMMUNITY}")
    lines.append("")
    for lab in sorted(by_label.keys()):
        lines.append(f"Cluster {lab}:")
        for t in by_label[lab]:
            lines.append(f"  • {t}")
        lines.append("")
    with open(SEMANTIC_TXT_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_merged_json(classified: Dict[str, Any], merged_clusters: Dict[str, Dict[str, Any]], patient_id: str, session_num_int: int) -> Path:
    """
    Keep top-level structure; replace 'clusters' with merged.
    Write to merged_json/<PXY>/<normalized_filename>.json
    Your save_classification created: cluster_{patient_id}_S{session_num}.json
    """
    SEMANTIC_JSON_BASE.mkdir(parents=True, exist_ok=True)
    patient_dir = SEMANTIC_JSON_BASE / patient_id  # patient_id like "P11"
    patient_dir.mkdir(parents=True, exist_ok=True)

    raw_name = f"cluster_{patient_id}_S{session_num_int}.json"
    norm_name = normalize_session_in_filename(raw_name)

    out_path = patient_dir / norm_name
    out_payload = dict(classified)
    out_payload["clusters"] = merged_clusters
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def run_semantic_merge_step(classified_labels_any: Any, patient_id: str, session_num_int: int) -> List[str]:
    """
    Accepts list[str], JSON-encoded str, or {'labels': [...]}. Returns merged label list.
    Also writes a TXT report and a merged JSON artifact.
    """
    labels_list = coerce_to_label_list(classified_labels_any)
    if not labels_list:
        print("Semantic merge skipped: no labels found.")
        return []

    # Compute embeddings & clusters
    emb = encode(labels_list)
    labs = cluster(emb)

    # Report (TXT)
    append_semantic_report_txt(f"{patient_id}_S{session_num_int}", labels_list, labs)

    # Merge into theme-shaped dict
    merged_clusters = merge_from_texts(labels_list, labs)

    # Derive a merged label list (representatives per cluster) in stable order
    merged_labels = [v.get("Theme", "").strip() for k, v in merged_clusters.items() if v.get("Theme")]

    # Prepare a sane payload to write; keep original labels and merged themes
    classified_payload = {
        "labels_original": labels_list,
        "labels_merged": merged_labels,
        "clusters": merged_clusters
    }
    out_path = write_merged_json(classified_payload, merged_clusters, patient_id, session_num_int)
    print(f"Semantic merge: wrote merged JSON to {out_path}")

    return merged_labels




def is_generic_label_output(output_str):
    def contains_blacklisted_item(items):
        for item in items:
            if item in black_list:
                print(f"Blacklisted item found: {item}")
                return True
        return False

    def starts_with_clients_phrase(items):
        for item in items:
            if isinstance(item, str) and item.strip().lower().startswith("the client's"):
                print(f"Item starts with 'The client's': {item}")
                return True
        return False

    def all_empty_strings(items):
        return all(isinstance(item, str) and item.strip() == "" for item in items)

    def contains_ellipsis(items):
        return any(item is Ellipsis or (isinstance(item, str) and "..." in item) for item in items)




    print(f"Input string: {output_str}")

    # Quick check for excessive empty string patterns in malformed input
    empty_quote_count = output_str.count('""') + output_str.count("''")
    if empty_quote_count >= 10:
        print(f"Returning True due to excessive empty string patterns: count={empty_quote_count}")
        return True

    # AST parsing
    try:
        parsed = ast.literal_eval(output_str)
        print(f"AST parsed result: {parsed}")
        if isinstance(parsed, list):
            if all_empty_strings(parsed):
                print("Returning True due to all items being empty strings (AST)")
                return True
            if contains_ellipsis(parsed):
                print("Returning True due to Ellipsis found in list (AST)")
                return True
            if contains_blacklisted_item(parsed):
                print("Returning True due to blacklist match (AST)")
                return True
            if starts_with_clients_phrase(parsed):
                print("Returning True due to 'The client's' match (AST)")
                return True
            if all(isinstance(item, str) and "theme" in item.lower() for item in parsed):
                print("Returning True due to 'theme' match in AST list")
                return True
    except (ValueError, SyntaxError) as e:
        print(f"AST parsing failed: {e}")

    # JSON parsing
    try:
        parsed = json.loads(output_str)
        print(f"JSON parsed result: {parsed}")
        if isinstance(parsed, list):
            if all_empty_strings(parsed):
                print("Returning True due to all items being empty strings (JSON)")
                return True
            if contains_blacklisted_item(parsed):
                print("Returning True due to blacklist match (JSON)")
                return True
            if contains_ellipsis(parsed):
                print("Returning True due to Ellipsis found (JSON)")
                return True
            if starts_with_clients_phrase(parsed):
                print("Returning True due to 'The client's' match (JSON)")
                return True
            if all(isinstance(item, str) and "theme" in item.lower() for item in parsed):
                print("Returning True due to 'theme' match in JSON list")
                return True
    except (json.JSONDecodeError, TypeError) as e:
        print(f"JSON parsing failed: {e}")

    # Fallback parsing
    output_str_cleaned = output_str.strip("[]").replace('"', '').replace("'", "")
    items = [item.strip() for item in output_str_cleaned.split(',')]
    valid_items = [item for item in items if item]

    print(f"Cleaned and split items: {valid_items}")

    if valid_items:
        if all(item == "" for item in valid_items):
            print("Returning True due to all items being empty strings (fallback)")
            return True
        if contains_blacklisted_item(valid_items):
            print("Returning True due to blacklist match (fallback)")
            return True
        if starts_with_clients_phrase(valid_items):
            print("Returning True due to 'The client's' match (fallback)")
            return True
        if all("theme" in item.lower() for item in valid_items):
            print("Returning True due to 'theme' match in fallback list")
            return True

    print("Returning False: No conditions met")
    return False



# --- Preprocess model output to remove structure/example description ---
def extract_json_from_output(output):
    # If model includes explanation + "**output:**", extract only what's after it
    if '**output:**' in output:
        output = output.split('**output:**', 1)[-1].strip()

    # Try to isolate the JSON block
    json_match = re.search(r'\{.*\}', output, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return output  # fallback, may still be useful

# Save generated labels to JSON
def save_labels(labels, session_id, patient_id):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"generated_labels_{patient_id}_{session_id}.json")

    with open(file_path, 'w') as f:
        json.dump({"labels": labels}, f, indent=2)

    print(f"Generated labels saved at: {file_path}")


# save classification to json
def save_classification(classification, session_id, patient_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"cluster_{patient_id}_S{session_id}.json")

    with open(file_path, 'w') as f:
        json.dump({"clusters": classification}, f, indent=2)

    print(f"Classification saved at: {file_path}")

# Chat function using vLLM
def chat(prompt, llm, temperature):
    sampling_params = SamplingParams(
        max_tokens=2000,
        temperature=temperature
    )
    print("Prompt:", prompt)
    outputs = llm.generate([prompt], sampling_params)
    #print("output before decoding:", outputs)
    response_origin = outputs[0].outputs[0].text.strip()  # Extract response
    print("Decoded output:", response_origin)
    return response_origin

def chat_classification(prompt, llm):
    sampling_params = SamplingParams(
        max_tokens=30544,
        temperature=0.0
    )
    print("Prompt:", prompt)
    outputs = llm.generate([prompt], sampling_params)
    #print("output before decoding:", outputs)
    response_origin = outputs[0].outputs[0].text.strip()  # Extract response
    print("Decoded output:", response_origin)
    return response_origin


# Initialize model
def initialize_model():
    model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        max_model_len=30544
    )
    return llm

# Function for generating labels
def generate_labels(dialogue_data, process_data):
    prompt = prompt_construct_generate_labels(dialogue_data, process_data)
    return prompt


# Function for classifying processes under generated labels
def classify_processes(dialogue_data, process_data, generated_labels):
    prompt = given_label_classification(dialogue_data,process_data,generated_labels)  # Define this function
    return prompt



def main(transcript, process_data):
    llm = initialize_model()

    # Each session that occures in the transcript file is processed
    for item in transcript:
        session_id = item["session"]  # Extract session ID
        dialogue_data = item["dialogues"]  # Extract dialogues

        # Example: session_id = "P15_S03"
        patient_id_dialogue = session_id.split("_")[0]  # "P15"
        session_str = session_id.split("_")[1]  # "S03"
        session_number_int = int(session_str[1:])  # Convert to integer

        print(f"\n---\nProcessing session_id: {session_id}")
        print(f"Extracted patient_id_dialogue: {patient_id_dialogue}, session_number_int: {session_number_int}")

        found_patient = False
        found_session = False

        for patient_id, sessions in process_data.items():
            if patient_id_dialogue.lower() == patient_id.lower():
                found_patient = True
                print(f"  Matched patient ID: {patient_id_dialogue} with {patient_id} (case-insensitive).")
                for session in sessions:
                    for pocess in session:
                        process_id = pocess["process_id"]  # e.g., "p15_3"
                        pid_parts = process_id.split("_")

                        try:
                            proc_patient_id = pid_parts[0]  # patient part (e.g., "p15")
                            proc_session_str = pid_parts[1]  # session part (e.g., "3")
                            proc_session_num = int(proc_session_str)  # Convert to integer
                        except ValueError:
                            print(f"  Cannot parse session or process number from {pid_parts}. Skipping.")
                            continue

                        print(f"  Checking process_id={process_id}: patient={proc_patient_id}, session_num={proc_session_num} vs. target_session={session_number_int}")
                        if proc_session_num == session_number_int:
                            found_session = True
                            print(f"    -> Session match found. Generating two stage prompt...")

                            # Step 1: Generate Labels that is later used to classify the processes under
                            print("Starting label generation process...")
                            start_time = time.time()
                            temperature = 0.0
                            max_temperature = 1.0
                            invalid_output = True

                            generated_labels_prompt = generate_labels(dialogue_data, session)
                            final_temperature = temperature

                            while invalid_output and temperature <= max_temperature:
                                print(f"Trying generation with temperature {temperature:.1f}...")
                                generated_labels_str = chat(generated_labels_prompt, llm, temperature=temperature)

                                if is_generic_label_output(generated_labels_str):
                                    print(
                                        f"Rejected output at temperature {temperature:.1f}. Reason: generic or blacklisted.")
                                    temperature += 0.2
                                else:
                                    invalid_output = False
                                    final_temperature = temperature  # Set once we find valid output

                            if invalid_output:
                                print("Warning: Max temperature reached but output may still be invalid.")
                                final_temperature = max_temperature

                            # Logging if temperature was increased
                            if final_temperature > 0.0:
                                print("Writing log")
                                log_entry = f"Increased temperature to {final_temperature:.1f} for {session_id}.\n"
                                with open("label_generation_log.txt", "a") as log_file:
                                    log_file.write(log_entry)

                            try:
                                generated_labels = json.loads(generated_labels_str)
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse generated labels JSON: {e}")
                                print(f"In {session_id}_{patient_id}")
                                # If the output of the model is not a correct json we still pare it for the next step of the clustering
                                generated_labels = generated_labels_str

                            print(f"Generated Labels: {generated_labels}")
                            save_labels(generated_labels, proc_session_num, proc_patient_id)
                            print(f"Total runtime: {time.time() - start_time:.2f} seconds")

                            # To prevent too similar Themes we use semantic clustering between the generated Theme Lables
                            print("Starting semantic clustering on generated labels...")
                            try:
                                merged_labels = run_semantic_merge_step(generated_labels, proc_patient_id,
                                                                        proc_session_num)
                                if merged_labels:
                                    labels_for_classification = merged_labels
                                else:
                                    # fallback to original if merge produced nothing
                                    labels_for_classification = coerce_to_label_list(generated_labels)
                            except Exception as e:
                                print(f"Semantic clustering failed for {proc_patient_id}_S{proc_session_num}: {e}")
                                labels_for_classification = coerce_to_label_list(generated_labels)

                            # Step 3: Classify Processes under (merged) Labels
                            print("Starting label classification process...")
                            start_time = time.time()
                            classification_result = classify_processes(dialogue_data, session,
                                                                       labels_for_classification)
                            classified_labels_str = chat_classification(classification_result, llm)

                            try:
                                    cleaned_output = extract_json_from_output(classified_labels_str)
                                    classified_labels = json.loads(cleaned_output)
                            except json.JSONDecodeError as e:
                                    print(f"Failed to parse classification Json : {e}")
                                    print(f"In {session_id}_{patient_id}")
                                    classified_labels = classified_labels_str

                            save_classification(classified_labels, proc_session_num, proc_patient_id)
                            print(f"Classification: {classified_labels}")
                            print(f"Total runtime: {time.time() - start_time:.2f} seconds")

                            print("Generated Labels END:", generated_labels)
                            print("Classification Result END:", classified_labels)

                            with open("Clustering", 'a') as file:
                                file.write(f"Session ID: {session_id}\n")
                                file.write(f"Generated Labels: {generated_labels}\n")
                                file.write(f"Classification Result: {classified_labels}\n")
                                file.write("\n\n")

                            break
                    if found_session:
                        break
                if found_session:
                    break

        if not found_patient:
            print(f"  No patient found with ID: {patient_id_dialogue}")
        elif not found_session:
            print(f"  No session found for patient: {patient_id_dialogue} with session number: {session_number_int}")
    return None



if __name__ == "__main__":
    start_time = time.time()
    transcript = load_transcripts("eating_disorder_data/minibatch/P004_dialog.json")  # Load transcript file
    process_data = load_transcripts("eating_disorder_data/minibatch/P004.json") # Loading Process file
    main(transcript, process_data)
    end_time = time.time()
    print(f"Time taken to cluster sentences: {end_time - start_time} seconds")


