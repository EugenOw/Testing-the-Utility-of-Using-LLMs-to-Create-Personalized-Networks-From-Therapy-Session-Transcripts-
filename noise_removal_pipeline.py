import json
import numpy as np
import time
from vllm import LLM, SamplingParams
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel
import torch
import os
from sentence_transformers import SentenceTransformer


# ------------Parameters----------------
# Define the model path
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"
os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"
# ----------------------------------------
def main():
    print("Pipeline started")
    #input_file = "output/modified_pbt_annotation_of_mini_session (1).json"
    input_file = "modified_pbt_annotation_of_mini_session (1).json"
    output_file = "slurm_clustering_clinicly/data/pre_processed_modified_pbt_annotation_of_mini_session.json"

    # JSON-Daten laden
    print("Loading output...")
    data = load_json(input_file)

    print("Processing output...")
    # Verarbeite alle Personen und ihre Sessions
    processed_data = process_all(data)

    # Ergebnisse speichern
    save_json(processed_data, output_file)


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json(data, file_path):
    def custom_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, default=custom_serializer)




# Processing Mini Sessions
def process_all(data):
    """
    # Processing all Patients and their Sessions
    """
    processed_data = {}
    for person_id, sessions in data.items():
        processed_sessions = process_person_sessions(person_id, sessions)
        processed_data[person_id] = processed_sessions
    return processed_data


def process_person_sessions(person_id, sessions):
    """
    Process all session for one Patient
    """
    processed_sessions = []
    for session in sessions:
        # Processing the entire session from the patient
        # Example of on session would be p15_1 or P11_13
        processed_session = process_single_session(session)
        processed_sessions.append(processed_session)
    return processed_sessions


def process_single_session(session):
    """
    Verarbeitet eine einzelne Session durch die Pipeline.
    """
    # print(session)
    # Step 1: Looking for Duplicates
    step_start = time.time()
    session, debug_log = check_duplicates(session)
    step_end = time.time()
    print(f"Step 1 (check_duplicates) took {step_end - step_start:.2f} seconds")
    print(debug_log)
    torch.cuda.empty_cache()  # Clear GPU memory
    # Step 2: Klassifikation (Filler vs. Non-Filler)
    step_start = time.time()
    session = classify_filler_with_llm(session)
    session = [process for process in session if process["filler_classification"] == "Not_filler"]
    step_end = time.time()
    print(f"Step 2 (classify_filler_with_llm) took {step_end - step_start:.2f} seconds")
    torch.cuda.empty_cache()  # Clear GPU memory
    # Step 3: Mega-Nodes durch semantische Gruppierung
    step_start = time.time()
    session, cluster_map = semantic_clustering(session)
    step_end = time.time()
    print(f"Step 3 (semantic_clustering) took {step_end - step_start:.2f} seconds")
    print(cluster_map)
    torch.cuda.empty_cache()  # Clear GPU memory

    return session


def merge_contexts(context_a, context_b):
    """
    Merges two context lists chronologically, avoiding duplicates.
    Handles overlapping timestamps by combining text and adjusting times.
    """
    combined = context_a + context_b
    combined = sorted(combined, key=lambda x: x["time_begin"])

    merged_context = []
    for entry in combined:
        if not merged_context:
            merged_context.append(entry)
            continue

        last_entry = merged_context[-1]

        # Check for duplicate or overlapping entries
        if (
            last_entry["speaker"] == entry["speaker"]
            and last_entry["time_end"] >= entry["time_begin"]
        ):
            # Merge overlapping entries
            last_entry["text"] += " " + entry["text"]
            last_entry["time_end"] = max(last_entry["time_end"], entry["time_end"])
        elif last_entry != entry:  # Avoid exact duplicates
            merged_context.append(entry)

    return merged_context


def check_duplicates(session):
    """
    Detects and resolves duplicate processes within the session.
    Merges processes only if one process's modified text contains another process's text.
    """
    process_texts = set()
    skip_ids = set()  # Track processes that are already merged
    was_modified = {process["process_id"]: False for process in session}  # Track if process was modified
    debug_logs = []

    for i, process_a in enumerate(session):
        if process_a["process_id"] in skip_ids:
            continue  # Skip already merged processes

        process_a_text = process_a["process"]
        process_a_context = process_a["context"]
        process_a_dimension = process_a.get("dimension", [])

        # Step 1: Detect and handle duplicates by modifying process_text
        prefix, suffix = "", ""
        added_context_texts = set()  # Track specific texts added as context
        for idx, context_entry in enumerate(process_a_context):
            if context_entry["text"] == process_a_text:
                speaker = context_entry["speaker"]

                # Extract prefix and suffix
                if idx > 0 and process_a_context[idx - 1]["speaker"] == speaker:
                    prefix = process_a_context[idx - 1]["text"] + " "
                    added_context_texts.add(process_a_context[idx - 1]["text"])
                if idx < len(process_a_context) - 1 and process_a_context[idx + 1]["speaker"] == speaker:
                    suffix = " " + process_a_context[idx + 1]["text"]
                    added_context_texts.add(process_a_context[idx + 1]["text"])
                break

        if process_a_text in process_texts:
            modified_process = f"{prefix}{process_a_text}{suffix}".strip()
            if modified_process != process_a_text:
                was_modified[process_a["process_id"]] = True  # Mark as modified
                debug_logs.append(
                    f"Modified process (ID: {process_a['process_id']}): '{process_a_text}' -> '{modified_process}'"
                )
                process_a["process"] = modified_process
        else:
            process_texts.add(process_a_text)

    # Step 2: Merge processes only if they were modified and added context matches another process
    for i, process_a in enumerate(session):
        if process_a["process_id"] in skip_ids or not was_modified[process_a["process_id"]]:
            continue  # Skip unmodified or already merged processes

        process_a_text = process_a["process"]
        process_a_context = process_a["context"]
        process_a_dimension = process_a.get("dimension", [])

        for j, process_b in enumerate(session):
            if i == j or process_b["process_id"] in skip_ids:
                continue  # Skip self-comparison and already merged processes

            process_b_text = process_b["process"]

            # Check if process_b_text was part of the added context in Step 1
            if process_b_text in added_context_texts:
                # Merge process_a and process_b
                unified_process_text = f"{process_a['process']} + {process_b['process']}"
                unified_context = merge_contexts(process_a_context, process_b["context"])
                unified_dimension = list(set(process_a_dimension + process_b.get("dimension", [])))

                # Update process_a with the merged result
                process_a.update({
                    "process": unified_process_text,
                    "context": unified_context,
                    "dimension": unified_dimension,
                })

                debug_logs.append(
                    f"Merged process (ID: {process_a['process_id']}): '{process_b['process_id']}' into '{process_a['process_id']}'"
                )

                # Mark process_b as merged
                skip_ids.add(process_b["process_id"])

    # Remove merged processes from the session
    session = [process for process in session if process["process_id"] not in skip_ids]

    return session, debug_logs





def classify_filler_with_llm(session):
    """
    Classification of each "process" in a session as Filler or Not_filler using an LLM
    The classification result will be added  as "filler_classification" into the json file for each process
    If the output is not feasible for a process then in default it will be labeled as "Not_filler

    return: modified session with filler_classification as new key
    """
    # Generate correct input for the model
    # Process with Context: {process_id, ,process, context_before, context_after}
    processes_with_context = [extract_context(process) for process in session]
    # The output will be a list of dictionaries with the key "sentence" and "answer"
    # results = [{"process_id": sentence["process_id"], "sentence": sentence["process"], "answer": final_classification}, ....]
    classification_results = filtering_sentences_majority_vote(processes_with_context)
    for process in session:
        for result in classification_results:
            if result["process_id"] == process["process_id"]:
                process["filler_classification"] = result["answer"]
                break
    return session


def extract_features(sentences):
    """
    Creates for the given list of sentences/processes the embeddings
    """
    # Load the model
    embeddings = model.encode(sentences)
    return np.array(embeddings)


def semantic_clustering(session, threshold=2.6):
    """
    Clustering processes in a session based on semantic similarity.
    Adds a semantic_cluster attribute to each process and returns a cluster map for quick lookups.
    """
    sentences = [entry["process"] for entry in session]
    features = extract_features(sentences)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    predicted_labels = clustering.fit_predict(features)

    # Group processes by their cluster labels
    clusters = {}
    for label, process in zip(predicted_labels, session):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(process["process_id"])

    # Add the semantic clusters to the session
    for process in session:
        process_id = process["process_id"]
        for cluster_id, process_ids in clusters.items():
            if process_id in process_ids:
                process["semantic_cluster"] = cluster_id
                break

    # Create a cluster map for easier searching
    cluster_map = {cluster_id: process_ids for cluster_id, process_ids in clusters.items()}

    return session, cluster_map



def extract_context(process_entry):
    """
    For a process the context before and after will be extarcted
    and returned as a trple:
    {process, context_before, context_after}
    """

    process_id = process_entry["process_id"]
    process_text = process_entry["process"]
    context = process_entry["context"]

    # Find the index of the process text in the context
    process_index = next((i for i, utterance in enumerate(context) if utterance["text"] == process_text), None)

    if process_index is None:
        # Handle case where process_text is not found in context
        print(f"Process text '{process_text}' not found in context")
        print("Returning full context:", context)
        relevant_context = context
    else:
        start = max(0, process_index - 4)
        end = min(len(context), process_index + 5)
        relevant_context = context[start:end]

    # Separate context into before, process, and after
    if process_index is not None:
        context_before = [{"text": utterance["text"], "speaker": utterance["speaker"]} for utterance in
                          context[:process_index]]
        context_after = [{"text": utterance["text"], "speaker": utterance["speaker"]} for utterance in
                         context[process_index + 1:]]
    else:
        context_before = []
        context_after = [{"text": utterance["text"], "speaker": utterance["speaker"]} for utterance in context]

    return {
        "process_id": process_id,
        "process": process_text,
        "context_before": context_before,
        "context_after": context_after,
    }


def filtering_sentences_majority_vote(sentences):
    results = []
    for sentence in sentences:
        prompts = [
            generate_prompt_model1(sentence),
            generate_prompt_model2(sentence),
            generate_prompt_model3(sentence)
        ]
        try:
            sampling_params = SamplingParams(
                max_tokens=300,
                temperature=0.1,
                top_k=50,
                top_p=0.09
            )
            outputs = [llm.generate([prompt], sampling_params)[0].outputs[0].text.strip() for prompt in prompts]
            parsed_outputs = [output.splitlines()[-1].strip().replace("Answer:", "").strip() for output in outputs]

            for i, parsed_output in enumerate(parsed_outputs):
                print(f"Model {i + 1} decision: {parsed_output}")

            # Majority vote
            vote_count = {"Filler": 0, "Not_filler": 0}
            for parsed_output in parsed_outputs:
                if parsed_output in vote_count:
                    vote_count[parsed_output] += 1

            if vote_count["Filler"] > vote_count["Not_filler"]:
                final_classification = "Filler"
            elif vote_count["Not_filler"] > vote_count["Filler"]:
                final_classification = "Not_filler"
            else:
                final_classification = "Not_filler"  # Default if undecidable

            results.append(
                {"process_id": sentence["process_id"], "sentence": sentence["process"], "answer": final_classification})
            print(f"Sentence: {sentence['process']}, Answer: {final_classification}")

        except Exception as e:
            print(f"Error processing sentence: {e}")
            results.append(None)
    return results


def generate_prompt_model1(process, ):
    process_text = process["process"]
    context_before = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_before"]])
    context_after = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_after"]])

    return f"""
A filler sentence in a counselling session is a statement used to fill a pause or create space in the dialogue, often without significant content.\n
**Task:** Classify a given sentence as a filler or a not_filler. In addition, you will be provided with the context before and after the given sentence.\n
The output format should be one word only: "Filler" or "Not_filler". Do not add explanations, notes, punctuation, or extra text.\n

It is part of a bigger dialogue. For better context, here is the surrounding dialogue:\n
Dialogue:\n
{context_before}
{process_text}  (our main utterance)
{context_after}

Classify the sentence that is marked as (our main sentence) as a filler or a not_filler.\n
{process_text} =>
"""


def generate_prompt_model2(process):
    process_text = process["process"]
    context_before = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_before"]])
    context_after = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_after"]])

    return f"""
A filler sentence in a counselling session is a statement used to fill a pause or create space in the dialogue, often without significant content.\n
**Task:** Classify a given sentence as a filler or a not_filler. In addition, you will be provided with the context before and after the given sentence.\n
The output format should be one word only: "Filler" or "Not_filler". Do not add explanations, notes, punctuation, or extra text.\n

A few examples:\n
- "you were saying like when you came up to your room it was easier to just sort of like here the thought is there and then I'm gonna do my thing." => Filler
- "Probably not as relevant now because I don't have a job, but also I think relevant because like, then I won't get out of the house." => Not_filler
- "And I'd like to do more of that, and I have observed myself thinking, well, it's going to be a whole lot easier for me to restart this whole thing than to go back to where I was." => Not_filler
- "So, homework were you able to do any, like, did you essentially the worksheet?" => Filler
- "all it would be a hold." => Filler


It is part of a bigger dialogue. For better context, here is the surrounding dialogue:\n
Dialogue:\n
{context_before}
{process_text}  (our main utterance)
{context_after}

Classify the sentence that is marked as (our main sentence) as a filler or a not_filler.
{process_text} =>
"""


def generate_prompt_model3(process):
    process_text = process["process"]
    context_before = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_before"]])
    context_after = "\n".join([f"{item['speaker']}: {item['text']}" for item in process["context_after"]])

    return f"""
A filler sentence in a counselling session is a statement used to fill a pause or create space in the dialogue, often without significant content.\n
**Task:** Classify a given sentence as "Filler" or "Not_filler".
The output format should be one word only: "Filler" or "Not_filler". Do not add explanations, notes, punctuation, or extra text.\n

A few examples:\n
- "you were saying like when you came up to your room it was easier to just sort of like here the thought is there and then I'm gonna do my thing." => Filler
- "Probably not as relevant now because I don't have a job, but also I think relevant because like, then I won't get out of the house." => Not_filler
- "And I'd like to do more of that, and I have observed myself thinking, well, it's going to be a whole lot easier for me to restart this whole thing than to go back to where I was." => Not_filler
- "So, homework were you able to do any, like, did you essentially the worksheet?" => Filler
- "all it would be a hold." => Filler

Classify the sentence  as a filler or a not_filler.
{process_text} =>
"""


if __name__ == "__main__":
    # Load Model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=24752
    )
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    main()

