import time
import os
import json
from itertools import permutations
from vllm import LLM, SamplingParams

os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"

llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.98,
    max_model_len=32672
)

def generate_linkage_between_paris(process_1, process_2):
    process_1_str = process_1["name"] + "\n"
    process_2_str = process_2["name"] + "\n"

    prompt = f"""

    **Task:** You are a clinical psychologist and your task is to identify relationships between psychological processes.
    Your task is to determine whether a relationship exists between **Process A** and **Process B**, and to generate information on the nature of that relationship.

        Definitions:
        - **Psychological process**: A psychological process refers to the mental activities and mechanisms through which individuals perceive, think, feel, and respond to their environment, influencing (overt) behavior, cognition, affect, attention, sense of self, motivation.
        - Connection:
            - 1: there is a relationship
            - 0: there is no relationship
        - **Type of relationships between two psychological processes**: 
            1) Excitatory as in one process amplifies or reinforces another process. (e.g. Process A leads to Process B, Process A causes Process B)
            2) Inhibitory as in  one process suppresses or counteracts another process. (e.g. Process A stops Process B, Process A diminishes Process B)
        - **Strength of Relationship**: 
            1) "Strong": The two processes are closely and consistently related across contexts. One strongly influences or is inherently linked to the other in psychological functioning.
            2) "Moderate": The two processes are meaningfully related, but the connection is less consistent or conditional. They are associated in many situations, but not tightly bound.
            3) "Weak": The relationship between the two processes is minimal, indirect, or highly context-dependent. They may co-occur at times, but the link is loose or peripheral.        

        Input: **Two psychological processes**

        Guidelines:
            1. Given two processes, determine whether or not a relationship exists.
            2. Provide a brief explanation of the connection. Avoid restating the processesthemselve

        Output Structure: If relationships exist from process A to process B, return a structured output containing details for each identified relationship: 
        json
        {{"relationship":
        [{{ "input_processes": ["Process A", "Process B"],
        "connection": [1],
        "relationship_type": ["excitatory", "inhibitory"],
        "strength_of_relationship": ["strong", "moderate", "weak"], 
        "explanation": "A concise natural language explanation of why this relationship exists."}}]
        }}

        If no relationships exist from Process A to Process B, return:
        json
        {{"relationship":
        [{{ "input_processes": ["Process A", "Process B"],
        "connection": [0]}}]
        }}

        Process A: {process_1_str}

        Process B: {process_2_str}
        Now, generate the clinically meaningful relationship between the provided Processes in the specified output format. 
        Only output the relationship.
    """
    return prompt

def cut_after_first_full_json(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None

def process_and_save_dialogues_with_cluster_linkage(clusters_data, output_file, temperature, batch_size=8):
    mismatch_log = []
    all_relationships = []
    failed_outputs = []
    clusters = clusters_data["clusters"]

    ordered_pairs = list(permutations(clusters, 2))
    pair_to_label = {}
    sorted_pairs = sorted(ordered_pairs, key=lambda pair: (pair[0]['name'], pair[1]['name']))
    for idx, (c1, c2) in enumerate(sorted_pairs, start=1):
        key = (c1['name'], c2['name'])
        pair_to_label[key] = idx

    prompts = []
    meta_infos = []

    for cluster1, cluster2 in ordered_pairs:
        label = pair_to_label[(cluster1['name'], cluster2['name'])]
        prompt = generate_linkage_between_paris(cluster1, cluster2)
        prompts.append(prompt)
        meta_infos.append((cluster1, cluster2, label))

    sampling_params = SamplingParams(max_tokens=2000, temperature=temperature)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_meta = meta_infos[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for out, (cluster1, cluster2, label) in zip(outputs, batch_meta):
            decoded_output = out.outputs[0].text.strip()
            json_string = cut_after_first_full_json(decoded_output)

            if json_string:
                try:
                    parsed_json = json.loads(json_string)
                    if parsed_json and parsed_json.get("relationship"):
                        for rel in parsed_json["relationship"]:
                            if isinstance(rel, dict):
                                expected_inputs = [cluster1["name"], cluster2["name"]]
                                actual_inputs = rel.get("input_processes", [])

                                if actual_inputs != expected_inputs:
                                    mismatch_log.append({
                                        "label": label,
                                        "expected": expected_inputs,
                                        "model_returned": actual_inputs,
                                        "raw_output": decoded_output
                                    })
                                    rel["input_processes"] = expected_inputs

                                rel["pair_label"] = label
                                all_relationships.append(rel)
                    else:
                        failed_outputs.append({
                            "label": label,
                            "cluster_pair": (cluster1["name"], cluster2["name"]),
                            "raw_output": decoded_output
                        })

                except json.JSONDecodeError:
                    failed_outputs.append({
                        "label": label,
                        "cluster_pair": (cluster1["name"], cluster2["name"]),
                        "raw_output": decoded_output
                    })
            else:
                failed_outputs.append({
                    "label": label,
                    "cluster_pair": (cluster1["name"], cluster2["name"]),
                    "raw_output": decoded_output
                })

    final_output = {"relationships": all_relationships}
    with open(output_file, 'w') as file:
        json.dump(final_output, file, indent=2)
    print(f"Saved {len(all_relationships)} relationships to {output_file}")

    if failed_outputs:
        failed_file = output_file.replace(".json", "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_outputs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(failed_outputs)} failed outputs to: {failed_file}")

    if mismatch_log:
        mismatch_file = output_file.replace(".json", "_input_mismatches.txt")
        with open(mismatch_file, "w", encoding="utf-8") as f:
            for entry in mismatch_log:
                f.write(f"Label: {entry['label']}\n")
                f.write(f"Expected: {entry['expected']}\n")
                f.write(f"Model Returned: {entry['model_returned']}\n")
                f.write(f"Raw Output:\n{entry['raw_output']}\n")
                f.write("-" * 80 + "\n")
        print(f"Saved {len(mismatch_log)} input mismatches to: {mismatch_file}")

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if isinstance(data.get("clusters"), dict):
        transformed_clusters = [
            {"name": theme_obj["Theme"]}
            for theme_obj in data["clusters"].values()
            if "Theme" in theme_obj
        ]
        return {"clusters": transformed_clusters}
    else:
        return data

def generate_linkage(input_folder, output_folder):
    start_time = time.time()
    temp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Original full temperature list

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            print(f"\nProcessing file: {input_file_path}")
            cluster_data = read_sentences_from_file(input_file_path)

            for temp in temp_list:
                print(f"\nRunning model with temperature: {temp} for file: {filename}")
                temp_str = f"{temp:.1f}"
                output_filename = f"{base_filename}_temperature_{temp_str}.json"
                output_file_path = os.path.join(output_folder, output_filename)

                process_and_save_dialogues_with_cluster_linkage(
                    clusters_data=cluster_data,
                    output_file=output_file_path,
                    temperature=temp,
                    batch_size=8  # You can tweak this if needed
                )

    print(f"Total time: {time.time() - start_time:.2f} seconds")

# Run pipeline
input_folder = "finalized_clusters/P13"
output_folder = "linkage_P13"
generate_linkage(input_folder, output_folder)
