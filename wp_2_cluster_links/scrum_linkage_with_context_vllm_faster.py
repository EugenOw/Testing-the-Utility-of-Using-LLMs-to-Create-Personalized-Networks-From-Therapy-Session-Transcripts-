import time
import re
import json
import os
from itertools import permutations
from vllm import LLM, SamplingParams

os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"

start_time = time.time()
llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.98,
    max_model_len=32672
)
print(f"Model load time: {time.time() - start_time:.2f} seconds")

def generate_linkage_between_paris(process_1, process_2):
    process_1_str = f"{process_1['name']}\n"
    process_2_str = f"{process_2['name']}\n"
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

def cut_after_first_full_json(text):
    start = text.find('{')
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        if brace_count == 0: return text[start:i+1]
    return None

def linkage(prompt, temperature):
    sampling_params = SamplingParams(max_tokens=150, temperature=temperature)
    outputs = llm.generate([prompt], sampling_params)
    decoded_output = outputs[0].outputs[0].text.strip()
    json_string = cut_after_first_full_json(decoded_output)
    if json_string:
        try:
            parsed_json = json.loads(json_string)
            return parsed_json, decoded_output
        except json.JSONDecodeError as e:
            print("JSON decode failed:", e)
    return None, decoded_output

def process_and_save_dialogues_with_cluster_linkage(clusters_data, output_file, temperature):
    from itertools import permutations

    clusters = clusters_data["clusters"]
    ordered_pairs = list(permutations(clusters, 2))

    # Assign stable labels based on sorted names
    pair_to_label = {}
    sorted_pairs = sorted(ordered_pairs, key=lambda pair: (pair[0]['name'], pair[1]['name']))
    for idx, (c1, c2) in enumerate(sorted_pairs, start=1):
        key = (c1['name'], c2['name'])
        pair_to_label[key] = idx

    relationships = []
    failed = []

    prompts = [generate_linkage_between_paris(p1, p2) for p1, p2 in ordered_pairs]

    def batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    for prompt_batch in batch(prompts, 4):  # safe batch size
        try:
            output_batch = llm.generate(prompt_batch, SamplingParams(max_tokens=150, temperature=temperature))
            for i, out in enumerate(output_batch):
                raw = out.outputs[0].text.strip()
                json_str = cut_after_first_full_json(raw)
                idx = prompts.index(prompt_batch[i])
                cluster_pair = ordered_pairs[idx]
                cluster_names = [cluster_pair[0]['name'], cluster_pair[1]['name']]
                label_key = tuple(cluster_names)
                pair_label = pair_to_label.get(label_key)

                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        rels = parsed.get("relationship", [])
                        for rel in rels:
                            # Overwrite with the correct input pair and label
                            rel["input_processes"] = cluster_names
                            rel["pair_label"] = pair_label
                            relationships.append(rel)
                    except Exception:
                        failed.append({"pair": cluster_names, "raw_output": raw})
                else:
                    failed.append({"pair": cluster_names, "raw_output": raw})
        except Exception as e:
            print(f"Batch failed: {e}")
            for prompt in prompt_batch:
                idx = prompts.index(prompt)
                cluster_pair = ordered_pairs[idx]
                cluster_names = [cluster_pair[0]['name'], cluster_pair[1]['name']]
                failed.append({"pair": cluster_names, "raw_output": "<batch failed>"})

    # Save successful relationships
    with open(output_file, 'w') as f:
        json.dump({"relationships": relationships}, f, indent=2)

    # Save failed outputs
    if failed:
        with open(output_file.replace(".json", "_failed.json"), 'w') as f:
            json.dump(failed, f, indent=2)


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
    temp_list = [0.0, 0.5, 1.0]

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            print(f"\nProcessing file: {input_file_path}")
            cluster_data = read_sentences_from_file(input_file_path)

            for temp in temp_list:
                print(f"\nRunning model with temperature: {temp} for file: {filename}")
                output_filename = f"{base_filename}_temperature_{temp:.1f}.json"
                output_file_path = os.path.join(output_folder, output_filename)

                process_and_save_dialogues_with_cluster_linkage(
                    clusters_data=cluster_data,
                    output_file=output_file_path,
                    temperature=temp
                )

    print(f"Total time: {time.time() - start_time:.2f} seconds")

input_folder = "finalized_clusters/P11"
output_folder = "linkage_P11"
generate_linkage(input_folder, output_folder)
