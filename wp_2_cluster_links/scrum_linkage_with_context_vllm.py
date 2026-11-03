import time
import re
from vllm import LLM, SamplingParams
import json
import os
from itertools import permutations
import ast

os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"
# Define the model path
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"

# Initialize the LLM using vLLM with tensor parallelism (for using two GPUs)
start_time = time.time()
llm = LLM(
    model=model_path,
    tensor_parallel_size=2,  # Reduce parallelism
    dtype="bfloat16",
    gpu_memory_utilization=0.98,
    max_model_len=32672  # Reduce sequence length
)
end_time = time.time()
print(f"Time taken to load the model: {end_time - start_time} seconds")


def generate_linkage_between_paris(process_1, process_2):

    process_1_str = ""
    process_1_name = process_1["name"]
    process_1_str += f"{process_1_name}\n"
    #for process in process_1["processes"]:
    #    process_1_str += f"- {process}\n"

    process_2_str = ""
    process_2_name = process_2["name"]
    process_2_str += f"{process_2_name}\n"
    #for process in process_2["processes"]:
    #    process_2_str += f"- {process}\n"

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
    return

    return prompt

def cut_after_first_full_json(text: str) -> str | None:
    """Returns a string containing the first complete JSON object from text."""
    start = text.find('{')
    if start == -1:
        return None  # No JSON found

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None  # No valid closing



def linkage(prompt, temperature):
    sampling_params = SamplingParams(
        max_tokens=150,
        temperature=temperature
    )
    outputs = llm.generate([prompt], sampling_params)
    decoded_output = outputs[0].outputs[0].text.strip()
    # Cut the string after first complete JSON block
    json_string = cut_after_first_full_json(decoded_output)


    if json_string:
        try:
            parsed_json = json.loads(json_string)
            return parsed_json, decoded_output
        except json.JSONDecodeError as e:
            print("JSON decode failed:", e)
            print("Extracted JSON string:\n", json_string)
    else:
        print("No valid JSON found in output.")

    print("Raw model output:\n", decoded_output)
    return None, decoded_output



def process_and_save_dialogues_with_cluster_linkage(clusters_data, output_file, temperature):
        mismatch_log = []
        all_relationships = []
        failed_outputs = []
        print(clusters_data)
        clusters = clusters_data["clusters"]

        # Step 1: generate all permutations and assign stable labels
        ordered_pairs = list(permutations(clusters, 2))
        pair_to_label = {}

        # Step2: sort based on cluster names to assign stable IDs
        sorted_pairs = sorted(ordered_pairs, key=lambda pair: (pair[0]['name'], pair[1]['name']))
        for idx, (c1, c2) in enumerate(sorted_pairs, start=1):
            key = (c1['name'], c2['name'])
            pair_to_label[key] = idx

        # TODO Add adtionaly a if question if the input cluster names are the same in the output if not then switch them with the realones
        for cluster1, cluster2 in ordered_pairs:
            label = pair_to_label[(cluster1['name'], cluster2['name'])]
            print(f"[{label}] Finding relationship between: '{cluster1['name']}' ↔ '{cluster2['name']}'")

            prompt = generate_linkage_between_paris(cluster1, cluster2)
            parsed_output, raw_output = linkage(prompt, temperature=temperature)

            if parsed_output and parsed_output.get("relationship"):
                for rel in parsed_output["relationship"]:
                    if isinstance(rel, dict):
                        expected_inputs = [cluster1["name"], cluster2["name"]]
                        actual_inputs = rel.get("input_processes", [])

                        if actual_inputs != expected_inputs:
                            mismatch_log.append({
                                "label": label,
                                "expected": expected_inputs,
                                "model_returned": actual_inputs,
                                "raw_output": raw_output
                            })
                            # Force correction
                            rel["input_processes"] = expected_inputs

                        rel["pair_label"] = label
                        all_relationships.append(rel)
            else:
                print(f"[{label}] No relationships found or JSON parsing failed.")
                failed_outputs.append({
                    "label": label,
                    "cluster_pair": (cluster1["name"], cluster2["name"]),
                    "raw_output": raw_output
                })


        final_output = {"relationships": all_relationships}
        with open(output_file, 'w') as file:
            json.dump(final_output, file, indent=2)

        print(f"Saved {len(all_relationships)} relationships to {output_file}")

        # Save failed outputs if any
        if failed_outputs:
            failed_file = output_file.replace(".json", "_failed.json")
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_outputs, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(failed_outputs)} failed outputs to: {failed_file}")

        # Save mismatches to a debug file
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


# input file = input cluster file
# output_dir = folder to save outputs
#base file = base filename for each run

def run_pipeline_with_temperatures(input_file, output_folder, base_filename, temperatures):
    os.makedirs(output_folder, exist_ok=True)
    cluster_data = read_sentences_from_file(input_file)

    for temp in temperatures:
        print(f"\n Running model with temperature: {temp}")
        output_file = os.path.join(output_folder, f"{base_filename}_tmp_{temp}.json")
        process_and_save_dialogues_with_cluster_linkage(cluster_data, output_file, temperature=temp)


def save_classification(classification, session_id, patient_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"classified_labels_{session_id}_{patient_id}.json")

    with open(file_path, 'w') as f:
        json.dump({"clusters": classification}, f, indent=2)

    print(f"Classification saved at: {file_path}")

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if isinstance(data.get("clusters"), dict):  # New format with Theme keys
        transformed_clusters = [
            {"name": theme_obj["Theme"]}
            for theme_obj in data["clusters"].values()
            if "Theme" in theme_obj
        ]
        return {"clusters": transformed_clusters}
    else:
        return data  # Already in the old format


def generate_linkage(input_folder, output_folder):
    start_time = time.time()
    temp_list = [0.0, 0.3, 0.6]

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            print(f"\nProcessing file: {input_file_path}")
            cluster_data = read_sentences_from_file(input_file_path)

            for temp in temp_list:
                print(f"\nRunning model with temperature: {temp} for file: {filename}")

                # Construct output file paths
                temp_str = f"{temp:.1f}"
                output_filename = f"{base_filename}_temperature_{temp_str}.json"
                failed_filename = f"{base_filename}_temperature_{temp_str}_failed.json"

                output_file_path = os.path.join(output_folder, output_filename)
                failed_file_path = os.path.join(output_folder, failed_filename)

                # Process and save
                process_and_save_dialogues_with_cluster_linkage(
                    clusters_data=cluster_data,
                    output_file=output_file_path,
                    temperature=temp
                )

    print(f"Total time: {time.time() - start_time:.2f} seconds")


input_folder = "finalized_clusters/P13"
output_folder = "linkage_P13"
generate_linkage(input_folder, output_folder)
