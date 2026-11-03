#!pip install -qU langchain-openai
#!pip install langchain-community
import datetime
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
import os
import json
from itertools import permutations
import re
from pydantic import BaseModel, Field
#!pip install --force-reinstall -v "openai==1.55.3"

class AnyOpenAILLM:
    def __init__(self, **kwargs):
      deployment = 'gpt-4o-mini'
      os.environ["AZURE_OPENAI_API_KEY"] = "nothing to see here"
      os.environ["AZURE_OPENAI_ENDPOINT"] = ("nothing to see here")

      self.model = AzureChatOpenAI(openai_api_version="2024-08-01-preview", azure_deployment=deployment, **kwargs['model_kwargs'])

def lmcall(temperature, prompt):
  llm = AnyOpenAILLM(model_name='gpt-4o-mini', model_kwargs = {'temperature': temperature})
  with get_openai_callback() as cb:
      generations_path = 'generations.txt'
      with open(generations_path, 'a') as generations_file: #saves output of models to a file, this way you have a history of everything and you don't need to rerun the same queries twice
      #(better to save this in google drive and not locally)
        generations_file.write("<PROMPT>"+str(prompt))
        generations_file.write("<MODEL>"+str('gpt-4o-mini'))
        response = llm.model.invoke(prompt)
        generations_file.write("<RESPONSE>"+str(response)+'\n')
        cost_path = 'cost_pbt.txt'
      with open(cost_path, 'a') as cost_file: #saves cost per call, you can run one call and multiply the cost by the total number of instances (to estimate future costs)
      #(better to save this in google drive and not locally)
        cost_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{cb.total_cost}\n")
      return response

# ---------- My functions-------
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
You are a clinical psychologist and your task is to identify relationships between psychological processes.

    Task: Your task is to determine whether a relationship exists between **Process A** and **Process B**, and to generate information on the nature of that relationship.

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
        - Given process A and process B, determine whether or not a relationship exists between two processes.
        - Provide a **brief explanation (a few words)** of the connection between Process A and Process B. Avoid restating the processes themselves.
        - Do not change the the provided process
        
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


def generate_linkage_between_pairs_new(process_1, process_2):
    print("Here is process one:", process_1)
    process_1_str = ""
    process_1_name = process_1["name"]
    process_1_str += f"{process_1_name}\n"
    print("Here is process_1_str", process_1_str)
    # for process in process_1["processes"]:
    #    process_1_str += f"- {process}\n"

    process_2_str = ""
    process_2_name = process_2["name"]
    process_2_str += f"{process_2_name}\n"
    # for process in process_2["processes"]:
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

    return prompt

def read_sentences_from_file_old(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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


from tqdm import tqdm
from itertools import permutations
import os
import json

def process_and_save_dialogues_with_cluster_linkage(clusters_data, output_file, temperature):
    all_relationships = []
    failed_outputs = []
    clusters = clusters_data["clusters"]

    # Step 1: generate all permutations and assign stable labels
    ordered_pairs = list(permutations(clusters, 2))
    pair_to_label = {}

    # Step2: sort based on cluster names to assign stable IDs
    sorted_pairs = sorted(ordered_pairs, key=lambda pair: (pair[0]['name'], pair[1]['name']))
    for idx, (c1, c2) in enumerate(sorted_pairs, start=1):
        key = (c1['name'], c2['name'])
        pair_to_label[key] = idx

    # Add tqdm here to track progress
    for cluster1, cluster2 in tqdm(ordered_pairs, desc="Processing cluster pairs", unit="pair"):
        label = pair_to_label[(cluster1['name'], cluster2['name'])]
        print(f"[{label}] Finding relationship between: '{cluster1['name']}' ↔ '{cluster2['name']}'")

        prompt = generate_linkage_between_pairs_new(cluster1, cluster2)
        resposne = lmcall(temperature, prompt)
        raw_content = resposne.content

        if raw_content.startswith("json\n"):
            raw_content = raw_content[5:]

        json_part = {}
        if raw_content:
            try:
                json_part = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print("JSON decode failed:", e)
                print("Extracted JSON string:", raw_content)
        else:
            print("No valid JSON found in output.")

        if json_part.get("relationship"):
            for rel in json_part["relationship"]:
                if isinstance(rel, dict):
                    rel["pair_label"] = label
                    all_relationships.append(rel)
                else:
                    all_relationships.append({
                        "relationship": rel,
                        "pair_label": label
                    })
        else:
            print(f"[{label}] Json Structure is wrong.")
            failed_outputs.append({
                "label": label,
                "cluster_pair": (cluster1["name"], cluster2["name"]),
                "raw_output": json_part
            })

    final_output = {"relationships": all_relationships}
    with open(output_file, 'w') as file:
        json.dump(final_output, file, indent=2)

    print(f"Saved {len(all_relationships)} relationships to {output_file}")

    if failed_outputs:
        failed_file = f"failed_output_{output_file}.json"
        os.makedirs(os.path.dirname(failed_file), exist_ok=True)
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_outputs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(failed_outputs)} failed outputs to: {failed_file}")



#example =  read_sentences_from_file("output/presentation_examples_real.json")

#process_and_save_dialogues_with_cluster_linkage(example, "chatgpt_output/presentation.json", temperature=0)


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    for filename in tqdm(json_files, desc="Processing files", unit="file"):
        input_file_path = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]
        print(f"\nProcessing file: {input_file_path}")
        cluster_data = read_sentences_from_file(input_file_path)

        print(f"Running GPT for file: {filename}")
        output_file = os.path.join(output_folder, f"{base_filename}.json")
        process_and_save_dialogues_with_cluster_linkage(cluster_data, output_file, temperature=0)


input_folder = "finalized_clusters_v2/P16"
output_folder = "output/linkage_gpt_p16"

main(input_folder, output_folder)