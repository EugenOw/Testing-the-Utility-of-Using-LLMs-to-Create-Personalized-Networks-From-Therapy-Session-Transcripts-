import os
import json
import argparse


def load_generated_labels(label_path):
    with open(label_path, 'r', encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    return labels

def prompt_construct_classify_processes(dialogue_data, process_data, generated_labels):


    prompt_str = ""
    for dialogue in dialogue_data:
        speaker = dialogue["speaker"]
        text = dialogue["text"]
        prompt_str += f"{speaker}: {text}\n"

        # Count the occurrences of each cluster
    cluster_counts = {}
    for process in process_data:
        cluster_label = process["semantic_cluster"]
        if cluster_label in cluster_counts:
            cluster_counts[cluster_label] += 1
        else:
            cluster_counts[cluster_label] = 1

    # Include only one process per cluster in the formatted output
    included_clusters = set()
    formatted_data_process_data = ""
    for process in process_data:
        cluster_label = process["semantic_cluster"]
        if cluster_label not in included_clusters:
            process_text = process["process"]
            frequency = cluster_counts[cluster_label]
            formatted_entry = f"""
            - {{utterance: \"{process_text}\", frequency: {frequency}}}

            """
            formatted_data_process_data += formatted_entry
            included_clusters.add(cluster_label)

    json_format_labels = json.dumps({
        "classified_processes": {
            "Label 1": ["Process A freq (X)", "Process B freq (X)"],
            "Label 2": ["Process C freq (X)", "Process D freq (X)"]
        }
    }, indent=4)

    prompt = f"""
**Task:** You are a clinical psychologist classifying psychological processes based on provided themes.

A clinically meaningful theme:A theme represents a meaningful pattern across psychological processes that reflects underlying functions or dynamics within the client’s experience.

**Input:** 
    1) A transcript excerpt from a therapy session (for context only). 
    2) A list of psychological processes to be clustered. 
    3) A set of themes under which the processesshould be categorized
    
Your goal:
    1) Read the transcript excerpt to understand the emotional and interpersonal context. Do not quote or extract specific utterances.
    2) Classify each listed psychological process under one or more of the provided themes based on their thematic relevance

**Guidelines for Clustering:**
    1. Use only the processes listed under “Listed Process to be Clustered”.
    2. Every process must appear at least in one cluster.
    3. A process may belong to multiple clusters only if it’s strongly relevant to thetheme.
    4. Each provided theme must have at least 2 relevant processes assigned to it.
    5. Do not put all processes in one cluster.
    
Output Structure: 
json
{{"Theme 1": {{
"Theme": "[Provided Theme A]",
"Processes": [
{{ "Process": "Process 2", "Frequency": X }},
{{ "Process": "Process 3", "Frequency": Y }} ]}},
"Theme 2": {{
"Theme": "[Provided Theme B]",
"Processes": [{{ "Process": "Process 1", "Frequency": Z }},
{{ "Process": "Process 4", "Frequency": N }}]}}
}}     
    
**Therapy Session Transcript (Context Only!):**
{prompt_str}
    
**List of Processes with Frequency (To Be Classified):**
{formatted_data_process_data}
    
**Provided Themes:**\n
{generated_labels}

Now, classify all of the given processes into the provided themes. Do not generate any explanation or modification to the provided data. Stick to the Guidelines for Clustering:
"""
    return prompt


def given_label_classification(dialogue_data, process_data, generated_labels):
    classification_prompt = prompt_construct_classify_processes(dialogue_data, process_data, generated_labels)
    return classification_prompt
