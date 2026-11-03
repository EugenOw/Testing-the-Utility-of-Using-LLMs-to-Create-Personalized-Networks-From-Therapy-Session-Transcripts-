import time
import re
from vllm import LLM, SamplingParams
import json
import os
from itertools import permutations
import ast


os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"


def chat(prompt, llm):
    sampling_params = SamplingParams(
        max_tokens=52000,
        temperature=0.0
    )
    print("Prompt:", prompt)
    outputs = llm.generate([prompt], sampling_params)
    #print("output before decoding:", outputs)
    response_origin = outputs[0].outputs[0].text.strip()  # Extract response
    print("Decoded output:", response_origin)
    return response_origin

def load_transcripts(transcript_path):
    with open(transcript_path, 'r') as file:
        data = json.load(file)
    return data

def save_classification(classification, session_id, patient_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"cluster_{patient_id}_S{session_id}.json")

    with open(file_path, 'w') as f:
        json.dump({"clusters": classification}, f, indent=2)

    print(f"Classification saved at: {file_path}")

def initialize_model():
    model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,  # Reduce parallelism
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        max_model_len=32672  # Reduce sequence length
    )
    return llm

def generate_prompt_baseline(dialogue_data, process_data):
    # Flatten the list of lists into a single list of strings

    prompt_str = ""
    for dialogue in dialogue_data:
        speaker = dialogue["speaker"]
        text = dialogue["text"]
        prompt_str += f"{speaker}: {text}\n"


    # Include only one process per cluster in the formatted output
    included_clusters = set()
    formatted_data_process_data = ""
    for process in process_data:

        process_text = process["semantic_cluster"]

        formatted_entry = f"""
    - {{Process: \"{process_text}\"}}

    """
    formatted_data_process_data += formatted_entry

    prompt = f"""
You are a clinical psychologist analyzing a therapy session transcript.
Your task is to perform a full, end-to-end clinical reasoning analysis based on a therapy dialogue and a list of psychological processes.

**Definitions:**
Psychological Process: A dynamic psychological mechanism that influences emotions, toughts, and behavior.
Clinically Meaningful Theme: A meaningful pattern across psychological processes that reflects underlying psychological functions or dynamics within the client’s experience.
Process Relationship:
- Connection: Connection: Indicates whether a meaningful relationship exists between two psychological processes (1 = yes, 0 = no).
- Relationship Type:
    - Excitatory: One clinically meaningful theme amplifies or reinforces another (e.g Theme A leads to Theme B, Theme A causes Theme B).
    - Inhibitory: One clinically meaningful theme suppresses or counteracts another (e.g Theme A stops Theme B, Theme A diminishes Theme B). 
- Strength of Relationship:
    - Strong: The two processes are closely and consistently related across contexts. One strongly influences or is inherently linked to the other in psychological functioning
    - Moderate: The two processes are meaningfully related, but the connection is less consistent or conditional. They are associated in many situations, but not tightly bound.
    - Weak: The relationship between the two processes is minimal, indirect, or highly context-dependent. They may co-occur at times, but the link is loose or peripheral. 
    
Input:
- Transcript: A therapy session dialogue (context only, do not extract utterances)
- Psychological Processes: A list of psychological processes     

Your Goals
1) Theme Generation:
Generate a list of clinically meaningful themes that capture psychologically relevant patterns across the provided processes.
- Themes must be specific, descriptive, and reflect underlying dynamics.
- Avoid generic or overly broad language.
- Do not include placeholders like "theme 1", "theme A" or reuse example themes   

2) Process Classification:
Assign each psychological process to one or more of the generated themes.
- Each process must belong to at least one theme.
- Each theme must include at least two processes.
- Use frequency as an indicator of importance, but do not overfit.
- Avoid creating or introducing new processes.

3) Inter-Theme Relationships:
For every pair of generated themes, determine whether a relationship exists, and if so:
- Specify the type (excitatory or inhibitory).
- Specify the strength (strong, moderate, or weak).
- Provide a concise explanation of the relationship between the two themes.

Output Format return your entire output as a single, valid JSON object structured like this:
{{
  "classified_processes": {{
    "Theme 1": {{
      "Title": "Theme 1",
      "Processes": [
        {"Process": "Process A", "Frequency": X},
        {"Process": "Process B", "Frequency": Y}
      ]
    }},
    "Theme 2": {
      "Title": "Theme 2",
      "Processes": [
        {"Process": "Process C", "Frequency": Z},
        {"Process": "Process D", "Frequency": W}
      ]
    }
  }},

  "theme_relationships": [
    {
      "input_themes": ["Theme 1", "Theme 2"],
      "connection": [1],
      "relationship_type": ["excitatory" | "inhibitory"],
      "strength_of_relationship": ["strong" | "moderate" | "weak"],
      "explanation": "Concise explanation of how theme 1 influence theme 2"
    },
    {
      "input_themes": ["Theme 1", "Theme 3"],
      "connection": [0]
    }
    // Continue for all pairs of themes
  ]
}}

Now begin your analysis.

Trancript:
{prompt_str}

Processes:
{formatted_data_process_data}

Only return your response as single complete JSON file using the structure above. Do not include commentary or multiple JSON blocks.

"""
    return prompt




def generate_prompt_with_context_old(process_data):
    formatted_data = ""
    for process in process_data:
        process_text = process["process"]
        context_before = "\n".join(process["context_before"])
        context_after = "\n".join(process["context_after"])

        # Format the process and context into a dialogue structure
        formatted_entry = f"""
This is **our main utterance**:  {process_text}
Dialogue:
{context_before}
{process_text}  (**our main utterance**)
{context_after}

"""
        formatted_data += formatted_entry

    prompt = f"""
Task:
You are a clinical psychologist. Below is a set of utterances from a therapy session, each presented along with its surrounding dialogue for context.\n

Your Task:\n
1. **Cluster the main utterances** into distinct, clinically meaningful groups based on thematic relevance.\n
2. Assign utterances to multiple clusters **only if they are highly relevant to more than one theme**.\n
3. Provide a **clear, concise, and descriptive title** for each cluster that reflects its overall theme.\n

Guidelines:\n
- The clusters should be cohesive and primarily non-overlapping. However, an utterance may belong to multiple themes if strongly related.\n
- Titles must be specific and informative, capturing the essence of the cluster's theme.\n
- **Do not cluster the context**; it is provided solely to help you understand the utterance in its dialogue setting.\n

Output Format:\n
Cluster 1: [Title of the Cluster]
- Utterance 1
- Utterance 2
...

Cluster 2: [Title of the Cluster]
- Utterance 1
- Utterance 2
...

Utterances and Context:\n
{formatted_data}

Provide the output in the specified format without additional explanations or modifications to the provided output.
"""
    return prompt

def main():
    llm = initialize_model()

    transcript = load_transcripts("minibatch/p11_dialog.json")  # Load transcript file
    process_data = load_transcripts("minibatch/p11.json")

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
                        # print(f"  Checking process_id={process_id}...")
                        pid_parts = process_id.split("_")
                        # print(pid_parts)
                        proc_patient_id, proc_session_str, proc_process_str = pid_parts

                        try:
                            proc_session_num = int(proc_session_str)
                        except ValueError:
                            print(f"  Cannot parse session number from {proc_session_str}. Skipping.")
                            continue

                        print(f"  Checking process_id={process_id}: patient={proc_patient_id}, session_num={proc_session_num} vs. target_session={session_number_int}")
                        if proc_session_num == session_number_int:
                            found_session = True
                            print(f" Matched patient ID: {patient_id_dialogue} with {patient_id} (case-insensitive).")

                            for session in sessions:
                                for pocess in session:
                                    process_id = pocess["process_id"]
                                    pid_parts = process_id.split("_")
                                    # print(pid_parts)
                                    proc_patient_id, proc_session_str, proc_process_str = pid_parts

                                    try:
                                        proc_session_num = int(proc_session_str)
                                    except ValueError:
                                        print(f"  Cannot parse session number from {proc_session_str}. Skipping.")
                                        continue

                                    print(f"  Checking process_id={process_id}: patient={proc_patient_id}, session_num={proc_session_num} vs. target_session={session_number_int}")

                                    if  proc_session_num == session_number_int:
                                        found_session = True
                                        print(f"    -> Session match found. Generating baseline...")


                                    baseline_prompt = generate_prompt_baseline(dialogue_data, session)
                                    baseline_str = chat(baseline_prompt, llm)

                                    try:
                                        generated_baseline = json.loads(baseline_str)
                                    except json.JSONDecodeError as e:
                                        print(f"Failed to parse generated labels JSON: {e}")
                                        print(f"In {session_id}_{patient_id}")
                                        generated_baseline = baseline_str

                                    save_classification(generated_baseline, proc_session_num, proc_patient_id)
                                    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


                                    with open(f"Baseline_P{proc_session_num}_S{proc_patient_id}", 'a') as file:
                                        file.write(f"Session ID: {session_id}\n")
                                        file.write(f"Generated Baseline: {generated_baseline}\n")
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


start_time = time.time()
main()
end_time = time.time()
print(f"Time taken to cluster sentences: {end_time - start_time} seconds")
