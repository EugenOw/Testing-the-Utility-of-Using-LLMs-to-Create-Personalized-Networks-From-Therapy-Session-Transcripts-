import time
import re
from vllm import LLM, SamplingParams
import json
import os
import csv

def cluster_sentences_old(process_data):
    start_time = time.time()
    prompt = generate_prompt_with_only_dialog(process_data)
    print(prompt)
    end_time = time.time()
    print(f"Time taken to load the model: {end_time - start_time} seconds")
    try:
        sampling_params = SamplingParams(
            max_tokens=24000,
            temperature=0.1,
            top_k=50,
            top_p=.09
        )
        start_time = time.time()
        # Run inference with vLLM
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()
        print(f"Time taken to generate response: {end_time - start_time} seconds")

        # Extract the first output
        start_time = time.time()
        decoded_output = outputs[0].outputs[0].text.strip()
        end_time = time.time()
        print(f"Time taken to decode response: {end_time - start_time} seconds")

        # Extract and organize clusters
        clusters = re.findall(r"Cluster \d+:.*?(?=(Cluster \d+:|$))", decoded_output, re.DOTALL)
        parsed_clusters = {i + 1: cluster.strip() for i, cluster in enumerate(clusters)}
        return parsed_clusters, decoded_output

    except Exception as e:
        print(f"Error processing sentences: {e}")
        return None

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


def generate_prompt_with_only_dialog(dialogue_data):
    prompt_str = ""
    for dialogue in dialogue_data:
        print(dialogue)
        speaker = dialogue["speaker"]
        text = dialogue["text"]
        prompt_str += f"{speaker}: {text}\n"

    prompt = f"""
**Task:**
You are a clinical psychologist analyzing a therapy session transcript.\n

**Your Task:**
1. **Identify and cluster** the utterances that are labeled with:
   - `[process freq=X]` at the beginning
   - `[/process]` at the end
2. Consider the **freq** (X) for each utterance as an indicator of how often it appeared in a similar form within the dialogue. 
   - A higher frequency suggests the utterance or theme is more important.
3. Provide **clear, concise titles** for each cluster that reflect clinically meaningful themes.


Guidelines:\n
- The entire dialogue is provided for context.
- **Only include utterances that appear between the labels** **[process freq=X] ... [/process]**.\n
- The clusters should be cohesive and primarily non-overlapping. However, an utterance may belong to multiple themes if strongly related.\n
- Titles should be specific, clinically relevant and informative, capturing the essence of the cluster theme.\n
- **Do not modify the original utterances**.
- **Use frequency** (X) to determine significance but remember all labeled utterances should still be clustered.

The Output Format:\n
Cluster 1: [Title of the Cluster]\n
- Utterance 1 [Freq = X] \n
- Utterance 2 [Freq = Y] \n
...

Cluster 2: [Title of the Cluster]\n
- Utterance 1 [Freq = Z] \n
- Utterance 2 [Freq = N] \n
...

Full Dialogue (Find and Cluster the Marked Utterances):
{prompt_str}


Provide the output in the specified format without additional explanations or modifications to the provided output.
"""
    return prompt


def generate_prompt_dialogue_and_nodes_old_v2(dialogue_data, process_data):
    # Flatten the list of lists into a single list of strings

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

    prompt = f"""
**Task:**
You are a clinical psychologist analyzing a therapy session transcript.

**Definitions:**
- **Process**: A dynamic psychological mechanism that influences emotions, thoughts, and behavior, affecting mental health and therapy outcomes.

**Input:** Includes transcript of a therapy session dialogue and a list of psychological processes.\n

**Your Goal:**
1. Review the provided dialogue for overall context. Do **not** extract utterances from the dialogue.
2. Cluster all listed processes into **distinct, clinically meaningful groups** based on thematic relevance.
3. Consider the **freq** (X) for each process as an indicator of how often it appeared in a similar form within the dialogue. 
   - A higher frequency suggests the process or theme is more important.
4. Provide **clear, concise titles** for each cluster, reflecting clinically meaningful themes.

**Guidelines:**
- Use only the processes listed under "Processes to be Clustered".
    - Do not include any new processes from the dialogue.
    - Do not omit any listed processes. Every process *must appear in at least one cluster*.
    - Considering also their frequency   
- Use the full dialogue **only** for context. You do **not** need to label or cluster the dialogue.
- A process may belong to multiple clusters only if it is strongly relevant to each theme.
    - Avoid excessive duplication—if a process appears in multiple clusters, it must have a distinct, meaningful connection to each.
- Titles must be **specific and descriptive**, capturing each cluster's essence.
- Ensure each cluster is meaningfully distinct rather than repeating the same processes under different labels.
- **Do not alter the original text** of the Processes in any way.


Output Structure: 
```json
{{
    "Theme 1": {
    "Title": "[Theme Title]",
        "Processes": [
            {"Process": "Process 1", "Frequency": X },
            {"Process": "Process 2", "Frequency": X }
        ]
    },
    "Theme 2": {
    "Title": "[Theme Title]",
        "Processes": [
            {"Process": "Process 1", "Frequency": Z },
            {"Process": "Process 2", "Frequency": N }
        ]
    }
}}
```

**Full Dialogue (Context Only!):**
{prompt_str}

**Listed Process to be Clustered:**
{formatted_data_process_data}

Generate the output strictly in the specified format without additional explanations an  no modifications to the provided output.
"""
    return prompt

def generate_prompt_dialogue_and_nodes_old_second_phase(dialogue_data, process_data):
    # Flatten the list of lists into a single list of strings

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

    prompt = f"""
    
You are a clinical psychologist analyzing a therapy session transcript. 
**Task:** Your task is to generate cluster with clinically meaningful themes with the provided processes.

**Definitions:**
- **Process**: A dynamic psychological mechanism that influences emotions, thoughts, and behavior, affecting mental health and therapy outcomes.
- **Clinically meaningful theme**: It represents a meaningful pattern across psychological processes that reflects underlying functions or dynamics within the client's experience.
                                    
**Input:**
- A transcript excerpt from a therapy session (for context only)
- A list of psychological processes.  

**Your Goal:**
1. Review the provided dialogue for overall context. Do **not** extract utterances from the dialogue.  
2. Cluster all listed processes into **distinct, clinically meaningful themes** based on thematic relevance.  
3. Consider the **freq** (X) for each process as an indicator of how often it appeared in a similar form within the dialogue.  
   - A higher frequency suggests the process or theme is more important.

**Guidelines for Clustering:**
- Use **only** the processes listed under "Processes to be Clustered".
  - Do **not** include any new processes from the dialogue.
- Every process **must appear in at least one cluster**.
- **Each theme must include at least two processes.**
- A process may belong to **multiple clusters** only if it is strongly relevant to each theme.
- Do not generate one big cluster with all processes


The following are **inspiration-only** themes to help you understand the type of output we aim for. 
**Do not reuse, rephrase, or incorporate these into your answer.**

Poor/Invalid Themes (Too broad, external, or unfocused) with explanation of why:
- "Professional Networking and Collaboration" -> Focuses on external activities, not internal processes or functions.
- "Breathing and Behavioral Patterns" -> Groups physical and behavioral elements without a unifying psychological mechanism.
- "Therapy and Reflection" -> Describes the setting or activity it is not a meaningful psychological theme.
- "Decision Making and Consequences" -> Overly broad and generic it lacks a shared internal function.
- "Reflection and Understanding" -> No specific mechanism or insight about how the processes function together.


Valid Inspiration-Only Themes (Do not reuse!):
- "Self confidence tied to competence in work role"
- "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others"
- "Evaluating thoughts to identify any cognitive distortions"
- "Assessing impact of reframed/alternate thought after identifying cognitive distortion"
- "Recognizing the role of thoughts and behaviors in emotional experiences"



Output Structure: 
```json
{{
    "Theme 1": {{
        "Theme": "[Theme 1]",
        "Processes": [
            {{ "Process": "Process 1", "Frequency": X }},
            {{ "Process": "Process 2", "Frequency": X }}
        ]
    }},
    "Theme 2": {{
        "Theme": "[Theme 2]",
        "Processes": [
            {{ "Process": "Process 1", "Frequency": Z }},
            {{ "Process": "Process 2", "Frequency": N }}
        ]
    }}
}}
```

**Full Dialogue (Context Only!):**
{prompt_str}

**Listed Process to be Clustered:**
{formatted_data_process_data}

Now, generate the clinically meaningful clusters with **all listed processes** in the specified format without additional explanations an no modifications to the provided output.
"""
    return prompt


def generate_prompt_dialogue_and_nodes(dialogue_data, process_data):
    # Flatten the list of lists into a single list of strings

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

    prompt = f"""

You are a clinical psychologist analyzing a therapy session transcript. 
**Task:** Your task is to group the provided psychological processes into clusters and generate a clear, one-sentence clinical description for each cluster.

**Definitions:**
- **Process**: A dynamic psychological mechanism that influences emotions, thoughts, and behavior, affecting mental health and therapy outcomes.
- **Clinically meaningful theme**: It represents a meaningful pattern across psychological processes that reflects underlying functions or dynamics within the client's experience.

**Input:**
- A transcript excerpt from a therapy session (for context only)
- A list of psychological processes.  

**Your Goal:**
1. Read the transcript excerpt to understand the emotional and interpersonal context. Do not quote or extract specific utterances. 
2. Analyze the list of psychological processes and organize them into thematically distinct, clinically meaningful clusters and provide a clinically meaningful theme.

**Guidelines for Writing Themes:**
- Each theme must be a complete, **short**, concise statement.
    - You will find under "Example" how good and bad themes look like
    - Do not start with "The client is ..." or "Exploring ..."
- Avoid generalities, labels that are not clinically informative.
- The **short** sentence must convey a specific psychological function, conflict, or transformation linking the processes in the cluster.
    - A strong theme goes beyond naming a topic. 
    - It provides clinical insight into how or why the processes are showing up and what they reflect in the client’s inner world.
- Do not simply list or rephrase the processes in the sentence.

**Guidelines for Clustering:**
- Use **only** the processes listed under "Listed Process to be Clustered".
- Every process **must appear at least in one cluster**.
- A process may belong to **multiple clusters** only if it is strongly relevant to each theme.
- Each cluster must:
    - Contain two or more processes.
    - Be grounded in what was clinically significant in the session (i.e., reflect key emotional or psychological dynamics the client is working through).
- Do not generate one big cluster.    
- Together, the clusters should cover the major psychological content and dynamics discussed in the transcript therapy session.

Example: The following are examples of good and bad themes. 
- "Self confidence tied to competence in work role" -> good
- "Professional Networking and Collaboration" -> bad
- "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others" -> good
- "Emotional regulation and self-compassion" -> bad
- "Evaluating thoughts to identify any cognitive distortions" -> good
- "Breathing and Behavioral Patterns" -> bad
- "Assessing impact of reframed/alternate thought after identifying cognitive distortion" -> good
- "Therapy and Reflection" -> bad
- "Decision Making and Consequences" -> bad
- "Recognizing the role of thoughts and behaviors in emotional experiences" -> good
- "Reflection and Understanding" -> bad
- "Emotional Intimacy and Relationships" -> bad
- "Difficulty with focus and concentration due to trauma and ADHD-like symptoms" -> good
- "Tension between value of achievement and family obligations" -> good
- "Therapy and Reflection" -> bad
- "Decision Making and Consequences" -> bad
- "Exploring the Impact of Surprises on Relationships" -> bad

**Do not reuse, rephrase, these examples**


Here is the Output Structure: 
json
{{"Theme 1": {{
"Theme": "[Full, clinically meaningful short sentence here]",
"Processes": [
{{ "Process": "Process 1", "Frequency": X }},
{{ "Process": "Process 2", "Frequency": X }} ]}},
"Theme 2": {{
"Theme": "[Full, clinically meaningful short sentence here]",
"Processes": [{{ "Process": "Process 1", "Frequency": Z }},
{{ "Process": "Process 2", "Frequency": N }}]}}
}}


**Transcript Therapy Session (Context Only!):**
{prompt_str}

**Listed Process to be Clustered:**
{formatted_data_process_data}

Now, generate clusters with clinically meaningful themes with **all listed processes** in the specified format without additional explanations an no modifications to the provided output. Stick to the Guidelines.
"""
    return prompt



def generate_prompt_dialogue_and_nodes_with_dimension(dialogue_data, process_data):
    # Flatten the list of lists into a single list of strings

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
            dimension = process["dimensions"]
            frequency = cluster_counts[cluster_label]
            formatted_entry = f"""
    - {{Process: \"{process_text}\", frequency: {frequency}, dimension: {dimension}}}

    """
            formatted_data_process_data += formatted_entry
            included_clusters.add(cluster_label)

    prompt = f"""
**Task:**
You are a clinical psychologist analyzing a therapy session transcript.

**Definitions:**
- **Process**: A dynamic psychological mechanism that influences emotions, thoughts, and behavior, affecting mental health and therapy outcomes.
- **Dimension**: A broad category representing different aspects of psychological function (e.g., cognitive, emotional, behavioral), within which processes operate.

**Your Goal:**
1. Review the provided dialogue for overall context.
2. Cluster the listed **processes** into distinct, clinically meaningful groups** based on thematic relevance.
3. Consider Frequency (Freq = X):
   - A higher frequency indicates a process appears more often and may be more significant.
4. Title Each Cluster Clearly, ensuring they accurately reflect the clinical themes.

**Guidelines:**
- Use the full dialogue only for context—do not label or cluster the entire dialogue.
- Cluster only the processes listed under "Processes to be Clustered," considering their frequency.
- A Process may belong to multiple clusters if **they strongly** fit more than one theme.
- Titles must be specific and descriptive, capturing each cluster’s essence.
- **Do not alter the original utterance** of the process in any way.
- Do NOT use Dimension for clustering. It is included in the output only for reference but should have zero impact on clustering decisions.


The Output Format:\n
Cluster 1: [Title of the Cluster]\n
- Process 1 [Freq = X, Dimension = L] \n
- Process 2 [Freq = X, Dimension = S] \n
...

Cluster 2: [Title of the Cluster]\n
- Process 1 [Freq = Z, Dimension = L] \n
- Process 2 [Freq = N, Dimension = L] \n
...

 **Full Dialogue (Context Only):**
{prompt_str}

**Processes to be Clustered:**
{formatted_data_process_data}

Provide the output in the specified format without additional explanations or modifications to the provided output.
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

def cluster_sentences(prompt):
    sampling_params = SamplingParams(
        max_tokens=24000,
        temperature=0.0
    )
    outputs = llm.generate([prompt], sampling_params)
    decoded_output = outputs[0].outputs[0].text.strip()
    print("Decoded output:", decoded_output)
    clusters = re.findall(r"Cluster \d+:.*?(?=(Cluster \d+:|$))", decoded_output, re.DOTALL)
    parsed_clusters = {i + 1: cluster.strip() for i, cluster in enumerate(clusters)}
    return parsed_clusters, decoded_output

def process_and_save_dialogues(dialogues, output_file):
    with open(output_file, 'w') as file:
        for item in dialogues:
            dialogue_data = item["dialogues"]
            prompt = generate_prompt_with_only_dialog(dialogue_data)
            print(prompt)
            cluster, answer_model = cluster_sentences(prompt)
            file.write(f"{item['session']}:\n")
            file.write(answer_model)
            file.write("\n\n")


def process_and_save_dialogues_with_nodes(dialogues, process_data, output_file, using_dimension):
    for item in dialogues:
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
                        #print(f"  Checking process_id={process_id}...")
                        pid_parts = process_id.split("_")
                        #print(pid_parts)
                        proc_patient_id, proc_session_str, proc_process_str = pid_parts

                        try:
                            proc_session_num = int(proc_session_str)
                        except ValueError:
                            print(f"  Cannot parse session number from {proc_session_str}. Skipping.")
                            continue

                        print(f"  Checking process_id={process_id}: patient={proc_patient_id}, session_num={proc_session_num} vs. target_session={session_number_int}")

                        if proc_session_num == session_number_int:
                            found_session = True
                            print(f"    -> Session match found. Generating prompt...")

                            if using_dimension:
                                prompt = generate_prompt_dialogue_and_nodes_with_dimension(dialogue_data, session)
                            else:
                                prompt = generate_prompt_dialogue_and_nodes(dialogue_data, session)

                            print(f"    Generated prompt: {prompt}")

                            cluster, answer_model = cluster_sentences(prompt)


                            with open(output_file, 'a') as file:
                                file.write(f"{session_id}:\n")
                                file.write(answer_model)
                                file.write("\n\n")

                            save_classification(answer_model, session_id, patient_id)


                            break
                    if found_session:
                        break
                if found_session:
                    break

        if not found_patient:
            print(f"  No patient match for {patient_id_dialogue} in process_data.")
        elif not found_session:
            print(f"  No matching session number {session_number_int} for patient {patient_id_dialogue} in process_data.")

    return None


def save_classification(classification, session_id, patient_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"cluster_{session_id}.json")

    with open(file_path, 'w') as f:
        json.dump({"clusters": classification}, f, indent=2)

    print(f"Classification saved at: {file_path}")

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

start_time = time.time()

# ---------Generating Cluster with Nodes without Dimension---------
#process_data = read_sentences_from_file('Second_Phase_Clustering.json')
#no_modfied_dialog_data = read_sentences_from_file('Second_Phase_no_modified_dialog.json')
#output_file2 = 'Second_Phase_Clustering'
#process_and_save_dialogues_with_nodes(no_modfied_dialog_data, process_data, output_file2, using_dimension=False)


#process_data = read_sentences_from_file('First_phase_clustering.json')
#no_modfied_dialog_data = read_sentences_from_file('no_modified_dialog.json')
#output_file2 = 'Fist_Phase_Finalized'
#process_and_save_dialogues_with_nodes(no_modfied_dialog_data, process_data, output_file2, using_dimension=False)


process_data = read_sentences_from_file('data/Third_Phase_Clustering.json')
no_modfied_dialog_data = read_sentences_from_file('data/Third_Phase_no_modified_dialog.json')
output_file2 = 'Third_Phase_clusters'
process_and_save_dialogues_with_nodes(no_modfied_dialog_data, process_data, output_file2, using_dimension=False)

end_time = time.time()
print(f"Time taken to cluster sentences: {end_time - start_time} seconds")




# Variables for generate_prompt_only_dialog
#process_data = read_sentences_from_file('14_3_process_data.json')
#modified_dialog_data = read_sentences_from_file('modified_dialog.json')
#output_file1 = 'output_only_dialogue.txt'
# Flatten rows and clean up new lines
#process_and_save_dialogues(modified_dialog_data, output_file1)



#Cluster 1: [Title of the Cluster]\n
#- Process 1 [Freq = X] \n
#- Process 2 [Freq = X] \n
#...

#Cluster 2: [Title of the Cluster]\n
#- Process 1 [Freq = Z] \n
#- Process 2 [Freq = N] \n
#...


#**Important Requirements for Clinically Meaningful Titles:**
#- **Titles must be specific, descriptive, and convey psychological insight.**
#  - They must capture **how and why** the processes interact as a title not just what they refer to.
#  - Each title must reflect a **shared underlying psychological function or mechanism**, not just similar content or terminology.
#- **Do NOT create vague or generic title titles.
#- Do **NOT** copy or replicate any example title below. Titles must be original and based on your analysis about the provided Input.