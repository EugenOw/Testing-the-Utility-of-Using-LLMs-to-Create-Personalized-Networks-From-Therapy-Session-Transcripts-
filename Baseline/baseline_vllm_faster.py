import time
import re
import json
import os
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

# Environment setup
os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"

# Constants
MAX_BATCH_SIZE = 2  # Tune based on VRAM capacity and prompt length

# Model initialization
def initialize_model():
    model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"
    return LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        max_model_len=32672
    )

# Load JSON files
def load_transcripts(path):
    with open(path, 'r') as f:
        return json.load(f)

# Save output JSON
def save_classification(classification, session_id, patient_id):
    path = os.path.join(os.path.dirname(__file__), f"cluster_{patient_id}_S{session_id}.json")
    with open(path, 'w') as f:
        json.dump({"clusters": classification}, f, indent=2)
    print(f"Saved: {path}")

# Prompt formatting
def generate_prompt_baseline(dialogue_data, process_data):
    prompt_str = "\n".join([f"{d['speaker']}: {d['text']}" for d in dialogue_data])

    formatted_data_process_data = "".join([
        f"\n    - {{Process: \"{p['process']}\"}}\n\n" for p in process_data
    ])

    print(formatted_data_process_data)

    return f"""
**Task:**  You are a clinical psychologist and your task is to perform a full, end-to-end clinical reasoning analysis based on a therapy dialogue and a list of psychological processes.

**Definitions:**
Psychological Process: A dynamic psychological mechanism that influences emotions, toughts, and behavior.
Clinically Meaningful Theme: A meaningful pattern across psychological processes that reflects underlying psychological functions or dynamics within the client’s experience.
Process Relationship:
- Connection: Indicates whether a meaningful relationship exists between two Clinically Meaningful Theme (1 = yes, 0 = no).
- Relationship Type:
    - Excitatory as in one process amplifies or reinforces another process. (e.g. Process A leads to Process B, Process A causes Process B)
    - Inhibitory as in  one process suppresses or counteracts another process. (e.g. Process A stops Process B, Process A diminishes Process B) 
- Strength of Relationship:
    - "Strong": The two themes are closely and consistently related across contexts. One strongly influences or is inherently linked to the other in psychological functioning.
    - "Moderate": The two themes are meaningfully related, but the connection is less consistent or conditional. They are associated in many situations, but not tightly bound.
    - "Weak": The relationship between the two themes is minimal, indirect, or highly context-dependent. They may co-occur at times, but the link is loose or peripheral. 

Input:
- Transcript: A therapy session dialogue (context only, do not extract utterances)
- Psychological Processes: A list of psychological processes     

Guidelines for wiriting Themes: 
(1) Each theme must be a short, complete sentencewith clinical insight. 
(2) The sentence must convey a specific psychological function,conflict, or transformation.

Guidelines for Process Classification:
(1) Classify all listed processes under one or morethemes based on thematic relevance. 
(2) Use only the processes listed in the input.
(3) Each process must appear in at least one theme. 
(4) Each theme must contain atleast two processes.

Inter-Theme Relationships:
(1) Determine whether a relationship existsbetween any two themes. 
(2) Use only the themes to generate relationships. 
(3) Provide a brief explanation (a few words) of the connection. 
(4) Analyze Theme A toTheme B and Theme B to Theme A separately.

Your Goals
(1) Theme Generation
(2) Process Classification
(3) Inter-Theme Relationships

**Examples of Themes:**
(Note: These examples are for understanding **style only**,  you must NOT reuse or adapt them!)
- "Self confidence tied to competence in work role" -> good
- "Professional Networking and Collaboration" -> bad
- "Impact of aspects of identity (professional role, age) on expectations for self and perceived expectations of others" -> good
- "Emotional regulation and self-compassion" -> bad
- "Evaluating thoughts to identify any cognitive distortions" -> good
- "Breathing and Behavioral Patterns" -> bad
- "Assessing impact of reframed/alternate thought after identifying cognitive distortion" -> good
- "Therapy and Reflection" -> bad
- "The client's struggle to articulate their feelings is a manifestation of the unexpressed emotions from their past." -> bad
- "Decision Making and Consequences" -> bad
- "Recognizing the role of thoughts and behaviors in emotional experiences" -> good
- "Reflection and Understanding" -> bad
- "Emotional Intimacy and Relationships" -> bad
- "Difficulty with focus and concentration due to trauma and ADHD-like symptoms" -> good
- "Tension between value of achievement and family obligations" -> good
- "Exploring the Impact of Surprises on Relationships" -> bad
- "The client's self-blame is a coping mechanism for feelings of overhelm and anxiety" -> bad

Output Format:
{{
  "classified_processes": {{
    "Theme 1": {{ "Title": "Theme A", "Processes": [{{"Process": "Process 1"}}, {{"Process": "Process 3"}}] }},
    "Theme 2": {{ "Title": "Theme B", "Processes": [{{"Process": "Process 2"}}] }}
  }},
  "theme_relationships": [
    {{ "input_themes": ["Theme A", "Theme B"], "connection": [1], "relationship_type": ["excitatory"], "strength_of_relationship": ["strong"], "explanation": "..." }},
    {{ "input_themes": ["Theme B", "Theme A"], "connection": [1], "relationship_type": ["inhibitory"], "strength_of_relationship": ["moderate"], "explanation": "..." }}
    {{ "input_themes": ["Theme A", "Theme C"], "connection": [0] }}
  ]
}}

Transcript:
{prompt_str}

Listed Process to be Clustered:
{formatted_data_process_data}

Do not reuse the provided examples, generate new Clinically Meaningful Themes.
Only return your response as single complete JSON file using the structure above. Do not include commentary or multiple JSON blocks.
Now begin your task:
"""

# Inference wrapper
def chat_batch(prompts, llm):
    sampling_params = SamplingParams(max_tokens=52000, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]

# Main script
def generate_baseline(dialiog_path, session_path):
    print("Im here")
    start_time = time.time()
    llm = initialize_model()
    transcript_data = load_transcripts(dialiog_path)
    process_data = load_transcripts(session_path)

    if os.path.exists("processed_sessions.json"):
        with open("processed_sessions.json") as f:
            processed = set(json.load(f))
    else:
        processed = set()

    for item in transcript_data:
        session_id = item["session"]
        if session_id in processed:
            continue

        dialogue_data = item["dialogues"]
        patient_id = session_id.split("_")[0]
        session_num = int(session_id.split("_")[1][1:])

        matched_sessions = []
        for pid, sessions in process_data.items():
            if pid.lower() == patient_id.lower():
                for session in sessions:
                    for proc in session:
                        try:
                            _, sess_str, _ = proc["process_id"].split("_")
                            if int(sess_str) == session_num:
                                matched_sessions = session
                                break
                        except Exception as e:
                            print(f"Skipping invalid process_id format: {proc.get('process_id')}, error: {e}")

        if not matched_sessions:
            continue

        # Prompt creation in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            prompts = list(executor.map(lambda s: generate_prompt_baseline(dialogue_data, matched_sessions), [matched_sessions]))

        # Batched inference
        responses = chat_batch(prompts, llm)
        print(responses)

        for resp in responses:
            try:
                result = json.loads(resp)
            except json.JSONDecodeError:
                result = resp
            save_classification(result, session_num, patient_id)
            processed.add(session_id)

        with open("processed_sessions.json", "w") as f:
            json.dump(list(processed), f)

    print(f"Total time: {time.time() - start_time:.2f} seconds")

dialog_path = "minibatch/p16_dialog.json"
session_path = "minibatch/p16.json"

print("We are starting")
generate_baseline(dialog_path, session_path)