import os
os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"

import time
import re
from vllm import LLM, SamplingParams

import csv
import json
import jsonlines
import random

# Define the model path
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"

# Initialize the LLM using vLLM with tensor parallelism (for using two GPUs)
start_time = time.time()
llm = LLM(model=model_path, tensor_parallel_size=2, dtype="bfloat16", gpu_memory_utilization=0.95, max_model_len=23000,  max_num_seqs=256)
end_time = time.time()
print(f"Time taken to load the model: {end_time - start_time} seconds")


newrows = []
ks = [0]

log_file = open("log.txt", "w")

#read ICL examples
examples = []
with open('train.csv', mode ='r', encoding="ISO-8859-1")as file:
  csvFile = csv.reader(file)
  for line_as_list in csvFile:
    examples.append(line_as_list)

test_data = []
with open('test.csv', mode ='r', encoding="ISO-8859-1")as file:
  csvFile = csv.reader(file)
  for line_as_list in csvFile:
    test_data.append(line_as_list)

for k in ks:
  for run in range(0, 1):
    random.shuffle(examples)
    examples_of_run = examples[:k]
    log_file.write("k: "+str(k)+" ___ run:"+str(run)+"\n")
    log_file.flush()
    BASE_prompt =f"""
    A psychological process is a mental or emotional function that reflects how a personperceives, 
    interprets, reacts to, or regulates their internal or external experiences. 
    This includes emotional responses, thoughts, decisions, memories, social interpretations,and personal reflections
    
    Task:You are a psychological process classifier. Your task is to analyze a dialogueutterance and determine:
    
    1. Whether it reflects a **psychological process**.
    2. If it does, classify it into one or more of the following types of psychological processes:

    - Cognition: How the patient thinks and assigns meaning to events.
    - Sense of Self: How the patient perceives and conceptualizes themselves.
    - (Overt) Behaviour: Observable or repetitive actions the patient takes.
    - Affect: How the patient feels about their situation.
    - Context/Moderators: Situational factors that are static or difficult to change.
    - Attention: How the patient directs or shifts focus during experiences.
    - Biophysiological: Aspects related to sleep, diet, exercise, and chronic health conditions.
    - Motivation: The goals or aims the patient pursues.
    - Sociocultural: The social and cultural relationships and contexts the patient engages with.

---

## Format for the output:
```json
{{
  "is_process": true or false,
  "types": ["<list of types if applicable, otherwise empty>"]
}}

"""
    for test_instance in test_data:
      print("Test_instance:", test_instance)
      intent = ""
      newrow={}
      newrow["id"] = test_instance[0]
      newrow["run"] = run
      newrow["k"] = k
      newrow["utterance"] = test_instance[1]
      newrow["gt_boolean"] = test_instance[4]
      newrow["gt_dimensions"] = test_instance[5]
      newrow["context_before"] = test_instance[2]
      newrow["context_after"] = test_instance[3]
      try:
        sampling_params = SamplingParams(
            temperature=0.0,max_tokens=300,stop=["}"]
        )
        conext_block = f""" Context: {test_instance[2]} Client: {test_instance[1]} {test_instance[3]}"""
        prompt = BASE_prompt+conext_block+"Classify this utterance: "+test_instance[1]+" Response:"
        outputs = llm.generate([prompt], sampling_params)
        log_file.write(str(outputs)+"\n")
        log_file.flush()
        decoded_output = outputs[0].outputs[0].text
        newrow["prompt"]= prompt
        newrow["response"]= decoded_output
        newrows.append(newrow)
      except Exception as e:
        log_file.write(f"Error processing sentences: {e}")
        log_file.flush()
        newrow["prompt"]= prompt
        newrow["response"] = "error"
        newrows.append(newrow)

fname = 'ICL_results_zs_with_definitions'

with jsonlines.open(fname+'.jsonl', 'w') as writer:
  writer.write_all(newrows)
import pandas
pandas.read_json(fname+'.jsonl', lines=True).to_excel(fname+".xlsx")
log_file.close()