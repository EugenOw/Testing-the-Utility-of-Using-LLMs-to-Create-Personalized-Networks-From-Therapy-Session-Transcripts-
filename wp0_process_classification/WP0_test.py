import os
os.environ["TRITON_CACHE_DIR"] = "/mnt/beegfs/work/cache"

import time
import re
from vllm import LLM, SamplingParams

import csv
import json
import jsonlines
import random


def token_len(s: str) -> int:
  return len(tokenizer(s).input_ids)

# Define the model path
model_path = "/mnt/beegfs/shared/shared_model_weights/models--llama-3.1/hf/70B-Instruct/"

# Initialize the LLM using vLLM with tensor parallelism (for using two GPUs)
start_time = time.time()
llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.98,  # give KV some headroom
    max_model_len=73568,          # enough for your ~2k prompt + output + slack
    max_num_seqs=1,               # you generate 1 prompt -> don't reserve for 256
    # Optional if your vLLM version supports it:
    # block_size=32,              # fewer blocks, less fragmentation
)

end_time = time.time()
print(f"Time taken to load the model: {end_time - start_time} seconds")


newrows = []
ks = [1, 5, 10, 50, 100]
#ks = [100]
log_file = open("log.txt", "w")

#read ICL examples
examples = []
with open('train.csv', mode ='r', encoding="ISO-8859-1")as file:
  csvFile = csv.reader(file)
  for line_as_list in csvFile:
    examples.append(line_as_list)

#reduce to 10 for testing
test_data = []
with open('test.csv', mode ='r', encoding="ISO-8859-1")as file:
  csvFile = csv.reader(file)
  for line_as_list in csvFile:
    test_data.append(line_as_list)

for k in ks:
  for run in range(0, 3):
    random.shuffle(examples)
    examples_of_run = examples[:k]
    log_file.write("k: "+str(k)+" ___ run:"+str(run)+"\n")
    log_file.flush()
    BASE_prompt =f"""You are a psychological process classifier. Your task is to analyze a dialogue utterance and determine:

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

**Definition of a psychological process**: A psychological process is a mental or emotional function that reflects how a person perceives, interprets, reacts to, or regulates their internal or external experiences. This includes emotional responses, thoughts, decisions, memories, social interpretations, and personal reflections.

---

## Format for the output:
{{
  "is_process": true or false,
  "types": ["<list of types if applicable, otherwise empty>"]
}}

For example:
"""
    for ex in examples_of_run:
      ex_utt = (ex[1] or "").strip()
      ex_bef = (ex[2] or "").strip()
      ex_aft = (ex[3] or "").strip()
      ex_proc = ex[4]
      ex_dims = ex[5]
      BASE_prompt += f"""
      Conext: {ex_bef}\n
      Client: {ex_utt}
      {ex_aft}\n
      Classify this utterance: {ex_utt}. 
      Response:
        {{
          "is_process": {ex_proc},
          "types": {ex_dims}
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
      newrow["gt_boolean"] = test_instance[4].lower()
      newrow["gt_dimensions"] = test_instance[5]
      newrow["context_before"] = test_instance[2]
      newrow["context_after"] = test_instance[3]
      try:
        sampling_params = SamplingParams(
          temperature=0.0,
          max_tokens=120,
          stop=["}", "<END>"]
        )
        conext_block = f""" \n Context: {test_instance[2]} Client: {test_instance[1]} {test_instance[3]}"""
        prompt = BASE_prompt+conext_block+"\n Now Classify this utterance: "+test_instance[1]+" \n Response:"
        # right before generate:
        tokenizer = llm.get_tokenizer()
        # right before generate:
        in_tokens = token_len(prompt)
        if in_tokens + sampling_params.max_tokens + 64 > 12288:
          print(f"[WARN] prompt({in_tokens}) + output budget exceeds max_model_len")

        outputs = llm.generate([prompt], sampling_params)
        log_file.write(str(outputs)+"\n")
        log_file.flush()
        decoded_output = outputs[0].outputs[0].text
        newrow["prompt"]= prompt
        newrow["response"]= decoded_output
        print("\n")
        print("Prompt:",prompt)
        print("\n")
        #print("Output is here:", outputs)
        #print("\n")
        print("Decoded Output is here:", decoded_output)
        print("\n")
        newrows.append(newrow)
      except Exception as e:
        log_file.write(f"Error processing sentences: {e}")
        log_file.flush()
        newrow["prompt"]= prompt
        newrow["response"] = "error"
        newrows.append(newrow)

fname = 'ICL_results_with_definitions'

with jsonlines.open(fname+'.jsonl', 'w') as writer:
  writer.write_all(newrows)
import pandas
pandas.read_json(fname+'.jsonl', lines=True).to_excel(fname+".xlsx")
log_file.close()