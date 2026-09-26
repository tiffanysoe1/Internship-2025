import transformers
import torch

model_path = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b"[cite: 1]
#modelpath in the folder[cite: 1]

pipeline = transformers.pipeline([cite: 1]
    "text-generation",[cite: 1]
    model=model_path,[cite: 1]
    model_kwargs={"torch_dtype": torch.bfloat16},[cite: 1]
    device_map="auto",[cite: 1]
)[cite: 1]

messages = [{"role": "user", "content": "What is LLM?"}][cite: 1]
terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")][cite: 1]

import time[cite: 1]
start_time = time.time()[cite: 1]

outputs = pipeline([cite: 1]
    messages,[cite: 1]
    max_new_tokens=256,         # Max completion tokens: 256[cite: 1]
    max_length=256,             # Max total sequence length: 256[cite: 1]
    batch_size=1024,            # Max batch size: 2048 tokens[cite: 1]
    eos_token_id=terminators,[cite: 1]
    do_sample=True,[cite: 1]
    temperature=0.6,[cite: 1]
    top_p=0.9[cite: 1]
)[cite: 1]

# Now calculate tokens[cite: 2]
input_tokens = len(pipeline.tokenizer.encode(messages[0]["content"]))[cite: 2]
full_tokens = len(pipeline.tokenizer.encode(assistant_response))[cite: 2]
output_tokens = full_tokens - input_tokens[cite: 2]
time_taken = end_time - start_time[cite: 2]
tokens_per_second = output_tokens / time_taken[cite: 2]

# Print all metrics[cite: 2]
print(f"Generated text: {assistant_response}")[cite: 2]
print(f"Prompt tokens: {input_tokens}")[cite: 2]
print(f"Completion tokens: {output_tokens}")[cite: 2]
print(f"Total tokens: {full_tokens}")[cite: 2]
print(f"Time taken: {time_taken:.2f} seconds")[cite: 2]
print(f"Tokens per second: {tokens_per_second:.2f}")[cite: 2]
