import torch
from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-1B" 
model_id = "meta-llama/Meta-Llama-3-8B" 



pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = "code for gcd of two numbers in python"
result = pipe(prompt, max_length=400, num_return_sequences=100)

print(f"Prompt: {prompt}\n")
print("Generated Text:")
print(result[0]['generated_text'])