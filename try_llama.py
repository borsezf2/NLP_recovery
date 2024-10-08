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

pipe("code for gcd of two number in python")
