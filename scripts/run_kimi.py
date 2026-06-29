#!/usr/bin/env python3
"""Run Kimi-VL server on GPUs 2,3 port 8010"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import base64
import io

class GenerateRequest(BaseModel):
    prompt: str
    images: Optional[List[str]] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_text: str
    model_name: str

app = FastAPI(title="Kimi-VL Server")

print("Loading Kimi-VL-A3B on GPUs 2,3...")
tokenizer = AutoTokenizer.from_pretrained(
    "moonshotai/Kimi-VL-A3B-Thinking-2506",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-VL-A3B-Thinking-2506",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory={0: "22GiB", 1: "22GiB"},
    attn_implementation="eager"  # Use eager attention to avoid SDPA issues
)

# Check memory usage
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem_gb = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} memory: {mem_gb:.2f} GB")
print("✓ Kimi-VL loaded successfully")

@app.get("/")
async def root():
    return {"model": "kimi-vl", "port": 8010, "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        with torch.no_grad():
            inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=512)
            # Move inputs to the correct device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False  # Disable cache to avoid DynamicCache issues
            )
            outputs_cpu = outputs.cpu()
            # Access input_ids from the dict, not as an attribute
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs_cpu[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return GenerateResponse(generated_text=generated_text, model_name="kimi-vl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
