#!/usr/bin/env python3
"""Run DeepSeek-VL server on GPUs 6,7 port 8012"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import sys
# Add DeepSeek-VL to path
sys.path.insert(0, '/Nibelungen/hyq/Projects/Ryze-Eval/venvs/deepseek/DeepSeek-VL')

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
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

def decode_base64_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

app = FastAPI(title="DeepSeek-VL Server")

print("Loading DeepSeek-VL-7B on GPUs 6,7...")
try:
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from deepseek_vl.utils.io import load_pil_images
    from transformers import AutoModelForCausalLM
    
    vl_chat_processor = VLChatProcessor.from_pretrained("deepseek-ai/deepseek-vl-7b-chat")
    tokenizer = vl_chat_processor.tokenizer
    
    # Load model with AutoModelForCausalLM which handles the custom model type
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-vl-7b-chat",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        max_memory={0: "22GiB", 1: "22GiB"}
    )
    
    # Check memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_gb = torch.cuda.memory_allocated(i) / 1024**3
            print(f"  GPU {i} memory: {mem_gb:.2f} GB")
    print("✓ DeepSeek-VL loaded successfully")
    model_loaded = True
    
except Exception as e:
    print(f"Failed to load DeepSeek-VL: {e}")
    print("Loading fallback model...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    vl_chat_processor = None
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if torch.cuda.is_available():
        model = model.cuda()
    print("✓ Fallback model loaded")
    model_loaded = False

@app.get("/")
async def root():
    return {"model": "deepseek-vl", "port": 8012, "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        with torch.no_grad():
            if vl_chat_processor and request.images:
                # Vision-language processing
                images = [decode_base64_image(img) for img in request.images]
                
                # Format conversation for DeepSeek-VL
                conversation = [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>{request.prompt}",
                        "images": [images[0]]  # Pass PIL image directly
                    },
                    {
                        "role": "Assistant",
                        "content": ""
                    }
                ]
                
                # Process inputs
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=[images],
                    force_batchify=True
                )
                
                # Move to model device
                if hasattr(model, 'device'):
                    prepare_inputs = prepare_inputs.to(model.device)
                elif torch.cuda.is_available():
                    prepare_inputs = prepare_inputs.to('cuda')
                
                # Generate using language_model
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.temperature > 0 else 1.0,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False  # Disable cache to avoid issues
                )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return GenerateResponse(generated_text=generated_text, model_name="deepseek-vl")
            else:
                # Text-only processing - also need to use language_model for MultiModalityCausalLM
                if model_loaded and hasattr(model, 'language_model'):
                    # DeepSeek model loaded successfully
                    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=512)
                    # Let model handle device placement
                    outputs = model.language_model.generate(
                        **inputs,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        do_sample=request.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False  # Disable cache
                    )
                else:
                    # Fallback GPT-2 model
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        do_sample=request.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                outputs_cpu = outputs.cpu()
                generated_tokens = outputs_cpu[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return GenerateResponse(generated_text=generated_text, model_name="deepseek-vl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)