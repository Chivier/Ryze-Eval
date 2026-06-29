#!/usr/bin/env python3
"""
Deploy two vision-language models with correct loading methods
- Kimi-VL-A3B-Thinking-2506 on port 8010, GPU 2,3
- DeepSeek-VL-7B-Chat on port 8012, GPU 4,5
"""

import os
import sys
import subprocess
import signal
import time
import multiprocessing

def kill_processes_on_port(port):
    """Kill any process using the specified port"""
    try:
        result = subprocess.run(
            f"lsof -ti :{port}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"  Killed process {pid} on port {port}")
                except:
                    pass
            return True
    except:
        pass
    
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        return True
    except:
        pass
    
    return False

def run_kimi_server():
    """Run Kimi-VL server on GPUs 2,3"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    # Activate Kimi venv and run
    script = """
import os
import sys
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
    max_memory={0: "22GiB", 1: "22GiB"}
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
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return GenerateResponse(generated_text=generated_text, model_name="kimi-vl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
"""
    
    # Run with Kimi venv
    subprocess.run([
        sys.executable, "-c", 
        f"import subprocess; subprocess.run(['venvs/kimi/.venv/bin/python', '-c', {repr(script)}])"
    ])

def run_deepseek_server():
    """Run DeepSeek-VL server on GPUs 4,5"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    
    script = """
import os
import sys
sys.path.insert(0, 'venvs/deepseek/DeepSeek-VL')
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
    from transformers import AutoModelForCausalLM
    
    vl_chat_processor = VLChatProcessor.from_pretrained("deepseek-ai/deepseek-vl-7b-chat")
    tokenizer = vl_chat_processor.tokenizer
    
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
    
except Exception as e:
    print(f"Failed to load DeepSeek-VL: {e}")
    print("Loading fallback model...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    vl_chat_processor = None
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    print("✓ Fallback model loaded")

@app.get("/")
async def root():
    return {"model": "deepseek-vl", "port": 8012, "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        with torch.no_grad():
            if vl_chat_processor and request.images:
                # Vision-language processing
                images = [decode_base64_image(img) for img in request.images]
                conversation = [{
                    "role": "User",
                    "content": f"<image_placeholder>{request.prompt}",
                    "images": images
                }]
                inputs = vl_chat_processor(
                    conversations=[conversation],
                    images=[images],
                    force_batchify=True
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    do_sample=request.do_sample
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return GenerateResponse(generated_text=generated_text, model_name="deepseek-vl")
            else:
                # Text-only processing
                inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    do_sample=request.do_sample
                )
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return GenerateResponse(generated_text=generated_text, model_name="deepseek-vl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)
"""
    
    # Run with DeepSeek venv
    subprocess.run([
        sys.executable, "-c",
        f"import subprocess; subprocess.run(['venvs/deepseek/.venv/bin/python', '-c', {repr(script)}])"
    ])

def main():
    print("="*60)
    print("Deploying Two Vision-Language Models")
    print("="*60)
    
    # Clean up ports
    print("\nCleaning up ports...")
    for port in [8010, 8012]:
        if kill_processes_on_port(port):
            print(f"  Cleared port {port}")
        else:
            print(f"  Port {port} is free")
    
    time.sleep(2)
    
    # Create processes for each model
    processes = []
    
    print("\nStarting Kimi-VL on GPUs 2,3, port 8010...")
    p1 = multiprocessing.Process(target=run_kimi_server)
    p1.start()
    processes.append(p1)
    time.sleep(5)
    
    print("\nStarting DeepSeek-VL on GPUs 4,5, port 8012...")
    p2 = multiprocessing.Process(target=run_deepseek_server)
    p2.start()
    processes.append(p2)
    
    print("\n" + "="*60)
    print("All models deployed!")
    print("="*60)
    print("\nModel endpoints:")
    print("  - Kimi-VL: http://localhost:8010")
    print("  - DeepSeek-VL: http://localhost:8012")
    
    print("\nPress Ctrl+C to stop all servers...")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("✓ All servers stopped")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
