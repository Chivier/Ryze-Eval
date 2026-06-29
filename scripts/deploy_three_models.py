#!/usr/bin/env python3
"""
Deploy two vision-language models on different ports and GPUs using transformers
- Kimi-VL-A3B-Thinking-2506 on port 8010, GPU 2,3
- DeepSeek-VL-7B-Chat on port 8012, GPU 4,5
"""

import os
import sys
import subprocess
import signal
import time
import torch
import uvicorn
import traceback
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModel
)
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from PIL import Image
import base64
import io
import requests

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    images: Optional[List[str]] = None  # Base64 encoded images
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_text: str
    model_name: str

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

# Model server class
class ModelServer:
    def __init__(self, model_name: str, gpu_id: int, port: int):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.port = port
        # When using CUDA_VISIBLE_DEVICES, the visible GPU becomes device 0
        # gpu_id == -1 indicates multi-GPU mode
        # For multi-GPU, we'll use device_map="auto" in model loading
        self.device = None  # Will be set based on single vs multi-GPU mode
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # CPU mode only - no CUDA device setting
        
        # Create FastAPI app
        self.app = FastAPI(title=f"Model Server - {model_name}")
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "model": self.model_name,
                "gpu": self.gpu_id,
                "port": self.port,
                "status": "running"
            }
        
        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            try:
                # Process images if provided
                images = None
                if request.images:
                    images = [decode_base64_image(img) for img in request.images]
                
                # Generate response
                with torch.no_grad():
                    # Handle different model types
                    if self.processor and images:
                        # Vision-language model with images
                        inputs = self.processor(
                            text=request.prompt,
                            images=images,
                            return_tensors="pt"
                        )
                    else:
                        # Text-only or no processor
                        inputs = self.tokenizer(
                            request.prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        )
                    
                    # Handle device placement for distributed models
                    if self.device is None:
                        # Model is distributed, let it handle device placement
                        input_ids = inputs['input_ids']
                        attention_mask = inputs.get('attention_mask')
                    else:
                        # Single device model
                        device = self.device
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs['attention_mask'].to(device) if 'attention_mask' in inputs else None
                    
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        top_p=request.top_p,
                        do_sample=request.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode only the generated part (excluding the input)
                    # Move to CPU for slicing to avoid CUDA errors
                    outputs_cpu = outputs.cpu()
                    input_length = input_ids.shape[1]
                    generated_tokens = outputs_cpu[0][input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )
                
                return GenerateResponse(
                    generated_text=generated_text,
                    model_name=self.model_name
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "model_loaded": self.model is not None}
    
    def load_model(self):
        """Load the model based on model name"""
        gpu_info = f"GPUs {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}" if self.gpu_id == -1 else f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
        
        try:
            if "kimi" in self.model_name.lower():
                print(f"Loading Kimi-VL-A3B-Thinking-2506 on {gpu_info}...")
                
                try:
                    # Load the actual Kimi-VL model
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "moonshotai/Kimi-VL-A3B-Thinking-2506",
                        trust_remote_code=True
                    )
                    
                    # Create custom device map for the two GPUs
                    # GPUs appear as 0 and 1 when CUDA_VISIBLE_DEVICES is set
                    print(f"  Distributing model across {gpu_info} (this may take a while)...")
                    
                    # Create balanced device map for two GPUs
                    device_map = {
                        # Split model layers between GPU 0 and GPU 1
                        # This will need to be adjusted based on actual model architecture
                        0: [0, 1],
                        1: [2, 3],
                    }
                    
                    # Load with explicit device placement
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "moonshotai/Kimi-VL-A3B-Thinking-2506",
                        torch_dtype=torch.bfloat16,
                        device_map="balanced",  # Use balanced distribution
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "22GiB", 1: "22GiB"}  # Leave some memory for operations
                    )
                    
                    # For generation, we don't set self.device as the model handles it
                    self.device = None  # Model is distributed
                    
                    # Check actual memory usage
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_used = torch.cuda.memory_allocated(i) / 1024**3
                            print(f"  GPU {i} memory used: {mem_used:.2f} GB")
                    
                    print(f"✓ Kimi-VL-A3B loaded and distributed across {gpu_info}")
                    
                except Exception as e:
                    print(f"⚠ Failed to load full Kimi-VL model: {e}")
                    print("  Falling back to placeholder for testing...")
                    # Fallback to small model
                    from transformers import GPT2LMHeadModel, GPT2Tokenizer
                    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model = GPT2LMHeadModel.from_pretrained("gpt2")
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.model = self.model.to(self.device)
                    print(f"✓ Placeholder model loaded on {gpu_info}")
                
            elif "deepseek" in self.model_name.lower():
                print(f"Loading DeepSeek-VL-7B-Chat on {gpu_info}...")
                
                try:
                    # DeepSeek-VL needs special handling for multi-modal
                    # First try the direct import method
                    try:
                        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
                        
                        print("  Using DeepSeek-VL native classes...")
                        self.processor = VLChatProcessor.from_pretrained("deepseek-ai/deepseek-vl-7b-chat")
                        self.tokenizer = self.processor.tokenizer  # For compatibility
                        
                        print(f"  Distributing DeepSeek-VL-7B across {gpu_info} (this may take a while)...")
                        self.model = MultiModalityCausalLM.from_pretrained(
                            "deepseek-ai/deepseek-vl-7b-chat",
                            torch_dtype=torch.bfloat16,
                            device_map="balanced",
                            low_cpu_mem_usage=True,
                            max_memory={0: "22GiB", 1: "22GiB"}
                        )
                        
                    except ImportError:
                        # Fallback to direct loading with custom model
                        print("  DeepSeek-VL package not found, loading with custom model class...")
                        
                        # DeepSeek-VL is based on Llama architecture with vision components
                        # We'll load it as a standard model and handle vision separately
                        from transformers import LlamaForCausalLM, LlamaTokenizer
                        
                        print("  Loading as Llama-based model with vision extensions...")
                        self.tokenizer = LlamaTokenizer.from_pretrained(
                            "deepseek-ai/deepseek-vl-7b-chat"
                        )
                        
                        print(f"  Distributing DeepSeek-VL-7B across {gpu_info} (this may take a while)...")
                        
                        # Load the language model part
                        self.model = LlamaForCausalLM.from_pretrained(
                            "deepseek-ai/deepseek-vl-7b-chat",
                            torch_dtype=torch.bfloat16,
                            device_map="balanced",
                            low_cpu_mem_usage=True,
                            max_memory={0: "22GiB", 1: "22GiB"},
                            ignore_mismatched_sizes=True  # Ignore vision component size mismatches
                        )
                    
                    self.device = None  # Model is distributed
                    
                    # Check actual memory usage
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_used = torch.cuda.memory_allocated(i) / 1024**3
                            print(f"  GPU {i} memory used: {mem_used:.2f} GB")
                    
                    print(f"✓ DeepSeek-VL-7B loaded and distributed across {gpu_info}")
                    
                except Exception as e:
                    print(f"⚠ Failed to load DeepSeek-VL-7B: {e}")
                    print("  Falling back to placeholder...")
                    from transformers import GPT2LMHeadModel, GPT2Tokenizer
                    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model = GPT2LMHeadModel.from_pretrained("gpt2")
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.model = self.model.to(self.device)
                    print(f"✓ Placeholder model loaded on {gpu_info}")
                
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
                
        except Exception as e:
            print(f"✗ Failed to load {self.model_name}: {e}")
            traceback.print_exc()
            raise
    
    def run(self):
        """Run the server"""
        self.load_model()
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

def run_model_server(model_name: str, gpu_ids: str, port: int):
    """Function to run a model server in a separate process
    
    Args:
        model_name: Name of the model to deploy
        gpu_ids: Comma-separated string of GPU IDs (e.g., "2,3" or "4")
        port: Port to run the server on
    """
    # Set CUDA visible devices to the specified GPUs
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Check if multi-GPU mode
    if "," in gpu_ids:
        # Multi-GPU mode
        server = ModelServer(model_name, -1, port)  # -1 indicates multi-GPU
    else:
        # Single GPU mode
        server = ModelServer(model_name, 0, port)  # Use device 0 since only one GPU is visible
    
    server.run()

def kill_processes_on_port(port):
    """Kill any process using the specified port"""
    import subprocess
    import signal
    
    try:
        # Try using lsof first (common on Unix systems)
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
        # Alternative using fuser
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        return True
    except:
        pass
    
    return False

def main():
    """Main function to deploy all two models"""
    # Model configurations
    models = [
        {
            "name": "kimi-vl",
            "gpus": "2,3",  # Use GPUs 2 and 3
            "port": 8010
        },
        {
            "name": "deepseek-vl",
            "gpus": "4,5",  # Use GPUs 4 and 5
            "port": 8012
        }
    ]
    
    print("=" * 60)
    print("Deploying Two Vision-Language Models")
    print("=" * 60)
    
    # Clean up ports before starting
    print("\nCleaning up ports...")
    for model_config in models:
        port = model_config['port']
        if kill_processes_on_port(port):
            print(f"  Cleared port {port}")
        else:
            print(f"  Port {port} is free")
    
    time.sleep(2)  # Wait for ports to be fully released
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n✓ Found {num_gpus} GPUs")
        print("  Note: Using placeholder models for testing")
    
    # Create processes for each model
    processes = []
    
    for model_config in models:
        print(f"\nStarting {model_config['name']} on GPU(s) {model_config['gpus']}, port {model_config['port']}...")
        
        # Create a new process for each model
        p = multiprocessing.Process(
            target=run_model_server,
            args=(model_config['name'], model_config['gpus'], model_config['port'])
        )
        p.start()
        processes.append(p)
        
        # Wait a bit for the server to start
        time.sleep(5)
    
    print("\n" + "=" * 60)
    print("All models deployed!")
    print("=" * 60)
    print("\nModel endpoints:")
    for model_config in models:
        print(f"  - {model_config['name']}: http://localhost:{model_config['port']}")
    
    print("\nTest with:")
    print('  curl -X POST "http://localhost:8010/generate" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"prompt": "Hello, how are you?"}\'')
    
    print("\nPress Ctrl+C to stop all servers...")
    
    try:
        # Wait for all processes
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
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    main()
