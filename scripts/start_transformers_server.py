#!/usr/bin/env python3
"""
Transformers Model Server for Lab-Bench Evaluation
Starts a FastAPI server to serve Transformers models with GPU selection and port configuration
"""

import argparse
import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig
)
import logging
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers Model Server")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerateResponse(BaseModel):
    text: str
    model: str

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class BatchGenerateResponse(BaseModel):
    texts: List[str]
    model: str

class ModelServer:
    def __init__(self, model_name: str, device: str = "cuda:0", quantization: Optional[str] = None, attn_implementation: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vision_model = False
        self.attn_implementation = attn_implementation
        
        self._load_model(quantization)
    
    def _load_model(self, quantization: Optional[str]):
        """Load the model with appropriate configuration"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Base model kwargs using the recommended approach
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",  # Use auto dtype selection
        }
        
        # Handle attention implementation
        if self.attn_implementation == "auto":
            # Try to check if flash_attention_2 is available
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2 implementation (auto-detected)")
            except ImportError:
                model_kwargs["attn_implementation"] = "eager"
                logger.info("flash_attn not available, using eager attention implementation")
        else:
            model_kwargs["attn_implementation"] = self.attn_implementation
            logger.info(f"Using {self.attn_implementation} attention implementation")
        
        # Handle device mapping
        if self.device != "auto":
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["device_map"] = "auto"
        
        # Configure quantization if requested
        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
            # Remove torch_dtype when using quantization
            model_kwargs.pop("torch_dtype", None)
        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            # Remove torch_dtype when using quantization
            model_kwargs.pop("torch_dtype", None)
        
        # Special handling for OpenVLA and VLA models - use AutoModel directly
        if "openvla" in self.model_name.lower() or "vla" in self.model_name.lower():
            self.is_vision_model = True
            try:
                logger.info("Loading model with AutoModel (vision/VLA model detected)")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Try to load processor, some models might not have it
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    logger.info("Processor loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load processor: {e}")
                    # Fall back to tokenizer if processor is not available
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("Using tokenizer instead of processor")
                
                logger.info("Vision/VLA model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load vision/VLA model: {e}")
                raise
                
        # Check if this is a generic vision model
        elif "vision" in self.model_name.lower() or "kimi" in self.model_name.lower():
            self.is_vision_model = True
            try:
                logger.info("Loading model with AutoModel (generic approach)")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Try to load processor first, then tokenizer
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        
            except Exception as e:
                logger.warning(f"Failed to load as vision model with AutoModel: {e}")
                logger.info("Attempting to load as standard causal LM...")
                self.is_vision_model = False
        
        # Standard text model loading
        if not self.is_vision_model:
            try:
                # First try AutoModel for maximum compatibility
                logger.info("Loading model with AutoModel (standard approach)")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception as e:
                logger.info(f"AutoModel failed, trying AutoModelForCausalLM: {e}")
                # Fall back to AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded successfully. Vision model: {self.is_vision_model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        try:
            if self.is_vision_model and self.processor:
                # For vision models, we'll treat text-only input
                inputs = self.processor(text=prompt, return_tensors="pt")
                if self.device != "auto":
                    inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                if self.device != "auto":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=kwargs.get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
                )
            
            if self.is_vision_model and self.processor:
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

# Global model server instance
model_server: Optional[ModelServer] = None

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": model_server.model_name if model_server else None,
        "device": model_server.device if model_server else None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not model_server:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = model_server.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        return GenerateResponse(text=text, model=model_server.model_name)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    if not model_server:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        texts = []
        for prompt in request.prompts:
            text = model_server.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
            texts.append(text)
        return BatchGenerateResponse(texts=texts, model=model_server.model_name)
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description='Start Transformers model server')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path (e.g., "openvla/openvla-7b", "meta-llama/Llama-2-7b-hf")')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID to use (e.g., "0", "1", "0,1") or "auto" for automatic')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--quantization', type=str, choices=['4bit', '8bit', 'none'], default='none',
                       help='Quantization mode for model loading')
    parser.add_argument('--attn-implementation', type=str, choices=['eager', 'flash_attention_2', 'auto'], default='auto',
                       help='Attention implementation to use (default: auto - will use flash_attention_2 if available)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set CUDA device
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = f"cuda:0"  # Always use cuda:0 after setting CUDA_VISIBLE_DEVICES
    else:
        device = "auto"
    
    # Initialize model server
    global model_server
    try:
        model_server = ModelServer(
            model_name=args.model,
            device=device,
            quantization=args.quantization if args.quantization != 'none' else None,
            attn_implementation=args.attn_implementation.replace('-', '_')  # Convert CLI arg format
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Quantization: {args.quantization}")
    
    uvicorn.run(
        "start_transformers_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()