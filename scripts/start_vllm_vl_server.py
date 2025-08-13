#!/usr/bin/env python3
"""
vLLM Vision-Language Model Server for Lab-Bench Evaluation
Optimized for serving vision-language models with GPU selection and advanced configuration
"""

import argparse
import os
import sys
import subprocess
import signal
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model-specific configurations
MODEL_CONFIGS = {
    "moonshotai/Kimi-VL-A3B-Thinking-2506": {
        "max_model_len": 32768,
        "max_num_batched_tokens": 32768,
        "limit_mm_per_prompt": {"image": 64},
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 8,
        "enforce_eager": False,
        "dtype": "auto",
        "trust_remote_code": True,
        "extra_args": ["--enable-prefix-caching"]
    },
    "deepseek-ai/deepseek-vl-7b-chat": {
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,
        "enforce_eager": False,
        "dtype": "auto",
        "trust_remote_code": True,
        "extra_args": []
    },
    "openvla/openvla-7b": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 128,
        "enforce_eager": True,  # May need eager mode for custom ops
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "extra_args": []
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get optimized configuration for specific model"""
    # Check exact match first
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
    
    # Check partial match
    for key in MODEL_CONFIGS:
        if key in model_name or model_name in key:
            logger.info(f"Using configuration for {key}")
            return MODEL_CONFIGS[key].copy()
    
    # Default configuration for unknown models
    logger.warning(f"No specific config for {model_name}, using defaults")
    return {
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,
        "enforce_eager": False,
        "dtype": "auto",
        "trust_remote_code": True,
        "extra_args": []
    }

def start_vllm_vl_server(
    model: str,
    port: int = 8001,
    host: str = "0.0.0.0",
    gpu_devices: str = "0",
    tensor_parallel_size: int = 1,
    override_config: Optional[Dict[str, Any]] = None,
    download_dir: Optional[str] = None,
    quantization: Optional[str] = None,
    served_model_name: Optional[str] = None
):
    """
    Start vLLM server optimized for vision-language models
    
    Args:
        model: Model name or path (HuggingFace model ID or local path)
        port: Port to run the server on
        host: Host to bind to
        gpu_devices: GPU device IDs to use (e.g., "0", "0,1", "0,1,2,3")
        tensor_parallel_size: Number of GPUs for tensor parallelism
        override_config: Override default configurations
        download_dir: Directory for model downloads
        quantization: Quantization method (awq, squeezellm, None)
        served_model_name: Name to serve the model as
    """
    
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    # Get model-specific configuration
    config = get_model_config(model)
    
    # Apply overrides if provided
    if override_config:
        config.update(override_config)
    
    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", config["dtype"],
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["max_num_seqs"])
    ]
    
    # Add served model name if specified
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    else:
        # Use a simplified name derived from model path
        served_name = model.split("/")[-1]
        cmd.extend(["--served-model-name", served_name])
    
    # Add max model length
    if "max_model_len" in config:
        cmd.extend(["--max-model-len", str(config["max_model_len"])])
    
    # Add max batched tokens for Kimi model
    if "max_num_batched_tokens" in config:
        cmd.extend(["--max-num-batched-tokens", str(config["max_num_batched_tokens"])])
    
    # Add multimodal limits (for models like Kimi-VL)
    if "limit_mm_per_prompt" in config:
        for media_type, limit in config["limit_mm_per_prompt"].items():
            cmd.extend([f"--limit-mm-per-prompt", f"{media_type}={limit}"])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if config.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    
    if download_dir:
        cmd.extend(["--download-dir", download_dir])
    
    if config.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    
    # Add any extra model-specific arguments
    for extra_arg in config.get("extra_args", []):
        cmd.append(extra_arg)
    
    logger.info(f"Starting vLLM vision-language server with command:")
    logger.info(" ".join(cmd))
    logger.info(f"Model: {model}")
    logger.info(f"Port: {port}")
    logger.info(f"GPU devices: {gpu_devices}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Model config: {config}")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nShutting down vLLM server...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info(f"\n✓ vLLM vision-language server is starting on http://{host}:{port}")
        logger.info(f"API endpoint: http://{host}:{port}/v1")
        logger.info("Press Ctrl+C to stop the server\n")
        
        # Wait for the process
        process.wait()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down vLLM server...")
        process.terminate()
        process.wait()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Start vLLM server optimized for vision-language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Vision-Language Models with Optimized Configs:
  - moonshotai/Kimi-VL-A3B-Thinking-2506
  - deepseek-ai/deepseek-vl-7b-chat
  - openvla/openvla-7b

Examples:
  # Kimi-VL on GPU 4, port 8010
  python start_vllm_vl_server.py --model moonshotai/Kimi-VL-A3B-Thinking-2506 --gpu 4 --port 8010

  # DeepSeek-VL on GPU 5, port 8011
  python start_vllm_vl_server.py --model deepseek-ai/deepseek-vl-7b-chat --gpu 5 --port 8011

  # OpenVLA on GPU 6, port 8012
  python start_vllm_vl_server.py --model openvla/openvla-7b --gpu 6 --port 8012

  # Multi-GPU with tensor parallelism
  python start_vllm_vl_server.py --model deepseek-ai/deepseek-vl-7b-chat --gpu 0,1 --tensor-parallel 2

  # With custom configuration
  python start_vllm_vl_server.py --model moonshotai/Kimi-VL-A3B-Thinking-2506 \\
    --gpu 0 --port 8010 --max-model-len 131072 --max-num-seqs 16
        """
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path (HuggingFace ID or local path)')
    
    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device IDs (e.g., "0", "0,1", "0,1,2,3")')
    parser.add_argument('--tensor-parallel', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    
    # Server configuration
    parser.add_argument('--port', type=int, default=8001,
                       help='Port to run the server on (default: 8001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--served-model-name', type=str, default=None,
                       help='Name to serve the model as (default: derived from model path)')
    
    # Model configuration overrides
    parser.add_argument('--max-model-len', type=int, default=None,
                       help='Override maximum context length')
    parser.add_argument('--dtype', type=str, choices=['auto', 'float16', 'bfloat16', 'float32'],
                       default=None, help='Override data type for model weights')
    parser.add_argument('--quantization', type=str, choices=['awq', 'squeezellm', 'gptq'],
                       default=None, help='Quantization method')
    parser.add_argument('--gpu-memory-utilization', type=float, default=None,
                       help='Override GPU memory utilization (0-1)')
    parser.add_argument('--max-num-seqs', type=int, default=None,
                       help='Override maximum number of sequences')
    parser.add_argument('--enforce-eager', action='store_true',
                       help='Force eager mode instead of CUDA graphs')
    
    # Other options
    parser.add_argument('--download-dir', type=str, default=None,
                       help='Directory for model downloads')
    parser.add_argument('--no-trust-remote-code', action='store_true',
                       help='Disable trusting remote code')
    
    args = parser.parse_args()
    
    # Validate tensor parallel size
    gpu_count = len(args.gpu.split(','))
    if args.tensor_parallel > gpu_count:
        logger.error(f"Tensor parallel size ({args.tensor_parallel}) cannot exceed number of GPUs ({gpu_count})")
        sys.exit(1)
    
    # Check if vLLM is installed
    try:
        import vllm
        logger.info(f"vLLM version: {vllm.__version__}")
    except ImportError:
        logger.error("vLLM is not installed. Please install it with:")
        logger.error("pip install vllm>=0.6.0")
        logger.error("For Kimi-VL, also install: pip install blobfile flash-attn --no-build-isolation")
        sys.exit(1)
    
    # Build override configuration
    override_config = {}
    if args.max_model_len:
        override_config["max_model_len"] = args.max_model_len
    if args.dtype:
        override_config["dtype"] = args.dtype
    if args.gpu_memory_utilization:
        override_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs:
        override_config["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        override_config["enforce_eager"] = True
    if args.no_trust_remote_code:
        override_config["trust_remote_code"] = False
    
    # Start the server
    start_vllm_vl_server(
        model=args.model,
        port=args.port,
        host=args.host,
        gpu_devices=args.gpu,
        tensor_parallel_size=args.tensor_parallel,
        override_config=override_config if override_config else None,
        download_dir=args.download_dir,
        quantization=args.quantization,
        served_model_name=args.served_model_name
    )

if __name__ == "__main__":
    main()