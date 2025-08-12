#!/usr/bin/env python3
"""
vLLM Model Server for Lab-Bench Evaluation
Starts a vLLM OpenAI-compatible server with GPU selection and advanced configuration
"""

import argparse
import os
import sys
import subprocess
import signal
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_vllm_server(
    model: str,
    port: int = 8001,
    host: str = "0.0.0.0",
    gpu_devices: str = "0",
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    trust_remote_code: bool = True,
    download_dir: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
    enforce_eager: bool = False
):
    """
    Start vLLM server with specified configuration
    
    Args:
        model: Model name or path (HuggingFace model ID or local path)
        port: Port to run the server on
        host: Host to bind to
        gpu_devices: GPU device IDs to use (e.g., "0", "0,1", "0,1,2,3")
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum context length
        dtype: Data type (auto, float16, bfloat16, float32)
        quantization: Quantization method (awq, squeezellm, None)
        trust_remote_code: Whether to trust remote code
        download_dir: Directory for model downloads
        gpu_memory_utilization: GPU memory utilization (0-1)
        max_num_seqs: Maximum number of sequences
        enforce_eager: Use eager mode instead of CUDA graphs
    """
    
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs)
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if download_dir:
        cmd.extend(["--download-dir", download_dir])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    # Additional optimizations for specific models
    if "llama" in model.lower():
        cmd.append("--enable-prefix-caching")
    
    logger.info(f"Starting vLLM server with command:")
    logger.info(" ".join(cmd))
    logger.info(f"Model: {model}")
    logger.info(f"Port: {port}")
    logger.info(f"GPU devices: {gpu_devices}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    
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
        
        logger.info(f"\n✓ vLLM server is starting on http://{host}:{port}")
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
        description='Start vLLM server for model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU with Llama model
  python start_vllm_server.py --model meta-llama/Llama-2-7b-hf --gpu 0 --port 8001

  # Multi-GPU with tensor parallelism
  python start_vllm_server.py --model meta-llama/Llama-2-70b-hf --gpu 0,1,2,3 --tensor-parallel 4

  # With quantization for large models
  python start_vllm_server.py --model TheBloke/Llama-2-70B-AWQ --gpu 0,1 --tensor-parallel 2 --quantization awq

  # Vision-language model
  python start_vllm_server.py --model llava-hf/llava-1.5-7b-hf --gpu 0 --trust-remote-code
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
    
    # Model configuration
    parser.add_argument('--max-model-len', type=int, default=None,
                       help='Maximum context length (default: model config)')
    parser.add_argument('--dtype', type=str, choices=['auto', 'float16', 'bfloat16', 'float32'],
                       default='auto', help='Data type for model weights')
    parser.add_argument('--quantization', type=str, choices=['awq', 'squeezellm', 'None'],
                       default=None, help='Quantization method')
    
    # Memory and performance
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (0-1, default: 0.9)')
    parser.add_argument('--max-num-seqs', type=int, default=256,
                       help='Maximum number of sequences (default: 256)')
    parser.add_argument('--enforce-eager', action='store_true',
                       help='Use eager mode instead of CUDA graphs')
    
    # Other options
    parser.add_argument('--trust-remote-code', action='store_true', default=True,
                       help='Trust remote code from HuggingFace (default: True)')
    parser.add_argument('--download-dir', type=str, default=None,
                       help='Directory for model downloads')
    
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
        logger.error("pip install vllm")
        sys.exit(1)
    
    # Start the server
    start_vllm_server(
        model=args.model,
        port=args.port,
        host=args.host,
        gpu_devices=args.gpu,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager
    )

if __name__ == "__main__":
    main()