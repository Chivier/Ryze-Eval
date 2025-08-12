#!/usr/bin/env python3
"""
Test script for vLLM integration
Tests both the vLLM server and the evaluation pipeline integration
"""

import os
import sys
import time
import requests
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_interface import ModelFactory

load_dotenv()

def test_vllm_health(base_url="http://localhost:8001"):
    """Test if the vLLM server is running"""
    print(f"\n1. Testing vLLM server health at {base_url}...")
    
    # Check OpenAI-compatible endpoint
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            if models:
                print(f"   ✓ vLLM server is healthy")
                print(f"   Available models:")
                for model in models:
                    print(f"     - {model['id']}")
                return True
            else:
                print(f"   ⚠️  vLLM server is running but no models loaded")
                return False
        else:
            print(f"   ✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Could not connect to vLLM server at {base_url}")
        print(f"\n   To start the vLLM server, run:")
        print(f"   python scripts/start_vllm_server.py --model meta-llama/Llama-2-7b-hf --port 8001")
        print(f"\n   For multi-GPU setup:")
        print(f"   python scripts/start_vllm_server.py --model meta-llama/Llama-2-70b-hf --gpu 0,1,2,3 --tensor-parallel 4")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_vllm_metrics(base_url="http://localhost:8001"):
    """Test vLLM metrics endpoint"""
    print(f"\n2. Testing vLLM metrics...")
    
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Metrics endpoint accessible")
            # Parse some basic metrics
            metrics = response.text
            if "vllm:num_requests_running" in metrics:
                print(f"   ✓ vLLM metrics available")
        else:
            print(f"   ⚠️  Metrics endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Could not access metrics: {e}")

def test_direct_api(base_url="http://localhost:8001"):
    """Test the vLLM OpenAI-compatible API directly"""
    print(f"\n3. Testing vLLM OpenAI-compatible API...")
    
    # Get model name
    model_name = None
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                model_name = models[0]["id"]
                print(f"   Using model: {model_name}")
    except Exception as e:
        print(f"   ✗ Could not get model list: {e}")
        return False
    
    if not model_name:
        print(f"   ✗ No models available")
        return False
    
    # Test completion endpoint
    print("   Testing /v1/completions endpoint...")
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["text"]
            print(f"   ✓ Completion successful")
            print(f"   Response: {text.strip()}")
        else:
            print(f"   ⚠️  Completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ⚠️  Error calling completions: {e}")
    
    # Test chat completion endpoint
    print("   Testing /v1/chat/completions endpoint...")
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "What is 2 + 2? Answer with just the number:"}
                ],
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"   ✓ Chat completion successful")
            print(f"   Response: {content.strip()}")
            return True
        else:
            print(f"   ⚠️  Chat completion returned status: {response.status_code}")
            # Some models may not support chat format
            if response.status_code == 400:
                print(f"   Note: This model may only support completion format, not chat format")
                return True  # Still consider it a success if completion worked
    except Exception as e:
        print(f"   ⚠️  Error calling chat completions: {e}")
    
    return True

def test_model_interface():
    """Test the VLLMInterface integration"""
    print("\n4. Testing VLLMInterface integration...")
    
    # Set environment variables
    os.environ["MODEL_PROVIDER"] = "vllm"
    
    try:
        model = ModelFactory.create_model("vllm")
        print("   ✓ VLLMInterface created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create VLLMInterface: {e}")
        return False
    
    # Test single generation
    print("   Testing generate() method...")
    try:
        prompt = "The quick brown fox"
        response = model.generate(prompt, max_tokens=20, temperature=0.5)
        if response:
            print(f"   ✓ Generate successful")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"   ⚠️  Generate returned empty response")
    except Exception as e:
        print(f"   ✗ Error in generate: {e}")
        return False
    
    # Test batch generation
    print("   Testing batch_generate() method...")
    try:
        prompts = [
            "Python is a",
            "Machine learning is"
        ]
        responses = model.batch_generate(prompts, max_tokens=30, temperature=0.5)
        if responses and all(responses):
            print(f"   ✓ Batch generate successful")
            print(f"   Generated {len(responses)} responses")
        else:
            print(f"   ⚠️  Batch generate returned empty responses")
    except Exception as e:
        print(f"   ✗ Error in batch_generate: {e}")
        return False
    
    return True

def test_evaluation_integration():
    """Test the full evaluation pipeline integration with vLLM"""
    print("\n5. Testing evaluation pipeline integration...")
    
    # Test if run_evaluation.py recognizes the vllm provider
    print("   Testing provider recognition...")
    
    from src.model_interface import ModelFactory
    providers = ModelFactory.get_available_providers()
    
    if "vllm" in providers:
        print(f"   ✓ 'vllm' provider is registered")
    else:
        print(f"   ✗ 'vllm' provider not found in {providers}")
        return False
    
    # Test creating model through evaluation flow
    try:
        os.environ["MODEL_PROVIDER"] = "vllm"
        model = ModelFactory.create_model()
        print(f"   ✓ Model created through factory with default provider")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    print("\n   You can now run the full evaluation with:")
    print("   python src/run_evaluation.py --provider vllm --subset CloningScenarios --limit 2")
    
    return True

def test_performance(base_url="http://localhost:8001"):
    """Test vLLM performance characteristics"""
    print("\n6. Testing vLLM performance...")
    
    # Get model name
    model_name = None
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                model_name = models[0]["id"]
    except:
        print("   ✗ Could not get model for performance test")
        return False
    
    if not model_name:
        return False
    
    # Test throughput with multiple requests
    print("   Testing throughput (5 concurrent-like requests)...")
    prompts = [
        "Explain quantum computing:",
        "What is machine learning?",
        "Describe the water cycle:",
        "How does photosynthesis work?",
        "What causes earthquakes?"
    ]
    
    start_time = time.time()
    responses = []
    
    for prompt in prompts:
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                timeout=30
            )
            if response.status_code == 200:
                responses.append(response.json())
        except:
            pass
    
    elapsed = time.time() - start_time
    
    if len(responses) == len(prompts):
        print(f"   ✓ Processed {len(prompts)} requests in {elapsed:.2f} seconds")
        print(f"   Average: {elapsed/len(prompts):.2f} seconds per request")
    else:
        print(f"   ⚠️  Only {len(responses)}/{len(prompts)} requests succeeded")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test vLLM integration')
    parser.add_argument('--base-url', type=str, default='http://localhost:8001',
                       help='Base URL of the vLLM server')
    parser.add_argument('--skip-server', action='store_true',
                       help='Skip server tests (test only ModelInterface)')
    parser.add_argument('--performance', action='store_true',
                       help='Include performance tests')
    
    args = parser.parse_args()
    
    print("="*60)
    print("vLLM INTEGRATION TEST")
    print("="*60)
    
    # Update environment variable if custom URL provided
    if args.base_url != 'http://localhost:8001':
        os.environ["VLLM_BASE_URL"] = args.base_url
    
    all_passed = True
    
    if not args.skip_server:
        # Test server health
        if not test_vllm_health(args.base_url):
            print("\n⚠️  vLLM server is not running.")
            print("\nTo start the vLLM server, run one of these commands:")
            print("\nFor single GPU:")
            print("python scripts/start_vllm_server.py --model meta-llama/Llama-2-7b-hf --gpu 0 --port 8001")
            print("\nFor multi-GPU with tensor parallelism:")
            print("python scripts/start_vllm_server.py --model meta-llama/Llama-2-70b-hf --gpu 0,1,2,3 --tensor-parallel 4")
            print("\nFor quantized models:")
            print("python scripts/start_vllm_server.py --model TheBloke/Llama-2-7B-AWQ --quantization awq --port 8001")
            all_passed = False
        else:
            # Test metrics
            test_vllm_metrics(args.base_url)
            
            # Test direct API
            if not test_direct_api(args.base_url):
                all_passed = False
            
            # Performance tests if requested
            if args.performance:
                test_performance(args.base_url)
    
    # Test ModelInterface (this will also check server connection)
    if not test_model_interface():
        all_passed = False
    
    # Test evaluation integration
    if not test_evaluation_integration():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    if all_passed:
        print("\n📝 Next steps:")
        print("1. Ensure your .env file has VLLM_BASE_URL and VLLM_MODEL_NAME set")
        print("2. Start the vLLM server with your desired model and configuration")
        print("3. Run evaluation: python src/run_evaluation.py --provider vllm --subset <subset>")
        print("\n💡 Tips:")
        print("- vLLM provides much faster inference than standard transformers")
        print("- Use tensor parallelism for large models across multiple GPUs")
        print("- Monitor GPU memory with nvidia-smi during inference")
        print("- Check /metrics endpoint for detailed performance stats")

if __name__ == "__main__":
    main()