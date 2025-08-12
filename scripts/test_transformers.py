#!/usr/bin/env python3
"""
Test script for Transformers integration
Tests both the server startup and the evaluation pipeline integration
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

def test_server_health(base_url="http://localhost:8000"):
    """Test if the Transformers server is running"""
    print(f"\n1. Testing server health at {base_url}...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Server is healthy")
            print(f"   Model: {data.get('model', 'unknown')}")
            print(f"   Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"   ✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Could not connect to server at {base_url}")
        print(f"   Please start the server first with:")
        print(f"   python scripts/start_transformers_server.py --model <model_name> --port 8000")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_direct_api(base_url="http://localhost:8000"):
    """Test the server's API directly"""
    print(f"\n2. Testing direct API calls...")
    
    # Test single generation
    print("   Testing /generate endpoint...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": "What is 2 + 2? Answer with just the number:",
                "max_tokens": 10,
                "temperature": 0.1,
                "do_sample": False
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Generation successful")
            print(f"   Response: {data['text'][:100]}...")
        else:
            print(f"   ✗ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Error calling generate: {e}")
        return False
    
    # Test batch generation
    print("   Testing /batch_generate endpoint...")
    try:
        response = requests.post(
            f"{base_url}/batch_generate",
            json={
                "prompts": [
                    "What is the capital of France? Answer in one word:",
                    "What is 3 + 5? Answer with just the number:"
                ],
                "max_tokens": 10,
                "temperature": 0.1,
                "do_sample": False
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Batch generation successful")
            print(f"   Responses: {len(data['texts'])} completions generated")
        else:
            print(f"   ✗ Batch generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error calling batch_generate: {e}")
        return False
    
    return True

def test_model_interface():
    """Test the TransformersInterface integration"""
    print("\n3. Testing ModelInterface integration...")
    
    # Set environment variables
    os.environ["MODEL_PROVIDER"] = "transformers"
    
    try:
        model = ModelFactory.create_model("transformers")
        print("   ✓ TransformersInterface created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create TransformersInterface: {e}")
        return False
    
    # Test single generation
    print("   Testing generate() method...")
    try:
        prompt = "Complete this sentence: The sky is"
        response = model.generate(prompt, max_tokens=20, temperature=0.5)
        if response:
            print(f"   ✓ Generate successful")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"   ✗ Generate returned empty response")
            return False
    except Exception as e:
        print(f"   ✗ Error in generate: {e}")
        return False
    
    # Test batch generation
    print("   Testing batch_generate() method...")
    try:
        prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms:"
        ]
        responses = model.batch_generate(prompts, max_tokens=50, temperature=0.5)
        if responses and all(responses):
            print(f"   ✓ Batch generate successful")
            print(f"   Generated {len(responses)} responses")
        else:
            print(f"   ✗ Batch generate returned empty responses")
            return False
    except Exception as e:
        print(f"   ✗ Error in batch_generate: {e}")
        return False
    
    return True

def test_evaluation_integration():
    """Test the full evaluation pipeline integration"""
    print("\n4. Testing evaluation pipeline integration...")
    
    # Test if run_evaluation.py recognizes the transformers provider
    print("   Testing provider recognition...")
    
    from src.model_interface import ModelFactory
    providers = ModelFactory.get_available_providers()
    
    if "transformers" in providers:
        print(f"   ✓ 'transformers' provider is registered")
    else:
        print(f"   ✗ 'transformers' provider not found in {providers}")
        return False
    
    # Test creating model through evaluation flow
    try:
        os.environ["MODEL_PROVIDER"] = "transformers"
        model = ModelFactory.create_model()
        print(f"   ✓ Model created through factory with default provider")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    print("\n   You can now run the full evaluation with:")
    print("   python src/run_evaluation.py --provider transformers --subset CloningScenarios --limit 2")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test Transformers integration')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000',
                       help='Base URL of the Transformers server')
    parser.add_argument('--skip-server', action='store_true',
                       help='Skip server tests (test only ModelInterface)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRANSFORMERS INTEGRATION TEST")
    print("="*60)
    
    # Update environment variable if custom URL provided
    if args.base_url != 'http://localhost:8000':
        os.environ["TRANSFORMERS_BASE_URL"] = args.base_url
    
    all_passed = True
    
    if not args.skip_server:
        # Test server health
        if not test_server_health(args.base_url):
            print("\n⚠️  Server is not running. Skipping API tests.")
            print("\nTo start the server, run:")
            print("python scripts/start_transformers_server.py --model meta-llama/Llama-2-7b-hf --port 8000")
            all_passed = False
        else:
            # Test direct API
            if not test_direct_api(args.base_url):
                all_passed = False
    
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
        print("1. Ensure your .env file has TRANSFORMERS_BASE_URL and TRANSFORMERS_MODEL_NAME set")
        print("2. Start the server: python scripts/start_transformers_server.py --model <model> --gpu 0 --port 8000")
        print("3. Run evaluation: python src/run_evaluation.py --provider transformers --subset <subset>")

if __name__ == "__main__":
    main()