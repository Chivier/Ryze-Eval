#!/usr/bin/env python3
"""
Test script for the deployed vision-language models
"""

import requests
import json
import base64
from PIL import Image
import io

def test_text_generation(port: int, model_name: str):
    """Test text generation for a model"""
    url = f"http://localhost:{port}/generate"
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} on port {port}")
    print(f"{'='*60}")
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result['generated_text'][:200]}...")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
        
        print("-" * 40)

def test_health_check(port: int, model_name: str):
    """Check if the model server is healthy"""
    url = f"http://localhost:{port}/health"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {model_name} (port {port}): {response.json()}")
        else:
            print(f"✗ {model_name} (port {port}): Error {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ {model_name} (port {port}): Connection failed - {e}")

def main():
    """Main test function"""
    models = [
        (8010, "Kimi-VL"),
        (8011, "OpenVLA"),
        (8012, "DeepSeek-VL")
    ]
    
    print("="*60)
    print("TESTING DEPLOYED MODELS")
    print("="*60)
    
    # First, check health of all servers
    print("\n🏥 Health Check:")
    for port, name in models:
        test_health_check(port, name)
    
    # Test text generation for each model
    print("\n📝 Text Generation Tests:")
    for port, name in models:
        test_text_generation(port, name)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    
    # Print curl examples
    print("\n📌 You can also test manually with curl:")
    print("\n# Test Kimi-VL:")
    print('''curl -X POST "http://localhost:8010/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "Tell me a joke", "max_new_tokens": 50}' ''')
    
    print("\n# Test OpenVLA:")
    print('''curl -X POST "http://localhost:8011/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "What is AI?", "max_new_tokens": 50}' ''')
    
    print("\n# Test DeepSeek-VL:")
    print('''curl -X POST "http://localhost:8012/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "Hello world", "max_new_tokens": 50}' ''')

if __name__ == "__main__":
    main()
