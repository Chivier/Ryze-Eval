#!/usr/bin/env python3
import requests
import json

# Test each model
models = [
    ("Kimi-VL", "http://localhost:8010"),
    ("DeepSeek-VL", "http://localhost:8012")
]

for name, url in models:
    print(f"\n{'='*50}")
    print(f"Testing {name} at {url}")
    print('='*50)
    
    # Test health
    try:
        health = requests.get(f"{url}/health").json()
        print(f"✓ Health check: {health}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        continue
    
    # Test generation
    try:
        response = requests.post(
            f"{url}/generate",
            json={
                "prompt": "What is the capital of France? Answer in one word.",
                "max_new_tokens": 20,
                "temperature": 0.1
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generation successful:")
            print(f"  Response: {result['generated_text'][:100]}")
        else:
            print(f"✗ Generation failed: {response.status_code}")
            print(f"  Error: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Generation error: {e}")

print("\n" + "="*50)
print("All tests completed!")
print("="*50)