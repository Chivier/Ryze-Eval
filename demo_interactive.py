#!/usr/bin/env python3
"""
Demo script showing the interactive TUI capabilities
"""
import os
import sys
import requests
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def detect_running_models(host="localhost", port_range=(8000, 8005)):
    """Detect models running on multiple ports"""
    running_models = []
    
    print("🔍 Scanning for running models...")
    print("-" * 50)
    
    for port in range(port_range[0], port_range[1]):
        base_url = f"http://{host}:{port}"
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data and models_data["data"]:
                    model_info = models_data["data"][0]
                    model_name = model_info.get("id", f"model-{port}")
                    
                    # Test with a simple request to get more info
                    try:
                        test_response = requests.post(
                            f"{base_url}/v1/chat/completions",
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": "Hi"}],
                                "max_tokens": 10,
                                "temperature": 0.1
                            },
                            timeout=10
                        )
                        status = "✅ Active" if test_response.status_code == 200 else "⚠️ Responding with errors"
                    except:
                        status = "⚠️ Connection only"
                    
                    running_models.append({
                        'port': port,
                        'url': base_url,
                        'model_name': model_name,
                        'status': status,
                        'models_data': models_data
                    })
                    
                    print(f"Port {port:4d}: {model_name:20s} - {status}")
                    
        except requests.exceptions.RequestException:
            # No service on this port, skip silently
            pass
    
    print("-" * 50)
    return running_models

def get_model_display_info(model_name):
    """Get display information for model based on name patterns"""
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        return "Qwen 2.5", "🧠"
    elif "llama" in model_name_lower:
        return "Llama 3.2", "🦙" 
    elif "gemma" in model_name_lower:
        return "Gemma 3", "💎"
    elif "kimi" in model_name_lower:
        return "Kimi VL", "👁️"
    elif "gpt" in model_name_lower:
        return "GPT OSS", "🤖"
    elif "deepseek" in model_name_lower:
        return "DeepSeek", "🔍"
    else:
        return "Unknown Model", "❓"

def main():
    print("🚀 Ryze Evaluation - Model Scanner Demo")
    print("=" * 60)
    
    # Detect running models
    running_models = detect_running_models()
    
    if not running_models:
        print("❌ No running models detected!")
        print("💡 Make sure you have models running on ports 8000-8004")
        return
    
    print(f"\n📋 Found {len(running_models)} running model(s)")
    print("=" * 60)
    
    # Display available models
    for i, model in enumerate(running_models, 1):
        model_type, icon = get_model_display_info(model['model_name'])
        print(f"{i:2d}. {icon} {model_type}")
        print(f"     Model: {model['model_name']}")
        print(f"     URL: {model['url']}")
        print(f"     Status: {model['status']}")
        print()
    
    print("🎯 Interactive Features:")
    print("  • Model selection from available ports")
    print("  • Quick test, medium test, or full evaluation")  
    print("  • Model-specific result directories")
    print("  • Progress tracking and error handling")
    print("\n💡 To run the full interactive evaluation:")
    print("     uv run python finish_ryze_eval.py")

if __name__ == "__main__":
    main()