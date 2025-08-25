#!/usr/bin/env python3
"""
Interactive TUI script to select and test models for Ryze evaluation
Supports multiple models running on different ports
"""
import os
import sys
import requests
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.model_interface import ModelFactory
from src.evaluator import LabBenchEvaluator
from datasets import load_dataset

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

def interactive_model_selection():
    """Interactive model selection interface"""
    print("🚀 Ryze Evaluation - Interactive Model Selection")
    print("=" * 60)
    
    # Detect running models
    running_models = detect_running_models()
    
    if not running_models:
        print("❌ No running models detected!")
        print("💡 Make sure you have models running on ports 8000-8004")
        return None
    
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
    
    # Get user selection
    while True:
        try:
            print(f"🎯 Select a model to evaluate (1-{len(running_models)}) or 'q' to quit:")
            choice = input("Enter your choice: ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("👋 Goodbye!")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(running_models):
                selected_model = running_models[choice_num - 1]
                model_type, icon = get_model_display_info(selected_model['model_name'])
                print(f"\n✅ Selected: {icon} {model_type} ({selected_model['model_name']})")
                print(f"📡 URL: {selected_model['url']}")
                return selected_model
            else:
                print(f"❌ Please enter a number between 1 and {len(running_models)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n👋 Selection cancelled")
            return None

def test_selected_model(selected_model, test_message="Hello! Can you tell me about yourself?"):
    """Test the selected model with a sample request"""
    print(f"\n🧪 Testing selected model...")
    print("-" * 40)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{selected_model['url']}/v1/chat/completions",
            json={
                "model": selected_model['model_name'],
                "messages": [{"role": "user", "content": test_message}],
                "max_tokens": 150,
                "temperature": 0.7
            },
            timeout=30
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Test successful! (Response time: {response_time:.2f}s)")
                print(f"📝 Question: {test_message}")
                print(f"🤖 Response: {content}")
                
                if "usage" in result:
                    usage = result["usage"]
                    print(f"📊 Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                          f"Completion: {usage.get('completion_tokens', 'N/A')}")
                
                return True
            else:
                print("❌ Unexpected response format")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def setup_environment_for_selected_model(selected_model):
    """Setup environment variables for selected model"""
    
    # Set environment variables for transformers provider
    os.environ["MODEL_PROVIDER"] = "transformers"
    os.environ["TRANSFORMERS_BASE_URL"] = selected_model['url']
    os.environ["TRANSFORMERS_MODEL_NAME"] = selected_model['model_name']
    os.environ["MAX_TOKENS"] = "2048"
    os.environ["TEMPERATURE"] = "0.7"
    
    print(f"🔧 Environment configured for {selected_model['model_name']} at {selected_model['url']}")


def get_evaluation_options():
    """Get evaluation options from user"""
    print("\n📊 Evaluation Options")
    print("-" * 30)
    
    # Ask about evaluation scope
    print("1. Quick test (5 questions per subset)")
    print("2. Medium test (20 questions per subset)")
    print("3. Full evaluation (all questions)")
    
    while True:
        try:
            choice = input("\nSelect evaluation scope (1-3): ").strip()
            if choice == "1":
                return {"limit": 5, "scope": "Quick test"}
            elif choice == "2":
                return {"limit": 20, "scope": "Medium test"}
            elif choice == "3":
                return {"limit": None, "scope": "Full evaluation"}
            else:
                print("❌ Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n👋 Evaluation cancelled")
            return None

def run_evaluation_with_selected_model(selected_model, eval_options):
    """Run Lab-Bench evaluation with the selected model"""
    print(f"\n{'='*80}")
    print(f"STARTING LAB-BENCH EVALUATION")
    print(f"Model: {selected_model['model_name']} at {selected_model['url']}")
    print(f"Scope: {eval_options['scope']}")
    if eval_options['limit']:
        print(f"Limit: {eval_options['limit']} questions per subset")
    print(f"{'='*80}")
    
    try:
        # Initialize model
        model = ModelFactory.create_model("transformers")
        print("✅ Model interface initialized")
        
        # Load Lab-Bench dataset
        print("\n📚 Loading Lab-Bench dataset...")
        configs = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 
                  'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
        
        dataset = {}
        for config in configs:
            print(f"  Loading {config}...")
            try:
                dataset[config] = load_dataset("futurehouse/lab-bench", config)['train']
            except Exception as e:
                print(f"  ⚠️  Failed to load {config}: {e}")
                continue
        
        if not dataset:
            print("❌ No datasets loaded successfully")
            return False
            
        print(f"✅ Loaded {len(dataset)} subsets")
        
        # Initialize evaluator
        evaluator = LabBenchEvaluator(model, verbose=True)
        
        # Run evaluation
        print(f"\n🎯 Evaluating {len(dataset)} subsets...")
        
        for subset_name in dataset.keys():
            print(f"\n📊 Evaluating {subset_name}...")
            subset_data = dataset[subset_name]
            evaluator.evaluate_subset(subset_data, subset_name, limit=eval_options['limit'])
        
        # Save results
        output_dir = f"./results_{selected_model['model_name'].replace('/', '_')}"
        print(f"\n💾 Saving results to {output_dir}...")
        report = evaluator.save_results(output_dir)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to {output_dir}/")
        print(f"  - Detailed results: detailed_results_*.json")
        print(f"  - Evaluation report: evaluation_report_*.json")
        
        # Show summary
        if report and 'overall' in report:
            overall = report['overall']
            print(f"\n📈 Summary Results:")
            print(f"  Accuracy: {overall.get('accuracy', 0):.1%}")
            print(f"  Coverage: {overall.get('coverage', 0):.1%}")
            print(f"  Avg Response Time: {overall.get('avg_response_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def main():
    """Main function with interactive interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Ryze evaluation with model selection")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for running models")
    parser.add_argument("--port-range", nargs=2, type=int, default=[8000, 8005], 
                       help="Port range to scan (default: 8000 8005)")
    parser.add_argument("--model-name", type=str, help="Specific model name to use (skips interactive selection)")
    parser.add_argument("--port", "-p", type=int, help="Specific port to use (skips interactive selection)")
    parser.add_argument("--host", default="localhost", help="Model host (default: localhost)")
    parser.add_argument("--limit", type=int, help="Limit questions per subset (overrides interactive selection)")
    
    args = parser.parse_args()
    
    try:
        # Check if specific model and port are provided
        if args.port and args.model_name:
            # Direct model specification mode
            selected_model = {
                'port': args.port,
                'url': f"http://{args.host}:{args.port}",
                'model_name': args.model_name,
                'status': 'Specified by user'
            }
            
            print(f"🎯 Using specified model:")
            model_type, icon = get_model_display_info(selected_model['model_name'])
            print(f"   {icon} {model_type} ({selected_model['model_name']})")
            print(f"   URL: {selected_model['url']}")
            
            # Test the specified model
            if not test_selected_model(selected_model):
                print("❌ Specified model test failed. Please check your model server.")
                return 1
                
        else:
            # Interactive model selection
            selected_model = interactive_model_selection()
            if not selected_model:
                return 0
        
        if args.scan_only:
            print("✅ Model scan completed!")
            return 0
        
        # Setup environment for selected model
        setup_environment_for_selected_model(selected_model)
        
        # Get evaluation options (skip if limit is provided)
        if args.limit is not None:
            if args.limit == 0:
                eval_options = {"limit": None, "scope": "Full evaluation"}
            else:
                eval_options = {"limit": args.limit, "scope": f"Limited to {args.limit} questions"}
            print(f"📊 Using specified evaluation scope: {eval_options['scope']}")
        else:
            eval_options = get_evaluation_options()
            if not eval_options:
                return 0
        
        # Confirm before starting evaluation (skip if running non-interactively)
        if not (args.port and args.model_name and args.limit is not None):
            model_type, icon = get_model_display_info(selected_model['model_name'])
            print(f"\n🚀 Ready to start evaluation:")
            print(f"   Model: {icon} {model_type} ({selected_model['model_name']})")
            print(f"   Scope: {eval_options['scope']}")
            
            confirm = input("\nProceed with evaluation? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("👋 Evaluation cancelled")
                return 0
        
        # Run evaluation
        if run_evaluation_with_selected_model(selected_model, eval_options):
            print("\n🎉 Ryze evaluation completed successfully!")
            return 0
        else:
            print("\n❌ Evaluation failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0

if __name__ == "__main__":
    exit(main())