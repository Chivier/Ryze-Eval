import argparse
import os
import sys
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_interface import ModelFactory
from src.evaluator import LabBenchEvaluator
from src.dataset_loader import LabBenchDatasetLoader

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Run Lab-Bench evaluation')
    parser.add_argument('--provider', type=str, default=None, 
                       help='Model provider (ollama, openai, deepseek, gemini, anthropic, transformers, vllm)')
    parser.add_argument('--subset', type=str, default='all',
                       help='Specific subset to evaluate or "all" for complete evaluation')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of questions per subset (for testing)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during evaluation')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download and save the dataset')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LAB-BENCH EVALUATION SYSTEM")
    print("="*80)
    
    if args.download_only:
        print("\n📥 Downloading Lab-Bench dataset...")
        loader = LabBenchDatasetLoader()
        dataset = loader.download_dataset()
        if dataset:
            loader.save_to_json()
            print("✓ Dataset downloaded and saved successfully")
        return
    
    provider = args.provider or os.getenv('MODEL_PROVIDER', 'ollama')
    print(f"\n🤖 Model Provider: {provider}")
    
    if provider == 'ollama':
        model_name = os.getenv('OLLAMA_MODEL', 'gemma3:12b')
    elif provider == 'openai':
        model_name = os.getenv('OPENAI_MODEL', 'gpt-5')
    elif provider == 'deepseek':
        model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-v3')
    elif provider == 'gemini':
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
    elif provider == 'anthropic':
        model_name = os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-1-20250805')
    elif provider == 'transformers':
        # For deployed transformers models, we specify the endpoint
        model_name = os.getenv('TRANSFORMERS_MODEL_NAME', 'kimi-vl')
        # You can also use environment variables for the endpoints:
        # TRANSFORMERS_ENDPOINT=http://localhost:8010 for Kimi-VL
        # TRANSFORMERS_ENDPOINT=http://localhost:8011 for OpenVLA  
        # TRANSFORMERS_ENDPOINT=http://localhost:8012 for DeepSeek-VL
    elif provider == 'vllm':
        model_name = os.getenv('VLLM_MODEL_NAME', 'meta-llama/Llama-2-7b-hf')
    else:
        model_name = 'unknown'
    
    print(f"📊 Model: {model_name}")
    
    try:
        model = ModelFactory.create_model(provider)
        print("✓ Model interface initialized")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        print("\nPlease check:")
        print("1. Your .env file is configured correctly")
        print("2. The model service is running (for Ollama)")
        print("3. API keys are valid (for cloud providers)")
        return
    
    print("\n📚 Loading Lab-Bench dataset...")
    try:
        # Lab-Bench requires loading each config separately
        configs = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 
                  'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
        
        dataset = {}
        for config in configs:
            print(f"  Loading {config}...")
            dataset[config] = load_dataset("futurehouse/lab-bench", config)['train']
        
        print(f"✓ Dataset loaded with {len(dataset)} subsets")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    evaluator = LabBenchEvaluator(model, verbose=args.verbose)
    
    subsets_to_evaluate = []
    if args.subset == 'all':
        subsets_to_evaluate = list(dataset.keys())
    elif args.subset in dataset:
        subsets_to_evaluate = [args.subset]
    else:
        print(f"✗ Subset '{args.subset}' not found in dataset")
        print(f"Available subsets: {', '.join(dataset.keys())}")
        return
    
    print(f"\n🎯 Evaluating subsets: {', '.join(subsets_to_evaluate)}")
    if args.limit:
        print(f"⚠️  Limited to {args.limit} questions per subset")
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    for subset_name in subsets_to_evaluate:
        subset_data = dataset[subset_name]
        evaluator.evaluate_subset(subset_data, subset_name, limit=args.limit)
    
    print("\n💾 Saving results...")
    report = evaluator.save_results(args.output_dir)
    
    print(f"\n✓ Results saved to {args.output_dir}/")
    print(f"  - Detailed results: detailed_results_*.json")
    print(f"  - Evaluation report: evaluation_report_*.json")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if args.limit:
        print("\n⚠️  Note: This was a limited evaluation. Run without --limit for complete results.")

if __name__ == "__main__":
    main()
