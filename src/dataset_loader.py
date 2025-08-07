import os
import json
from datasets import load_dataset
from tqdm import tqdm

class LabBenchDatasetLoader:
    def __init__(self, cache_dir="./data"):
        self.cache_dir = cache_dir
        self.dataset = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_dataset(self):
        print("Downloading Lab-Bench dataset from Hugging Face...")
        try:
            # Lab-Bench requires loading each config separately
            configs = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 
                      'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
            
            self.dataset = {}
            for config in configs:
                print(f"  Loading {config}...")
                self.dataset[config] = load_dataset("futurehouse/lab-bench", 
                                                   config, 
                                                   cache_dir=self.cache_dir)['train']
            
            print(f"Dataset downloaded successfully!")
            self._print_dataset_info()
            return self.dataset
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def _print_dataset_info(self):
        if self.dataset:
            print("\n=== Dataset Information ===")
            for split in self.dataset.keys():
                print(f"{split}: {len(self.dataset[split])} examples")
                if len(self.dataset[split]) > 0:
                    print(f"  Columns: {self.dataset[split].column_names}")
    
    def save_to_json(self, output_dir="./data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.dataset:
            print("No dataset loaded. Please download first.")
            return
        
        for split in self.dataset.keys():
            output_file = os.path.join(output_dir, f"lab_bench_{split}.json")
            print(f"Saving {split} split to {output_file}...")
            
            data = []
            for item in tqdm(self.dataset[split]):
                data.append(item)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(data)} examples to {output_file}")
    
    def load_from_json(self, split="test", data_dir="./data/processed"):
        json_file = os.path.join(data_dir, f"lab_bench_{split}.json")
        
        if not os.path.exists(json_file):
            print(f"File {json_file} not found. Please download and save the dataset first.")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples from {split} split")
        return data
    
    def get_sample(self, n=5, split="test"):
        if self.dataset and split in self.dataset:
            samples = []
            for i in range(min(n, len(self.dataset[split]))):
                samples.append(self.dataset[split][i])
            return samples
        return None

if __name__ == "__main__":
    loader = LabBenchDatasetLoader()
    
    dataset = loader.download_dataset()
    
    if dataset:
        loader.save_to_json()
        
        print("\n=== Sample Data ===")
        samples = loader.get_sample(n=2)
        if samples:
            for i, sample in enumerate(samples):
                print(f"\nSample {i+1}:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")