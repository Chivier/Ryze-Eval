import json
import os
from collections import defaultdict
from datasets import load_dataset
from typing import Dict, List, Any
import pandas as pd

class LabBenchAnalyzer:
    def __init__(self):
        self.dataset = None
        self.subsets = [
            'LitQA2', 'DbQA', 'SuppQA', 'FigQA', 
            'TableQA', 'ProtocolQA', 'SeqQA', 'CloningScenarios'
        ]
        self.analysis_results = {}
    
    def load_dataset(self):
        print("Loading Lab-Bench dataset from Hugging Face...")
        try:
            # Lab-Bench requires loading each config separately
            configs = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 
                      'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
            
            self.dataset = {}
            for config in configs:
                print(f"  Loading {config}...")
                self.dataset[config] = load_dataset("futurehouse/lab-bench", 
                                                   config)['train']
            
            print(f"✓ Dataset loaded successfully\n")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def analyze_subset(self, subset_name: str) -> Dict[str, Any]:
        if subset_name not in self.dataset:
            return {"error": f"Subset {subset_name} not found"}
        
        subset = self.dataset[subset_name]
        analysis = {
            "name": subset_name,
            "num_samples": len(subset),
            "columns": subset.column_names,
            "subtasks": set(),
            "has_images": False,
            "image_samples": [],
            "question_lengths": [],
            "num_choices": [],
            "sample_questions": []
        }
        
        for idx, item in enumerate(subset):
            if 'subtask' in item:
                analysis["subtasks"].add(item['subtask'])
            
            # Check for images and collect sample info
            if 'image' in item and item['image'] is not None:
                analysis["has_images"] = True
                if len(analysis["image_samples"]) < 2:
                    from PIL import Image
                    import io
                    try:
                        img = item['image']
                        if isinstance(img, Image.Image):
                            analysis["image_samples"].append({
                                "id": item.get('id', 'N/A'),
                                "format": img.format if hasattr(img, 'format') else 'Unknown',
                                "size": f"{img.size[0]}x{img.size[1]}" if hasattr(img, 'size') else 'Unknown',
                                "mode": img.mode if hasattr(img, 'mode') else 'Unknown',
                                "question_preview": item.get('question', '')[:100] + "..."
                            })
                    except:
                        pass
            
            if 'table_image' in item and item['table_image'] is not None:
                analysis["has_images"] = True
                if len(analysis["image_samples"]) < 2:
                    from PIL import Image
                    try:
                        img = item['table_image']
                        if isinstance(img, Image.Image):
                            analysis["image_samples"].append({
                                "id": item.get('id', 'N/A'),
                                "format": img.format if hasattr(img, 'format') else 'Unknown',
                                "size": f"{img.size[0]}x{img.size[1]}" if hasattr(img, 'size') else 'Unknown',
                                "mode": img.mode if hasattr(img, 'mode') else 'Unknown',
                                "question_preview": item.get('question', '')[:100] + "..."
                            })
                    except:
                        pass
            
            if 'question' in item:
                analysis["question_lengths"].append(len(item['question']))
            
            if 'distractors' in item and item['distractors']:
                num_choices = len(item['distractors']) + 1
                analysis["num_choices"].append(num_choices)
            
            if idx < 2:
                sample = {
                    "id": item.get('id', 'N/A'),
                    "question": item.get('question', 'N/A')[:200] + "..." if len(item.get('question', '')) > 200 else item.get('question', 'N/A'),
                    "ideal": item.get('ideal', 'N/A')[:100] + "..." if len(item.get('ideal', '')) > 100 else item.get('ideal', 'N/A'),
                    "num_distractors": len(item.get('distractors', [])),
                    "subtask": item.get('subtask', 'N/A')
                }
                analysis["sample_questions"].append(sample)
        
        analysis["subtasks"] = list(analysis["subtasks"])
        
        if analysis["question_lengths"]:
            analysis["avg_question_length"] = sum(analysis["question_lengths"]) / len(analysis["question_lengths"])
        else:
            analysis["avg_question_length"] = 0
        
        if analysis["num_choices"]:
            analysis["avg_num_choices"] = sum(analysis["num_choices"]) / len(analysis["num_choices"])
        else:
            analysis["avg_num_choices"] = 0
        
        del analysis["question_lengths"]
        del analysis["num_choices"]
        
        return analysis
    
    def analyze_all_subsets(self):
        print("=" * 80)
        print("LAB-BENCH DATASET COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        total_samples = 0
        all_subtasks = set()
        
        for subset_name in self.dataset.keys():
            print(f"\n{'='*40}")
            print(f"Analyzing: {subset_name}")
            print(f"{'='*40}")
            
            analysis = self.analyze_subset(subset_name)
            self.analysis_results[subset_name] = analysis
            
            print(f"✓ Samples: {analysis['num_samples']}")
            print(f"✓ Columns: {', '.join(analysis['columns'])}")
            print(f"✓ Has Images: {'Yes' if analysis['has_images'] else 'No'}")
            print(f"✓ Avg Question Length: {analysis['avg_question_length']:.0f} chars")
            print(f"✓ Avg Number of Choices: {analysis['avg_num_choices']:.1f}")
            print(f"✓ Subtasks ({len(analysis['subtasks'])}): {', '.join(sorted(analysis['subtasks']))}")
            
            # Print image sample info if available
            if analysis.get('image_samples'):
                print("\nImage/Table Examples:")
                for img_sample in analysis['image_samples']:
                    print(f"  - ID: {img_sample['id']}")
                    print(f"    Size: {img_sample['size']}, Format: {img_sample['format']}, Mode: {img_sample['mode']}")
                    print(f"    Question: {img_sample['question_preview']}")
            
            total_samples += analysis['num_samples']
            all_subtasks.update(analysis['subtasks'])
            
            print("\nSample Questions:")
            for i, sample in enumerate(analysis['sample_questions'], 1):
                print(f"\n  Sample {i}:")
                print(f"    ID: {sample['id']}")
                print(f"    Subtask: {sample['subtask']}")
                print(f"    Question: {sample['question']}")
                print(f"    Ideal Answer: {sample['ideal']}")
                print(f"    Distractors: {sample['num_distractors']}")
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        print(f"✓ Total Subsets: {len(self.dataset.keys())}")
        print(f"✓ Total Samples: {total_samples}")
        print(f"✓ Total Unique Subtasks: {len(all_subtasks)}")
        print(f"✓ Subtasks List: {', '.join(sorted(all_subtasks))}")
        
        return self.analysis_results
    
    def save_analysis(self, output_file: str = "dataset_analysis.json"):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Analysis saved to {output_file}")
    
    def create_summary_table(self):
        summary_data = []
        for subset_name, analysis in self.analysis_results.items():
            summary_data.append({
                'Subset': subset_name,
                'Samples': analysis['num_samples'],
                'Has Images': '✓' if analysis['has_images'] else '✗',
                'Avg Q Length': f"{analysis['avg_question_length']:.0f}",
                'Avg Choices': f"{analysis['avg_num_choices']:.1f}",
                'Subtasks': len(analysis['subtasks'])
            })
        
        df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        df.to_csv('dataset_summary.csv', index=False)
        print("\n✓ Summary table saved to dataset_summary.csv")

def main():
    analyzer = LabBenchAnalyzer()
    
    if analyzer.load_dataset():
        analyzer.analyze_all_subsets()
        analyzer.save_analysis()
        
        if analyzer.analysis_results:
            analyzer.create_summary_table()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("✓ Detailed analysis saved to: dataset_analysis.json")
        print("✓ Summary table saved to: dataset_summary.csv")
        print("✓ Ready to proceed with evaluation implementation")

if __name__ == "__main__":
    main()