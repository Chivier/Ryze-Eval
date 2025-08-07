#!/usr/bin/env python3
"""
Test script for Lab-Bench dataset loading with proper config handling.

This script tests:
1. Loading each subset individually with correct config names
2. Combining all subsets into a single dataset object
3. Verifying the structure and data is loaded correctly
4. Printing detailed results for each subset
"""

import os
from datasets import load_dataset
from tqdm import tqdm

class LabBenchDatasetTester:
    def __init__(self, cache_dir="./data"):
        self.cache_dir = cache_dir
        self.datasets = {}
        self.config_names = [
            'CloningScenarios', 
            'DbQA', 
            'FigQA', 
            'LitQA2', 
            'ProtocolQA', 
            'SeqQA', 
            'SuppQA', 
            'TableQA'
        ]
        os.makedirs(cache_dir, exist_ok=True)
    
    def test_individual_config_loading(self):
        """Test loading each config individually."""
        print("=== Testing Individual Config Loading ===\n")
        
        for config_name in self.config_names:
            print(f"Loading config: {config_name}")
            try:
                dataset = load_dataset(
                    "futurehouse/lab-bench", 
                    config_name, 
                    cache_dir=self.cache_dir
                )
                self.datasets[config_name] = dataset
                print(f"  ✓ Successfully loaded {config_name}")
                
                # Print basic info about each split
                for split in dataset.keys():
                    num_examples = len(dataset[split])
                    columns = dataset[split].column_names
                    print(f"    {split}: {num_examples} examples, columns: {columns}")
                
                print()
                
            except Exception as e:
                print(f"  ✗ Failed to load {config_name}: {e}")
                print()
                return False
        
        print(f"Successfully loaded all {len(self.datasets)} configs!\n")
        return True
    
    def combine_datasets(self):
        """Combine all subsets into a single dataset structure."""
        print("=== Combining All Datasets ===\n")
        
        if not self.datasets:
            print("No datasets loaded. Cannot combine.")
            return None
        
        combined = {}
        
        # Get all unique splits across all configs
        all_splits = set()
        for dataset in self.datasets.values():
            all_splits.update(dataset.keys())
        
        print(f"Found splits: {list(all_splits)}")
        
        # For each split, combine data from all configs
        for split in all_splits:
            combined_data = []
            
            for config_name, dataset in self.datasets.items():
                if split in dataset:
                    # Add config name to each example for tracking
                    for example in dataset[split]:
                        example_with_config = dict(example)
                        example_with_config['config_name'] = config_name
                        combined_data.append(example_with_config)
            
            combined[split] = combined_data
            print(f"  {split}: {len(combined_data)} total examples")
        
        print(f"\nCombined dataset contains {sum(len(data) for data in combined.values())} total examples\n")
        return combined
    
    def verify_data_structure(self):
        """Verify the structure and content of loaded data."""
        print("=== Verifying Data Structure ===\n")
        
        # Check each config's structure
        for config_name, dataset in self.datasets.items():
            print(f"Config: {config_name}")
            
            for split in dataset.keys():
                if len(dataset[split]) > 0:
                    # Get first example to check structure
                    first_example = dataset[split][0]
                    print(f"  {split} split structure:")
                    for key, value in first_example.items():
                        value_type = type(value).__name__
                        if isinstance(value, str):
                            value_preview = value[:50] + "..." if len(value) > 50 else value
                            print(f"    {key}: {value_type} - '{value_preview}'")
                        else:
                            print(f"    {key}: {value_type} - {value}")
                else:
                    print(f"  {split} split: empty")
            print()
    
    def print_sample_data(self, n_samples=2):
        """Print sample data from each config."""
        print(f"=== Sample Data (first {n_samples} examples per config) ===\n")
        
        for config_name, dataset in self.datasets.items():
            print(f"Config: {config_name}")
            
            for split in dataset.keys():
                if len(dataset[split]) > 0:
                    print(f"  {split} split samples:")
                    
                    for i in range(min(n_samples, len(dataset[split]))):
                        example = dataset[split][i]
                        print(f"    Sample {i+1}:")
                        
                        for key, value in example.items():
                            if isinstance(value, str) and len(value) > 100:
                                print(f"      {key}: {value[:100]}...")
                            else:
                                print(f"      {key}: {value}")
                        print()
                else:
                    print(f"  {split} split: empty")
            print("-" * 50)
    
    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        print("Lab-Bench Dataset Loading Test\n")
        print("=" * 50)
        
        # Test 1: Load individual configs
        success = self.test_individual_config_loading()
        if not success:
            print("Individual config loading failed. Aborting.")
            return False
        
        # Test 2: Verify data structure
        self.verify_data_structure()
        
        # Test 3: Combine datasets
        combined = self.combine_datasets()
        if combined is None:
            print("Dataset combination failed.")
            return False
        
        # Test 4: Print sample data
        self.print_sample_data(n_samples=1)
        
        # Test 5: Summary statistics
        self.print_summary_statistics()
        
        print("✓ All tests completed successfully!")
        return True
    
    def print_summary_statistics(self):
        """Print summary statistics for all loaded datasets."""
        print("=== Summary Statistics ===\n")
        
        total_examples = 0
        config_stats = {}
        
        for config_name, dataset in self.datasets.items():
            config_total = sum(len(dataset[split]) for split in dataset.keys())
            config_stats[config_name] = config_total
            total_examples += config_total
        
        print("Examples per config:")
        for config_name, count in sorted(config_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_examples) * 100 if total_examples > 0 else 0
            print(f"  {config_name}: {count} examples ({percentage:.1f}%)")
        
        print(f"\nTotal examples across all configs: {total_examples}")
        print(f"Number of configs loaded: {len(self.datasets)}")
        
        # Check splits distribution
        split_stats = {}
        for dataset in self.datasets.values():
            for split in dataset.keys():
                if split not in split_stats:
                    split_stats[split] = 0
                split_stats[split] += len(dataset[split])
        
        print(f"\nExamples per split:")
        for split, count in split_stats.items():
            print(f"  {split}: {count} examples")
        print()


def main():
    """Main function to run the dataset loading test."""
    print("Starting Lab-Bench Dataset Loading Test...\n")
    
    try:
        tester = LabBenchDatasetTester(cache_dir="./data")
        success = tester.run_comprehensive_test()
        
        if success:
            print("\n🎉 Dataset loading test completed successfully!")
            print("\nThe Lab-Bench dataset can now be loaded properly using individual configs.")
            print("Each config should be loaded separately and then combined if needed.")
        else:
            print("\n❌ Dataset loading test failed.")
            print("Please check the error messages above for details.")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()