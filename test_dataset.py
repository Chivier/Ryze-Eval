#!/usr/bin/env python3
from datasets import load_dataset
import traceback

try:
    print("Loading CloningScenarios...")
    ds = load_dataset('futurehouse/lab-bench', 'CloningScenarios', cache_dir='./data')['train']
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()