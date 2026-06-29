#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import sys
sys.path.insert(0, '/Nibelungen/hyq/Projects/Ryze-Eval/venvs/deepseek/DeepSeek-VL')

import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-vl-7b-chat",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    low_cpu_mem_usage=True,
    max_memory={0: "22GiB", 1: "22GiB"}
)

print(f"Model type: {type(model)}")
print(f"Model class name: {model.__class__.__name__}")
print(f"Has generate method: {hasattr(model, 'generate')}")

# Try to see what methods it has
methods = [m for m in dir(model) if not m.startswith('_')]
print(f"Available methods: {methods[:10]}...")  # First 10 methods