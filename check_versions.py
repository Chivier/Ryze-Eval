#!/usr/bin/env python3
"""
Check versions of installed packages
"""

import sys
import importlib

def check_package(name, import_name=None):
    """Check if a package is installed and get its version"""
    if import_name is None:
        import_name = name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: not installed")
        return False

def main():
    print("Checking installed packages...\n")
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("einops", "einops"),
        ("safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "google.protobuf"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("PIL", "PIL"),
        ("timm", "timm"),
        ("opencv-python", "cv2"),
        ("scipy", "scipy"),
        ("qwen-vl-utils", "qwen_vl_utils"),
        ("tiktoken", "tiktoken"),
        ("attrdict", "attrdict"),
        ("gym", "gym"),
        ("dm-tree", "tree"),
        ("ftfy", "ftfy"),
        ("regex", "regex"),
        ("xformers", "xformers"),
        ("flash-attn", "flash_attn"),
        ("bitsandbytes", "bitsandbytes"),
    ]
    
    failed = []
    for package in packages:
        if len(package) == 2:
            name, import_name = package
        else:
            name = import_name = package
        
        if not check_package(name, import_name):
            failed.append(name)
    
    if failed:
        print(f"\n✗ Missing packages: {', '.join(failed)}")
        print("\nTo install missing packages, run:")
        print(f"uv pip install {' '.join(failed)}")
    else:
        print("\n✓ All required packages are installed!")

if __name__ == "__main__":
    main()
