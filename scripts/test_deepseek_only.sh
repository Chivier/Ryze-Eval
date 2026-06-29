#!/bin/bash

# Test only the DeepSeek-VL model which is already running
echo "Testing deepseek-vl at port 8012..."
TRANSFORMERS_ENDPOINT="http://localhost:8012" MAX_TOKENS=4096 \
    python -m src.run_evaluation --provider transformers \
    --output-dir "./results/deepseek-vl" --verbose --limit 2
