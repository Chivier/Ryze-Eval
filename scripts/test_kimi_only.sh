#!/bin/bash

# Test only the Kimi-VL model which is now running
echo "Testing kimi-vl at port 8010..."
TRANSFORMERS_ENDPOINT="http://localhost:8010" MAX_TOKENS=4096 \
    python -m src.run_evaluation --provider transformers \
    --output-dir "./results/kimi-vl" --verbose --limit 1
