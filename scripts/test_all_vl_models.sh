#!/bin/bash

# Test vision-language models sequentially (excluding OpenVLA)
for model in "kimi-vl:8010" "deepseek-vl:8012"; do
    name="${model%:*}"
    port="${model#*:}"
    echo "Testing $name at port $port..."
    TRANSFORMERS_ENDPOINT="http://localhost:$port" MAX_TOKENS=4096 \
        python -m src.run_evaluation --provider transformers \
        --output-dir "./results/$name" --verbose
done
