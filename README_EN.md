### Ryze-Eval

A lightweight toolkit to evaluate and visualize the Lab-Bench multimodal QA dataset for scientific reasoning.

![framework](./imgs/framework.png)

---

## Highlights

- **Dataset management**: Download each Lab-Bench subset separately and save to JSON
- **Evaluation**: Run MCQ evaluation on any subset or all subsets, with optional sampling limit
- **Model providers**: Unified interface for `ollama / openai / deepseek / gemini / anthropic`
- **Results**: Store detailed records and summary metrics (accuracy, coverage, avg latency)
- **Visualization**: Generate subset comparison, subtask heatmap, and response time plots
- **Sample export**: Extract example images from FigQA/TableQA for quick inspection

---

## Project Layout

- `src/run_evaluation.py` entry point to run evaluations
- `src/dataset_loader.py` dataset download/save/load utilities
- `src/evaluator.py` evaluation logic and metrics aggregation
- `src/model_interface.py` unified interfaces to model providers
- `src/visualizer.py` plots and summary report generation
- `src/analyze_dataset.py` dataset-wide stats and summaries
- `src/extract_images.py` export sample images
- `src/test_dataset_loading.py` verify per-config loading correctness
- `requirements.txt` Python dependencies

---

## Installation

1) Python: 3.10+
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Environment variables: create a `.env` in the repository root

```bash
# Default provider: ollama / openai / deepseek / gemini / anthropic
MODEL_PROVIDER=ollama

# Optional common knobs
MAX_TOKENS=4096
TEMPERATURE=0.7

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# OpenAI
# OPENAI_API_KEY=your_key
# OPENAI_MODEL=gpt-4o

# DeepSeek
# DEEPSEEK_API_KEY=your_key
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# DEEPSEEK_MODEL=deepseek-v3

# Gemini
# GEMINI_API_KEY=your_key
# GEMINI_MODEL=gemini-2.5-pro

# Anthropic
# ANTHROPIC_API_KEY=your_key
# ANTHROPIC_MODEL=claude-3-5-sonnet
```

---

## Quick Start

- Download and save the dataset only (to `./data/processed`):

```bash
python -m src.run_evaluation --download-only
```

- Run an evaluation (example: FigQA only, first 50 examples, save to `./results`):

```bash
python -m src.run_evaluation --provider ollama --subset FigQA --limit 50 --output-dir ./results --verbose
```

- Evaluate all subsets:

```bash
python -m src.run_evaluation --provider openai --subset all --output-dir ./results
```

- Visualize the latest results (artifacts saved under `results/visualizations/`):

```bash
python -m src.visualizer --results-dir ./results
```

- Run dataset-wide analysis (produces `dataset_analysis.json` and `dataset_summary.csv`):

```bash
python -m src.analyze_dataset
```

- Export sample images from FigQA/TableQA to `./sample_images/`:

```bash
python -m src.extract_images
```

- Verify per-config dataset loading:

```bash
python -m src.test_dataset_loading
```

---

## Outputs

- `results/detailed_results_*.json`: per-question raw responses and judgements
- `results/evaluation_report_*.json`: overall, per-subset, and per-subtask metrics
- `results/visualizations/`: plots and a text summary report
- `dataset_analysis.json`, `dataset_summary.csv`: dataset analysis and summary
- `sample_images/`: exported example images and `image_info.json`

---

## Troubleshooting

- Model init errors: check `.env` variables, ensure Ollama service is running or cloud API keys are valid
- Low coverage: increase `MAX_TOKENS` or verify the parser returns a valid choice letter (A–H)

---

## License

This project is licensed under the terms described in the `LICENSE` file.


