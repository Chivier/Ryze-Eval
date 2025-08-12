### Ryze-Eval | [English](./README_EN.md)

一个用于评测科研类多模态问答数据集 Lab-Bench 的轻量级评测与可视化工具集。

![framework](./imgs/framework.png)

---

## 功能概览

- **数据获取与管理**: 按配置分别下载 Lab-Bench 各子集并保存为 JSON
- **评测执行**: 支持对单个或全部子集进行选择题评测，可限制条数快速验证
- **多模型对接**: 通过统一接口对接 `ollama / openai / deepseek / gemini / anthropic / transformers / vllm`
- **本地模型支持**: 支持通过 Transformers 和 vLLM 加载本地模型，包括视觉语言模型
- **结果保存**: 自动输出明细与汇总指标（准确率、覆盖率、平均响应时长等）
- **结果可视化**: 一键生成子集对比、子任务热力图、响应时长分布图
- **样例导出**: 从 FigQA / TableQA 子集导出示例图片，便于快速查看

---

## 目录结构

- `src/run_evaluation.py` 运行评测入口
- `src/dataset_loader.py` 数据集下载/保存/加载
- `src/evaluator.py` 评测逻辑与指标统计
- `src/model_interface.py` 各模型提供商的统一接口
- `src/visualizer.py` 结果可视化与摘要报告
- `src/analyze_dataset.py` 数据概览与统计
- `src/extract_images.py` 抽取样例图片
- `src/test_dataset_loading.py` 数据加载验证脚本
- `scripts/start_transformers_server.py` Transformers 模型服务启动脚本
- `scripts/start_vllm_server.py` vLLM 高性能推理服务启动脚本
- `scripts/test_transformers.py` Transformers 集成测试
- `scripts/test_vllm.py` vLLM 集成测试
- `requirements.txt` 依赖

---

## 安装与环境

1) Python 版本: 3.10+
2) 安装依赖:

```bash
pip install -r requirements.txt
```

3) 配置环境变量（在项目根目录创建 `.env`）:

```bash
# 选择默认提供商: ollama / openai / deepseek / gemini / anthropic / transformers / vllm
MODEL_PROVIDER=ollama

# 通用可选项
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

# Transformers (本地模型)
# TRANSFORMERS_BASE_URL=http://localhost:8000
# TRANSFORMERS_MODEL_NAME=openvla/openvla-7b

# vLLM (高性能推理)
# VLLM_BASE_URL=http://localhost:8001
# VLLM_MODEL_NAME=meta-llama/Llama-2-7b-hf
```

---

## 快速开始

- 仅下载并保存数据集（到 `./data/processed`）

```bash
python -m src.run_evaluation --download-only
```

- 运行评测（示例：只评测 FigQA，限制前 50 条，保存到 `./results`）

```bash
python -m src.run_evaluation --provider ollama --subset FigQA --limit 50 --output-dir ./results --verbose
```

- 评测所有子集

```bash
python -m src.run_evaluation --provider openai --subset all --output-dir ./results
```

- 可视化最近一次评测结果（图片与摘要报告将保存在 `results/visualizations/`）

```bash
python -m src.visualizer --results-dir ./results
```

- 数据集全量分析与概览（会生成 `dataset_analysis.json` 与 `dataset_summary.csv`）

```bash
python -m src.analyze_dataset
```

- 抽取 FigQA/TableQA 示例图片到 `./sample_images/`

```bash
python -m src.extract_images
```

- 验证数据加载配置是否正确

```bash
python -m src.test_dataset_loading
```

---

## Lab-Bench 数据集结构

| 子数据集 | 样本数 | 描述 | 特点 |
|---------|--------|------|------|
| **LitQA2** | 199 | 文献问答 | 从科学文献中提取信息 |
| **DbQA** | 520 | 数据库问答 | 从生物数据库检索信息 |
| **SuppQA** | 82 | 补充材料问答 | 从补充材料中查找信息 |
| **FigQA** | 181 | 图像问答 | 包含图像，需要视觉推理 |
| **TableQA** | 244 | 表格问答 | 包含表格图像，需要表格理解 |
| **ProtocolQA** | 108 | 协议问答 | 生物实验协议故障排除 |
| **SeqQA** | 600 | 序列问答 | DNA/蛋白质序列操作 |
| **CloningScenarios** | 33 | 克隆场景 | 复杂分子克隆工作流 |

---

## 评分方案

- **Accuracy（准确率）**: 正确答案数 / 总问题数
- **Precision（精确率）**: 正确答案数 / 尝试回答的问题数
- **Coverage（覆盖率）**: 尝试回答的问题数 / 总问题数
- **Avg Response Time（平均响应时长）**

---

## 输出产物

- `results/detailed_results_*.json`: 每道题的原始响应与判断
- `results/evaluation_report_*.json`: 汇总指标（总体/子集/子任务）
- `results/visualizations/`: 可视化图片与摘要报告
- `dataset_analysis.json`、`dataset_summary.csv`: 数据集分析与汇总
- `sample_images/`: 抽取的示例图片与 `image_info.json`

---

## 使用本地模型

### 使用 Ollama

1) 确保 Ollama 已安装并运行：

```bash
ollama serve
```

2) 拉取模型：

```bash
ollama pull gemma3:12b
```

3) 小规模评测验证：

```bash
python -m src.run_evaluation --limit 5 --verbose
```

### 使用 Transformers（支持视觉语言模型）

1) 启动 Transformers 服务器：

```bash
# 标准文本模型
python scripts/start_transformers_server.py --model meta-llama/Llama-2-7b-hf --gpu 0 --port 8000

# 视觉语言模型（如 OpenVLA）
python scripts/start_transformers_server.py --model openvla/openvla-7b --gpu 0 --port 8000

# 量化模型（节省显存）
python scripts/start_transformers_server.py --model meta-llama/Llama-2-70b-hf --gpu 0 --quantization 4bit
```

2) 运行评测：

```bash
python -m src.run_evaluation --provider transformers --subset FigQA --limit 10
```

3) 测试集成：

```bash
python scripts/test_transformers.py
```

### 使用 vLLM（高性能推理）

1) 安装 vLLM：

```bash
pip install vllm
```

2) 启动 vLLM 服务器：

```bash
# 单 GPU
python scripts/start_vllm_server.py --model meta-llama/Llama-2-7b-hf --gpu 0 --port 8001

# 多 GPU 张量并行（大模型）
python scripts/start_vllm_server.py --model meta-llama/Llama-2-70b-hf --gpu 0,1,2,3 --tensor-parallel 4

# 量化模型
python scripts/start_vllm_server.py --model TheBloke/Llama-2-7B-AWQ --quantization awq --port 8001
```

3) 运行评测：

```bash
python -m src.run_evaluation --provider vllm --subset all --output-dir ./results
```

4) 测试集成：

```bash
python scripts/test_vllm.py --performance  # 包含性能测试
```

---

## 注意事项

1) FigQA 和 TableQA 需要多模态能力，纯文本模型可能表现不佳  
2) 首次运行会下载数据集（约几百 MB）  
3) 完整评测包含 1967 个问题，可能需要较长时间  
4) 建议先用 `--limit` 参数进行小规模测试
5) Transformers 和 vLLM 需要 GPU 支持，请确保有足够的显存
6) vLLM 提供更快的推理速度，适合批量评测和生产环境

---

## 常见问题

- 初始化模型报错：检查 `.env` 是否配置、Ollama 服务是否启动、云端 API Key 是否有效
- 评测未覆盖全部题目：可提高 `MAX_TOKENS` 或检查输出解析是否返回合法选项字母（A–H）
- Transformers 服务连接失败：确保服务已启动，检查端口是否正确（默认 8000）
- vLLM 显存不足：尝试使用量化模型或减少 `--gpu-memory-utilization` 参数
- 多 GPU 配置：vLLM 使用 `--tensor-parallel` 参数，Transformers 目前仅支持单 GPU

---

## 许可

本项目遵循 `LICENSE` 文件所述许可协议。