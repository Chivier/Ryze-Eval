# Lab-Bench Evaluation System 使用指南

## 项目概述

本项目实现了对 Lab-Bench 数据集的完整评测系统，支持多种模型接口（Ollama、OpenAI、DeepSeek、Gemini、Anthropic）。

## Lab-Bench 数据集结构

Lab-Bench 包含 8 个子数据集，共 30 个子任务：

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

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，配置你的模型：

```env
# 选择模型提供商
MODEL_PROVIDER=ollama  # 可选: ollama, openai, deepseek, gemini, anthropic

# Ollama 配置（本地模型）
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# 其他云端模型配置...
```

### 3. 运行评测

#### 分析数据集结构
```bash
python src/analyze_dataset.py
```

#### 运行完整评测
```bash
# 使用默认模型（Ollama）
python src/run_evaluation.py

# 使用特定模型
python src/run_evaluation.py --provider openai

# 评测特定子集
python src/run_evaluation.py --subset DbQA

# 限制样本数（用于测试）
python src/run_evaluation.py --limit 10 --verbose
```

#### 可视化结果
```bash
python src/visualizer.py
```

## 评分方案

### 评测指标

1. **Accuracy（准确率）**: 正确答案数 / 总问题数
2. **Precision（精确率）**: 正确答案数 / 尝试回答的问题数
3. **Coverage（覆盖率）**: 尝试回答的问题数 / 总问题数
4. **Response Time（响应时间）**: 平均回答时间

### 评分特点

- **多选题格式**: 每个问题包含1个正确答案和多个干扰项
- **分层评测**: 提供整体、子集、子任务三个层级的评测结果
- **特殊处理**: FigQA和TableQA需要图像理解能力（多模态）

## 输出文件

运行评测后，会在 `results/` 目录生成：

- `detailed_results_*.json`: 详细的每题结果
- `evaluation_report_*.json`: 评测报告摘要
- `visualizations/`: 可视化图表
  - `subset_performance_*.png`: 子集性能对比
  - `subtask_heatmap_*.png`: 子任务热力图
  - `response_times_*.png`: 响应时间分布
  - `summary_report_*.txt`: 文字总结报告

## 使用 Ollama 测试

1. 确保 Ollama 已安装并运行：
```bash
ollama serve
```

2. 拉取模型：
```bash
ollama pull gemma3:12b
```

3. 运行小规模测试：
```bash
python src/run_evaluation.py --limit 5 --verbose
```

## 支持的模型

- **Ollama**: 本地模型（gemma3:12b 等）
- **OpenAI**: GPT-5
- **DeepSeek**: DeepSeek-V3
- **Gemini**: Gemini 2.5 Pro
- **Anthropic**: Claude Opus 4.1

## 注意事项

1. FigQA 和 TableQA 需要多模态能力，纯文本模型可能表现不佳
2. 首次运行会下载数据集（约几百MB）
3. 完整评测包含 1967 个问题，可能需要较长时间
4. 建议先用 `--limit` 参数进行小规模测试