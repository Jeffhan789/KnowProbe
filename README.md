# KnowProbe v2.0（知探）

> English first. 中文版见后文。

[English](#english) | [中文](#中文)

> **Knowledge-Grounded Question Generation and RAG Evaluation Platform**
>
> A production-ready, engineering-grade toolkit for LLM-based question generation from structured knowledge, multi-strategy evaluation, and RAG pipeline assessment.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)

---

## English

### Overview

KnowProbe v2 is a complete engineering upgrade of the original COMP390 Final Year Project research prototype. It transforms the controlled study of knowledge-base question generation into a deployable, extensible platform with:

- **Multi-backend LLM support** — Ollama, vLLM, Transformers, OpenAI, DeepSeek, Claude
- **Rich prompt strategy library** — Zero-shot, Few-shot, Chain-of-Thought, Self-Consistency, ReAct
- **Comprehensive evaluation suite** — BLEU-4, ROUGE, BERTScore, LLM-as-Judge, multi-dimensional quality scoring
- **Full RAG pipeline evaluation** — Retrieval + Generation end-to-end assessment
- **RESTful API** — FastAPI with structured logging, metrics, and health checks
- **Interactive Dashboard** — Streamlit-based experiment management and visualization
- **Production tooling** — CLI, Docker, CI/CD, database persistence, configuration management

### Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/Jeffhan789/KnowProbe.git
cd KnowProbe

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[all]"
```

#### CLI Usage

```bash
# Generate a single question
kp generate single \
  --knowledge "(Marie Curie, won, Nobel Prize in Physics)" \
  --type factual \
  --strategy cot \
  --model llama3.1:8b

# Run a full experiment
kp experiment run \
  --name "strategy_comparison" \
  --models llama3.1:8b,qwen2.5:7b \
  --strategies zero_shot,few_shot,cot \
  --types factual,schema

# Evaluate generated questions
kp evaluate batch \
  --questions outputs/questions.json \
  --references data/references.json \
  --metrics bleu,rouge,bert_score

# Start API server
kp serve --reload

# Start Dashboard
kp dashboard
```

#### Docker Deployment

```bash
# Start API + Dashboard
docker-compose -f docker/docker-compose.yml up --build

# API available at http://localhost:8000
# Dashboard available at http://localhost:8501
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        KnowProbe v2.0                        │
├─────────────────────────────────────────────────────────────┤
│  CLI (Typer)  │  API (FastAPI)  │  Dashboard (Streamlit)     │
├───────────────┼─────────────────┼────────────────────────────┤
│               │    ┌─────────────┴─────────────┐              │
│  generate     │    │    Core Services           │              │
│  evaluate     │    │  • Question Generator      │              │
│  experiment   │    │  • Evaluator Suite         │              │
│  serve        │    │  • RAG Pipeline            │              │
│  dashboard    │    │  • Experiment Runner       │              │
│               │    │  • Prompt Strategy Engine  │              │
│               │    └─────────────┬─────────────┘              │
├───────────────┴──────────────────┼─────────────────────────────┤
│  Infrastructure                                                │
│  • Multi-Backend LLM Client (Ollama/HF/OpenAI/Claude/DS)    │
│  • Knowledge Base Parser (Triple/Schema/Text/Entity)         │
│  • Vector Store & Embeddings (sentence-transformers)         │
│  • Database (SQLAlchemy + SQLite/PostgreSQL)                 │
│  • Structured Logging (structlog)                             │
│  • Configuration Management (Pydantic Settings)                 │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
knowprobe/
├── src/knowprobe/
│   ├── core/           # Configuration, data models, base types
│   ├── parsers/        # Knowledge input parsers (triple, schema, text, entity)
│   ├── llm/            # Unified LLM client (multi-backend abstraction)
│   ├── prompts/        # Prompt strategy engine (5 strategies + example management)
│   ├── generators/     # Question generation engine
│   ├── evaluators/     # Evaluation metrics, quality scoring, experiment runner
│   ├── rag/            # RAG pipeline (retrieval, generation, evaluation)
│   ├── db/             # Database models (SQLAlchemy ORM)
│   ├── api/            # FastAPI RESTful service
│   ├── cli/            # Command-line interface (Typer)
│   ├── dashboard/      # Streamlit interactive dashboard
│   └── utils/          # Logging, validation utilities
├── configs/            # Configuration templates and prompt templates
├── docker/             # Docker and docker-compose files
├── tests/              # Unit and integration tests
├── docs/               # Documentation (MkDocs)
└── examples/           # Example datasets and scripts
```

### Features

#### 1. Knowledge Input Processing
- Parse **factual triples** `(S, P, O)`, **schema relations**, **free text**, and **entity descriptions**
- Auto-detect input type with heuristic classification
- Batch processing with error isolation

#### 2. Multi-Backend LLM Generation
- **Local**: Ollama, vLLM, Hugging Face Transformers
- **Cloud**: OpenAI, DeepSeek, Anthropic Claude
- Unified client with retry logic, connection pooling, and async support

#### 3. Prompt Strategy Engine
- **Zero-shot**: Direct instruction
- **Few-shot**: Dynamic example selection (Random / Similarity / Diversity-MMR)
- **Chain-of-Thought**: Step-by-step reasoning guidance
- **Self-Consistency**: N-sample majority voting
- **ReAct**: Thought-Action-Observation reasoning loop
- Jinja2 templates with hot-reload support

#### 4. Comprehensive Evaluation
- **Automatic metrics**: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, BERTScore, Self-BLEU, Distinct-N
- **Quality dimensions**: Relevance, Type Consistency, Answerability, Fluency, Structural Grounding
- **LLM-as-Judge**: Configurable criteria and rubrics
- **Statistical analysis**: t-tests, Cohen's d, confidence intervals

#### 5. RAG Pipeline Evaluation
- **Retrieval metrics**: Precision@K, Recall@K, MRR, NDCG, HitRate
- **Generation metrics**: Faithfulness, Answer Relevance, Context Precision
- **End-to-end latency tracking**

#### 6. Experiment Management
- Controlled experiment execution with factorial design
- Full provenance tracking (model, strategy, prompt, parameters)
- Statistical comparison across conditions
- Export to JSON, CSV, Markdown, LaTeX

### Configuration

KnowProbe uses a layered configuration system:

1. **Default config** (`configs/default.yaml`)
2. **Local config** (`configs/local.yaml` — gitignored)
3. **Environment variables** (`KNOWPROBE_*` prefix)
4. **Runtime overrides** (CLI flags, API parameters)

```yaml
# Example local.yaml
models:
  api:
    openai:
      api_key: "sk-..."  # Or set OPENAI_API_KEY env var

generation:
  temperature: 0.7
  max_length: 256

evaluation:
  metrics: [bleu, rouge, bert_score]
```

### Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/ tests/
ruff format src/ tests/

# Run type checking
mypy src/

# Run tests
pytest --cov=knowprobe

# Build docs
mkdocs serve
```

### License

MIT License — see [LICENSE](LICENSE) for details.

### Acknowledgments

- Original research conducted as COMP390 Final Year Project
- AI agents assisted in prototyping, scripting, and documentation
- Human oversight on research design, experimental control, and interpretation

### Changelog

#### v2.0.0 (2025-07)
- Complete engineering upgrade from research prototype
- Added multi-backend LLM client, prompt strategy engine, RAG evaluation
- Added RESTful API, Streamlit dashboard, CLI, Docker support
- Added database persistence, configuration management, CI/CD

#### v1.0.0 (Original Research)
- Controlled study of LLM question generation from structured knowledge
- Model comparison: Llama-3.1-8B, Qwen-2.5-7B, Flan-T5-Large
- Prompt strategies: Zero-shot, Few-shot, Chain-of-Thought
- BLEU-4 evaluation on factual vs schema questions

---

## 中文

### 概述

KnowProbe v2 是 COMP390 本科毕业设计研究原型的完整工程化升级。它将知识库问题生成的受控研究转化为可部署、可扩展的平台，具备：

- **多后端 LLM 支持** — Ollama、vLLM、Transformers、OpenAI、DeepSeek、Claude
- **丰富的提示策略库** — Zero-shot、Few-shot、Chain-of-Thought、Self-Consistency、ReAct
- **全面的评估套件** — BLEU-4、ROUGE、BERTScore、LLM Judge、多维度质量评分
- **完整 RAG 流水线评估** — 检索 + 生成端到端评估
- **RESTful API** — FastAPI，结构化日志、监控指标、健康检查
- **交互式 Dashboard** — 基于 Streamlit 的实验管理与可视化
- **生产级工具链** — CLI、Docker、CI/CD、数据库持久化、配置管理

### 快速开始

#### 安装

```bash
# 克隆仓库
git clone https://github.com/Jeffhan789/KnowProbe.git
cd KnowProbe

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[all]"
```

#### CLI 使用

```bash
# 生成单个问题
kp generate single \
  --knowledge "(Marie Curie, won, Nobel Prize in Physics)" \
  --type factual \
  --strategy cot \
  --model llama3.1:8b

# 运行完整实验
kp experiment run \
  --name "strategy_comparison" \
  --models llama3.1:8b,qwen2.5:7b \
  --strategies zero_shot,few_shot,cot \
  --types factual,schema

# 评估生成的问题
kp evaluate batch \
  --questions outputs/questions.json \
  --references data/references.json \
  --metrics bleu,rouge,bert_score

# 启动 API 服务
kp serve --reload

# 启动 Dashboard
kp dashboard
```

#### Docker 部署

```bash
# 启动 API + Dashboard
docker-compose -f docker/docker-compose.yml up --build

# API 访问地址 http://localhost:8000
# Dashboard 访问地址 http://localhost:8501
```

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        KnowProbe v2.0                        │
├─────────────────────────────────────────────────────────────┤
│  CLI (Typer)  │  API (FastAPI)  │  Dashboard (Streamlit)     │
├───────────────┼─────────────────┼────────────────────────────┤
│               │    ┌─────────────┴─────────────┐              │
│  generate     │    │    核心服务                 │              │
│  evaluate     │    │  • 问题生成引擎              │              │
│  experiment   │    │  • 评估套件                 │              │
│  serve        │    │  • RAG 流水线               │              │
│  dashboard    │    │  • 实验运行器                │              │
│               │    │  • 提示策略引擎              │              │
│               │    └─────────────┬─────────────┘              │
├───────────────┴──────────────────┼─────────────────────────────┤
│  基础设施                                                      │
│  • 多后端 LLM 客户端 (Ollama/HF/OpenAI/Claude/DS)          │
│  • 知识库解析器 (三元组/模式/文本/实体)                    │
│  • 向量存储与嵌入 (sentence-transformers)                  │
│  • 数据库 (SQLAlchemy + SQLite/PostgreSQL)                 │
│  • 结构化日志 (structlog)                                   │
│  • 配置管理 (Pydantic Settings)                            │
└─────────────────────────────────────────────────────────────┘
```

### 项目结构

```
knowprobe/
├── src/knowprobe/
│   ├── core/           # 配置、数据模型、基础类型
│   ├── parsers/        # 知识输入解析器（三元组、模式、文本、实体）
│   ├── llm/            # 统一 LLM 客户端（多后端抽象）
│   ├── prompts/        # 提示策略引擎（5 种策略 + 示例管理）
│   ├── generators/     # 问题生成引擎
│   ├── evaluators/     # 评估指标、质量评分、实验运行器
│   ├── rag/            # RAG 流水线（检索、生成、评估）
│   ├── db/             # 数据库模型（SQLAlchemy ORM）
│   ├── api/            # FastAPI RESTful 服务
│   ├── cli/            # 命令行界面（Typer）
│   ├── dashboard/      # Streamlit 交互式仪表板
│   └── utils/          # 日志、验证工具
├── configs/            # 配置模板和提示模板
├── docker/             # Docker 和 docker-compose 文件
├── tests/              # 单元测试和集成测试
├── docs/               # 文档（MkDocs）
└── examples/           # 示例数据集和脚本
```

### 功能特性

#### 1. 知识输入处理
- 解析**事实三元组** `(S, P, O)`、**模式关系**、**自由文本**和**实体描述**
- 启发式分类自动检测输入类型
- 批量处理，错误隔离

#### 2. 多后端 LLM 生成
- **本地**：Ollama、vLLM、Hugging Face Transformers
- **云端**：OpenAI、DeepSeek、Anthropic Claude
- 统一客户端，支持重试逻辑、连接池和异步操作

#### 3. 提示策略引擎
- **Zero-shot**：直接指令
- **Few-shot**：动态示例选择（随机 / 相似度 / 多样性-MMR）
- **Chain-of-Thought**：逐步推理引导
- **Self-Consistency**：N 样本多数投票
- **ReAct**：思考-行动-观察推理循环
- Jinja2 模板，支持热重载

#### 4. 全面评估
- **自动指标**：BLEU-1/2/3/4、ROUGE-1/2/L、METEOR、BERTScore、Self-BLEU、Distinct-N
- **质量维度**：相关性、类型一致性、可回答性、流畅性、结构 grounding
- **LLM-as-Judge**：可配置的标准和评分规则
- **统计分析**：t 检验、Cohen's d、置信区间

#### 5. RAG 流水线评估
- **检索指标**：Precision@K、Recall@K、MRR、NDCG、HitRate
- **生成指标**：忠实度、答案相关性、上下文精确度
- **端到端延迟追踪**

#### 6. 实验管理
- 因子化设计的受控实验执行
- 完整来源追踪（模型、策略、提示、参数）
- 跨条件统计比较
- 导出为 JSON、CSV、Markdown、LaTeX

### 配置

KnowProbe 使用分层配置系统：

1. **默认配置** (`configs/default.yaml`)
2. **本地配置** (`configs/local.yaml` — gitignored)
3. **环境变量** (`KNOWPROBE_*` 前缀)
4. **运行时覆盖**（CLI 标志、API 参数）

```yaml
# 示例 local.yaml
models:
  api:
    openai:
      api_key: "sk-..."  # 或设置 OPENAI_API_KEY 环境变量

generation:
  temperature: 0.7
  max_length: 256

evaluation:
  metrics: [bleu, rouge, bert_score]
```

### 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码检查
ruff check src/ tests/
ruff format src/ tests/

# 运行类型检查
mypy src/

# 运行测试
pytest --cov=knowprobe

# 构建文档
mkdocs serve
```

### 许可证

MIT 许可证 — 详情见 [LICENSE](LICENSE)。

### 致谢

- 原始研究作为 COMP390 本科毕业设计进行
- AI 智能体协助原型开发、脚本编写和文档撰写
- 人工监督研究设计、实验控制和结果解释

### 更新日志

#### v2.0.0 (2025-07)
- 从研究原型到完整工程化升级
- 新增多后端 LLM 客户端、提示策略引擎、RAG 评估
- 新增 RESTful API、Streamlit 仪表板、CLI、Docker 支持
- 新增数据库持久化、配置管理、CI/CD

#### v1.0.0 (原始研究)
- LLM 从结构化知识生成问题的受控研究
- 模型对比：Llama-3.1-8B、Qwen-2.5-7B、Flan-T5-Large
- 提示策略：Zero-shot、Few-shot、Chain-of-Thought
- 事实与模式问题的 BLEU-4 评估
