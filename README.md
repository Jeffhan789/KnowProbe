# 知探 KnowProbe v2.0

> **Knowledge-Grounded Question Generation and RAG Evaluation Platform**
>
> A production-ready, engineering-grade toolkit for LLM-based question generation from structured knowledge, multi-strategy evaluation, and RAG pipeline assessment.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)

[English](#overview) | [中文](#概述)

---

## Overview

KnowProbe v2 is a complete engineering upgrade of the original COMP390 Final Year Project research prototype. It transforms the controlled study of knowledge-base question generation into a deployable, extensible platform with:

- **Multi-backend LLM support** — Ollama, vLLM, Transformers, OpenAI, DeepSeek, Claude
- **Rich prompt strategy library** — Zero-shot, Few-shot, Chain-of-Thought, Self-Consistency, ReAct
- **Comprehensive evaluation suite** — BLEU-4, ROUGE, BERTScore, LLM-as-Judge, multi-dimensional quality scoring
- **Full RAG pipeline evaluation** — Retrieval + Generation end-to-end assessment
- **RESTful API** — FastAPI with structured logging, metrics, and health checks
- **Interactive Dashboard** — Streamlit-based experiment management and visualization
- **Production tooling** — CLI, Docker, CI/CD, database persistence, configuration management

---

## 概述

KnowProbe v2 是 COMP390 本科毕业设计研究原型的完整工程化升级。它将知识库问题生成的受控研究转化为可部署、可扩展的平台，具备：

- **多后端 LLM 支持** — Ollama、vLLM、Transformers、OpenAI、DeepSeek、Claude
- **丰富的提示策略库** — Zero-shot、Few-shot、Chain-of-Thought、Self-Consistency、ReAct
- **全面的评估套件** — BLEU-4、ROUGE、BERTScore、LLM Judge、多维度质量评分
- **完整 RAG 流水线评估** — 检索 + 生成端到端评估
- **RESTful API** — FastAPI，结构化日志、监控指标、健康检查
- **交互式 Dashboard** — 基于 Streamlit 的实验管理与可视化
- **生产级工具链** — CLI、Docker、CI/CD、数据库持久化、配置管理

---

## Quick Start

### Installation

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

### CLI Usage

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

### Docker Deployment

```bash
# Start API + Dashboard
docker-compose -f docker/docker-compose.yml up --build

# API available at http://localhost:8000
# Dashboard available at http://localhost:8501
```

---

## Architecture

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

---

## Project Structure

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

---

## Features

### 1. Knowledge Input Processing
- Parse **factual triples** `(S, P, O)`, **schema relations**, **free text**, and **entity descriptions**
- Auto-detect input type with heuristic classification
- Batch processing with error isolation

### 2. Multi-Backend LLM Generation
- **Local**: Ollama, vLLM, Hugging Face Transformers
- **Cloud**: OpenAI, DeepSeek, Anthropic Claude
- Unified client with retry logic, connection pooling, and async support

### 3. Prompt Strategy Engine
- **Zero-shot**: Direct instruction
- **Few-shot**: Dynamic example selection (Random / Similarity / Diversity-MMR)
- **Chain-of-Thought**: Step-by-step reasoning guidance
- **Self-Consistency**: N-sample majority voting
- **ReAct**: Thought-Action-Observation reasoning loop
- Jinja2 templates with hot-reload support

### 4. Comprehensive Evaluation
- **Automatic metrics**: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, BERTScore, Self-BLEU, Distinct-N
- **Quality dimensions**: Relevance, Type Consistency, Answerability, Fluency, Structural Grounding
- **LLM-as-Judge**: Configurable criteria and rubrics
- **Statistical analysis**: t-tests, Cohen's d, confidence intervals

### 5. RAG Pipeline Evaluation
- **Retrieval metrics**: Precision@K, Recall@K, MRR, NDCG, HitRate
- **Generation metrics**: Faithfulness, Answer Relevance, Context Precision
- **End-to-end latency tracking**

### 6. Experiment Management
- Controlled experiment execution with factorial design
- Full provenance tracking (model, strategy, prompt, parameters)
- Statistical comparison across conditions
- Export to JSON, CSV, Markdown, LaTeX

---

## Configuration

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

---

## Development

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

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Original research conducted as COMP390 Final Year Project
- AI agents assisted in prototyping, scripting, and documentation
- Human oversight on research design, experimental control, and interpretation

---

## Changelog

### v2.0.0 (2025-07)
- Complete engineering upgrade from research prototype
- Added multi-backend LLM client, prompt strategy engine, RAG evaluation
- Added RESTful API, Streamlit dashboard, CLI, Docker support
- Added database persistence, configuration management, CI/CD

### v1.0.0 (Original Research)
- Controlled study of LLM question generation from structured knowledge
- Model comparison: Llama-3.1-8B, Qwen-2.5-7B, Flan-T5-Large
- Prompt strategies: Zero-shot, Few-shot, Chain-of-Thought
- BLEU-4 evaluation on factual vs schema questions
