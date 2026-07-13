# KnowProbe v3.0（知探）

> English first. A Chinese version follows below.
>
> 中文版见后文。

[English](#english) | [中文](#中文)

> **Knowledge-Grounded Question Generation and RAG Evaluation Platform**
>
> A production-minded toolkit for LLM-based question generation from structured knowledge, multi-strategy evaluation, and **GraphRAG / Agentic RAG / adversarial** pipeline assessment.

[![CI](https://github.com/Jeffhan789/KnowProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/Jeffhan789/KnowProbe/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v3.0.0-blueviolet)](https://github.com/Jeffhan789/KnowProbe/releases)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)

---

## English

### Overview

KnowProbe v3 is an **engineering upgrade** of a Final Year Project (COMP390) research prototype. It transforms the controlled study of knowledge-base question generation into a **deployable, extensible, and pedagogically-oriented platform**.

**What makes it different:**
- **Pedagogical focus** — Every module includes detailed docstrings and inline comments that explain *why*, not just *how*. Three Chinese tutorials cover KG construction, Agentic RAG, and multi-hop / adversarial evaluation.
- **Complete engineering stack** — CLI (Typer), REST API (FastAPI), Dashboard (Streamlit), Docker, CI, database persistence, configuration management.
- **Frontier RAG coverage** — Vector RAG, **GraphRAG** (ego-graph & path retrieval), **Agentic RAG** (ReAct reasoning loop), and **Adversarial Evaluation** (robustness testing).

**Who is this for?**
- **Developers and teams** who need a modular reference implementation for question generation and retrieval evaluation.
- **Learners** who want to understand RAG internals through clean, well-commented code.
- **Researchers** who need a reproducible evaluation framework for LLM question generation and RAG benchmarking.

### Quick Start

```bash
# Clone
git clone https://github.com/Jeffhan789/KnowProbe.git
cd KnowProbe

# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run a demo (no LLM backend required)
PYTHONPATH=src python examples/graphrag_demo/graphrag_demo.py
PYTHONPATH=src python examples/agentic_demo/agentic_demo.py

# Start API + Dashboard
docker compose -f docker/docker-compose.yml up --build
# API at http://localhost:8000  |  Dashboard at http://localhost:8501
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KnowProbe v3.0                                   │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   CLI (Typer)   │  API (FastAPI)  │   Dashboard (Streamlit)         │
├─────────────────┼─────────────────┼─────────────────────────────────┤
│                 │    ┌────────────┴─────────────┐                  │
│  generate       │    │   Core Services           │                  │
│  evaluate       │    │ • Question Generator      │                  │
│  experiment     │    │ • Evaluator Suite         │                  │
│  serve          │    │ • Experiment Runner       │                  │
│  dashboard      │    │ • Prompt Strategy Engine  │                  │
│                 │    └────────────┬─────────────┘                  │
├─────────────────┴─────────────────┼─────────────────────────────────┤
│  RAG & Knowledge Infrastructure                                   │
│  • Vector Retrieval (Dense / Hybrid / BM25)                       │
│  • Knowledge Graph (Rule-based & LLM-based construction)          │
│  • GraphRAG Retrievers (EgoGraph / Path / HybridGraph)            │
│  • Agentic RAG (ReAct loop: Think → Act → Observe)                │
│  • Multi-hop Benchmarks (HotpotQA-style / Synthetic)              │
│  • Adversarial Evaluator (Distractor / Negation / EdgeCase)       │
├───────────────────────────────────┼───────────────────────────────┤
│  Infrastructure                                                  │
│  • Multi-Backend LLM (Ollama/HF/OpenAI/Claude/DS)               │
│  • Vector Store & Embeddings (sentence-transformers)             │
│  • Database (SQLAlchemy + SQLite/PostgreSQL)                     │
│  • Structured Logging (structlog)                                 │
│  • Configuration (Pydantic Settings)                              │
└───────────────────────────────────┴───────────────────────────────┘
```

### Learning Path (Pedagogical)

KnowProbe is designed as a **learning scaffold**. Each module teaches one concept:

| Module | What you learn | Key files |
|--------|---------------|-----------|
| `src/knowprobe/kg/` | How Knowledge Graphs work; GraphRAG and vector-retrieval trade-offs | `graph.py`, `builder.py`, `retriever.py` |
| `src/knowprobe/agentic/` | ReAct pattern; limits of fixed retrieval pipelines on complex queries | `agent.py` |
| `src/knowprobe/benchmarks/` | Multi-hop reasoning; HotpotQA-style evaluation | `multihop.py` |
| `src/knowprobe/adversarial/` | Systematic robustness testing; RAG failure modes | `generator.py` |
| `docs/tutorials/` | 3 Chinese tutorials with architecture discussion notes | `01_knowledge_graph.md`, `02_agentic_rag.md`, `03_multihop_and_adversarial.md` |

### Key Features

#### v3.0 New — Knowledge Graph & GraphRAG
- **RuleBasedBuilder** — Zero-cost KG extraction from text using regex patterns.
- **LLMBasedBuilder** — Configurable LLM-powered entity and relation extraction.
- **EgoGraphRetriever** — k-hop subgraph retrieval for local aggregation queries.
- **PathRetriever** — Multi-hop path retrieval for relationship queries.
- **HybridGraphRetriever** — Combines dense vector + graph traversal.

#### v3.0 New — Agentic RAG
- **ReAct Loop** — Thought → Action → Observation cycle with configurable max iterations.
- **Rule-based planner** — Works without LLM backend (for demos and teaching).
- **LLM-based planner** — Dynamic retrieval strategy selection via LLM.

#### v3.0 New — Multi-hop & Adversarial Benchmarks
- **Synthetic multi-hop generator** — Create test questions from any triple set.
- **Four adversarial strategies** — Distractor, Negation, Multi-hop Variation, Edge Case.
- **Robustness scoring** — Attack success rate per strategy.

#### Existing v2.0 Features
- Multi-backend LLM support (Ollama, vLLM, HF, OpenAI, DeepSeek, Claude)
- Prompt strategy engine (Zero-shot, Few-shot, CoT, Self-Consistency, ReAct)
- Comprehensive evaluation (BLEU, ROUGE, BERTScore, LLM-as-Judge)
- RESTful API with structured logging and health checks
- Streamlit dashboard
- Docker deployment

### Architecture Notes

> These are built into the code comments and tutorials. Study them to review the project architecture.

**"Why does this platform exist?"**
> The original research prototype was difficult to reproduce and covered too few evaluation dimensions. KnowProbe turns that work into a tested platform and adds graph, agentic, multi-hop, and adversarial evaluation paths behind consistent interfaces.

**"Why GraphRAG?"**
> Vector retrieval can underperform on multi-hop questions when intermediate facts are semantically dissimilar. A knowledge graph explicitly models entity relationships, enabling path-based retrieval across documents.

**"Why Agentic RAG?"**
> Fixed RAG pipelines retrieve once regardless of query complexity. Agentic RAG uses a ReAct loop to dynamically decide: "Do I need more retrieval? Should I decompose this query? Is my answer confident enough?"

**"How do you evaluate robustness?"**
> We generate adversarial variants (negation, entity confusion, superlative changes) and measure attack success rate. This reveals failure modes that standard benchmarks miss.

### Development

```bash
pip install -e ".[dev]"
pytest --cov=knowprobe
ruff check src/ tests/
mkdocs serve
```

### License

MIT License — see [LICENSE](LICENSE).

---

## 中文

### 概述

KnowProbe v3 是本科毕业设计（COMP390）研究原型的**工程化升级**。它将知识库问题生成的受控研究转化为**可部署、可扩展、偏教学的实践平台**。

**核心差异点：**
- **教学导向** — 每个模块都有详细中文注释，解释"为什么"而不仅是"怎么做"。三篇中文教程覆盖知识图谱、Agentic RAG、多跳/对抗性评估。
- **完整工程栈** — CLI (Typer)、REST API (FastAPI)、Dashboard (Streamlit)、Docker、CI、数据库持久化、配置管理。
- **前沿 RAG 覆盖** — 向量 RAG、**GraphRAG**（子图/路径检索）、**Agentic RAG**（ReAct 推理循环）、**对抗性评估**（鲁棒性测试）。

**适用人群：**
- **开发者** — 需要问题生成与检索评估模块化参考实现。
- **学习者** — 通过整洁、注释充分的代码理解 RAG 内部原理。
- **研究者** — 需要可复现的 LLM 问题生成与 RAG 基准测试框架。

### 快速开始

```bash
# 克隆
git clone https://github.com/Jeffhan789/KnowProbe.git
cd KnowProbe

# 安装
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 运行 Demo（无需 LLM 后端）
PYTHONPATH=src python examples/graphrag_demo/graphrag_demo.py
PYTHONPATH=src python examples/agentic_demo/agentic_demo.py
```

### 学习路径（教学设计）

KnowProbe 设计为**渐进式学习脚手架**。每个模块教授一个核心概念：

| 模块 | 学习内容 | 关键文件 |
|------|---------|---------|
| `src/knowprobe/kg/` | 知识图谱原理；为什么 GraphRAG 优于纯向量 RAG | `graph.py`, `builder.py`, `retriever.py` |
| `src/knowprobe/agentic/` | ReAct 模式；为什么固定 RAG 流水线在复杂查询上会失败 | `agent.py` |
| `src/knowprobe/benchmarks/` | 多跳推理；HotpotQA 式评估方法 | `multihop.py` |
| `src/knowprobe/adversarial/` | 系统性鲁棒性测试；RAG 失效模式 | `generator.py` |
| `docs/tutorials/` | 三篇中文教程，含架构要点 | `01_knowledge_graph.md`, `02_agentic_rag.md`, `03_multihop_and_adversarial.md` |

### v3.0 新功能

#### 知识图谱 & GraphRAG
- **RuleBasedBuilder** — 基于正则规则的零成本文本提取，适合教学演示。
- **LLMBasedBuilder** — LLM 驱动的实体关系提取，生产级精度。
- **EgoGraphRetriever** — k-hop 子图检索，用于局部信息聚合查询。
- **PathRetriever** — 多跳路径检索，用于关系推理查询。
- **HybridGraphRetriever** — 向量检索 + 图遍历的混合检索。

#### Agentic RAG
- **ReAct 循环** — 思考 → 行动 → 观察，可配置最大迭代次数。
- **规则决策器** — 无需 LLM 后端即可运行，用于教学和演示。
- **LLM 决策器** — 通过 LLM 动态选择检索策略。

#### 多跳 & 对抗性基准
- **合成多跳生成器** — 从任意三元组集合生成测试问题。
- **四种对抗策略** — 混淆实体、否定形式、多跳变体、边界情况。
- **鲁棒性评分** — 按策略统计攻击成功率。

### 架构要点

> 以下说明给出主要架构选择及其取舍，并可作为深入阅读代码和教程的索引。

**"你为什么做这个项目？"**
> 原始研究聚焦 LLM 从结构化知识生成问题，但原型复现成本高、评估维度单一。KnowProbe 将其工程化为经过测试的平台，并用统一接口加入图检索、智能体、多跳和对抗评估路径。

**"为什么需要 GraphRAG？"**
> 向量检索在多跳问题上会失败，因为中间事实在语义上不相似。知识图谱显式建模实体关系，支持跨文档的路径检索。

**"为什么需要 Agentic RAG？"**
> 固定 RAG 流水线无论查询多复杂都只检索一次。Agentic RAG 用 ReAct 循环动态决定："需要更多检索吗？应该分解查询吗？答案足够可信吗？"

**"如何评估鲁棒性？"**
> 我们生成对抗性变体（否定、实体混淆、最高级变化）并统计攻击成功率。这能揭示标准基准测试遗漏的失效模式。

### 更新日志

#### v3.0.0 (2026-07)
- 新增知识图谱模块（RuleBased/LLMBased Builder、EgoGraph/Path/Hybrid Retriever）
- 新增 Agentic RAG 模块（ReAct 循环、规则/LLM 决策器）
- 新增多跳评估基准（合成生成器、HotpotQA 风格加载器）
- 新增对抗性评估（四种策略生成器 + 鲁棒性评分）
- 新增三篇中文教学教程 + 两个可运行 Demo

#### v2.0.0 (2025-07)
- 完整工程化升级：多后端 LLM、提示策略引擎、RAG 评估
- RESTful API、Streamlit Dashboard、CLI、Docker 支持
- 数据库持久化、配置管理、CI

#### v1.0.0 (原始研究)
- COMP390 本科毕业设计：LLM 问题生成的受控研究
- 模型对比：Llama-3.1-8B、Qwen-2.5-7B、Flan-T5-Large
- BLEU-4 评估
