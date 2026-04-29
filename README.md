# Large Language Model for Question Generation from Knowledge Bases

> This repository is a public-facing project showcase for my COMP390 Final Year Project. The full dissertation, source code, experimental outputs, and datasets will be released only after the formal university submission process is completed and when it is academically appropriate to disclose further details.

## Project Overview

This project investigates how Large Language Models can generate natural-language questions from structured knowledge-base inputs. The main research focus is the transformation from knowledge graph / knowledge base structures into usable questions for evaluation, search, and question-answering scenarios.

Instead of building a general chatbot, this project focuses on a more controlled NLP task: given factual triples or schema-level relations, can an LLM generate clear, relevant, and structurally grounded questions?

## Motivation

Enterprise knowledge bases, intelligent search systems, and question-answering applications often require high-quality question samples for evaluation, testing, or data augmentation. Manually writing such questions can be expensive and limited in coverage.

This project explores whether LLM-based question generation can support this process by automatically producing questions from structured knowledge representations.

The project studies three core questions:

- How do different local LLMs perform on knowledge-base question generation?
- How do Zero-shot, Few-shot, and Chain-of-Thought prompting strategies affect generation quality?
- Are factual triples easier for LLMs than schema-level relations when generating questions?

## Methodology

The project follows a controlled experimental workflow:

1. Prepare structured knowledge inputs  
   The project uses factual triples and schema-level relations as structured inputs.

2. Design prompt strategies  
   Zero-shot, Few-shot, and Chain-of-Thought prompt templates are used to guide question generation.

3. Generate questions with local LLMs  
   Multiple local or locally callable models are compared under the same task setting.

4. Evaluate and analyse outputs  
   BLEU-4 is used as the primary automatic evaluation metric, with qualitative analysis used to interpret model behaviour and failure cases.

## Experimental Design

| Dimension | Setting |
|---|---|
| Models | Llama-3.1-8B, Qwen-2.5-7B, Flan-T5-Large |
| Prompting Strategies | Zero-shot, Few-shot, Chain-of-Thought |
| Question Types | Factual questions, Schema questions |
| Experimental Conditions | 3 × 3 × 2 = 18 |
| Generated Questions | 4,500 |
| Primary Metric | BLEU-4 |

## Preliminary Findings

The current experimental analysis suggests:

- Llama-3.1-8B achieved the strongest overall BLEU-4 performance.
- Chain-of-Thought prompting generally performed better than the other prompting strategies.
- Factual questions were easier to generate than schema-level questions.
- Schema-level question generation requires stronger abstraction and structural understanding.

These findings indicate that question generation from knowledge bases depends not only on model capability, but also on input structure, prompt strategy, and question type.

## Relevance

This project is relevant to:

- Enterprise knowledge-base question answering
- AI search evaluation
- RAG evaluation dataset construction
- Knowledge graph to natural language question generation
- Prompt engineering and LLM application evaluation
- Applied NLP systems

## Technical Keywords

- Large Language Models
- Knowledge Bases
- Knowledge Graphs
- Question Generation
- Prompt Engineering
- Chain-of-Thought
- BLEU-4 Evaluation
- Local LLM Deployment
- RAG Evaluation
- Applied NLP

## Current Disclosure Scope

To avoid academic integrity risks and premature disclosure before formal submission, this repository currently does not include:

- The full dissertation
- The full source code
- Raw datasets
- Generated experimental outputs
- University submission materials
- Supervisor feedback or assessment-related documents
- Any files containing private information or internal paths

After the formal university submission process is completed, and when it is appropriate to share further details, this repository may be updated with:

- A cleaned demo version
- Example prompt templates
- Non-sensitive sample inputs and outputs
- Simplified experiment workflow
- Result visualisations
- Reproducible documentation

## Project Status

Status: Final Year Project in progress.

Full technical details will be disclosed at an appropriate time after the academic submission process is complete.

## Personal Note

AI agents were used to support code prototyping, experiment scripting, documentation organisation, and dissertation drafting. My main focus has been on research framing, experimental design, variable control, result interpretation, system understanding, and presentation.

This project reflects my interest in applied LLM systems, knowledge-base question answering, RAG evaluation, and agent-assisted development workflows.

---

# 基于知识库生成问题的大语言模型研究

> 本仓库是 COMP390 本科毕业设计的公开展示页。完整论文、源代码、实验输出和数据集将在学校正式提交流程完成后，并在学术上适合公开的时间再进一步披露。

## 项目简介

本项目研究如何利用大语言模型从知识库或知识图谱结构化信息中自动生成自然语言问题。项目关注的核心任务是：给定知识库中的事实三元组或 schema 级关系，模型能否生成清晰、相关、结构上合理的问题，用于评估、搜索或问答场景。

与普通聊天机器人不同，本项目更关注一个受控的 NLP 任务：如何把结构化知识输入转换成自然语言问题。

## 研究动机

企业知识库、智能搜索系统和问答应用通常需要大量高质量问题样本，用于评估、测试或数据增强。人工编写这些问题成本高，覆盖面也有限。

本项目探索能否通过大语言模型自动生成问题，从而支持知识库问答和搜索评估流程。

本项目主要研究三个问题：

- 不同本地大语言模型在知识库问题生成任务中的表现是否存在明显差异？
- Zero-shot、Few-shot 和 Chain-of-Thought 提示策略对生成质量有什么影响？
- 与 factual triples 相比，schema-level relations 是否更难生成高质量问题？

## 方法框架

项目采用受控实验流程：

1. 整理结构化知识输入  
   项目使用 factual triples 和 schema-level relations 作为结构化输入。

2. 设计提示策略  
   使用 Zero-shot、Few-shot 和 Chain-of-Thought 三类 Prompt 模板引导问题生成。

3. 使用本地大语言模型生成问题  
   在相同任务设定下对比多个本地部署或本地调用模型。

4. 评估与分析输出结果  
   使用 BLEU-4 作为主要自动评估指标，并结合定性分析解释模型行为和失败案例。

## 实验设计

| 维度 | 设置 |
|---|---|
| 模型 | Llama-3.1-8B、Qwen-2.5-7B、Flan-T5-Large |
| 提示策略 | Zero-shot、Few-shot、Chain-of-Thought |
| 问题类型 | Factual question、Schema question |
| 实验条件 | 3 × 3 × 2 = 18 组 |
| 生成规模 | 4,500 条问题 |
| 主要指标 | BLEU-4 |

## 阶段性发现

当前实验分析显示：

- Llama-3.1-8B 在整体 BLEU-4 表现上较优。
- Chain-of-Thought 提示策略整体表现优于其他提示策略。
- Factual question 比 schema-level question 更容易生成。
- Schema-level question 生成需要更强的抽象能力和结构理解能力。

这些结果说明，知识库问题生成不仅取决于模型能力，也受到输入结构、提示策略和问题类型的明显影响。

## 应用价值

本项目与以下方向相关：

- 企业知识库问答
- AI 搜索评估
- RAG 评估数据构建
- 知识图谱到自然语言问题生成
- Prompt 工程与 LLM 应用评估
- 应用型 NLP 系统

## 技术关键词

- Large Language Models
- Knowledge Bases
- Knowledge Graphs
- Question Generation
- Prompt Engineering
- Chain-of-Thought
- BLEU-4 Evaluation
- Local LLM Deployment
- RAG Evaluation
- Applied NLP

## 当前公开范围

为避免学术诚信风险和正式提交前过早公开，本仓库当前不包含：

- 完整论文
- 完整源代码
- 原始数据集
- 生成的实验输出
- 学校提交材料
- 导师反馈或评分相关文件
- 任何包含个人信息或内部路径的文件

在学校正式提交流程完成后，并在适合公开的时间，本仓库可能会进一步更新：

- 清理后的 demo 版本
- Prompt 模板示例
- 非敏感输入输出样例
- 简化实验流程
- 结果可视化图表
- 可复现说明文档

## 项目状态

当前状态：本科毕业设计进行中。

完整技术细节将在学校正式提交完成后，于合适时间进一步公开。

## 个人说明

本项目使用 AI Agent 辅助完成代码原型、实验脚本、文档整理和论文写作。本人重点负责研究问题设定、实验设计、变量控制、结果解释、系统理解与答辩表达。

本项目体现了我对 LLM 应用落地、知识库问答、RAG 评估和 Agent 辅助开发流程的持续兴趣。
