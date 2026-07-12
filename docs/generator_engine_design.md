# KnowProbe 问题生成器主引擎 — 设计文档

> 文档版本: v1.0  
> 对应代码: `src/knowprobe/generators/` (1,793 行 Python + 295 行 Jinja2 模板)

---

## 一、架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QuestionGeneratorEngine (主引擎)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ PromptBuilder│→│ ModelClient │→│ OutputParser│→│ Confidence  │ │
│  │  提示词构建   │  │  模型调用   │  │  输出解析   │  │  置信度估计  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         ↓                    ↓
   ┌─────────────┐    ┌─────────────┐
   │ Jinja2模板   │    │ Ollama / HF │
   │  (磁盘/内置) │    │ / OpenAI    │
   └─────────────┘    └─────────────┘
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **策略与实现解耦** | `PromptBuilder` 负责提示工程，`ModelClient` 负责推理，引擎负责编排 |
| **多后端统一抽象** | 单一 `BaseModelClient` 接口覆盖 Ollama / Transformers / OpenAI-Compatible |
| **渐进式模板加载** | 磁盘模板 → 内置模板 → 智能降级（zero_shot 兜底） |
| **完整的可追溯性** | 每个 `GeneratedQuestion` 包含知识源、策略、模型、原始输出、超参数 |
| **生产级错误处理** | 分层异常体系 + tenacity 自动重试 + 结构化日志 |

---

## 二、模块详解

### 2.1 `base.py` — 抽象基类与异常体系

**职责**: 定义所有问题生成器必须遵循的接口契约，以及分层异常体系。

#### 异常层次

```
Exception
├── GenerationError          # 生成逻辑失败（模型返回了，但后续处理出错）
│   └── PromptBuildError     # 模板渲染失败（从 prompt_builder.py 引入）
├── ModelUnavailableError    # 模型完全不可达（网络/服务未启动/密钥错误）
└── PromptBuildError         # 提示词构建失败（模板缺失/渲染异常）
```

| 异常类 | 触发场景 | 包含的上下文 |
|--------|---------|-------------|
| `GenerationError` | 模型调用成功但解析失败、批处理部分失败 | `details` 字典包含 model, strategy, question_type, source_id |
| `ModelUnavailableError` | Ollama 未启动、API 密钥无效、Transformers 模型加载失败 | `provider`, `model` 字段 |
| `PromptBuildError` | 模板文件缺失、Jinja2 语法错误、变量未定义 | `template_key` 字段 |

#### `BaseQuestionGenerator` 接口契约

```python
class BaseQuestionGenerator(ABC):
    # 生命周期
    async def initialize(self) -> None        # 必须在此之后才能调用 generate
    async def shutdown(self) -> None          # 释放资源，幂等

    # 生成接口
    async def generate(...) -> GeneratedQuestion      # 单条生成
    async def generate_batch(...) -> list[GeneratedQuestion]  # 批量生成

    # 可观测性
    async def health_check(self) -> dict[str, Any]    # 返回 {status: ok|degraded|unavailable}

    # 支持 async with 语法
    async def __aenter__(self) -> Self
    async def __aexit__(self, ...) -> None
```

**关键约束**:
- `generate()` / `generate_batch()` 在 `initialize()` 之前调用必须抛出 `RuntimeError`
- `shutdown()` 必须幂等（多次调用无副作用）
- `health_check()` 不抛异常，总是返回结构化字典

---

### 2.2 `prompt_builder.py` — 策略驱动的提示词构建器

**职责**: 将 `(KnowledgeInput, QuestionType, PromptStrategy)` 三元组渲染为模型可直接消费的提示字符串。

#### 模板解析优先级

```
用户调用 build()
    │
    ▼
┌─────────────────┐
│ 1. 内存缓存查找  │ ← 已编译的 Template 对象，避免重复解析
└─────────────────┘
    │ 命中 → 直接渲染
    │ 未命中
    ▼
┌─────────────────┐
│ 2. 磁盘模板查找  │ ← FileSystemLoader 从 configs/prompts/ 加载 *.jinja2
└─────────────────┘
    │ 命中 → 编译 → 缓存 → 渲染
    │ 未命中
    ▼
┌─────────────────┐
│ 3. 内置模板兜底  │ ← 嵌入 Python 代码的字符串模板
└─────────────────┘
    │ 命中 → 编译 → 缓存 → 渲染
    │ 未命中
    ▼
┌─────────────────┐
│ 4. 策略降级     │ ← e.g. self_consistency → zero_shot 同类型
└─────────────────┘
    │ 命中 → 编译 → 缓存 → 渲染
    │ 未命中 → 抛出 PromptBuildError
```

#### 模板命名约定

```
{strategy}_{question_type}.jinja2

示例:
  zero_shot_factual.jinja2
  zero_shot_schema.jinja2
  few_shot_factual.jinja2
  few_shot_schema.jinja2
  cot_factual.jinja2
  cot_schema.jinja2
  self_consistency_factual.jinja2
  self_consistency_schema.jinja2
```

#### 模板上下文变量

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `knowledge` | `dict` | 包含 `source_id`, `input_type`, `content`, `structured`, `metadata` |
| `examples` | `list[dict]` | Few-shot 示例列表，每项有 `knowledge` 和 `question` |
| `reasoning_steps` | `list[str]` | CoT 的预定义推理步骤 |
| `self_consistency_n` | `int` | 自一致性采样的候选数（默认 5） |
| `**extra` | 任意 | 调用者传入的额外变量 |

#### 内置模板设计要点

每个内置模板都是**内聚**的 Python 字符串常量，确保即使磁盘文件全部丢失，系统仍可运行。模板遵循以下规范:

1. **角色锚定**: 开头明确指定专家角色（"知识问答生成专家" / "Schema分析专家"）
2. **约束清单**: 用 bullet points 列出格式约束（一个问题、问号结尾、无解释）
3. **条件渲染**: 根据 `knowledge.input_type` 选择不同的展示格式（triple / text / schema）
4. **元信息附注**: `knowledge.metadata` 以键值对形式追加

---

### 2.3 `model_client.py` — 统一多后端模型客户端

**职责**: 将 `generate(prompt, **params)` 调用翻译为不同后端的原生协议，并返回标准化的 `ModelResponse`。

#### 支持的 Backend

| 客户端类 | 后端协议 | 适用模型 | 并发策略 |
|---------|---------|---------|---------|
| `OllamaClient` | HTTP `/api/generate` | Llama-3.1-8B, Qwen-2.5-7B (本地) | `asyncio.Semaphore(4)` |
| `TransformersClient` | `transformers.AutoModelForCausalLM` | Flan-T5-Large, 任意 HuggingFace | 线程池执行器 (`run_in_executor`) |
| `OpenAICompatibleClient` | HTTP `/chat/completions` | GPT-4o, DeepSeek, Claude | `asyncio.Semaphore(8)` |

#### `ModelResponse` 标准化结构

```python
@dataclass(frozen=True)
class ModelResponse:
    text: str           # 已 strip 的生成文本
    usage: dict | None  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float   # 墙钟耗时（毫秒）
    model: str          # 模型标识符
```

**关键设计**: `frozen=True` 保证响应对象不可变，适合作为日志/审计的可靠记录。

#### 重试策略（tenacity）

| 客户端 | 最大重试 | 等待策略 | 重试条件 |
|--------|---------|---------|---------|
| `OllamaClient` | 3 次 | 指数退避 (2s~10s) | `httpx.HTTPError`, `TimeoutException`, `ConnectError` |
| `TransformersClient` | 2 次 | 指数退避 (1s~5s) | 任意异常 |
| `OpenAICompatibleClient` | 3 次 | 指数退避 (2s~10s) | `httpx.HTTPError`, `TimeoutException` |

**不 retry 的场景**: `ValueError`（参数错误）、`RuntimeError`（未初始化）、HTTP 4xx 客户端错误（除 429 Rate Limit 外）。

#### `TransformersClient` 的特殊处理

由于 HuggingFace `transformers` 的 `generate()` 是 **CPU/GPU 密集型阻塞调用**，必须使用 `asyncio.run_in_executor()` 将其委托给线程池，避免阻塞事件循环。

```python
async def generate(self, prompt: str, **params) -> ModelResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._generate_sync, prompt, params)
```

`_generate_sync` 内部使用 `torch.no_grad()` 上下文管理器，并仅解码新生成的 token（`outputs[0][input_ids.shape[1]:]`），避免重复解码 prompt。

#### `ModelClientFactory` 工厂模式

```python
client = ModelClientFactory.create(
    ModelProvider.OLLAMA,      # 或 TRANSFORMERS / OPENAI / DEEPSEEK / CLAUDE
    "llama3.1:8b",
    base_url="http://localhost:11434",
)
```

工厂从 `Settings` 自动拉取默认配置（`configs/default.yaml` + 环境变量），同时允许运行时覆盖。

---

### 2.4 `question_generator.py` — 核心编排引擎

**职责**: 作为 **Facade / Orchestrator**，将 PromptBuilder、ModelClient、Parser、ConfidenceEstimator 组合成完整的生成流水线。

#### 生成流水线（单条）

```
KnowledgeInput ──→ _build_prompt() ──→ ModelClient.generate()
                                              │
                                              ▼
                                        ModelResponse
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              _parse_output()           _estimate_confidence()    构建 GeneratedQuestion
                    │                         │                         │
                    ▼                         ▼                         ▼
              清理后的问题文本          质量启发分 [0,1]         完整溯源对象
```

#### 批量生成优化

批量模式的核心优化在于 **prompt 构建的批量化** + **模型调用的并发化**:

```python
# 1. 所有 prompt 先构建完（纯CPU，无IO）
prompts = [self._build_prompt(k, ...) for k in knowledges]

# 2. 一次性交给 ModelClient.generate_batch()
#    - Ollama: Semaphore(4) 限制并发，避免压垮本地GPU
#    - API: Semaphore(8) 限制并发，避免触发限流
#    - Transformers: run_in_executor 顺序执行
responses = await self._model_client.generate_batch(prompts, ...)

# 3. 结果解析（并行无IO，可顺序执行）
results = [self._build_result(k, r) for k, r in zip(knowledges, responses)]
```

#### 输出解析器 `_parse_output()`

解析器是**策略感知**的，针对不同 `PromptStrategy` 执行不同的提取逻辑:

| 策略 | 解析逻辑 |
|------|---------|
| `ZERO_SHOT` / `FEW_SHOT` | 直接 strip，取第一个以 `?`/`？` 结尾的行 |
| `CHAIN_OF_THOUGHT` | 扫描 "最终问题:" / "Final question:" / "问题:" 标记，提取标记后的内容 |
| `SELF_CONSISTENCY` | 扫描 "最终选择" / "Final answer:" / "最佳问题:" 标记 |
| 通用后处理 | 去除 `<think>...` 标签、引号包裹、确保以问号结尾 |

#### 置信度估计 `_estimate_confidence()`

这是一个**后验启发式**评分器，而非模型 logits 概率。适用于自动筛选和排名:

| 信号 | 影响 |
|------|------|
| 长度 < 10 字符 | ×0.5（可能生成失败） |
| 长度 > 200 字符 | ×0.8（可能包含多余内容） |
| 长度 20~100 字符 | ×1.05（黄金长度奖励） |
| 无结尾问号 | ×0.7 |
| Schema 类型 | ×0.95（Schema问题更难自动验证） |
| 输出 token 数 ≥ max_length × 0.95 | ×0.85（可能截断） |

最终分数截断到 `[0, 1]`，保留 3 位小数。

#### 默认 Few-shot 示例

引擎内置了 4 个默认示例（2 factual + 2 schema），确保即使调用者不提供 examples，Few-shot 策略仍能工作:

**Factual 示例 1**:
- 知识: `(巴黎, 首都, 法国) — 巴黎是法国的首都，位于塞纳河畔。`
- 问题: `法国的首都是哪座城市？`

**Schema 示例 1**:
- 知识: `Class: Person | Properties: name (string), birthDate (date) | Relations: worksAt (Company)`
- 问题: `Person类的worksAt关系的range是什么类型？`

---

### 2.5 `__init__.py` — 包导出

统一导出所有公共符号，确保外部模块只需:

```python
from knowprobe.generators import (
    QuestionGeneratorEngine,  # 主引擎
    PromptBuilder,            # 提示词构建
    ModelClientFactory,       # 模型客户端工厂
    GenerationError,          # 异常
)
```

---

## 三、Jinja2 模板设计

### 3.1 模板文件清单

位于 `configs/prompts/`，共 8 个模板:

| 文件 | 策略 | 问题类型 | 行数 |
|------|------|---------|------|
| `zero_shot_factual.jinja2` | Zero-shot | Factual | 31 |
| `zero_shot_schema.jinja2` | Zero-shot | Schema | 29 |
| `few_shot_factual.jinja2` | Few-shot | Factual | 39 |
| `few_shot_schema.jinja2` | Few-shot | Schema | 37 |
| `cot_factual.jinja2` | CoT | Factual | 42 |
| `cot_schema.jinja2` | CoT | Schema | 41 |
| `self_consistency_factual.jinja2` | Self-Consistency | Factual | 39 |
| `self_consistency_schema.jinja2` | Self-Consistency | Schema | 37 |

### 3.2 模板设计规范

每条模板遵循 **SIT 结构**:

```
S — System/Role:    定义专家角色和任务目标
I — Instruction:    明确的约束清单（格式、数量、语气）
T — Target:         知识/Schema 内容插槽 + 输出生成指令
```

**条件渲染示例**（`zero_shot_factual.jinja2`）:

```jinja2
{% if knowledge.input_type == 'triple' %}
知识三元组：{{ knowledge.content }}
{% if knowledge.structured %}
结构化数据：{{ knowledge.structured | tojson }}
{% endif %}
{% elif knowledge.input_type == 'text' %}
文本内容：{{ knowledge.content }}
{% else %}
{{ knowledge.content }}
{% endif %}
```

这确保了无论输入是三元组、纯文本还是 schema，模板都能正确展示。

---

## 四、接口使用示例

### 4.1 基础用法（单条生成）

```python
import asyncio
from knowprobe.core.models import KnowledgeInput, QuestionType, PromptStrategy, ModelProvider
from knowprobe.generators import QuestionGeneratorEngine

async def main():
    engine = QuestionGeneratorEngine(
        model_name="llama3.1:8b",
        model_provider=ModelProvider.OLLAMA,
    )
    async with engine:
        question = await engine.generate(
            knowledge=KnowledgeInput(
                source_id="kg_001",
                input_type="triple",
                content="(巴黎, 首都, 法国)",
            ),
            question_type=QuestionType.FACTUAL,
            prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        )
        print(question.question_text)
        # → "法国的首都是哪座城市？"
        print(question.confidence)
        # → 0.997

asyncio.run(main())
```

### 4.2 实验批处理

```python
from knowprobe.core.models import ExperimentConfig

async def run_experiment():
    config = ExperimentConfig(
        experiment_id="exp_001",
        name="Llama3.1 vs Qwen2.5",
        models=["llama3.1:8b", "qwen2.5:7b"],
        prompt_strategies=[
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.FEW_SHOT,
            PromptStrategy.CHAIN_OF_THOUGHT,
        ],
        question_types=[QuestionType.FACTUAL, QuestionType.SCHEMA],
        evaluation_metrics=["bleu", "rouge", "bert_score"],
        knowledge_sources=["wikidata_triples", "dbpedia_schema"],
    )

    all_questions = []
    for model_name in config.models:
        engine = QuestionGeneratorEngine(
            model_name=model_name,
            model_provider=ModelProvider.OLLAMA,
        )
        async with engine:
            for strategy in config.prompt_strategies:
                for qtype in config.question_types:
                    questions = await engine.generate_batch(
                        knowledges=knowledge_items,
                        question_type=qtype,
                        prompt_strategy=strategy,
                    )
                    all_questions.extend(questions)
```

### 4.3 自定义模板目录

```python
engine = QuestionGeneratorEngine(
    model_name="gpt-4o-mini",
    model_provider=ModelProvider.OPENAI,
    templates_dir="/path/to/custom/prompts",
)
```

### 4.4 运行时参数覆盖

```python
question = await engine.generate(
    knowledge=knowledge,
    question_type=QuestionType.FACTUAL,
    prompt_strategy=PromptStrategy.FEW_SHOT,
    temperature=0.3,          # 覆盖默认 0.7
    max_tokens=128,           # 覆盖默认 256
    examples=custom_examples, # 自定义 few-shot 示例
)
```

---

## 五、错误处理与可观测性

### 5.1 日志规范

所有模块使用 `structlog` 结构化日志，统一格式:

```python
self._logger.info(
    "Batch generation completed",
    batch_size=batch_size,
    total_latency_ms=round(total_latency, 2),
    avg_latency_ms=round(avg_latency, 2),
)
```

关键日志事件:
- `Generator engine initialised` — 引擎就绪
- `Starting generation` / `Starting batch generation` — 生成开始
- `Model generation failed` — 模型调用异常（含重试后仍失败）
- `Batch generation completed` — 批量完成
- `Generator engine shut down` — 资源释放

### 5.2 健康检查

```python
health = await engine.health_check()
# {
#     "status": "ok",           # ok | degraded | unavailable
#     "model": "llama3.1:8b",
#     "provider": "ollama",
#     "client_details": {...}
# }
```

---

## 六、扩展指南

### 6.1 添加新的 PromptStrategy

1. 在 `core/models.py` 的 `PromptStrategy` 枚举中添加新值
2. 在 `configs/prompts/` 创建 `{strategy}_factual.jinja2` 和 `{strategy}_schema.jinja2`
3. 在 `prompt_builder.py` 的 `_BUILTIN_TEMPLATES` 中添加内置兜底模板
4. 在 `question_generator.py` 的 `_parse_output()` 中添加对应的解析逻辑

### 6.2 添加新的 ModelProvider

1. 在 `core/models.py` 的 `ModelProvider` 枚举中添加新值
2. 创建新的 `BaseModelClient` 子类（参考 `OllamaClient` 的结构）
3. 在 `ModelClientFactory.create()` 中添加新的分支

### 6.3 添加新的 QuestionType

1. 在 `core/models.py` 的 `QuestionType` 枚举中添加新值
2. 创建对应的 Jinja2 模板文件
3. 更新 `_estimate_confidence()` 中的类型启发式（如有需要）

---

## 七、文件清单

### Python 模块

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/knowprobe/generators/__init__.py` | 42 | 包导出 |
| `src/knowprobe/generators/base.py` | 202 | 抽象基类 + 异常体系 |
| `src/knowprobe/generators/prompt_builder.py` | 361 | Jinja2 模板系统 |
| `src/knowprobe/generators/model_client.py` | 599 | 多后端模型客户端 |
| `src/knowprobe/generators/question_generator.py` | 589 | 主引擎 + 流水线编排 |

### Jinja2 模板

| 文件 | 行数 | 策略 | 类型 |
|------|------|------|------|
| `configs/prompts/zero_shot_factual.jinja2` | 31 | Zero-shot | Factual |
| `configs/prompts/zero_shot_schema.jinja2` | 29 | Zero-shot | Schema |
| `configs/prompts/few_shot_factual.jinja2` | 39 | Few-shot | Factual |
| `configs/prompts/few_shot_schema.jinja2` | 37 | Few-shot | Schema |
| `configs/prompts/cot_factual.jinja2` | 42 | CoT | Factual |
| `configs/prompts/cot_schema.jinja2` | 41 | CoT | Schema |
| `configs/prompts/self_consistency_factual.jinja2` | 39 | Self-Consistency | Factual |
| `configs/prompts/self_consistency_schema.jinja2` | 37 | Self-Consistency | Schema |

---

## 八、质量保证 checklist

- [x] **类型注解完整**: 所有函数参数和返回值均有 `typing` 注解
- [x] **`from __future__ import annotations`**: 所有模块文件开头均包含，支持 `X | Y` 语法在 Python 3.9+ 下工作
- [x] **异常分层**: 3 个自定义异常类，各有明确的触发场景和上下文
- [x] **自动重试**: `tenacity` 配置覆盖所有网络调用
- [x] **资源管理**: `async with` 上下文管理器 + `shutdown()` 幂等释放
- [x] **结构化日志**: `structlog` 全链路集成，无 `print()`
- [x] **配置兼容**: 完全遵循 `Settings` 类接口，支持运行时覆盖
- [x] **模型无关**: 统一 `ModelResponse` 抽象，新增后端无需改动引擎
- [x] **渐进降级**: 磁盘模板 → 内置模板 → 策略回退
- [x] **AST 验证**: 所有文件通过 `py_compile` 语法检查
