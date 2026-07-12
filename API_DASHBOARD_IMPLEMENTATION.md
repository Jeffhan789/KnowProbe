# KnowProbe API + Dashboard 工程化实现文档

> 项目路径: `/Users/jeffhan/Library/Mobile Documents/com~apple~CloudDocs/9.浮生杂藏/02_修技砺能_职业技能/knowprobe_v2`
> 开发者: KnowProbe 工程化团队
> 版本: 2.0.0

---

## 一、总体架构设计

### 1.1 模块划分

```
knowprobe/
├── api/                    # FastAPI RESTful API 服务
│   ├── main.py            # 应用入口 + 生命周期管理
│   ├── dependencies.py    # 依赖注入 (Settings, RequestID, 分页)
│   ├── middleware.py      # 中间件 (请求ID, 日志, 异常, CORS)
│   ├── schemas.py         # API 专用请求/响应模型
│   └── routes/            # 路由模块
│       ├── health.py      # 健康检查 (/health, /health/ready, /health/live)
│       ├── generation.py  # 问题生成 (/generate, /generate/batch, /strategies, /types)
│       ├── evaluation.py  # 评估 (/evaluate, /evaluate/metrics, /evaluate/batch)
│       ├── experiments.py # 实验管理 (CRUD + /{id}/run)
│       └── rag.py         # RAG评估 (/rag/query, /rag/evaluate)
│
├── dashboard/              # Streamlit 交互式仪表盘
│   ├── app.py             # Streamlit 主入口 + 页面导航
│   ├── components.py      # 可复用 UI 组件 (图表, 卡片, 表格)
│   ├── utils.py           # API 客户端 + 格式化工具 + session 管理
│   └── pages/             # 页面模块
│       ├── generation.py  # 问题生成交互页面
│       ├── evaluation.py  # 评估结果可视化页面
│       ├── experiments.py # 实验管理页面 (Create/List/Results)
│       └── rag.py         # RAG 评估交互页面
│
└── scripts/                # 启动脚本
    ├── start_api.py       # Uvicorn API 启动器
    └── start_dashboard.py # Streamlit Dashboard 启动器
```

### 1.2 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| API 框架 | FastAPI | ≥0.111.0 | RESTful API + 自动文档 |
| ASGI 服务器 | Uvicorn | ≥0.30.0 | 异步 HTTP 服务 |
| Dashboard | Streamlit | ≥1.35.0 | 交互式 Web 界面 |
| 数据模型 | Pydantic v2 | ≥2.7.0 | 请求校验 + 响应序列化 |
| 可视化 | Plotly | ≥5.22.0 | 图表渲染 |
| 监控 | Prometheus | ≥0.20.0 | /metrics 指标暴露 |
| 日志 | structlog | ≥24.1.0 | 结构化日志 |

---

## 二、API 模块详细实现

### 2.1 `api/main.py` — FastAPI 应用入口

**职责**: 应用工厂模式创建 FastAPI 实例，管理生命周期，注册路由和中间件。

**实现要求**:

1. **应用工厂 `create_app()`**:
   - 从 `get_settings()` 加载全局配置
   - 标题/版本从 `settings.app.name/version` 获取
   - Debug 模式下启用 `/docs`, `/redoc`, `/openapi.json`；生产环境关闭
   - 使用 `@asynccontextmanager lifespan` 管理启动/关闭事件

2. **生命周期管理**:
   - `startup`: 调用 `configure_logging()` 初始化 structlog
   - `shutdown`: 记录关闭日志

3. **中间件注册顺序** (至关重要):
   ```
   RequestIdMiddleware → LoggingMiddleware → ExceptionHandlerMiddleware → CORS
   ```
   - `RequestIdMiddleware` 必须在最外层，确保后续所有中间件都能访问 `request.state.request_id`
   - `ExceptionHandlerMiddleware` 在 CORS 之前捕获未处理异常

4. **异常处理器**:
   - 注册 `RequestValidationError` → 返回结构化 `ErrorResponse` (422)
   - 全局异常由 `ExceptionHandlerMiddleware` 处理 (500)

5. **Prometheus 指标**:
   - 挂载 `/metrics` 端点，使用 `make_asgi_app()`

6. **路由注册**:
   ```python
   app.include_router(health.router)       # prefix=/health
   app.include_router(generation.router)   # prefix=/generate
   app.include_router(evaluation.router)   # prefix=/evaluate
   app.include_router(experiments.router)  # prefix=/experiments
   app.include_router(rag.router)          # prefix=/rag
   ```

**类型注解要求**:
- `create_app() -> FastAPI`
- `lifespan(app: FastAPI) -> AsyncGenerator[None, None]`

---

### 2.2 `api/dependencies.py` — 依赖注入

**职责**: 提供 FastAPI `Depends` 注入项，实现关注点分离。

**核心依赖**:

| 依赖名 | 类型 | 说明 |
|--------|------|------|
| `SettingsDep` | `Annotated[Settings, Depends(get_app_settings)]` | 全局配置注入，使用 `@lru_cache` 缓存避免重复加载 |
| `RequestIdDep` | `Annotated[str, Depends(get_request_id)]` | 从请求头 `X-Request-ID` 或新生成的 UUID 获取 |
| `CommonParamsDep` | `Annotated[CommonQueryParams, Depends()]` | 分页参数 `page`, `per_page`, `sort_by`, `sort_order`，带校验 |
| `OptionalAuthDep` | `Annotated[bool, Depends(optional_api_key)]` | 可选 API Key 验证 (当前开放) |

**实现要求**:
- `get_app_settings()` 使用 `@lru_cache(maxsize=1)` 缓存
- `CommonQueryParams.__init__` 中对 `page` 和 `per_page` 做边界校验，非法值抛 `HTTPException(400)`
- 所有注入函数必须有完整的类型注解

---

### 2.3 `api/middleware.py` — 中间件层

**职责**: 横向切面的请求处理逻辑。

#### 2.3.1 `RequestIdMiddleware`
- 为每个请求生成 UUID4 作为 `request_id`
- 写入 `request.state.request_id`
- 在响应头中返回 `X-Request-ID`

#### 2.3.2 `LoggingMiddleware`
- 使用 `structlog` 记录请求生命周期
- 记录字段: `method`, `path`, `query`, `client`, `request_id`
- 计算并记录响应时间 `duration_ms`
- 在响应头中返回 `X-Response-Time-Ms`
- **错误处理**: 捕获下游异常，记录错误后重新抛出

#### 2.3.3 `ExceptionHandlerMiddleware`
- 捕获任何未处理的异常
- 返回 JSON 格式的 `ErrorResponse` (500)
- 包含 `request_id` 便于追踪

#### 2.3.4 `setup_cors()`
- 读取 `settings.api.cors_origins`
- 开发模式自动追加常用本地端口 (`:3000`, `:5173`, `:8000`, `:8501`)
- 暴露头: `X-Request-ID`, `X-Response-Time-Ms`

**类型注解要求**:
- 所有 `dispatch` 方法签名: `async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response`

---

### 2.4 `api/schemas.py` — API 数据模型

**职责**: 定义所有端点的请求体和响应体模型，与 `core.models.py` 解耦。

**设计原则**:
- API 模型继承自 `pydantic.BaseModel`，不直接使用 `core.models` 的 ORM 模型
- 扩展核心模型以适应 API 场景 (如添加 `success`, `latency_ms`, `error` 包装字段)

**模型清单**:

| 模型 | 用途 | 核心字段 |
|------|------|----------|
| `HealthResponse` | 健康检查 | `status`, `version`, `timestamp`, `environment` |
| `GenerateQuestionRequest` | 单条生成请求 | `knowledge`, `question_type`, `prompt_strategy`, `model_name`, `generation_params` |
| `GenerateBatchRequest` | 批量生成请求 | `knowledge_items` (列表) |
| `GenerationResponse` | 生成响应 | `success`, `data: GeneratedQuestion`, `latency_ms` |
| `BatchGenerationResponse` | 批量生成响应 | `results`, `total_count`, `success_count`, `failed_count` |
| `StrategyInfo` / `TypeInfo` | 元数据 | `name`, `value`, `description` |
| `EvaluateRequest` | 评估请求 | `question`, `reference_question`, `metrics` |
| `EvaluateBatchRequest` | 批量评估 | `questions`, `references`, `metrics` |
| `EvaluationResponse` | 评估响应 | `success`, `question_id`, `scores: list[EvaluationResult]` |
| `CreateExperimentRequest` | 创建实验 | 继承 `ExperimentConfig` |
| `ExperimentResponse` | 实验操作响应 | `success`, `experiment_id`, `data` |
| `ExperimentListResponse` | 实验列表 | `experiments`, `total` |
| `RAGQueryRequest` | RAG查询 | `query`, `documents`, `top_k`, `retriever_type` |
| `RAGQueryResponse` | RAG响应 | `success`, `result: RAGResult`, `latency_ms` |
| `ErrorResponse` | 错误响应 | `success=False`, `error`, `details`, `request_id` |
| `PaginatedResponse` | 分页基础 | `total`, `page`, `per_page`, `total_pages`, `items` |

---

### 2.5 `api/routes/` — 路由模块

#### 2.5.1 `health.py` — 健康检查

**端点**:
- `GET /health` — 服务健康状态 (`healthy`)
- `GET /health/ready` — 就绪探针 (`ready`)，检查依赖加载
- `GET /health/live` — 存活探针 (`alive`)，进程存活检查

**实现要求**:
- 所有端点返回 `HealthResponse`
- 状态码 200
- 轻量级，适合 K8s 探针使用

#### 2.5.2 `generation.py` — 问题生成

**端点**:
- `GET /generate/strategies` — 返回 `StrategyInfo` 列表
- `GET /generate/types` — 返回 `TypeInfo` 列表
- `POST /generate` — 单条生成，返回 `GenerationResponse` (201)
- `POST /generate/batch` — 批量生成，返回 `BatchGenerationResponse` (201)

**实现要求**:
1. **模型名称校验**: 非空且长度 ≤ 100，否则 400
2. **提供商推断**: 根据 `model_name` 自动推断 `ModelProvider` (OLLAMA/OPENAI/DEEPSEEK/CLAUDE)
3. **参数合并**: `generation_params` 与 `settings.generation` 默认值合并
4. **批处理限制**: 检查 batch size 不超过 `settings.generation.batch_size * 10`
5. **逐条错误处理**: 批量模式下单条失败不影响其他条目
6. **延迟计算**: 使用 `time.perf_counter()` 精确计时，返回 `latency_ms`
7. **日志**: 每个请求记录开始/完成/失败事件，包含 `request_id`

**当前占位逻辑**:
- `_mock_generate_question()` 生成基于输入内容的占位问题文本
- 后续由 `generators.question_generator` 替换为真实 LLM 调用

#### 2.5.3 `evaluation.py` — 评估

**端点**:
- `GET /evaluate/metrics` — 返回可用指标列表及描述
- `POST /evaluate` — 单条评估，返回 `EvaluationResponse`
- `POST /evaluate/batch` — 批量评估，返回 `list[EvaluationResponse]`

**实现要求**:
1. **指标校验**: 请求的 `metrics` 必须在 `settings.evaluation.metrics` 中，否则 400
2. **参考文本可选**: `reference_question` 为 `None` 时，基于参考的指标 (BLEU/ROUGE/BERTScore) 返回 0
3. **LLM Judge**: 不需要参考文本，返回固定评分占位
4. **批量校验**: `questions` 和 `references` 长度必须匹配，否则 400

**当前占位逻辑**:
- `_compute_metrics()` 返回合成评分: BLEU=0.42, ROUGE=0.55, BERTScore=0.78, LLM_Judge=4.2
- 后续由 `evaluators.metrics` 替换为真实计算

#### 2.5.4 `experiments.py` — 实验管理

**端点**:
- `POST /experiments` — 创建实验，返回 `ExperimentResponse` (201)
- `GET /experiments` — 列表查询，支持分页/排序
- `GET /experiments/{id}` — 获取详情
- `DELETE /experiments/{id}` — 删除实验
- `POST /experiments/{id}/run` — 运行实验 (202 Accepted)

**实现要求**:
1. **存储**: 当前使用内存字典 `_experiments` 存储，生产环境需替换为数据库
2. **ID 唯一性**: 创建时检查重复，返回 409 Conflict
3. **字段校验**: `models`, `prompt_strategies`, `question_types` 非空检查
4. **dry_run**: `/run` 支持 dry_run 模式，返回估计结果而不执行
5. **结果生成**: `_mock_generate_for_experiment()` 按笛卡尔积生成组合问题

#### 2.5.5 `rag.py` — RAG 评估

**端点**:
- `POST /rag/query` — 执行 RAG 查询
- `POST /rag/evaluate` — 评估 RAG 结果质量

**实现要求**:
1. **文档非空校验**: `documents` 为空时返回 400
2. **检索逻辑**: `_retrieve_documents()` 当前使用关键词匹配，后续替换为向量检索
3. **答案生成**: `_generate_rag_answer()` 基于检索文档拼接上下文
4. **评估指标**: `retrieval_accuracy`, `answer_relevance`, `faithfulness`, `context_precision`

---

## 三、Dashboard 模块详细实现

### 3.1 `dashboard/app.py` — Streamlit 主入口

**职责**: 页面配置 + 侧边栏导航 + 页面分发。

**实现要求**:

1. **页面配置**:
   ```python
   st.set_page_config(
       page_title=settings.dashboard.title,    # "KnowProbe Dashboard"
       page_icon=settings.dashboard.page_icon, # "🔍"
       layout="wide",
       initial_sidebar_state="expanded",
   )
   ```

2. **侧边栏导航**:
   - 标题: `🔍 KnowProbe`
   - 导航: Question Generation / Evaluation / Experiments / RAG Evaluation
   - 底部信息: Version, Environment

3. **页面映射**:
   ```python
   PAGE_MAP = {
       "Question Generation": generation,
       "Evaluation": evaluation,
       "Experiments": experiments,
       "RAG Evaluation": rag,
   }
   ```
   - 每个模块必须有 `render()` 函数作为入口

4. **页头/页脚**: 调用 `components.render_header()` 和 `render_footer()`

---

### 3.2 `dashboard/components.py` — UI 组件

**职责**: 封装可复用的可视化组件，统一视觉风格。

**设计规范**:
- **配色**: 低饱和度、暖色调，禁用蓝紫渐变
- **布局**: 充足的留白 (whitespace)，清晰的信息层级
- **图表库**: Plotly (交互式，支持缩放/悬停)

**组件清单**:

| 函数 | 用途 | 参数 |
|------|------|------|
| `render_header()` | 页面标题 + 描述 | — |
| `render_footer()` | 底部时间戳 + 版本 | — |
| `metric_card(title, value, delta)` | 指标卡片 | 支持 delta 变化 |
| `info_card(title, content)` | 信息卡片 | 带边框容器 |
| `render_bar_chart(data, ...)` | 水平条形图 | `dict[str, float]` |
| `render_grouped_bar_chart(data, ...)` | 分组条形图 | `dict[str, dict[str, float]]` |
| `render_radar_chart(categories, values, ...)` | 雷达图 | 至少3个维度 |
| `render_comparison_heatmap(data, ...)` | 对比热力图 | `RdYlGn` 色阶 |
| `render_data_table(data, ...)` | 数据表格 | `list[dict]` |

**热力图特别说明**:
- 色阶: `RdYlGn` (红-黄-绿)，`zmin=0`, `zmax=1`
- 文本显示: `texttemplate="%{text}"`，保留两位小数

---

### 3.3 `dashboard/utils.py` — 工具函数

**职责**: API 客户端、格式化、Session 状态管理。

**核心功能**:

1. **API 客户端**:
   - `api_get(endpoint, params)` — 带 `@st.cache_data(ttl=60)` 缓存的 GET 请求
   - `api_post(endpoint, payload)` — POST 请求 (无缓存，保证实时性)
   - 基地址: `http://{settings.api.host}:{settings.api.port}`
   - 超时: GET=10s, POST=60s
   - 失败时返回 `None`，由调用方处理

2. **格式化**:
   - `format_strategy_label(strategy: str) -> str` — 枚举值 → 可读标签
   - `format_question_type_label(qtype: str) -> str`
   - `format_score(score, metric) -> str` — LLM Judge 显示为 `x.x/5`，其他显示为 `0.xxx`

3. **Session 状态**:
   - `ensure_session_state(key, default)` — 安全地初始化 session state
   - `clear_session_state(pattern)` — 清除匹配 key

---

### 3.4 `dashboard/pages/` — 页面模块

#### 3.4.1 `generation.py` — 问题生成页面

**UI 布局**:
1. **顶部控制区** (两列):
   - Prompt Strategy 选择框
   - Question Type 选择框
2. **模型输入**: `model_name` 文本框
3. **知识输入区**:
   - Input Type 选择 (`triple`/`schema`/`text`/`entity`)
   - Source ID 输入
   - Knowledge Content 文本域
4. **高级参数** (折叠面板):
   - Temperature (0.0-1.5)
   - Top-P (0.0-1.0)
   - Max Length (16-1024)
5. **生成按钮** (主按钮样式)
6. **批量生成区**:
   - 多行输入文本域
   - 批量生成按钮
7. **历史记录区**: 显示 `session_state["gen_history"]` 中的结果

**交互逻辑**:
- 点击生成 → 调用 `POST /generate` → 显示结果 + 进度条(confidence) + 添加到历史
- 批量生成 → 调用 `POST /generate/batch` → 显示汇总统计

#### 3.4.2 `evaluation.py` — 评估页面

**UI 布局**:
1. **评估输入区** (两列):
   - Generated Question 文本域
   - Reference Question 文本域 (可选)
2. **指标选择**: 多选框 (默认 BLEU + ROUGE)
3. **评估按钮**
4. **结果可视化**:
   - 指标卡片行 (metric 卡片)
   - 条形图 (最新评分)
   - 历史表格
   - 雷达图 (≥3 个指标时)
5. **指标说明区**: 从 `GET /evaluate/metrics` 加载并显示

#### 3.4.3 `experiments.py` — 实验管理页面

**Tab 导航**:
1. **Create Tab**:
   - Experiment ID, Name, Description 输入
   - Models 多选 (llama3.1:8b, qwen2.5:7b, flan-t5-large, gpt-4o-mini, deepseek-chat)
   - Strategies 多选
   - Types 多选
   - Metrics 多选
   - Knowledge Sources 文本域
   - 创建按钮 → `POST /experiments`

2. **List Tab**:
   - 从 `GET /experiments` 加载列表
   - 每个实验显示为卡片: 名称 + 模型数/策略数/类型数 + Run 按钮
   - 汇总表格

3. **Results Tab**:
   - 选择实验下拉框
   - 分组条形图 (Model × Strategy)
   - 生成问题表格

#### 3.4.4 `rag.py` — RAG 评估页面

**UI 布局**:
1. **文档集合区**:
   - 文本域输入 (格式: `title | content`，每行一个)
   - 预置5条关于爱因斯坦的示例文档
2. **查询区**:
   - Query Text 输入
   - Expected Answer 输入 (可选)
   - Top-K 滑动条 (1-10)
3. **按钮区** (两列):
   - Execute RAG Query (主按钮)
   - Evaluate RAG Result
4. **结果显示**:
   - 检索到的文档卡片列表
   - 生成的答案信息框
5. **历史记录**: 折叠面板展示过往查询

---

## 四、启动脚本

### 4.1 `scripts/start_api.py`

```python
# 使用 uvicorn 启动 FastAPI
# 自动读取 settings.api.host/port/workers
# Debug 模式下启用 reload，workers 强制为 1
```

### 4.2 `scripts/start_dashboard.py`

```python
# 使用 subprocess 调用 streamlit run
# 自动读取 settings.dashboard.port
# 禁用 usage stats 收集
```

**CLI 集成**:
- 项目已有的 `kp serve` 命令直接导入 `knowprobe.api.main:app`
- 项目已有的 `kp dashboard` 命令查找 `dashboard/app.py`
- 启动脚本作为独立入口，供不想使用 CLI 的用户使用

---

## 五、错误处理规范

### 5.1 HTTP 状态码使用

| 场景 | 状态码 | 说明 |
|------|--------|------|
| 成功 | 200 | GET, POST 评估 |
| 创建成功 | 201 | POST 生成 |
| 接受执行 | 202 | POST 运行实验 (异步) |
| 参数错误 | 400 | 校验失败 |
| 验证错误 | 422 | Pydantic 校验失败 |
| 资源不存在 | 404 | 实验 ID 不存在 |
| 资源冲突 | 409 | 实验 ID 重复 |
| 服务器错误 | 500 | 未捕获异常 |

### 5.2 错误响应格式

```json
{
  "success": false,
  "error": "Validation error",
  "details": [
    {"field": "knowledge_items", "message": "must not be empty", "type": "validation_error"}
  ],
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 六、类型注解与工程标准

### 6.1 类型注解要求

- 所有函数必须有返回类型注解 (`-> T`)
- 所有参数必须有类型注解 (`param: T`)
- 使用 `from __future__ import annotations` 延迟注解求值
- 复杂类型使用 `typing` 模块: `dict[str, Any]`, `list[GeneratedQuestion]`, `str | None`
- mypy 严格模式: `disallow_untyped_defs = true`

### 6.2 代码风格

- 行长度: 100 字符 (`tool.ruff.line-length = 100`)
- 目标 Python 版本: 3.11+
- Ruff 规则: `E, F, W, I, N, D, UP, B, C4, SIM`
- 文档字符串: 所有公共模块/类/函数必须有 docstring

### 6.3 日志规范

- 使用 `structlog` 结构化日志
- 事件名使用 snake_case: `generate_question_start`, `evaluate_batch_complete`
- 所有日志包含 `request_id`
- 关键路径记录延迟: `latency_ms`
- 错误日志记录异常详情: `error`, `error_type`

---

## 七、与现有基础设施的集成

### 7.1 配置集成
- API 和 Dashboard 均通过 `get_settings()` 读取 `configs/default.yaml`
- API host/port 从 `settings.api.*` 获取
- Dashboard port/title 从 `settings.dashboard.*` 获取
- 模型默认名称从 `settings.models.local.default_model` 获取

### 7.2 模型复用
- API schemas 扩展 `core.models` 中的 `GeneratedQuestion`, `EvaluationResult` 等
- 不修改 `core/models.py`，保持向后兼容

### 7.3 日志集成
- 所有模块使用 `from knowprobe.utils.logging import get_logger`
- 命名空间: `api.routes.generation`, `dashboard.pages.evaluation`

### 7.4 后续替换点

| 占位模块 | 当前实现 | 后续替换为 |
|----------|----------|-----------|
| `_mock_generate_question` | 字符串拼接 | `generators.question_generator.generate()` |
| `_compute_metrics` | 合成评分 | `evaluators.metrics.compute_bleu()`, `compute_rouge()`, etc. |
| `_retrieve_documents` | 关键词匹配 | `rag.retriever.Retriever.retrieve()` |
| `_generate_rag_answer` | 上下文拼接 | `rag.rag_generator.generate()` |
| `_experiments` dict | 内存存储 | `db` 模块 + SQLAlchemy ORM |

---

## 八、文件清单

### 新建/修改文件

```
src/knowprobe/api/__init__.py                          # 包初始化
src/knowprobe/api/main.py                              # FastAPI 应用入口
src/knowprobe/api/dependencies.py                      # 依赖注入
src/knowprobe/api/middleware.py                        # 中间件
src/knowprobe/api/schemas.py                           # API 数据模型
src/knowprobe/api/routes/__init__.py                   # 路由包初始化
src/knowprobe/api/routes/health.py                     # 健康检查
src/knowprobe/api/routes/generation.py                 # 问题生成
src/knowprobe/api/routes/evaluation.py                 # 评估
src/knowprobe/api/routes/experiments.py                # 实验管理
src/knowprobe/api/routes/rag.py                        # RAG评估
src/knowprobe/dashboard/__init__.py                    # 包初始化
src/knowprobe/dashboard/app.py                         # Streamlit 主入口
src/knowprobe/dashboard/components.py                  # UI 组件
src/knowprobe/dashboard/utils.py                       # 工具函数
src/knowprobe/dashboard/pages/__init__.py              # 页面包初始化
src/knowprobe/dashboard/pages/generation.py            # 生成页面
src/knowprobe/dashboard/pages/evaluation.py            # 评估页面
src/knowprobe/dashboard/pages/experiments.py           # 实验页面
src/knowprobe/dashboard/pages/rag.py                   # RAG页面
scripts/start_api.py                                   # API 启动脚本
scripts/start_dashboard.py                             # Dashboard 启动脚本
```

---

## 九、启动方式

### 方式一: 直接脚本启动
```bash
# API
python scripts/start_api.py

# Dashboard
python scripts/start_dashboard.py
```

### 方式二: CLI 命令
```bash
# API
kp serve --reload

# Dashboard
kp dashboard
```

### 方式三: 开发模式
```bash
# API (带热重载)
uvicorn knowprobe.api.main:app --reload --port 8000

# Dashboard
streamlit run src/knowprobe/dashboard/app.py --server.port 8501
```

---

## 十、API 端点总览

| 方法 | 路径 | 说明 | 状态码 |
|------|------|------|--------|
| GET | `/health` | 健康检查 | 200 |
| GET | `/health/ready` | 就绪探针 | 200 |
| GET | `/health/live` | 存活探针 | 200 |
| GET | `/generate/strategies` | 策略列表 | 200 |
| GET | `/generate/types` | 类型列表 | 200 |
| POST | `/generate` | 单条生成 | 201 |
| POST | `/generate/batch` | 批量生成 | 201 |
| GET | `/evaluate/metrics` | 指标列表 | 200 |
| POST | `/evaluate` | 单条评估 | 200 |
| POST | `/evaluate/batch` | 批量评估 | 200 |
| POST | `/experiments` | 创建实验 | 201 |
| GET | `/experiments` | 实验列表 | 200 |
| GET | `/experiments/{id}` | 获取实验 | 200 |
| DELETE | `/experiments/{id}` | 删除实验 | 200 |
| POST | `/experiments/{id}/run` | 运行实验 | 202 |
| POST | `/rag/query` | RAG查询 | 200 |
| POST | `/rag/evaluate` | RAG评估 | 200 |
| GET | `/metrics` | Prometheus指标 | 200 |
