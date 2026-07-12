# KnowProbe 知识库输入处理器 — 开发计划

## Stage 1: 异常体系设计
- 文件: `src/knowprobe/parsers/exceptions.py`
- 内容: 定义解析/验证专用的异常类层次结构

## Stage 2: 解析器核心实现
- 文件: `src/knowprobe/parsers/knowledge_parser.py`
- 内容: KnowledgeParser ABC + TripleParser + SchemaParser + TextParser + EntityParser
- 要求: 完整类型注解、结构化日志、策略模式

## Stage 3: 验证器实现
- 文件: `src/knowprobe/parsers/validators.py`
- 内容: InputValidator ABC + 格式验证器 + 语义验证器

## Stage 4: 主处理器实现
- 文件: `src/knowprobe/parsers/knowledge_processor.py`
- 内容: KnowledgeInputProcessor 调度器、管道编排、批处理

## Stage 5: 工具函数与辅助
- 文件: `src/knowprobe/parsers/utils.py`
- 内容: 正则模式、文本清洗、ID生成、辅助函数

## Stage 6: 初始化与接口暴露
- 文件: `src/knowprobe/parsers/__init__.py`
- 内容: 统一导出公共API

## Stage 7: 单元测试
- 文件: `tests/parsers/test_*.py`
- 内容: 完整测试覆盖

## 设计约束
1. 严格遵循 models.py 中的 KnowledgeInput 接口
2. 使用 logging.py 中的 get_logger 记录结构化日志
3. input_type 取值范围: "triple" | "schema" | "text" | "entity"
4. 所有解析器返回 `KnowledgeInput` 实例
5. 支持批处理流水线（batch processing pipeline）
6. 完善的错误处理与回退机制
