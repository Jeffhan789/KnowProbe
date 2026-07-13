# Contributing to KnowProbe

首先，感谢你抽出时间为 KnowProbe 做出贡献！🎉

## 开发环境设置

1. Fork 本仓库并克隆到本地：
   ```bash
   git clone https://github.com/<your-username>/KnowProbe.git
   cd KnowProbe
   ```

2. 创建并激活虚拟环境（推荐 Python 3.11+）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. 安装开发依赖：
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## 代码规范

- **格式化**：使用 `ruff` 进行代码格式化和检查
  ```bash
  ruff check src/ tests/
  ruff format src/ tests/
  ```
- **类型检查**：使用 `mypy`
  ```bash
  mypy src/
  ```
- **测试**：使用 `pytest`，确保所有测试通过
  ```bash
  pytest
  ```
- **提交信息**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat:` 新功能
  - `fix:` 修复 bug
  - `chore:` 构建/工具变动
  - `docs:` 文档更新
  - `test:` 测试相关
  - `refactor:` 重构

## 提交 Pull Request 的流程

1. 从 `main` 分支创建新分支：`git checkout -b feat/your-feature-name`
2. 编写代码和测试，确保 CI 检查通过
3. 提交并推送到你的 Fork
4. 在 GitHub 上发起 Pull Request，填写 PR 模板
5. 等待审查并处理反馈

## 报告 Bug

请使用 `.github/ISSUE_TEMPLATE/bug_report.md` 模板，提供尽可能详细的复现步骤和环境信息。

## 许可证

通过提交代码，你同意将其授权给本项目，遵循 MIT 许可证。
