"""Configuration management for KnowProbe."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "KnowProbe"
    version: str = "2.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/knowprobe.db"
    echo: bool = False


class LocalModelConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    timeout: int = 300


class ApiProviderConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""


class ModelsConfig(BaseModel):
    local: LocalModelConfig = Field(default_factory=LocalModelConfig)
    api: dict[str, ApiProviderConfig] = Field(default_factory=dict)


class GenerationConfig(BaseModel):
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    batch_size: int = 8


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge", "bert_score"])
    llm_judge: dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    retriever: str = "dense"


class PromptsConfig(BaseModel):
    strategy: str = "cot"
    templates_dir: str = "configs/prompts"
    few_shot_examples: int = 3
    self_consistency_samples: int = 5


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_key: str = ""
    allow_unauthenticated: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"]
    )


class DashboardConfig(BaseModel):
    port: int = 8501
    title: str = "KnowProbe Dashboard"
    page_icon: str = "🔍"


class Settings(BaseSettings):
    """Application settings loaded from environment and config files."""

    model_config = SettingsConfigDict(
        env_prefix="KNOWPROBE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


_settings: Settings | None = None


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from config file and environment variables."""
    global _settings
    if _settings is not None:
        return _settings

    config_data: dict[str, Any] = {}
    if config_path is None:
        # Try default locations
        for path in ["configs/default.yaml", "configs/local.yaml"]:
            p = Path(path)
            if p.exists():
                config_path = p
                break

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

    # Merge API keys from env vars
    for provider in ["openai", "deepseek", "claude"]:
        env_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
        if env_key:
            if "models" not in config_data:
                config_data["models"] = {}
            if "api" not in config_data["models"]:
                config_data["models"]["api"] = {}
            if provider not in config_data["models"]["api"]:
                config_data["models"]["api"][provider] = {}
            config_data["models"]["api"][provider]["api_key"] = env_key

    _settings = Settings(**config_data)
    return _settings


def get_settings() -> Settings:
    """Get current settings."""
    if _settings is None:
        return load_settings()
    return _settings
