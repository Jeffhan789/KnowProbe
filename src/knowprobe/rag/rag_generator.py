"""RAG answer generation using retrieved context."""

import time
from abc import ABC, abstractmethod
from typing import Any

from knowprobe.core.models import RAGChunk, RAGQuery
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationBackend(ABC):
    """Abstract backend for text generation."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...


class OllamaBackend(GenerationBackend):
    """Ollama API backend for generation."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Any = None
        logger.info("generator.ollama_init", model=model, base_url=base_url)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError as e:
                raise RuntimeError("httpx is required for Ollama backend") from e
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        client = self._get_client()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 256),
            },
        }
        try:
            response = client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error("generator.ollama_error", error=str(e))
            raise


class TransformersBackend(GenerationBackend):
    """Hugging Face Transformers backend for generation."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None
        logger.info("generator.transformers_init", model=model_name)

    def _load_model(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                logger.info("generator.transformers_loaded", model=self.model_name)
            except ImportError as e:
                raise RuntimeError(
                    "transformers and torch are required for Transformers backend"
                ) from e
        return self._tokenizer, self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        tokenizer, model = self._load_model()
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            import torch

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 256),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return generated.strip()
        except Exception as e:
            logger.error("generator.transformers_error", error=str(e))
            raise


class RAGPromptBuilder:
    """Build prompts for RAG generation with context."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "If the context doesn't contain the answer, say 'I don't have enough information.'"
    )

    def __init__(self, max_context_length: int = 3000) -> None:
        self.max_context_length = max_context_length
        logger.info("prompt_builder.init", max_context=max_context_length)

    def build(self, query: RAGQuery, retrieved_chunks: list[RAGChunk], **kwargs: Any) -> str:
        """Build a RAG prompt with retrieved context."""
        # Build context string
        context_parts: list[str] = []
        current_length = 0

        for i, chunk in enumerate(retrieved_chunks, 1):
            part = f"[Document {i}]\n{chunk.content}\n"
            if current_length + len(part) > self.max_context_length:
                break
            context_parts.append(part)
            current_length += len(part)

        context = "\n".join(context_parts)

        # Build prompt
        system = kwargs.get("system_prompt", self.SYSTEM_PROMPT)
        prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query.query_text}\n\nAnswer:"

        return prompt


class RAGGenerator:
    """Generate answers using RAG with retrieved context."""

    def __init__(
        self,
        backend: GenerationBackend | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        logger.info("rag_generator.init")

    def generate(
        self,
        query: RAGQuery,
        retrieved_chunks: list[RAGChunk],
        **generation_kwargs: Any,
    ) -> tuple[str, str, float]:
        """
        Generate an answer using RAG.

        Returns:
            Tuple of (answer, prompt, latency_ms)
        """
        if not retrieved_chunks:
            logger.warning("generator.no_context", query_id=query.query_id)
            return "No relevant context found.", "", 0.0

        prompt = self.prompt_builder.build(query, retrieved_chunks, **generation_kwargs)

        if self.backend is None:
            logger.warning("generator.no_backend", query_id=query.query_id)
            return "[No generation backend configured]", prompt, 0.0

        start_time = time.perf_counter()
        try:
            answer = self.backend.generate(prompt, **generation_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "generator.complete",
                query_id=query.query_id,
                latency_ms=latency_ms,
                answer_len=len(answer),
            )
            return answer, prompt, latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "generator.error",
                query_id=query.query_id,
                error=str(e),
                latency_ms=latency_ms,
            )
            raise
