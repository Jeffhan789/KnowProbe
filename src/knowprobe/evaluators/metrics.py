"""Evaluation metrics computation for KnowProbe.

Implements automatic evaluation metrics for question generation quality:
- BLEU-1/2/3/4 (n-gram precision with brevity penalty)
- ROUGE-1/2/L (recall-oriented n-gram overlap)
- METEOR (stemmed/synonym-aware matching)
- BERTScore (semantic similarity via contextual embeddings)
- Self-BLEU (generation diversity)
- Distinct-N (lexical diversity)
- Grammar correctness (heuristic checks)
"""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from transformers import logging as transformers_logging

from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)
transformers_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricScore:
    """Score returned by a single metric."""

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            logger.warning(
                "metric_out_of_range",
                metric=self.name,
                value=self.value,
            )


@dataclass
class AggregateScore:
    """Aggregated scores across multiple samples."""

    metric_name: str
    mean: float
    std: float
    median: float
    min: float
    max: float
    count: int
    raw_scores: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Base metric
# ---------------------------------------------------------------------------


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    metric_name: ClassVar[str] = ""

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        """Compute the metric for a batch of predictions and references.

        Args:
            predictions: Generated texts (one per sample).
            references: Reference texts. Either one string per sample or
                multiple references per sample (list of lists).
            **kwargs: Additional metric-specific parameters.

        Returns:
            A list of MetricScore objects, one per sample or one aggregate.
        """

    def _normalize_text(self, text: str) -> str:
        """Lower-case, strip punctuation, and collapse whitespace."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def _get_ngrams(self, text: str, n: int) -> Counter[str]:
        """Extract n-gram counts from a whitespace-tokenized string."""
        tokens = text.split()
        if len(tokens) < n:
            return Counter()
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)


# ---------------------------------------------------------------------------
# BLEU metric
# ---------------------------------------------------------------------------


class BLEUMetric(BaseMetric):
    """BLEU metric with configurable max n-gram order (default 4)."""

    metric_name = "bleu"

    def __init__(self, max_order: int = 4, smooth: bool = True) -> None:
        self.max_order = max_order
        self.smooth = smooth
        try:
            import sacrebleu

            self._sacrebleu = sacrebleu
            self._use_sacrebleu = True
            logger.info("bleu_using_sacrebleu", max_order=max_order)
        except ImportError:
            self._use_sacrebleu = False
            logger.warning("bleu_fallback_to_naive_impl")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        # Normalize references to list-of-lists
        refs: list[list[str]]
        if references and isinstance(references[0], str):
            refs = [[r] for r in references]  # type: ignore[arg-type]
        else:
            refs = references  # type: ignore[assignment]

        if self._use_sacrebleu:
            return self._compute_sacrebleu(predictions, refs, **kwargs)
        return self._compute_naive(predictions, refs)

    def _compute_sacrebleu(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        """Use sacrebleu for accurate corpus-level BLEU."""
        # sacrebleu expects refs transposed: [[r1_1, r1_2], [r2_1, r2_2]] -> [[r1_1, r2_1], [r1_2, r2_2]]
        num_refs = max(len(r) for r in references)
        ref_lists: list[list[str]] = [[] for _ in range(num_refs)]
        for sample_refs in references:
            for i in range(num_refs):
                ref_lists[i].append(sample_refs[i] if i < len(sample_refs) else "")

        # Corpus-level BLEU
        metric = self._sacrebleu.metrics.BLEU(
            smooth_method="exp" if self.smooth else "none",
            max_ngram_order=self.max_order,
        )
        bleu = metric.corpus_score(predictions, ref_lists)
        score = bleu.score / 100.0  # sacrebleu returns 0-100
        details = {
            "bleu_score_100": bleu.score,
            "bp": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
            "precisions": bleu.precisions,
        }
        return [MetricScore(name=f"bleu-{self.max_order}", value=score, details=details)]

    def _compute_naive(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> list[MetricScore]:
        """Naive implementation when sacrebleu is unavailable."""
        scores: list[float] = []
        for pred, refs in zip(predictions, references, strict=False):
            pred_norm = self._normalize_text(pred)
            ref_norms = [self._normalize_text(r) for r in refs]
            score = self._bleu_naive_single(pred_norm, ref_norms)
            scores.append(score)

        avg_score = float(np.mean(scores)) if scores else 0.0
        return [
            MetricScore(
                name=f"bleu-{self.max_order}", value=avg_score, details={"num_samples": len(scores)}
            )
        ]

    def _bleu_naive_single(self, prediction: str, references: list[str]) -> float:
        """Compute BLEU for a single prediction against multiple references."""
        best_bp = 0.0

        for ref in references:
            precisions = []
            for n in range(1, self.max_order + 1):
                pred_ngrams = self._get_ngrams(prediction, n)
                ref_ngrams = self._get_ngrams(ref, n)
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                matches = sum((pred_ngrams & ref_ngrams).values())
                precision = matches / sum(pred_ngrams.values())
                precisions.append(precision)

            # Brevity penalty
            pred_len = len(prediction.split())
            ref_len = len(ref.split())
            if pred_len > ref_len:
                bp = 1.0
            elif pred_len == 0:
                bp = 0.0
            else:
                bp = np.exp(1 - ref_len / pred_len) if ref_len > 0 else 0.0

            geo_mean = (
                np.exp(np.mean([np.log(max(p, 1e-10)) for p in precisions])) if precisions else 0.0
            )
            bleu = bp * geo_mean
            if bleu > best_bp:
                best_bp = bleu

        return best_bp


# ---------------------------------------------------------------------------
# ROUGE metric
# ---------------------------------------------------------------------------


class ROUGEMetric(BaseMetric):
    """ROUGE-L (and optionally ROUGE-1/2) metric for text overlap."""

    metric_name = "rouge"

    def __init__(self, use_stemmer: bool = True, rouge_types: list[str] | None = None) -> None:
        self.use_stemmer = use_stemmer
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        try:
            from rouge_score import rouge_scorer

            self._scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)
            self._use_lib = True
            logger.info("rouge_using_rouge_score_lib", types=self.rouge_types)
        except ImportError:
            self._use_lib = False
            logger.warning("rouge_fallback_to_naive_impl")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        if self._use_lib:
            return self._compute_with_lib(predictions, references)
        return self._compute_naive(predictions, references)

    def _compute_with_lib(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
    ) -> list[MetricScore]:
        scores: dict[str, list[float]] = {t: [] for t in self.rouge_types}

        for pred, refs in zip(predictions, references, strict=False):
            # Use the first reference if multiple are provided
            ref = refs if isinstance(refs, str) else str(refs[0])
            result = self._scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                scores[rouge_type].append(result[rouge_type].fmeasure)

        metric_scores: list[MetricScore] = []
        for rouge_type, values in scores.items():
            mean_score = float(np.mean(values)) if values else 0.0
            metric_scores.append(
                MetricScore(
                    name=rouge_type,
                    value=mean_score,
                    details={
                        "mean": mean_score,
                        "std": float(np.std(values)) if values else 0.0,
                        "num_samples": len(values),
                    },
                )
            )
        return metric_scores

    def _compute_naive(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
    ) -> list[MetricScore]:
        """Naive LCS-based ROUGE-L implementation."""
        scores: list[float] = []
        for pred, refs in zip(predictions, references, strict=False):
            ref = refs if isinstance(refs, str) else str(refs[0])
            pred_toks = self._normalize_text(pred).split()
            ref_toks = self._normalize_text(ref).split()
            lcs_len = self._lcs_length(pred_toks, ref_toks)
            if not pred_toks or not ref_toks:
                scores.append(0.0)
                continue
            precision = lcs_len / len(pred_toks)
            recall = lcs_len / len(ref_toks)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            scores.append(f1)

        mean_score = float(np.mean(scores)) if scores else 0.0
        return [
            MetricScore(
                name="rougeL",
                value=mean_score,
                details={
                    "mean": mean_score,
                    "std": float(np.std(scores)) if scores else 0.0,
                    "num_samples": len(scores),
                },
            )
        ]

    @staticmethod
    def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]


# ---------------------------------------------------------------------------
# METEOR metric
# ---------------------------------------------------------------------------


class METEORMetric(BaseMetric):
    """METEOR metric with synonym and stem matching."""

    metric_name = "meteor"

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        try:
            import nltk
            from nltk.corpus import wordnet

            self._nltk = nltk
            self._wordnet = wordnet
            # Ensure wordnet is available
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            self._use_nltk = True
            logger.info("metor_using_nltk_wordnet")
        except ImportError:
            self._use_nltk = False
            logger.warning("meteor_fallback_no_synonyms")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        scores: list[float] = []
        for pred, refs in zip(predictions, references, strict=False):
            ref = refs if isinstance(refs, str) else str(refs[0])
            score = self._meteor_single(pred, ref)
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        return [
            MetricScore(
                name="meteor",
                value=mean_score,
                details={
                    "mean": mean_score,
                    "std": float(np.std(scores)) if scores else 0.0,
                    "num_samples": len(scores),
                    "alpha": self.alpha,
                },
            )
        ]

    def _meteor_single(self, prediction: str, reference: str) -> float:
        """Compute METEOR for a single prediction-reference pair."""
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Exact matches
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        exact_matches = sum((pred_counts & ref_counts).values())

        # Synonym matches (if NLTK available)
        synonym_matches = 0
        if self._use_nltk:
            pred_unmatched = list((pred_counts - ref_counts).elements())
            ref_unmatched = list((ref_counts - pred_counts).elements())
            synonym_matches = self._count_synonym_matches(pred_unmatched, ref_unmatched)

        total_matches = exact_matches + synonym_matches
        precision = total_matches / len(pred_tokens) if pred_tokens else 0.0
        recall = total_matches / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            return 0.0

        f_mean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        # Fragmentation penalty
        pred_chunks = self._count_chunks(pred_tokens, ref_tokens)
        penalty = (
            self.gamma * ((pred_chunks / total_matches) ** self.beta) if total_matches > 0 else 0.0
        )

        return f_mean * (1 - penalty)

    def _count_synonym_matches(self, pred_tokens: list[str], ref_tokens: list[str]) -> int:
        """Count synonym matches between two token lists."""
        matches = 0
        matched_ref = set()
        for p_tok in pred_tokens:
            for i, r_tok in enumerate(ref_tokens):
                if i in matched_ref:
                    continue
                if self._are_synonyms(p_tok, r_tok):
                    matches += 1
                    matched_ref.add(i)
                    break
        return matches

    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms using WordNet."""
        if word1 == word2:
            return True
        try:
            synsets1 = self._wordnet.synsets(word1)
            synsets2 = self._wordnet.synsets(word2)
            for syn1 in synsets1:
                for syn2 in synsets2:
                    if syn1 == syn2:
                        return True
                    # Check lemma names overlap
                    if set(syn1.lemma_names()) & set(syn2.lemma_names()):
                        return True
            return False
        except Exception:
            return False

    def _count_chunks(self, pred_tokens: list[str], ref_tokens: list[str]) -> int:
        """Count contiguous matching chunks."""
        # Simplified: count contiguous exact matches
        ref_set = set(ref_tokens)
        chunks = 0
        in_chunk = False
        for tok in pred_tokens:
            if tok in ref_set:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False
        return chunks


# ---------------------------------------------------------------------------
# BERTScore metric
# ---------------------------------------------------------------------------


class BERTScoreMetric(BaseMetric):
    """BERTScore for semantic similarity using contextual embeddings."""

    metric_name = "bert_score"

    def __init__(
        self,
        model_type: str = "bert-base-uncased",
        lang: str = "en",
        device: str | None = None,
        batch_size: int = 32,
        rescale_with_baseline: bool = False,
    ) -> None:
        self.model_type = model_type
        self.lang = lang
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline
        self._bertscore = None

    def _load_bertscore(self) -> Any:
        """Lazy-load bert-score library."""
        if self._bertscore is None:
            try:
                from bert_score import score as bert_score_fn

                self._bertscore = bert_score_fn
                logger.info("bert_score_loaded", model=self.model_type)
            except ImportError as e:
                logger.error("bert_score_import_failed", error=str(e))
                raise RuntimeError(
                    "bert-score library is required. Install with: pip install bert-score"
                ) from e
        return self._bertscore

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        # Use first reference per sample
        refs: list[str] = []
        for r in references:
            if isinstance(r, list):
                refs.append(r[0] if r else "")
            else:
                refs.append(r)

        score_fn = self._load_bertscore()
        try:
            P, R, F1 = score_fn(
                predictions,
                refs,
                model_type=self.model_type,
                lang=self.lang,
                device=self.device,
                verbose=False,
                batch_size=self.batch_size,
                rescale_with_baseline=self.rescale_with_baseline,
            )
            p_scores = P.tolist()
            r_scores = R.tolist()
            f1_scores = F1.tolist()

            mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
            return [
                MetricScore(
                    name="bert_score_f1",
                    value=mean_f1,
                    details={
                        "precision_mean": float(np.mean(p_scores)) if p_scores else 0.0,
                        "recall_mean": float(np.mean(r_scores)) if r_scores else 0.0,
                        "f1_mean": mean_f1,
                        "model_type": self.model_type,
                        "num_samples": len(f1_scores),
                    },
                )
            ]
        except Exception as e:
            logger.error("bert_score_computation_failed", error=str(e))
            return [MetricScore(name="bert_score_f1", value=0.0, details={"error": str(e)})]


# ---------------------------------------------------------------------------
# Self-BLEU metric (diversity)
# ---------------------------------------------------------------------------


class SelfBLEUMetric(BaseMetric):
    """Self-BLEU measures diversity within a set of generated texts.

    Lower Self-BLEU indicates higher diversity (less similarity between
    generated samples).
    """

    metric_name = "self_bleu"

    def __init__(self, max_order: int = 4) -> None:
        self.max_order = max_order

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]] | None = None,
        **kwargs: Any,
    ) -> list[MetricScore]:
        """Compute Self-BLEU across all predictions.

        For each prediction, compute BLEU against all other predictions
        as references. Average these scores.
        """
        if len(predictions) < 2:
            return [
                MetricScore(name="self_bleu", value=0.0, details={"reason": "insufficient_samples"})
            ]

        normalized = [self._normalize_text(p) for p in predictions]
        scores: list[float] = []

        for i, pred in enumerate(normalized):
            refs = normalized[:i] + normalized[i + 1 :]
            score = self._bleu_against_refs(pred, refs)
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        # Invert for intuitive scoring: higher = more diverse
        diversity_score = 1.0 - mean_score
        return [
            MetricScore(
                name="self_bleu",
                value=mean_score,
                details={
                    "mean": mean_score,
                    "std": float(np.std(scores)) if scores else 0.0,
                    "diversity_score": diversity_score,
                    "num_samples": len(scores),
                },
            )
        ]

    def _bleu_against_refs(self, prediction: str, references: list[str]) -> float:
        """Compute BLEU of prediction against a list of references."""
        best = 0.0
        for ref in references:
            precisions = []
            for n in range(1, self.max_order + 1):
                pred_ngrams = self._get_ngrams(prediction, n)
                ref_ngrams = self._get_ngrams(ref, n)
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                matches = sum((pred_ngrams & ref_ngrams).values())
                precisions.append(matches / sum(pred_ngrams.values()))

            geo_mean = (
                np.exp(np.mean([np.log(max(p, 1e-10)) for p in precisions])) if precisions else 0.0
            )
            if geo_mean > best:
                best = geo_mean
        return best


# ---------------------------------------------------------------------------
# Distinct-N metric (lexical diversity)
# ---------------------------------------------------------------------------


class DistinctNMetric(BaseMetric):
    """Distinct-N measures the ratio of unique n-grams to total n-grams."""

    metric_name = "distinct_n"

    def __init__(self, n: int = 2) -> None:
        self.n = n

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]] | None = None,
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        all_ngrams: list[str] = []
        for text in predictions:
            normalized = self._normalize_text(text)
            ngrams = self._get_ngrams(normalized, self.n)
            all_ngrams.extend(ngrams.keys())

        total = len(all_ngrams)
        unique = len(set(all_ngrams))
        ratio = unique / total if total > 0 else 0.0

        return [
            MetricScore(
                name=f"distinct-{self.n}",
                value=ratio,
                details={
                    "unique_ngrams": unique,
                    "total_ngrams": total,
                    "num_sentences": len(predictions),
                },
            )
        ]


# ---------------------------------------------------------------------------
# Grammar correctness metric (heuristic)
# ---------------------------------------------------------------------------


class GrammarMetric(BaseMetric):
    """Heuristic grammar correctness score.

    Checks for common issues like repeated punctuation, missing spaces,
    unbalanced quotes, etc. This is a lightweight proxy; for production
    use, consider integrating LanguageTool or a fine-tuned classifier.
    """

    metric_name = "grammar"

    def __init__(self) -> None:
        self._patterns = [
            (re.compile(r"[.!?]{2,}"), 0.3, "repeated_punctuation"),
            (re.compile(r"\s[,.!?;:]"), 0.2, "missing_space_before_punctuation"),
            (re.compile(r"[a-zA-Z][A-Z]{2,}[a-zA-Z]"), 0.1, "unlikely_caps"),
            (re.compile(r"\([^)]*$"), 0.4, "unbalanced_paren"),
            (re.compile(r'"[^"]*$'), 0.4, "unbalanced_quote"),
            (re.compile(r"\s{2,}"), 0.1, "multiple_spaces"),
            (re.compile(r"[\u4e00-\u9fff]"), 0.0, "chinese_chars"),  # Neutral for Chinese
        ]

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]] | None = None,
        **kwargs: Any,
    ) -> list[MetricScore]:
        if not predictions:
            return []

        scores: list[float] = []
        issue_counts: Counter[str] = Counter()

        for text in predictions:
            score, issues = self._check_text(text)
            scores.append(score)
            for issue in issues:
                issue_counts[issue] += 1

        mean_score = float(np.mean(scores)) if scores else 0.0
        return [
            MetricScore(
                name="grammar_score",
                value=mean_score,
                details={
                    "mean": mean_score,
                    "std": float(np.std(scores)) if scores else 0.0,
                    "issue_counts": dict(issue_counts),
                    "num_samples": len(scores),
                },
            )
        ]

    def _check_text(self, text: str) -> tuple[float, list[str]]:
        """Check a single text for grammar issues. Return score and issue list."""
        penalty = 0.0
        issues: list[str] = []
        for pattern, weight, issue_name in self._patterns:
            if pattern.search(text):
                penalty += weight
                issues.append(issue_name)
        score = max(0.0, 1.0 - penalty)
        return score, issues


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


class MetricRegistry:
    """Registry for all available metrics."""

    _metrics: ClassVar[dict[str, type[BaseMetric]]] = {}

    @classmethod
    def register(cls, metric_class: type[BaseMetric]) -> type[BaseMetric]:
        """Register a metric class."""
        cls._metrics[metric_class.metric_name] = metric_class
        logger.debug("metric_registered", name=metric_class.metric_name)
        return metric_class

    @classmethod
    def get(cls, name: str) -> BaseMetric:
        """Get an instance of a registered metric by name."""
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available: {available}")
        return cls._metrics[name]()

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())

    @classmethod
    def build_metrics(cls, names: list[str]) -> list[BaseMetric]:
        """Build a list of metric instances from names."""
        return [cls.get(name) for name in names]


# Register default metrics
MetricRegistry.register(BLEUMetric)
MetricRegistry.register(ROUGEMetric)
MetricRegistry.register(METEORMetric)
MetricRegistry.register(BERTScoreMetric)
MetricRegistry.register(SelfBLEUMetric)
MetricRegistry.register(DistinctNMetric)
MetricRegistry.register(GrammarMetric)
