"""Evaluation reporter for KnowProbe.

Generates structured statistical reports from evaluation results:
- Grouped statistics (by model, strategy, question type)
- Significance testing (t-tests, effect sizes)
- Ranking tables
- CSV/Markdown/JSON export

Designed for thesis reporting and reproducible research.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticRow:
    """A single row in a statistics table."""

    group: str
    subgroup: str | None
    metric: str
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    ci_lower: float | None = None
    ci_upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "subgroup": self.subgroup,
            "metric": self.metric,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "ci_lower": round(self.ci_lower, 4) if self.ci_lower is not None else None,
            "ci_upper": round(self.ci_upper, 4) if self.ci_upper is not None else None,
        }

    def to_csv_row(self) -> dict[str, str | int | float | None]:
        return self.to_dict()


@dataclass(frozen=True)
class ComparisonRow:
    """A single comparison result row."""

    comparison_type: str
    baseline: str
    comparison: str
    metric: str
    baseline_mean: float
    comp_mean: float
    diff: float
    percent_change: float
    p_value: float | None
    effect_size: float | None
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_type": self.comparison_type,
            "baseline": self.baseline,
            "comparison": self.comparison,
            "metric": self.metric,
            "baseline_mean": round(self.baseline_mean, 4),
            "comp_mean": round(self.comp_mean, 4),
            "diff": round(self.diff, 4),
            "percent_change": round(self.percent_change, 2),
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None else None,
            "significant": self.significant,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    title: str
    generated_at: datetime
    statistics: list[StatisticRow] = field(default_factory=list)
    comparisons: list[ComparisonRow] = field(default_factory=list)
    rankings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "statistics": [s.to_dict() for s in self.statistics],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rankings": self.rankings,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def to_markdown(self) -> str:
        """Generate a Markdown report suitable for thesis documentation."""
        lines: list[str] = [
            f"# {self.title}",
            "",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Metadata
        if self.metadata:
            lines.append("## Metadata")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Statistics
        if self.statistics:
            lines.append("## Descriptive Statistics")
            lines.append("")
            lines.append(self._statistics_to_markdown())
            lines.append("")

        # Comparisons
        if self.comparisons:
            lines.append("## Comparative Analysis")
            lines.append("")
            lines.append(self._comparisons_to_markdown())
            lines.append("")

        # Rankings
        if self.rankings:
            lines.append("## Rankings")
            lines.append("")
            for ranking_name, items in self.rankings.items():
                lines.append(f"### {ranking_name}")
                lines.append("")
                lines.append("| Rank | Name | Score |")
                lines.append("|------|------|-------|")
                for i, item in enumerate(items, 1):
                    name = item.get("name", "N/A")
                    score = item.get("score", 0.0)
                    lines.append(f"| {i} | {name} | {score:.4f} |")
                lines.append("")

        return "\n".join(lines)

    def _statistics_to_markdown(self) -> str:
        """Convert statistics to Markdown table."""
        if not self.statistics:
            return "No statistics available."

        lines = [
            "| Group | Subgroup | Metric | N | Mean | Std | Median | Min | Max |",
            "|-------|----------|--------|---|------|-----|--------|-----|-----|",
        ]
        for row in self.statistics:
            subgroup = row.subgroup or "-"
            lines.append(
                f"| {row.group} | {subgroup} | {row.metric} | {row.count} | "
                f"{row.mean:.4f} | {row.std:.4f} | {row.median:.4f} | "
                f"{row.min:.4f} | {row.max:.4f} |"
            )
        return "\n".join(lines)

    def _comparisons_to_markdown(self) -> str:
        """Convert comparisons to Markdown table."""
        if not self.comparisons:
            return "No comparisons available."

        lines = [
            "| Type | Baseline | Comparison | Metric | Baseline Mean | Comp Mean | Diff | % Change | p-value | Effect Size | Significant |",
            "|------|----------|------------|--------|---------------|-----------|------|----------|---------|-------------|-------------|",
        ]
        for row in self.comparisons:
            p_val = f"{row.p_value:.4f}" if row.p_value is not None else "N/A"
            es = f"{row.effect_size:.4f}" if row.effect_size is not None else "N/A"
            sig = "✓" if row.significant else "✗"
            lines.append(
                f"| {row.comparison_type} | {row.baseline} | {row.comparison} | {row.metric} | "
                f"{row.baseline_mean:.4f} | {row.comp_mean:.4f} | {row.diff:.4f} | "
                f"{row.percent_change:.2f}% | {p_val} | {es} | {sig} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class EvaluationReporter:
    """Generate statistical reports from evaluation data.

    Supports grouped statistics, significance testing, and multi-format export.
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def build_report(
        self,
        title: str,
        raw_data: dict[str, dict[str, list[float]]],
        comparisons: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Build a complete evaluation report from raw data.

        Args:
            title: Report title.
            raw_data: Nested dict of {group: {metric: [scores]}}.
            comparisons: Optional pre-computed comparison results.
            metadata: Additional metadata for the report.
        """
        statistics = self._compute_statistics(raw_data)
        comparison_rows = self._build_comparison_rows(comparisons or [])
        rankings = self._compute_rankings(raw_data)

        return EvaluationReport(
            title=title,
            generated_at=datetime.now(),
            statistics=statistics,
            comparisons=comparison_rows,
            rankings=rankings,
            metadata=metadata or {},
        )

    def _compute_statistics(
        self,
        raw_data: dict[str, dict[str, list[float]]],
    ) -> list[StatisticRow]:
        """Compute descriptive statistics for all groups and metrics."""
        rows: list[StatisticRow] = []

        for group, metrics in raw_data.items():
            for metric, scores in metrics.items():
                if not scores:
                    continue
                arr = np.array(scores)
                n = len(arr)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                median = float(np.median(arr))
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))

                # Confidence interval
                ci_lower, ci_upper = None, None
                if n > 1 and std > 0:
                    se = std / np.sqrt(n)
                    t_val = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
                    ci_lower = mean - t_val * se
                    ci_upper = mean + t_val * se

                rows.append(
                    StatisticRow(
                        group=group,
                        subgroup=None,
                        metric=metric,
                        count=n,
                        mean=mean,
                        std=std,
                        median=median,
                        min=min_val,
                        max=max_val,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                    )
                )

        return rows

    def _build_comparison_rows(
        self,
        comparisons: list[dict[str, Any]],
    ) -> list[ComparisonRow]:
        """Build comparison rows from raw comparison data."""
        rows: list[ComparisonRow] = []
        for comp in comparisons:
            rows.append(
                ComparisonRow(
                    comparison_type=comp.get("comparison_type", "unknown"),
                    baseline=comp.get("baseline", ""),
                    comparison=comp.get("comparison", ""),
                    metric=comp.get("metric", ""),
                    baseline_mean=comp.get("baseline_mean", 0.0),
                    comp_mean=comp.get("comp_mean", 0.0),
                    diff=comp.get("diff", 0.0),
                    percent_change=comp.get("percent_change", 0.0),
                    p_value=comp.get("p_value"),
                    effect_size=comp.get("effect_size"),
                    significant=comp.get("significant", False),
                )
            )
        return rows

    def _compute_rankings(
        self,
        raw_data: dict[str, dict[str, list[float]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Compute rankings for each metric."""
        rankings: dict[str, list[dict[str, Any]]] = {}

        # Collect all metrics
        all_metrics: set[str] = set()
        for metrics in raw_data.values():
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            items: list[dict[str, Any]] = []
            for group, metrics in raw_data.items():
                if metric in metrics and metrics[metric]:
                    mean_score = float(np.mean(metrics[metric]))
                    items.append({"name": group, "score": mean_score})

            # Sort by score descending
            items.sort(key=lambda x: x["score"], reverse=True)
            rankings[metric] = items

        return rankings

    # -----------------------------------------------------------------------
    # Grouped statistics from experiment data
    # -----------------------------------------------------------------------

    def build_grouped_statistics(
        self,
        data: list[dict[str, Any]],
        group_by: list[str],
        metric_fields: list[str],
    ) -> list[StatisticRow]:
        """Build statistics grouped by multiple fields.

        Args:
            data: List of data dictionaries (e.g., condition results).
            group_by: Fields to group by (e.g., ["model_name", "strategy"]).
            metric_fields: Fields to compute statistics for (e.g., ["overall_score"]).
        """
        # Build nested groups
        grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for item in data:
            key = tuple(str(item.get(f, "unknown")) for f in group_by)
            for metric_field in metric_fields:
                value = item.get(metric_field)
                if isinstance(value, (int, float)):
                    grouped[key][metric_field].append(float(value))
                elif isinstance(value, list) and value:
                    grouped[key][metric_field].extend(
                        float(v) for v in value if isinstance(v, (int, float))
                    )

        rows: list[StatisticRow] = []
        for key, metrics in grouped.items():
            group_name = " / ".join(key)
            for metric, scores in metrics.items():
                if not scores:
                    continue
                arr = np.array(scores)
                n = len(arr)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                median = float(np.median(arr))
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))

                ci_lower, ci_upper = None, None
                if n > 1 and std > 0:
                    se = std / np.sqrt(n)
                    t_val = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
                    ci_lower = mean - t_val * se
                    ci_upper = mean + t_val * se

                rows.append(
                    StatisticRow(
                        group=group_name,
                        subgroup=None,
                        metric=metric,
                        count=n,
                        mean=mean,
                        std=std,
                        median=median,
                        min=min_val,
                        max=max_val,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                    )
                )

        return rows

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_csv(
        self,
        report: EvaluationReport,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Export statistics to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = output_path / fname

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if report.statistics:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "group",
                        "subgroup",
                        "metric",
                        "count",
                        "mean",
                        "std",
                        "median",
                        "min",
                        "max",
                        "ci_lower",
                        "ci_upper",
                    ],
                )
                writer.writeheader()
                for row in report.statistics:
                    writer.writerow(row.to_csv_row())

        logger.info("report_csv_exported", filepath=str(filepath))
        return filepath

    def export_json(
        self,
        report: EvaluationReport,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Export full report to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / fname

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        logger.info("report_json_exported", filepath=str(filepath))
        return filepath

    def export_markdown(
        self,
        report: EvaluationReport,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Export report to Markdown."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = output_path / fname

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        logger.info("report_md_exported", filepath=str(filepath))
        return filepath

    def export_all(
        self,
        report: EvaluationReport,
        output_dir: str | Path,
        basename: str | None = None,
    ) -> dict[str, Path]:
        """Export report in all formats (CSV, JSON, Markdown)."""
        base = basename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "csv": self.export_csv(report, output_dir, f"{base}.csv"),
            "json": self.export_json(report, output_dir, f"{base}.json"),
            "markdown": self.export_markdown(report, output_dir, f"{base}.md"),
        }

    # -----------------------------------------------------------------------
    # Statistical helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_confidence_interval(
        data: list[float],
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute confidence interval for a dataset."""
        if len(data) < 2:
            return (0.0, 0.0)
        arr = np.array(data)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        se = std / np.sqrt(len(arr))
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha / 2, df=len(arr) - 1)
        margin = t_val * se
        return (mean - margin, mean + margin)

    @staticmethod
    def compute_effect_size(
        group1: list[float],
        group2: list[float],
    ) -> float:
        """Compute Cohen's d effect size between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

    @staticmethod
    def compute_pairwise_comparisons(
        groups: dict[str, list[float]],
        metric_name: str = "score",
    ) -> list[dict[str, Any]]:
        """Compute all pairwise t-test comparisons between groups."""
        results: list[dict[str, Any]] = []
        group_names = list(groups.keys())

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1_name, g2_name = group_names[i], group_names[j]
                g1_data, g2_data = groups[g1_name], groups[g2_name]

                if not g1_data or not g2_data:
                    continue

                if len(g1_data) == len(g2_data) and len(g1_data) > 1:
                    t_stat, p_value = stats.ttest_rel(g1_data, g2_data)
                else:
                    t_stat, p_value = stats.ttest_ind(g1_data, g2_data, equal_var=False)

                mean1, mean2 = np.mean(g1_data), np.mean(g2_data)
                diff = mean2 - mean1
                pct_change = (diff / mean1 * 100) if mean1 != 0 else 0.0
                effect_size = EvaluationReporter.compute_effect_size(g1_data, g2_data)

                results.append(
                    {
                        "comparison_type": "pairwise",
                        "baseline": g1_name,
                        "comparison": g2_name,
                        "metric": metric_name,
                        "baseline_mean": mean1,
                        "comp_mean": mean2,
                        "diff": diff,
                        "percent_change": pct_change,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "significant": p_value < 0.05 if p_value is not None else False,
                    }
                )

        return results

    @staticmethod
    def format_for_thesis(
        report: EvaluationReport,
        section: str = "results",
    ) -> str:
        """Generate a LaTeX-friendly formatted section for thesis inclusion."""
        lines: list[str] = []

        if section == "results":
            lines.append("\\section{Evaluation Results}")
            lines.append("")

            # Summary table
            if report.statistics:
                lines.append("\\begin{table}[h]")
                lines.append("\\centering")
                lines.append("\\caption{Descriptive Statistics by Condition}")
                lines.append("\\begin{tabular}{lllcccc}")
                lines.append("\\hline")
                lines.append("Group & Metric & N & Mean & Std & Median & Min/Max \\\\")
                lines.append("\\hline")
                for row in report.statistics:
                    lines.append(
                        f"{row.group} & {row.metric} & {row.count} & "
                        f"{row.mean:.3f} & {row.std:.3f} & {row.median:.3f} & "
                        f"{row.min:.3f}/{row.max:.3f} \\\\"
                    )
                lines.append("\\hline")
                lines.append("\\end{tabular}")
                lines.append("\\end{table}")
                lines.append("")

            # Comparison table
            if report.comparisons:
                lines.append("\\begin{table}[h]")
                lines.append("\\centering")
                lines.append("\\caption{Statistical Comparisons}")
                lines.append("\\begin{tabular}{lllcccc}")
                lines.append("\\hline")
                lines.append(
                    "Comparison & Metric & $\\Delta$ Mean & \\% Change & "
                    "$p$-value & Effect Size \\\\"
                )
                lines.append("\\hline")
                for comp in report.comparisons:
                    sig = "*" if comp.significant else ""
                    p_val = f"{comp.p_value:.4f}{sig}" if comp.p_value is not None else "N/A"
                    es = f"{comp.effect_size:.3f}" if comp.effect_size is not None else "N/A"
                    lines.append(
                        f"{comp.baseline} vs {comp.comparison} & {comp.metric} & "
                        f"{comp.diff:+.3f} & {comp.percent_change:+.1f}\\% & "
                        f"{p_val} & {es} \\\\"
                    )
                lines.append("\\hline")
                lines.append("\\end{tabular}")
                lines.append("\\end{table}")
                lines.append("")

        return "\n".join(lines)
