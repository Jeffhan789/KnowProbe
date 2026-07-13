"""Utility functions for knowledge input parsing and processing."""

import re
import uuid
from typing import Any

# ─────────────────────────── Regex Patterns ───────────────────────────

# Triple patterns: (Subject, Predicate, Object) or Subject|Predicate|Object
TRIPLE_PATTERN = re.compile(r"\((?P<s>[^,|]+?)\s*,\s*(?P<p>[^,|]+?)\s*,\s*(?P<o>[^,|]+?)\s*\)")
TRIPLE_PIPE_PATTERN = re.compile(r"(?P<s>[^|]+?)\s*\|\s*(?P<p>[^|]+?)\s*\|\s*(?P<o>[^|]+)")

# Schema pattern: Type:Property -> Value
SCHEMA_PATTERN = re.compile(r"(?P<entity_type>\w+)\s*:\s*(?P<property>\w+)\s*->\s*(?P<value>.+)")

# Entity pattern: EntityName [Type] {properties}
ENTITY_PATTERN = re.compile(
    r"(?P<name>[^{\[]+)\s*(?:\[(?P<type>\w+)\])?\s*(?:\{(?P<props>[^}]*)\})?"
)

# Whitespace and normalization
WHITESPACE_PATTERN = re.compile(r"\s+")
NEWLINE_PATTERN = re.compile(r"[\r\n]+")

# ─────────────────────────── Helper Functions ───────────────────────────


def generate_source_id(prefix: str = "kp") -> str:
    """Generate a unique source ID with optional prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_text(text: str) -> str:
    """Normalize text: collapse whitespace, strip, lowercase first char only for consistency."""
    text = NEWLINE_PATTERN.sub(" ", text.strip())
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text


def clean_triple_content(content: str) -> str:
    """Clean triple format content by removing extra whitespace and normalizing separators."""
    content = normalize_text(content)
    # Normalize various separators to a consistent format
    content = re.sub(r"\s*[,\|]\s*", " | ", content)
    # Remove surrounding parentheses if present
    content = re.sub(r"^\(|\)$", "", content)
    return content.strip()


def extract_triples(content: str) -> list[dict[str, str]]:
    """Extract all triples from content using regex patterns.

    Supports both parenthesized and pipe-delimited formats.
    Returns a list of dicts with keys: subject, predicate, object.
    """
    triples: list[dict[str, str]] = []
    seen: set[str] = set()

    # Try parenthesized format: (Subject, Predicate, Object)
    for match in TRIPLE_PATTERN.finditer(content):
        triple = {
            "subject": match.group("s").strip(),
            "predicate": match.group("p").strip(),
            "object": match.group("o").strip(),
        }
        key = f"{triple['subject']}|{triple['predicate']}|{triple['object']}"
        if key not in seen:
            seen.add(key)
            triples.append(triple)

    # Try pipe-delimited format: Subject | Predicate | Object
    # Only if no parenthesized triples were found, or if there are additional
    # pipe-delimited triples not in parentheses
    for match in TRIPLE_PIPE_PATTERN.finditer(content):
        triple = {
            "subject": match.group("s").strip(),
            "predicate": match.group("p").strip(),
            "object": match.group("o").strip(),
        }
        key = f"{triple['subject']}|{triple['predicate']}|{triple['object']}"
        if key not in seen:
            seen.add(key)
            triples.append(triple)

    # Fallback: if content contains commas and no pipes/parentheses matched,
    # try splitting by comma into exactly 3 parts
    if not triples and content.count(",") >= 2 and "|" not in content:
        parts = [p.strip() for p in content.split(",") if p.strip()]
        if len(parts) >= 3:
            # Take parts in groups of 3
            for i in range(0, len(parts) - 2, 3):
                triple = {
                    "subject": parts[i],
                    "predicate": parts[i + 1],
                    "object": parts[i + 2],
                }
                key = f"{triple['subject']}|{triple['predicate']}|{triple['object']}"
                if key not in seen:
                    seen.add(key)
                    triples.append(triple)

    return triples


def extract_schema_entries(content: str) -> list[dict[str, str]]:
    """Extract schema entries from content.

    Returns a list of dicts with keys: entity_type, property, value.
    """
    entries: list[dict[str, str]] = []
    # Split by newlines or semicolons to handle multi-line input
    lines = re.split(r"[\n;]+", content)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = SCHEMA_PATTERN.match(line)
        if match:
            entries.append(
                {
                    "entity_type": match.group("entity_type").strip(),
                    "property": match.group("property").strip(),
                    "value": match.group("value").strip(),
                }
            )
    return entries


def extract_entities(content: str) -> list[dict[str, Any]]:
    """Extract entity definitions from content.

    Returns a list of dicts with keys: name, type, properties.
    """
    entities: list[dict[str, Any]] = []
    # Split by newlines to handle multi-line input
    lines = re.split(r"[\n;]+", content)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = ENTITY_PATTERN.match(line)
        if match:
            entity: dict[str, Any] = {
                "name": match.group("name").strip(),
            }
            if match.group("type"):
                entity["type"] = match.group("type").strip()
            if match.group("props"):
                props = {}
                for prop in match.group("props").split(","):
                    if ":" in prop:
                        k, v = prop.split(":", 1)
                        props[k.strip()] = v.strip()
                if props:
                    entity["properties"] = props
            entities.append(entity)
    return entities


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap sentences
            overlap_sentences: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_length
        current_chunk.append(sentence)
        current_length += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def estimate_token_count(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count using character heuristic."""
    return len(text) // chars_per_token


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Sanitize metadata values for JSON serialization."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [
                v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for v in value
            ]
        elif isinstance(value, dict):
            clean[key] = sanitize_metadata(value)
        else:
            clean[key] = str(value)
    return clean


def is_empty_content(content: str) -> bool:
    """Check if content is effectively empty."""
    return (
        not content
        or not content.strip()
        or content.strip().lower() in {"null", "none", "nan", "-"}
    )
