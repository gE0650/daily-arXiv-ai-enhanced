"""Reuse rules for AI summaries produced earlier the same day.

Re-running a failed day used to recompute every summary, wasting tokens and
time. Entries written by this module carry an ``ai_meta`` stamp, so a cached
summary is only reused when it was built from the same input and the same
configuration: same abstract, model, language and prompts.
"""

import hashlib
import json
import os
import sys
from typing import Dict, Optional

CACHE_VERSION = 1

AI_FIELDS = ("tldr", "motivation", "method", "result", "conclusion")

# Fallback text written when summarization could not run. It must never be
# reused as a real summary.
PLACEHOLDER_VALUES = frozenset({
    "Summary generation failed",
    "Motivation analysis unavailable",
    "Method extraction failed",
    "Result analysis unavailable",
    "Conclusion extraction failed",
    "Processing failed",
})
PLACEHOLDER_MARKERS = ("has not passed the compliance test", "未通过合规检测")


def _is_placeholder(value: str) -> bool:
    if value in PLACEHOLDER_VALUES:
        return True
    lowered = value.lower()
    return any(marker.lower() in lowered for marker in PLACEHOLDER_MARKERS)


def prompt_hash(system: str, template: str) -> str:
    """Fingerprint the prompts so a prompt edit invalidates cached summaries."""
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\0")
    digest.update(template.encode("utf-8"))
    return digest.hexdigest()[:12]


def build_ai_meta(model: str, language: str, system: str, template: str) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "model": model,
        "language": language,
        "prompt_hash": prompt_hash(system, template),
    }


def is_reusable(entry: Optional[dict], item: dict, ai_meta: dict) -> bool:
    """Return True when a cached summary still describes this exact paper."""
    if not isinstance(entry, dict):
        return False

    summary = item.get("summary")
    if not summary or entry.get("summary") != summary:
        return False

    ai = entry.get("AI")
    if not isinstance(ai, dict):
        return False
    for field in AI_FIELDS:
        value = ai.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
        if _is_placeholder(value.strip()):
            return False

    meta = entry.get("ai_meta")
    if not isinstance(meta, dict):
        return False
    return all(
        meta.get(key) == ai_meta.get(key)
        for key in ("cache_version", "model", "language", "prompt_hash")
    )


def load_cache(path: Optional[str]) -> Dict[str, dict]:
    """Load ``id -> entry`` from a previous run, keeping the last entry per id."""
    if not path or not os.path.exists(path):
        return {}

    cache: Dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                identifier = str(entry.get("id", "")).strip()
                if identifier:
                    cache[identifier] = entry
    except OSError as error:
        print(f"Could not read summary cache {path}: {error}", file=sys.stderr)
        return {}

    return cache
