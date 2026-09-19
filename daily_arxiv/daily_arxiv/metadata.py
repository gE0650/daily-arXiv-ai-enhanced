"""arXiv metadata lookups with batched requests and a same-day cache.

The listing pages only expose paper ids, so every paper needs a metadata
lookup. One request per paper is serialized by the arXiv client rate limit
(3 seconds per request), which turns a 500 paper crawl into a 25 minute wait.
The API accepts a list of ids per request, so batching 100 ids at a time cuts
that to a handful of requests. Re-using the previous run's ``data/<date>.jsonl``
skips the requests that are already known.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import arxiv

CHUNK_SIZE = 100
REQUIRED_FIELDS = ("title", "authors", "summary", "categories")
_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_id(raw_id: str) -> str:
    """Reduce an arXiv id or URL to its versionless form.

    ``https://arxiv.org/abs/2609.19680v2`` and ``2609.19680v2`` both become
    ``2609.19680``, which is what the listing pages hand us.
    """
    identifier = (raw_id or "").strip().rstrip("/").split("/")[-1]
    return _VERSION_SUFFIX.sub("", identifier)


def is_usable_metadata(entry: dict) -> bool:
    """Return True when a cached entry carries everything the site needs."""
    if not isinstance(entry, dict):
        return False
    if not all(entry.get(field) for field in REQUIRED_FIELDS):
        return False
    return isinstance(entry["authors"], (list, tuple)) and len(entry["authors"]) > 0


def load_metadata_cache(path: str | None) -> Dict[str, dict]:
    """Load ``id -> metadata`` from a previous run of the same day."""
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
                identifier = normalize_id(str(entry.get("id", "")))
                if identifier and is_usable_metadata(entry):
                    cache[identifier] = entry
    except OSError as error:
        print(f"Could not read metadata cache {path}: {error}", file=sys.stderr)
        return {}

    return cache


def chunked(items: Sequence[str], size: int = CHUNK_SIZE) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def metadata_from_result(result) -> dict:
    """Convert an arXiv API result into the fields stored in ``data/*.jsonl``."""
    return {
        "authors": [author.name for author in result.authors],
        "title": result.title,
        "categories": list(result.categories),
        "comment": result.comment,
        "summary": result.summary,
    }


def _fetch_single(client, identifier: str) -> dict | None:
    """Fallback for ids the batch response left out (withdrawn, renamed)."""
    try:
        for result in client.results(arxiv.Search(id_list=[identifier])):
            parsed = metadata_from_result(result)
            return parsed if is_usable_metadata(parsed) else None
    except Exception as error:  # noqa: BLE001 - one bad id must not kill the crawl
        print(f"Metadata request failed for {identifier}: {error}", file=sys.stderr)
    return None


def fetch_metadata(
    ids: Iterable[str],
    cache: Dict[str, dict] | None = None,
    client=None,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[Dict[str, dict], List[str]]:
    """Resolve metadata for ``ids``, reusing ``cache`` and batching the rest.

    Returns ``(metadata_by_id, missing_ids)``; ids that arXiv cannot resolve are
    reported in ``missing_ids`` so the caller can log them.
    """
    cache = cache or {}
    order: List[str] = []
    seen = set()
    for raw_id in ids:
        identifier = normalize_id(str(raw_id))
        if identifier and identifier not in seen:
            seen.add(identifier)
            order.append(identifier)

    metadata: Dict[str, dict] = {}
    pending: List[str] = []
    for identifier in order:
        cached = cache.get(identifier)
        if is_usable_metadata(cached):
            metadata[identifier] = cached
        else:
            pending.append(identifier)

    if not pending:
        return metadata, []

    client = client or arxiv.Client(chunk_size)
    missing: List[str] = []
    for chunk in chunked(pending, chunk_size):
        try:
            for result in client.results(arxiv.Search(id_list=chunk)):
                identifier = normalize_id(result.get_short_id())
                if identifier not in seen or identifier in metadata:
                    continue
                parsed = metadata_from_result(result)
                if is_usable_metadata(parsed):
                    metadata[identifier] = parsed
        except Exception as error:  # noqa: BLE001 - fall back to per-id requests
            print(
                f"Batch metadata request failed for {len(chunk)} id(s): {error}",
                file=sys.stderr,
            )

        for identifier in chunk:
            if identifier in metadata:
                continue
            parsed = _fetch_single(client, identifier)
            if parsed:
                metadata[identifier] = parsed
            else:
                missing.append(identifier)

    return metadata, missing
