import json
import os
import tempfile
import unittest

from daily_arxiv.metadata import (
    chunked,
    fetch_metadata,
    is_usable_metadata,
    load_metadata_cache,
    normalize_id,
)


def make_entry(identifier: str, **overrides) -> dict:
    entry = {
        "id": identifier,
        "title": f"Title {identifier}",
        "authors": ["Author One"],
        "summary": f"Abstract {identifier}",
        "categories": ["cs.CV"],
        "comment": None,
    }
    entry.update(overrides)
    return entry


class FakeAuthor:
    def __init__(self, name: str):
        self.name = name


class FakeResult:
    def __init__(self, short_id: str):
        self._short_id = short_id
        self.authors = [FakeAuthor("Author One")]
        self.title = f"Title {short_id}"
        self.summary = f"Abstract {short_id}"
        self.comment = None
        self.categories = ["cs.CV"]

    def get_short_id(self) -> str:
        return self._short_id


class FakeClient:
    """Stands in for arxiv.Client and records every requested id list."""

    def __init__(self, known=(), unresolvable=()):
        self.known = set(known)
        self.unresolvable = set(unresolvable)
        self.requests = []

    def results(self, search):
        requested = list(search.id_list)
        self.requests.append(requested)
        for identifier in requested:
            if identifier in self.known and identifier not in self.unresolvable:
                yield FakeResult(f"{identifier}v1")


class NormalizeIdTests(unittest.TestCase):
    def test_strips_url_prefix_and_version_suffix(self):
        self.assertEqual("2609.19680", normalize_id("https://arxiv.org/abs/2609.19680v2"))
        self.assertEqual("2609.19680", normalize_id("2609.19680v12"))
        self.assertEqual("2609.19680", normalize_id(" 2609.19680 "))
        self.assertEqual("0701001", normalize_id("cs/0701001"))


class UsableMetadataTests(unittest.TestCase):
    def test_complete_entry_is_usable(self):
        self.assertTrue(is_usable_metadata(make_entry("2609.00001")))

    def test_missing_required_field_is_not_usable(self):
        for field in ("title", "authors", "summary", "categories"):
            with self.subTest(field=field):
                entry = make_entry("2609.00001", **{field: None})
                self.assertFalse(is_usable_metadata(entry))

    def test_non_dict_is_not_usable(self):
        self.assertFalse(is_usable_metadata(None))
        self.assertFalse(is_usable_metadata("nope"))


class ChunkedTests(unittest.TestCase):
    def test_splits_into_fixed_size_batches(self):
        self.assertEqual([3, 3, 1], [len(chunk) for chunk in chunked([str(i) for i in range(7)], 3)])

    def test_empty_input_yields_nothing(self):
        self.assertEqual([], list(chunked([], 100)))


class LoadMetadataCacheTests(unittest.TestCase):
    def test_missing_file_returns_empty_cache(self):
        self.assertEqual({}, load_metadata_cache("/tmp/definitely-not-here.jsonl"))
        self.assertEqual({}, load_metadata_cache(None))

    def test_skips_unusable_entries_and_broken_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(make_entry("2609.00001")) + "\n")
                handle.write("{not json}\n")
                handle.write(json.dumps({"id": "2609.00002", "title": "T"}) + "\n")
                handle.write("\n")

            cache = load_metadata_cache(path)

        self.assertEqual(["2609.00001"], list(cache))


class FetchMetadataTests(unittest.TestCase):
    def test_cached_ids_are_never_requested(self):
        identifiers = ["2609.00001", "2609.00002", "2609.00003"]
        cache = {identifier: make_entry(identifier) for identifier in identifiers}
        client = FakeClient(known=identifiers)

        metadata, missing = fetch_metadata(identifiers, cache=cache, client=client)

        self.assertEqual([], client.requests)
        self.assertEqual(set(identifiers), set(metadata))
        self.assertEqual([], missing)

    def test_misses_are_batched(self):
        identifiers = [f"2609.{i:05d}" for i in range(250)]
        client = FakeClient(known=identifiers)

        metadata, missing = fetch_metadata(identifiers, cache={}, client=client)

        self.assertEqual([100, 100, 50], [len(request) for request in client.requests])
        self.assertEqual(len(identifiers), len(metadata))
        self.assertEqual([], missing)

    def test_ids_missing_from_a_batch_fall_back_to_single_requests(self):
        identifiers = [f"2609.{i:05d}" for i in range(150)]
        client = FakeClient(known=identifiers, unresolvable={"2609.00120"})

        metadata, missing = fetch_metadata(identifiers, cache={}, client=client)

        self.assertEqual([100, 50], [len(r) for r in client.requests[:2]])
        self.assertEqual(["2609.00120"], client.requests[2])
        self.assertEqual(["2609.00120"], missing)
        self.assertNotIn("2609.00120", metadata)

    def test_version_suffix_in_api_response_maps_back_to_requested_id(self):
        client = FakeClient(known={"2609.00001"})

        metadata, _ = fetch_metadata(["2609.00001"], cache={}, client=client)

        self.assertIn("2609.00001", metadata)
        self.assertEqual(["cs.CV"], metadata["2609.00001"]["categories"])


if __name__ == "__main__":
    unittest.main()
