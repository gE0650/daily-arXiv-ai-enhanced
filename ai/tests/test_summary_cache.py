import json
import os
import tempfile
import unittest

from summary_cache import (
    build_ai_meta,
    is_reusable,
    load_cache,
    prompt_hash,
)

SYSTEM = "You are a paper analyst."
TEMPLATE = "{language} {content}"


def make_item(identifier: str = "2609.00001", summary: str = "An abstract.") -> dict:
    return {"id": identifier, "summary": summary}


def make_entry(identifier: str = "2609.00001", summary: str = "An abstract.", **overrides) -> dict:
    entry = {
        "id": identifier,
        "summary": summary,
        "AI": {
            "tldr": "一句话摘要",
            "motivation": "动机",
            "method": "方法",
            "result": "结果",
            "conclusion": "结论",
        },
        "ai_meta": build_ai_meta("deepseek-flash", "Chinese", SYSTEM, TEMPLATE),
    }
    entry.update(overrides)
    return entry


class PromptHashTests(unittest.TestCase):
    def test_prompt_change_changes_the_hash(self):
        self.assertNotEqual(
            prompt_hash(SYSTEM, TEMPLATE),
            prompt_hash(SYSTEM + " Be brief.", TEMPLATE),
        )

    def test_hash_is_stable(self):
        self.assertEqual(prompt_hash(SYSTEM, TEMPLATE), prompt_hash(SYSTEM, TEMPLATE))


class IsReusableTests(unittest.TestCase):
    def setUp(self):
        self.meta = build_ai_meta("deepseek-flash", "Chinese", SYSTEM, TEMPLATE)
        self.item = make_item()

    def test_matching_entry_is_reusable(self):
        self.assertTrue(is_reusable(make_entry(), self.item, self.meta))

    def test_entry_without_ai_meta_is_not_reusable(self):
        """Catches reusing summaries from files written before this feature existed."""
        entry = make_entry()
        entry.pop("ai_meta")

        self.assertFalse(is_reusable(entry, self.item, self.meta))

    def test_placeholder_summary_is_not_reusable(self):
        for field in ("tldr", "motivation", "method", "result", "conclusion"):
            with self.subTest(field=field):
                entry = make_entry()
                entry["AI"][field] = "Summary generation failed"
                self.assertFalse(is_reusable(entry, self.item, self.meta))

    def test_compliance_refusal_is_not_reusable(self):
        entry = make_entry()
        entry["AI"]["tldr"] = "This content has not passed the compliance test and has been hidden."

        self.assertFalse(is_reusable(entry, self.item, self.meta))

    def test_changed_abstract_is_not_reusable(self):
        entry = make_entry(summary="The authors revised the abstract.")

        self.assertFalse(is_reusable(entry, self.item, self.meta))

    def test_changed_model_language_or_prompt_is_not_reusable(self):
        for meta in (
            build_ai_meta("deepseek-v4-pro", "Chinese", SYSTEM, TEMPLATE),
            build_ai_meta("deepseek-flash", "English", SYSTEM, TEMPLATE),
            build_ai_meta("deepseek-flash", "Chinese", SYSTEM + " Be brief.", TEMPLATE),
        ):
            with self.subTest(meta=meta):
                self.assertFalse(is_reusable(make_entry(), self.item, meta))

    def test_entry_without_ai_is_not_reusable(self):
        entry = make_entry()
        entry.pop("AI")

        self.assertFalse(is_reusable(entry, self.item, self.meta))


class LoadCacheTests(unittest.TestCase):
    def test_missing_path_returns_empty(self):
        self.assertEqual({}, load_cache(None))
        self.assertEqual({}, load_cache("/tmp/definitely-not-here.jsonl"))

    def test_last_entry_wins_and_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(make_entry()) + "\n")
                handle.write("\n")
                handle.write("{broken json}\n")
                handle.write(json.dumps(make_entry(summary="updated")) + "\n")

            cache = load_cache(path)

        self.assertEqual(["2609.00001"], list(cache))
        self.assertEqual("updated", cache["2609.00001"]["summary"])


if __name__ == "__main__":
    unittest.main()
