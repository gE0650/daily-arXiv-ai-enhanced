import os
import unittest
from unittest.mock import patch

from scrapy.http import HtmlResponse, Request

os.environ.setdefault("CATEGORIES", "cs.CV")

from daily_arxiv.spiders.arxiv import ArxivSpider  # noqa: E402

LISTING_TEMPLATE = """<html><body>
<div id="dlpage"><ul><li><a href="#item3">3</a></li></ul></div>
<dl>
  <dt><a name="item1"></a><a title="Abstract" href="/abs/{first}">Abstract</a></dt>
  <dd><div class="list-subjects"><span class="primary-subject">Computer Vision (cs.CV)</span></div></dd>
  <dt><a name="item2"></a><a title="Abstract" href="/abs/{second}">Abstract</a></dt>
  <dd><div class="list-subjects"><span class="primary-subject">Robotics ({second_subject})</span></div></dd>
</dl>
</body></html>"""


def make_response(
    first: str = "2609.00001",
    second: str = "2609.00002",
    second_subject: str = "cs.CV",
) -> HtmlResponse:
    url = "https://arxiv.org/list/cs.CV/new"
    body = LISTING_TEMPLATE.format(
        first=first, second=second, second_subject=second_subject
    ).encode()
    return HtmlResponse(url=url, body=body, encoding="utf-8", request=Request(url=url))


def make_metadata(identifiers) -> dict:
    return {
        identifier: {
            "title": f"Title {identifier}",
            "authors": ["Author One"],
            "summary": f"Abstract {identifier}",
            "categories": ["cs.CV"],
            "comment": None,
        }
        for identifier in identifiers
    }


class SpiderSetupTests(unittest.TestCase):
    def test_duplicate_categories_collapse_to_one_listing_page(self):
        """Duplicate pages would be dropped by the dupefilter and emit nothing."""
        with patch.dict(os.environ, {"CATEGORIES": "cs.CV, cs.CV , cs.CL"}):
            spider = ArxivSpider()

        self.assertEqual(
            ["https://arxiv.org/list/cs.CV/new", "https://arxiv.org/list/cs.CL/new"],
            spider.start_urls,
        )
        self.assertEqual(2, spider.pending_pages)


class SpiderEmitTests(unittest.TestCase):
    def _spider(self, categories: str = "cs.CV") -> ArxivSpider:
        with patch.dict(os.environ, {"CATEGORIES": categories}):
            return ArxivSpider()

    def test_last_listing_page_emits_papers_with_metadata(self):
        spider = self._spider("cs.CV")
        metadata = make_metadata(["2609.00001", "2609.00002"])

        with patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata", return_value=(metadata, [])
        ) as fetch, patch(
            "daily_arxiv.spiders.arxiv.load_metadata_cache", return_value={}
        ):
            items = list(spider.parse(make_response()))

        self.assertEqual(["2609.00001", "2609.00002"], [item["id"] for item in items])
        self.assertEqual(
            "https://arxiv.org/abs/2609.00001", items[0]["abs"]
        )
        self.assertEqual("Title 2609.00001", items[0]["title"])
        self.assertEqual(0, spider.pending_pages)
        fetch.assert_called_once()

    def test_earlier_pages_do_not_emit_before_the_last_one(self):
        spider = self._spider("cs.CV,cs.CL")
        metadata = make_metadata(["2609.00001", "2609.00002"])

        with patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata", return_value=(metadata, [])
        ) as fetch, patch(
            "daily_arxiv.spiders.arxiv.load_metadata_cache", return_value={}
        ):
            first = list(spider.parse(make_response()))
            second = list(spider.parse(make_response()))

        self.assertEqual([], first)
        self.assertEqual(2, len(second))
        self.assertEqual(1, fetch.call_count)

    def test_papers_outside_the_target_categories_are_skipped(self):
        spider = self._spider("cs.CV")
        metadata = make_metadata(["2609.00001", "2609.00002"])

        with patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata", return_value=(metadata, [])
        ), patch("daily_arxiv.spiders.arxiv.load_metadata_cache", return_value={}):
            items = list(spider.parse(make_response(second_subject="cs.RO")))

        self.assertEqual(["2609.00001"], [item["id"] for item in items])

    def test_unresolved_ids_are_skipped_without_killing_the_batch(self):
        spider = self._spider("cs.CV")
        metadata = make_metadata(["2609.00001"])

        with patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata",
            return_value=(metadata, ["2609.00002"]),
        ), patch("daily_arxiv.spiders.arxiv.load_metadata_cache", return_value={}):
            items = list(spider.parse(make_response()))

        self.assertEqual(["2609.00001"], [item["id"] for item in items])

    def test_failed_listing_page_still_emits_the_collected_papers(self):
        spider = self._spider("cs.CV")
        metadata = make_metadata(["2609.00001", "2609.00002"])

        class Failure:
            value = RuntimeError("listing page timed out")

        with patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata", return_value=(metadata, [])
        ), patch("daily_arxiv.spiders.arxiv.load_metadata_cache", return_value={}):
            items = list(spider.errback(Failure()))

        self.assertEqual([], items)
        self.assertEqual(0, spider.pending_pages)

    def test_metadata_cache_path_is_passed_through(self):
        spider = self._spider("cs.CV")
        sentinel = {"/tmp/cache.jsonl": {"2609.00001": {"title": "cached"}}}

        with patch(
            "daily_arxiv.spiders.arxiv.load_metadata_cache",
            side_effect=lambda path: sentinel.get(path, {}),
        ) as load, patch(
            "daily_arxiv.spiders.arxiv.fetch_metadata", return_value=({}, [])
        ), patch.dict(os.environ, {"ARXIV_META_CACHE": "/tmp/cache.jsonl"}):
            list(spider.parse(make_response()))

        load.assert_called_once_with("/tmp/cache.jsonl")


if __name__ == "__main__":
    unittest.main()
