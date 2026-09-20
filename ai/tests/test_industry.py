import unittest
from datetime import datetime, timezone

from collect_industry import (
    INDUSTRY_FEEDS,
    INDUSTRY_PAPERS,
    clean_text,
    collect_blogs,
    parse_date,
    parse_feed,
)

RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Fresh post</title>
    <link>https://example.com/fresh</link>
    <pubDate>Fri, 18 Sep 2026 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description>
  </item>
  <item>
    <title>Stale post</title>
    <link>https://example.com/stale</link>
    <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    <description>Old news</description>
  </item>
</channel></rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Atom post</title>
    <link rel="alternate" href="https://example.com/atom"/>
    <published>2026-09-15T17:05:57Z</published>
    <summary>Atom summary text</summary>
  </entry>
</feed>"""

TODAY = datetime(2026, 9, 20, tzinfo=timezone.utc)


class TextHelpersTests(unittest.TestCase):
    def test_clean_text_strips_markup_and_collapses_space(self):
        self.assertEqual("Hello world", clean_text("<p>Hello   <b>world</b></p>"))
        self.assertEqual("", clean_text(None))

    def test_parse_date_handles_both_formats(self):
        self.assertEqual("2026-09-18", parse_date("Fri, 18 Sep 2026 12:00:00 GMT"))
        self.assertEqual("2026-09-15", parse_date("2026-09-15T17:05:57Z"))
        self.assertEqual("", parse_date("not a date"))
        self.assertEqual("", parse_date(None))


class ParseFeedTests(unittest.TestCase):
    def test_parses_rss_items(self):
        entries = parse_feed(RSS_SAMPLE)

        self.assertEqual(2, len(entries))
        self.assertEqual("Fresh post", entries[0]["title"])
        self.assertEqual("https://example.com/fresh", entries[0]["url"])
        self.assertEqual("2026-09-18", entries[0]["date"])
        self.assertEqual("Hello world", entries[0]["summary"])

    def test_parses_atom_entries(self):
        entries = parse_feed(ATOM_SAMPLE)

        self.assertEqual(1, len(entries))
        self.assertEqual("Atom post", entries[0]["title"])
        self.assertEqual("https://example.com/atom", entries[0]["url"])
        self.assertEqual("2026-09-15", entries[0]["date"])
        self.assertEqual("Atom summary text", entries[0]["summary"])

    def test_items_without_title_or_link_are_skipped(self):
        broken = b"""<rss><channel><item><title>No link</title></item></channel></rss>"""

        self.assertEqual([], parse_feed(broken))


class CollectBlogsTests(unittest.TestCase):
    def test_keeps_recent_posts_and_drops_stale_ones(self):
        feeds = [{"source": "Example", "url": "https://example.com/feed"}]

        posts = collect_blogs(feeds, days=30, fetch=lambda url: RSS_SAMPLE, today=TODAY)

        self.assertEqual(["Fresh post"], [post["title"] for post in posts])
        self.assertEqual("blog", posts[0]["kind"])
        self.assertEqual("Example", posts[0]["source"])
        self.assertEqual("https://example.com/fresh", posts[0]["abs"])
        self.assertEqual([], posts[0]["authors"])
        self.assertTrue(posts[0]["id"].startswith("blog-example-"))

    def test_caps_posts_per_source(self):
        feeds = [{"source": "Example", "url": "https://example.com/feed"}]
        multi = b"""<rss><channel>
          <item><title>A</title><link>https://e.com/a</link><pubDate>Fri, 18 Sep 2026 12:00:00 GMT</pubDate></item>
          <item><title>B</title><link>https://e.com/b</link><pubDate>Thu, 17 Sep 2026 12:00:00 GMT</pubDate></item>
          <item><title>C</title><link>https://e.com/c</link><pubDate>Wed, 16 Sep 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>"""

        posts = collect_blogs(feeds, days=30, max_per_source=2, fetch=lambda url: multi, today=TODAY)

        self.assertEqual(["A", "B"], [post["title"] for post in posts])

    def test_broken_feed_does_not_stop_the_others(self):
        def fetch(url):
            if "broken" in url:
                raise RuntimeError("connection reset")
            return RSS_SAMPLE

        feeds = [
            {"source": "Broken", "url": "https://example.com/broken"},
            {"source": "Working", "url": "https://example.com/feed"},
        ]

        posts = collect_blogs(feeds, days=30, fetch=fetch, today=TODAY)

        self.assertEqual(["Fresh post"], [post["title"] for post in posts])
        self.assertEqual("Working", posts[0]["source"])

    def test_posts_are_sorted_newest_first(self):
        feeds = [
            {"source": "One", "url": "https://one.example/feed"},
            {"source": "Two", "url": "https://two.example/feed"},
        ]
        older = b"""<rss><channel><item><title>Older</title><link>https://e.com/o</link>
          <pubDate>Mon, 14 Sep 2026 12:00:00 GMT</pubDate></item></channel></rss>"""
        newer = b"""<rss><channel><item><title>Newer</title><link>https://e.com/n</link>
          <pubDate>Sat, 19 Sep 2026 12:00:00 GMT</pubDate></item></channel></rss>"""
        payloads = {"https://one.example/feed": older, "https://two.example/feed": newer}

        posts = collect_blogs(feeds, days=30, fetch=lambda url: payloads[url], today=TODAY)

        self.assertEqual(["Newer", "Older"], [post["title"] for post in posts])


class CuratedListTests(unittest.TestCase):
    def test_arxiv_ids_are_well_formed_and_unique(self):
        ids = [arxiv_id for arxiv_id, _, _ in INDUSTRY_PAPERS]

        self.assertEqual(len(ids), len(set(ids)), "duplicate arXiv id in INDUSTRY_PAPERS")
        for arxiv_id in ids:
            self.assertRegex(arxiv_id, r"^\d{4}\.\d{4,5}$")

    def test_every_entry_has_a_source_label(self):
        for arxiv_id, source, label in INDUSTRY_PAPERS:
            self.assertTrue(source, f"missing source for {arxiv_id}")
            self.assertTrue(label, f"missing label for {arxiv_id}")

    def test_feeds_are_unique_and_https(self):
        urls = [feed["url"] for feed in INDUSTRY_FEEDS]

        self.assertEqual(len(urls), len(set(urls)), "duplicate feed url")
        for feed in INDUSTRY_FEEDS:
            self.assertTrue(feed["source"])
            self.assertTrue(feed["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
