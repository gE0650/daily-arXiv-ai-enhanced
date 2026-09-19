import os
import re

import scrapy

from ..metadata import fetch_metadata, load_metadata_cache


class ArxivSpider(scrapy.Spider):
    name = "arxiv"  # 爬虫名称
    allowed_domains = ["arxiv.org"]  # 允许爬取的域名

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = os.environ.get("CATEGORIES", "cs.CV").split(",")
        # 去重是必须的：重复的分类页会被 Scrapy 的 dupefilter 丢掉，
        # 那样 pending_pages 永远归不了零，整次抓取会一条数据都产不出来。
        self.target_categories = list(
            dict.fromkeys(cat.strip() for cat in categories if cat.strip())
        )
        self.start_urls = [
            f"https://arxiv.org/list/{cat}/new" for cat in self.target_categories
        ]  # 起始URL（计算机科学领域的最新论文）
        if not self.start_urls:
            self.logger.warning(
                "CATEGORIES is empty; nothing will be crawled. "
                "Set the CATEGORIES repository variable, e.g. 'cs.CV,cs.CL'."
            )
        self.pending_pages = len(self.start_urls)
        # 保留列表页出现顺序，同时去重
        self.collected_ids: "dict[str, bool]" = {}
        self.emitted = False

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, errback=self.errback)

    def errback(self, failure):
        """A failed listing page must still let the batch run."""
        self.logger.error(f"Listing page failed: {failure.value}")
        yield from self.finish_page()

    def parse(self, response):
        # 提取每篇论文的信息
        anchors = []
        for li in response.css("div[id=dlpage] ul li"):
            href = li.css("a::attr(href)").get()
            if href and "item" in href:
                anchors.append(int(href.split("item")[-1]))

        # 遍历每篇论文的详细信息
        for paper in response.css("dl dt"):
            paper_anchor = paper.css("a[name^='item']::attr(name)").get()
            if not paper_anchor:
                continue
                
            paper_id = int(paper_anchor.split("item")[-1])
            if anchors and paper_id >= anchors[-1]:
                continue

            # 获取论文ID
            abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
            if not abstract_link:
                continue
                
            arxiv_id = abstract_link.split("/")[-1]
            
            # 获取对应的论文描述部分 (dd元素)
            paper_dd = paper.xpath("following-sibling::dd[1]")
            if not paper_dd:
                continue
            
            # 提取论文分类信息 - 在subjects部分
            subjects_text = paper_dd.css(".list-subjects .primary-subject::text").get()
            if not subjects_text:
                # 如果找不到主分类，尝试其他方式获取分类
                subjects_text = paper_dd.css(".list-subjects::text").get()
            
            if subjects_text:
                # 解析分类信息，通常格式如 "Computer Vision and Pattern Recognition (cs.CV)"
                # 提取括号中的分类代码
                categories_in_paper = re.findall(r'\(([^)]+)\)', subjects_text)
                
                # 检查论文分类是否与目标分类有交集
                paper_categories = set(categories_in_paper)
                if paper_categories.intersection(self.target_categories):
                    self.collected_ids.setdefault(arxiv_id, True)
                    self.logger.info(f"Found paper {arxiv_id} with categories {paper_categories}")
                else:
                    self.logger.debug(f"Skipped paper {arxiv_id} with categories {paper_categories} (not in target {self.target_categories})")
            else:
                # 如果无法获取分类信息，记录警告但仍然返回论文（保持向后兼容）
                self.logger.warning(f"Could not extract categories for paper {arxiv_id}, including anyway")
                self.collected_ids.setdefault(arxiv_id, True)

        yield from self.finish_page()

    def finish_page(self):
        """Emit every collected paper once the last listing page is done.

        Metadata is fetched in batches here instead of one request per paper:
        the arXiv client sleeps 3s between requests, so a per-paper lookup for
        ~500 papers costs ~25 minutes.
        """
        self.pending_pages -= 1
        if self.pending_pages > 0 or self.emitted:
            return
        self.emitted = True

        identifiers = list(self.collected_ids)
        if not identifiers:
            self.logger.warning("No papers collected from the listing pages")
            return

        cache = load_metadata_cache(os.environ.get("ARXIV_META_CACHE"))
        metadata, missing = fetch_metadata(identifiers, cache=cache)
        cached_hits = sum(1 for identifier in identifiers if identifier in cache)

        self.logger.info(
            f"Collected {len(identifiers)} papers: {len(metadata)} with metadata "
            f"({cached_hits} from cache, {len(missing)} unresolved)"
        )
        if missing:
            self.logger.warning(f"arXiv returned no metadata for: {', '.join(missing[:10])}")

        for identifier in identifiers:
            entry = metadata.get(identifier)
            if not entry:
                continue
            yield {
                "id": identifier,
                "pdf": f"https://arxiv.org/pdf/{identifier}",
                "abs": f"https://arxiv.org/abs/{identifier}",
                **entry,
            }
