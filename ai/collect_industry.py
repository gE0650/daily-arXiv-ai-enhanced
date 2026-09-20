#!/usr/bin/env python3
"""
Collect industry papers and company blog posts (工业界板块)

功能:
1. 抓取一批已校验的工业界/初创公司经典论文的 arXiv 元数据
2. 抓取各大公司实验室博客的 RSS/Atom 最新文章
3. 使用 LLM (与 enhance.py 相同) 生成 TLDR / motivation / method / result / conclusion
4. 输出 data/industry.jsonl, 供前端"工业界 · 论文与动态"区展示

说明:
- 每次运行都会重新生成（博客是滚动更新的），单篇 LLM 失败时不会写入占位符，只保留原文
- 未设置 OPENAI_API_KEY 时跳过摘要，仍然输出元数据与博客描述
- 单个博客源抓取失败不影响其他源
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# 工业界与初创公司论文清单 (arXiv ID, 公司/机构, 备注), 全部经 arXiv API 校验
INDUSTRY_PAPERS = [
    # OpenAI
    ("2005.14165", "OpenAI", "GPT-3"),
    ("2103.00020", "OpenAI", "CLIP"),
    ("2203.02155", "OpenAI", "InstructGPT"),
    ("2212.04356", "OpenAI", "Whisper"),
    ("2206.11795", "OpenAI", "Video PreTraining"),
    # Google / DeepMind
    ("1706.03762", "Google", "Transformer"),
    ("1810.04805", "Google", "BERT"),
    ("2010.11929", "Google", "ViT"),
    ("2204.02311", "Google", "PaLM"),
    ("2203.15556", "DeepMind", "Chinchilla"),
    ("2204.14198", "DeepMind", "Flamingo"),
    ("2205.06175", "DeepMind", "Gato"),
    ("2203.07814", "DeepMind", "AlphaCode"),
    ("2204.01691", "Google", "SayCan"),
    ("2303.03378", "Google", "PaLM-E"),
    ("2212.06817", "Google", "RT-1"),
    ("2307.15818", "Google DeepMind", "RT-2"),
    # Meta
    ("2302.13971", "Meta", "LLaMA"),
    ("2307.09288", "Meta", "Llama 2"),
    ("2407.21783", "Meta", "Llama 3"),
    ("2104.14294", "Meta", "DINO"),
    ("2304.07193", "Meta", "DINOv2"),
    ("2304.02643", "Meta", "Segment Anything"),
    ("2305.05665", "Meta", "ImageBind"),
    ("2408.00714", "Meta", "SAM 2"),
    # Microsoft
    ("2106.09685", "Microsoft", "LoRA"),
    ("2306.11644", "Microsoft", "Phi-1"),
    ("1910.02054", "Microsoft", "DeepSpeed / ZeRO"),
    ("2301.02111", "Microsoft", "VALL-E"),
    # NVIDIA
    ("1909.08053", "NVIDIA", "Megatron-LM"),
    ("1812.04948", "NVIDIA", "StyleGAN"),
    ("2310.12931", "NVIDIA", "Eureka"),
    # Anthropic
    ("2212.08073", "Anthropic", "Constitutional AI"),
    ("2401.05566", "Anthropic", "Sleeper Agents"),
    # 初创公司
    ("2410.24164", "Physical Intelligence", "pi0"),
    ("2309.17080", "Wayve", "GAIA-1"),
    ("2310.06825", "Mistral AI", "Mistral 7B"),
    ("2401.04088", "Mistral AI", "Mixtral"),
    # 国内公司
    ("2412.19437", "DeepSeek", "DeepSeek-V3"),
    ("2501.12948", "DeepSeek", "DeepSeek-R1"),
    ("2407.10671", "Alibaba", "Qwen2"),
    ("2406.12793", "Zhipu AI", "ChatGLM / GLM-4"),
]

# 公司实验室博客 (只取最近 N 天的文章, 每个来源限量)
INDUSTRY_FEEDS = [
    {"source": "OpenAI", "url": "https://openai.com/news/rss.xml"},
    {"source": "Google DeepMind", "url": "https://deepmind.google/blog/rss.xml"},
    {"source": "Google Research", "url": "https://research.google/blog/rss/"},
    {"source": "Meta AI", "url": "https://ai.meta.com/blog/rss/"},
    {"source": "NVIDIA", "url": "https://blogs.nvidia.com/feed/"},
    {"source": "Microsoft Research", "url": "https://www.microsoft.com/en-us/research/feed/"},
    {"source": "AWS ML", "url": "https://aws.amazon.com/blogs/machine-learning/feed/"},
    {"source": "Apple ML", "url": "https://machinelearning.apple.com/rss.xml"},
    {"source": "Hugging Face", "url": "https://huggingface.co/blog/feed.xml"},
    {"source": "BAIR", "url": "https://bair.berkeley.edu/blog/feed.xml"},
]

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_text(raw: Optional[str]) -> str:
    """Strip HTML tags and collapse whitespace from a feed field."""
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    return _SPACE_RE.sub(" ", text).strip()


def parse_date(raw: Optional[str]) -> str:
    """Return YYYY-MM-DD for RFC822 or ISO8601 feed dates, '' when unparseable."""
    if not raw:
        return ""
    value = raw.strip()
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        pass
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else ""


def _find_text(element, names: Iterable[str]) -> str:
    """Find the first child whose tag matches one of ``names`` (namespace-insensitive)."""
    wanted = set(names)
    for child in element:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in wanted:
            return (child.text or "").strip()
    return ""


def _find_link(element) -> str:
    """Extract the item link from RSS (text) or Atom (href attribute)."""
    fallback = ""
    for child in element:
        if child.tag.rsplit("}", 1)[-1] != "link":
            continue
        href = child.attrib.get("href")
        rel = child.attrib.get("rel", "alternate")
        if href and rel == "alternate":
            return href.strip()
        if href and not fallback:
            fallback = href.strip()
        if child.text and child.text.strip() and not fallback:
            fallback = child.text.strip()
    return fallback


def parse_feed(raw_xml: bytes) -> List[Dict[str, str]]:
    """Parse an RSS 2.0 or Atom feed into {title, url, date, summary}."""
    root = ET.fromstring(raw_xml)
    entries = root.findall(".//item")
    if not entries:
        entries = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "entry"]

    parsed: List[Dict[str, str]] = []
    for entry in entries:
        title = clean_text(_find_text(entry, ("title",)))
        url = _find_link(entry)
        if not title or not url:
            continue
        summary = clean_text(
            _find_text(entry, ("description", "summary", "encoded", "content"))
        )
        date = parse_date(
            _find_text(entry, ("pubDate", "published", "updated", "date"))
        )
        parsed.append({"title": title, "url": url, "date": date, "summary": summary})
    return parsed


def fetch_feed(url: str, timeout: int = 20) -> bytes:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "daily-arxiv-ai-enhanced (+https://arxiv.0-range.cn/)"},
    )
    response.raise_for_status()
    return response.content


def collect_blogs(
    feeds: List[Dict[str, str]],
    days: int = 30,
    max_per_source: int = 3,
    fetch=fetch_feed,
    today: Optional[datetime] = None,
) -> List[Dict]:
    """Collect recent posts from the configured feeds, newest first."""
    today = today or datetime.now(timezone.utc)
    cutoff = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    collected: List[Dict] = []

    for feed in feeds:
        source = feed["source"]
        try:
            entries = parse_feed(fetch(feed["url"]))
        except Exception as error:  # noqa: BLE001 - one broken feed must not stop the rest
            print(f"Feed failed for {source}: {error}", file=sys.stderr)
            continue

        entries.sort(key=lambda item: item.get("date") or "", reverse=True)
        recent = [
            entry
            for entry in entries
            if entry.get("date") and entry["date"] >= cutoff
        ][:max_per_source]

        for entry in recent:
            collected.append({
                "id": "blog-{}-{}".format(
                    re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-"),
                    hashlib.sha1(entry["url"].encode("utf-8")).hexdigest()[:10],
                ),
                "kind": "blog",
                "source": source,
                "title": entry["title"],
                "authors": [],
                "categories": [],
                "summary": entry["summary"],
                "abs": entry["url"],
                "date": entry["date"],
            })

    collected.sort(key=lambda item: item.get("date") or "", reverse=True)
    return collected


def fetch_arxiv_metadata(arxiv_ids: List[str]) -> List[Dict]:
    """Fetch metadata for every curated industry paper in one batched request."""
    params = {"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}
    response = requests.get(ARXIV_API, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: List[Dict] = []
    for entry in root.findall("a:entry", ATOM):
        arxiv_id = entry.find("a:id", ATOM).text.split("/abs/")[-1]
        base_id = arxiv_id.split("v")[0]
        title = clean_text(entry.find("a:title", ATOM).text)
        summary = clean_text(entry.find("a:summary", ATOM).text)
        authors = [
            clean_text(author.find("a:name", ATOM).text)
            for author in entry.findall("a:author", ATOM)
            if clean_text(author.find("a:name", ATOM).text)
        ]
        categories = [
            category.attrib.get("term")
            for category in entry.findall("a:category", ATOM)
            if category.attrib.get("term")
        ]
        published = (entry.find("a:published", ATOM).text or "")[:10]
        papers.append({
            "id": base_id,
            "kind": "paper",
            "source": "",
            "title": title,
            "authors": authors,
            "categories": categories,
            "summary": summary,
            "abs": f"https://arxiv.org/abs/{base_id}",
            "date": published,
        })
    return papers


def attach_sources(papers: List[Dict], labels: Dict[str, str]) -> List[Dict]:
    for paper in papers:
        paper["source"] = labels.get(paper["id"], "Industry")
    return papers


def build_chain(model_name: str):
    """Build the same structured-output chain used by enhance.py."""
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain_openai import ChatOpenAI
    from runtime import build_chat_openai_kwargs
    from structure import Structure

    template = open("template.txt", "r").read()
    system = open("system.txt", "r").read()

    llm = ChatOpenAI(
        **build_chat_openai_kwargs(
            model_name=model_name,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    ).with_structured_output(Structure, method="function_calling")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template),
    ])
    return prompt | llm


def summarize(chain, item: Dict, language: str) -> None:
    """Attach item['AI'] in place; on failure leave it out instead of faking it."""
    content = item["summary"]
    if item.get("kind") == "blog":
        content = f"Title: {item['title']}\n\n{content}"
    try:
        response = chain.invoke({"language": language, "content": content})
        item["AI"] = response.model_dump()
    except Exception as error:  # noqa: BLE001 - keep the raw text when summarization fails
        print(f"AI failed for {item.get('id', 'unknown')}: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Collect industry papers and blog posts")
    parser.add_argument("--out", type=str, default="../data/industry.jsonl", help="Output jsonl path")
    parser.add_argument("--days", type=int, default=30, help="Keep blog posts from the last N days")
    parser.add_argument("--max-per-source", type=int, default=3, help="Max blog posts per source")
    parser.add_argument("--no-ai", action="store_true", help="Skip LLM summarization")
    args = parser.parse_args()

    model_name = os.environ.get("MODEL_NAME", "deepseek-flash")
    language = os.environ.get("LANGUAGE", "Chinese")

    labels = {arxiv_id: source for arxiv_id, source, _ in INDUSTRY_PAPERS}
    arxiv_ids = [arxiv_id for arxiv_id, _, _ in INDUSTRY_PAPERS]

    print(f"Fetching metadata for {len(arxiv_ids)} industry papers...", file=sys.stderr)
    papers = attach_sources(fetch_arxiv_metadata(arxiv_ids), labels)
    print(f"Fetched {len(papers)} papers", file=sys.stderr)

    blogs = collect_blogs(INDUSTRY_FEEDS, days=args.days, max_per_source=args.max_per_source)
    print(f"Collected {len(blogs)} blog posts from {len(INDUSTRY_FEEDS)} feeds", file=sys.stderr)

    items = papers + blogs
    if not items:
        print("ERROR: nothing collected, keeping the previous file", file=sys.stderr)
        sys.exit(1)

    if not args.no_ai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set, writing metadata without summaries", file=sys.stderr)
        else:
            chain = build_chain(model_name)
            for item in items:
                summarize(chain, item, language)
            summarized = sum(1 for item in items if item.get("AI"))
            print(f"Summarized {summarized}/{len(items)} items", file=sys.stderr)

    # 论文按新旧倒序，博客已在 collect_blogs 里排好
    papers.sort(key=lambda item: item.get("date") or "", reverse=True)
    ordered = papers + blogs

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done. Wrote {len(ordered)} items to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
