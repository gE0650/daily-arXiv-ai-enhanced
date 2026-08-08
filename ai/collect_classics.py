#!/usr/bin/env python3
"""
AI 抓取 VLA 经典论文 (Collect classic VLA papers)

功能:
1. 从 arXiv API 抓取经典 VLA (Vision-Language-Action) 论文的元数据
2. 使用 LLM (与 enhance.py 相同) 为每篇论文生成 TLDR / motivation / method / result / conclusion
3. 输出 data/classic.jsonl, 供前端"经典论文"区展示

说明:
- 默认只生成一次 (workflow 中检测到文件已存在则跳过), 已读并删除的论文不会再次被抓取或推送
- 可通过环境变量 CLASSIC_ARXIV_IDS 覆盖默认的经典论文列表 (逗号分隔)
- 本地测试可用 --no-ai 跳过 LLM 摘要, 仅抓取元数据
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import requests

# 经典 VLA 论文清单 (arXiv ID, 已通过 arXiv API 校验)
DEFAULT_CLASSIC_IDS = [
    "2109.12098",  # CLIPort (2021)
    "2202.02005",  # BC-Z (2022)
    "2204.01691",  # SayCan (2022)
    "2205.06175",  # GATO (2022)
    "2209.05451",  # PerAct / Perceiver-Actor (2022)
    "2212.06817",  # RT-1 (2022)
    "2303.03378",  # PaLM-E (2023)
    "2303.04137",  # Diffusion Policy (2023)
    "2305.15021",  # EmbodiedGPT (2023)
    "2306.11706",  # RoboCat (2023)
    "2307.05973",  # VoxPoser (2023)
    "2307.15818",  # RT-2 (2023)
    "2310.08864",  # Open X-Embodiment / RT-X (2023)
    "2405.12213",  # Octo (2024)
    "2406.09246",  # OpenVLA (2024)
    "2410.07864",  # RDT-1B (2024)
    "2410.24164",  # pi0 (2024)
    "2503.20020",  # Gemini Robotics (2025)
]

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def get_classic_ids() -> List[str]:
    """返回经典论文 ID 列表, 支持环境变量覆盖。"""
    override = os.environ.get("CLASSIC_ARXIV_IDS", "").strip()
    if override:
        ids = [i.strip() for i in override.split(",") if i.strip()]
        if ids:
            return ids
    return DEFAULT_CLASSIC_IDS


def fetch_arxiv_metadata(arxiv_ids: List[str]) -> List[Dict]:
    """从 arXiv API 抓取论文元数据。"""
    params = {"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers: List[Dict] = []
    for entry in root.findall("a:entry", ATOM):
        arxiv_id = entry.find("a:id", ATOM).text.split("/abs/")[-1]
        # 去掉版本号 (如 2212.06817v2 -> 2212.06817)
        base_id = arxiv_id.split("v")[0]

        title = " ".join((entry.find("a:title", ATOM).text or "").split())
        summary = " ".join((entry.find("a:summary", ATOM).text or "").split())
        authors = [
            author.find("a:name", ATOM).text
            for author in entry.findall("a:author", ATOM)
            if author.find("a:name", ATOM).text
        ]
        categories = [
            cat.attrib.get("term")
            for cat in entry.findall("a:category", ATOM)
            if cat.attrib.get("term")
        ]
        published = (entry.find("a:published", ATOM).text or "")[:10]

        papers.append({
            "id": base_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "summary": summary,
            "abs": f"https://arxiv.org/abs/{base_id}",
            "date": published,
            "classic": True,
            "topic": "VLA",
        })
    return papers


def is_sensitive(content: str) -> bool:
    """与 enhance.py 保持一致的内容合规检测。"""
    try:
        resp = requests.post(
            "https://spam.dw-dengwei.workers.dev",
            json={"text": content},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("sensitive", True)
        return True
    except Exception as e:
        print(f"Sensitive check error: {e}", file=sys.stderr)
        return True


def enhance_item(item: Dict, chain, language: str, default_ai_fields: Dict) -> Optional[Dict]:
    """使用 LLM 为单篇论文生成结构化摘要。"""
    if is_sensitive(item.get("summary", "")):
        return None

    try:
        response = chain.invoke({
            "language": language,
            "content": item["summary"],
        })
        item["AI"] = response.model_dump()
    except Exception as e:
        print(f"AI processing failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item["AI"] = dict(default_ai_fields)

    for field in default_ai_fields:
        if field not in item["AI"]:
            item["AI"][field] = default_ai_fields[field]

    for value in item.get("AI", {}).values():
        if is_sensitive(str(value)):
            return None
    return item


def build_chain(model_name: str):
    """构建与 enhance.py 相同的 LLM chain。"""
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from structure import Structure

    template = open("template.txt", "r").read()
    system = open("system.txt", "r").read()

    llm = ChatOpenAI(
        model=model_name,
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}
    ).with_structured_output(Structure, method="function_calling")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template),
    ])
    return prompt_template | llm, {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed",
    }


def main():
    parser = argparse.ArgumentParser(description="Collect classic VLA papers from arXiv with AI summaries")
    parser.add_argument("--out", type=str, default="../data/classic.jsonl", help="Output jsonl file path")
    parser.add_argument("--no-ai", action="store_true", help="Skip LLM summarization (fetch metadata only)")
    args = parser.parse_args()

    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")
    arxiv_ids = get_classic_ids()

    print(f"Fetching {len(arxiv_ids)} classic VLA papers from arXiv API...", file=sys.stderr)
    papers = fetch_arxiv_metadata(arxiv_ids)
    print(f"Fetched metadata for {len(papers)} papers", file=sys.stderr)

    if not papers:
        print("ERROR: no papers fetched from arXiv API", file=sys.stderr)
        sys.exit(1)

    # 按发表日期排序 (旧 -> 新)
    papers.sort(key=lambda p: p.get("date", ""))

    if not args.no_ai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY is not set, using raw abstracts only", file=sys.stderr)
        else:
            print(f"Enhancing {len(papers)} papers with model {model_name}...", file=sys.stderr)
            chain, default_ai_fields = build_chain(model_name)
            enhanced = []
            for item in papers:
                result = enhance_item(item, chain, language, default_ai_fields)
                if result is not None:
                    enhanced.append(result)
            papers = enhanced
            print(f"Enhanced {len(papers)} papers", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in papers:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(papers)} classic papers to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
