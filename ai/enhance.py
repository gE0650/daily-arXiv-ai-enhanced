import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
import requests
import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure
from content_filter import is_sensitive
from runtime import build_chat_openai_kwargs, raise_if_processing_failed
from summary_cache import AI_FIELDS, build_ai_meta, is_reusable, load_cache

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

# Fallback text used when summarization cannot run. summary_cache treats these
# values as placeholders, so they are never reused as real summaries.
DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed",
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="jsonline file with AI results from an earlier run of the same day",
    )
    return parser.parse_args()


def check_github_code(content: str) -> Dict:
    """提取并验证 GitHub 链接"""
    code_info = {}

    # 1. 优先匹配 github.com/owner/repo 格式
    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)

    if match:
        owner, repo = match.groups()
        # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
        repo = repo.rstrip(".git").rstrip(".,)")

        full_url = f"https://github.com/{owner}/{repo}"
        code_info["code_url"] = full_url

        # 尝试调用 GitHub API 获取信息
        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            # API 调用失败不影响主流程
            pass
        return code_info

    # 2. 如果没有 github.com，尝试匹配 github.io
    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)

    if match_io:
        url = match_io.group(0)
        # 清理末尾标点
        url = url.rstrip(".,)")
        code_info["code_url"] = url
        # github.io 不进行 star 和 update 判断

    return code_info


def prepare_item(item: Dict) -> bool:
    """做不依赖 LLM 的预处理：合规检查 + 代码链接检测。

    返回 False 表示这篇论文应当被丢弃。
    """
    if is_sensitive(item.get("summary", "")):
        return False

    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)
    return True


def summarize_item(chain, item: Dict, language: str) -> bool:
    """调用 LLM 填充 item['AI']，解析失败时退回占位字段。

    返回 False 表示生成内容未通过合规检查，应当丢弃。
    """
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}

        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)

        # Merge partial data with defaults to ensure all fields exist
        item['AI'] = {**DEFAULT_AI_FIELDS, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        raise RuntimeError(
            f"AI request failed for {item.get('id', 'unknown')}: {e}"
        ) from e

    # Final validation to ensure all required fields exist
    for field in DEFAULT_AI_FIELDS.keys():
        if field not in item['AI']:
            item['AI'][field] = DEFAULT_AI_FIELDS[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return False
    return True


def process_single_item(chain, item: Dict, language: str) -> Dict:
    if not prepare_item(item):
        return None
    if not summarize_item(chain, item, language):
        return None
    return item


def reuse_item(item: Dict, entry: Dict, ai_meta: Dict) -> Dict:
    """复用上一轮生成的摘要，但代码链接检测与合规检查仍然重新做一遍。"""
    if not prepare_item(item):
        return None

    cached_ai = entry["AI"]
    item["AI"] = {field: cached_ai[field] for field in AI_FIELDS}
    return item


def build_chain(model_name: str):
    llm = ChatOpenAI(
        **build_chat_openai_kwargs(
            model_name=model_name,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    ).with_structured_output(Structure, method="function_calling")

    print('Connect to:', model_name, file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    return prompt_template | llm


def process_all_items(
    data: List[Dict],
    model_name: str,
    language: str,
    max_workers: int,
    cache: Dict[str, Dict] = None,
    ai_meta: Dict = None,
    on_item=None,
) -> List[Dict]:
    """复用缓存里仍然有效的摘要，其余并行调用 LLM。

    on_item 会在每条结果完成时立刻回调，便于边跑边落盘。
    """
    cache = cache or {}
    processed_data: List[Dict] = [None] * len(data)

    def publish(idx: int, result) -> None:
        # 打上缓存戳：只有带戳且内容匹配的条目才会被下一次运行复用
        if result is not None and ai_meta:
            result["ai_meta"] = ai_meta
        processed_data[idx] = result
        if on_item:
            on_item(result)

    reuse_jobs: List = []
    llm_jobs: List = []
    for idx, item in enumerate(data):
        entry = cache.get(str(item.get("id", "")))
        if ai_meta and is_reusable(entry, item, ai_meta):
            reuse_jobs.append((idx, item, entry))
        else:
            llm_jobs.append((idx, item))

    errors_by_idx: Dict[int, str] = {}
    if reuse_jobs or llm_jobs:
        chain = build_chain(model_name) if llm_jobs else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, item, entry in reuse_jobs:
                future_to_idx[executor.submit(reuse_item, item, entry, ai_meta)] = idx
            for idx, item in llm_jobs:
                future_to_idx[executor.submit(process_single_item, chain, item, language)] = idx

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Processing items"
            ):
                idx = future_to_idx[future]
                try:
                    publish(idx, future.result())
                except Exception as e:
                    print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                    errors_by_idx[idx] = str(e)

        # 失败项串行重试一次：多数是限流或网关抖动之类的瞬时错误
        if errors_by_idx:
            entries_by_idx = {idx: entry for idx, _, entry in reuse_jobs}
            print(f"Retrying {len(errors_by_idx)} failed paper(s)...", file=sys.stderr)
            for idx in sorted(errors_by_idx):
                try:
                    if idx in entries_by_idx:
                        # 复用任务失败（例如临时网络问题），重试复用而不是改调 LLM
                        publish(idx, reuse_item(data[idx], entries_by_idx[idx], ai_meta))
                    else:
                        publish(idx, process_single_item(chain, data[idx], language))
                    errors_by_idx.pop(idx)
                    print(f"Retry succeeded for {data[idx].get('id', idx)}", file=sys.stderr)
                except Exception as e:
                    errors_by_idx[idx] = str(e)
                    print(f"Retry failed for {data[idx].get('id', idx)}: {e}", file=sys.stderr)

    raise_if_processing_failed(list(errors_by_idx.values()), total=len(data))

    reused = sum(1 for idx, _, _ in reuse_jobs if processed_data[idx] is not None)
    kept = sum(1 for item in processed_data if item is not None)
    if reused:
        print(f"Reused {reused} summaries from cache", file=sys.stderr)
    print(
        f"Summaries: {reused} reused, {kept - reused} generated, "
        f"{len(errors_by_idx)} failed, {len(data) - kept} filtered",
        file=sys.stderr,
    )
    return processed_data


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    language = os.environ.get("LANGUAGE", 'Chinese')

    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')

    # 先读缓存：--cache 有可能就是 target_file 本身
    cache = load_cache(args.cache)
    if args.cache:
        print(f'Cache: {args.cache} ({len(cache)} entries)', file=sys.stderr)

    # 检查并删除目标文件
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    ai_meta = build_ai_meta(model_name, language, system, template)

    # 边处理边落盘：中途失败时磁盘上留下已完成的部分，下一次运行可以直接复用
    with open(target_file, "w", encoding="utf-8") as handle:
        def on_item(item) -> None:
            if item is None:
                return
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()

        processed_data = process_all_items(
            data,
            model_name,
            language,
            args.max_workers,
            cache=cache,
            ai_meta=ai_meta,
            on_item=on_item,
        )

    # 全部成功后再按输入顺序重写一次，保证输出顺序稳定
    with open(target_file, "w", encoding="utf-8") as handle:
        for item in processed_data:
            if item is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
