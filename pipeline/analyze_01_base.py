"""
对论文进行基础分析，并判断是否非常切合当前研究主题
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import random

import arxiv
import requests
from bs4 import BeautifulSoup

from tqdm import tqdm

# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.ai import get_ai_clients
from config.prompt import (
    SYSTEM_CN_JSON,
    SYSTEM_CN_RELEVANCE,
    build_user_prompt_step03_relevance_cn,
    build_user_prompt_step03_summary_cn,
)

def _parse_json_obj(text: str) -> Dict[str, Any]:
    """
    解析 LLM 输出为 JSON 对象：
    - 先直接 json.loads
    - 失败则从文本中提取第一个 {...} 再 json.loads
    """
    t = (text or "").strip()
    obj = json.loads(t)
    if isinstance(obj, dict):
        return obj

    # 有些模型会输出数组或其它结构；这里保持最简单：强制要求 dict
    raise ValueError("LLM JSON is not an object")


def _parse_json_obj_relaxed(text: str) -> Dict[str, Any]:
    try:
        return _parse_json_obj(text)
    except Exception:
        t = (text or "").strip()
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise
        return _parse_json_obj(m.group(0))


def _fetch_arxiv_metadata_from_page(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    从 arxiv.org/abs/{id} 静态页面抓取元数据。
    比 arXiv API 宽松得多，不容易触发限流。
    返回 None 表示抓取失败（由调用方决定是否回退）。
    """
    pid = (paper_id or "").strip()
    pid_no_ver = pid.split("v", 1)[0]

    url = f"https://arxiv.org/abs/{pid_no_ver}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 标题：<meta name="citation_title">
    title = ""
    tag = soup.find("meta", attrs={"name": "citation_title"})
    if tag and tag.get("content"):
        title = tag["content"].replace("\n", " ").strip()

    # 摘要：<meta name="abstract"> 或 <blockquote class="abstract">
    abstract = ""
    tag = soup.find("meta", attrs={"name": "abstract"})  # noqa: F841
    if tag and tag.get("content"):
        abstract = tag["content"].replace("\n", " ").strip()
    else:
        block = soup.find("blockquote", class_="abstract")
        if block:
            # 去掉开头的 "Abstract:" 标签
            text = block.get_text(separator=" ", strip=True)
            if text.lower().startswith("abstract"):
                text = re.sub(r"^abstract\s*:?\s*", "", text, flags=re.IGNORECASE)
            abstract = text.replace("\n", " ").strip()

    # 作者：<meta name="citation_author">
    authors = []
    for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
        if tag.get("content"):
            authors.append(tag["content"].strip())

    # 发布日期：<meta name="citation_date">
    published = ""
    tag = soup.find("meta", attrs={"name": "citation_date"})
    if tag and tag.get("content"):
        published = tag["content"].strip()

    # 分类
    categories = []
    span = soup.find("span", class_="primary-subject")
    if span:
        categories.append(span.get_text(strip=True))

    if not title and not abstract:
        return None

    return {
        "paperID": pid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": published,
        "updated": published,
        "arxiv_url": f"https://arxiv.org/abs/{pid_no_ver}",
        "pdf_url": f"https://arxiv.org/pdf/{pid_no_ver}",
        "categories": categories,
    }


def _fetch_arxiv_metadata_from_api(
    paper_id: str,
    *,
    client: Optional[arxiv.Client] = None,
    max_attempts: int = 6,
    backoff_base: float = 10.0,
) -> Dict[str, Any]:
    """通过 arXiv API 抓取元数据（带重试），作为静态页面的兜底方案。"""
    pid = (paper_id or "").strip()
    pid_no_ver = pid.split("v", 1)[0]

    if client is None:
        client = arxiv.Client(delay_seconds=3, num_retries=3)

    search = arxiv.Search(id_list=[pid_no_ver], max_results=1)

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            r = next(iter(client.results(search)), None)
            if r is None:
                raise RuntimeError(f"arXiv returned no result for {pid}")
            return {
                "paperID": pid,
                "title": (r.title or "").replace("\n", " ").strip(),
                "abstract": (r.summary or "").replace("\n", " ").strip(),
                "authors": [a.name for a in (r.authors or [])],
                "published": r.published.isoformat(),
                "updated": r.updated.isoformat(),
                "arxiv_url": getattr(r, "entry_id", "") or "",
                "pdf_url": getattr(r, "pdf_url", "") or "",
                "categories": list(getattr(r, "categories", []) or []),
            }
        except arxiv.HTTPError as e:
            last_err = e
            status = getattr(e, "status", None)
            if status == 429 and attempt < max_attempts - 1:
                sleep_s = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                print(f"[WARN] arXiv API 429 fetching {pid}; sleep {sleep_s:.1f}s ({attempt+1}/{max_attempts})", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
            else:
                raise
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                sleep_s = min(backoff_base * (2 ** attempt), 60.0) + random.uniform(0, 1)
                print(f"[WARN] arXiv API error fetching {pid}; sleep {sleep_s:.1f}s ({attempt+1}/{max_attempts}): {e}", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
            else:
                raise

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to fetch arXiv metadata for {pid}")


def fetch_arxiv_metadata(
    paper_id: str,
    *,
    client: Optional[arxiv.Client] = None,
) -> Dict[str, Any]:
    """
    获取 arXiv 元数据，优先级：本地缓存 → 静态页面 → API 兜底。
    """
    pid = (paper_id or "").strip()
    if not pid:
        raise ValueError("paper_id is empty")

    # 1) 本地缓存
    meta_path = _ROOT / "storage" / "fetch-arxiv" / "meta" / f"{pid}.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2) 静态页面（arxiv.org/abs/{id}），限流宽松
    try:
        result = _fetch_arxiv_metadata_from_page(pid)
        if result is not None:
            return result
    except Exception as e:
        print(f"[WARN] page fetch failed for {pid}: {e}", file=sys.stderr, flush=True)

    # 3) API 兜底
    print(f"[INFO] falling back to API for {pid}", file=sys.stderr, flush=True)
    return _fetch_arxiv_metadata_from_api(pid, client=client)


def analyze_one(paper_id: str, interest_description: str, sleep_s: float = 0.0, *, arxiv_client: Optional[arxiv.Client] = None) -> Dict[str, Any]:
    cfg, llm, _ocr = get_ai_clients()

    meta = fetch_arxiv_metadata(paper_id, client=arxiv_client)
    abstract = meta.get("abstract", "")

    # Step 1: 结构化摘要（JSON）
    messages_summary = [
        {"role": "system", "content": SYSTEM_CN_JSON},
        {"role": "user", "content": build_user_prompt_step03_summary_cn(abstract)},
    ]
    summary_text = llm.chat_text(messages_summary, response_json=True)
    summary_obj = _parse_json_obj_relaxed(summary_text)

    # Step 2: 相关性判断（是/否）
    messages_rel = [
        {"role": "system", "content": SYSTEM_CN_RELEVANCE},
        {"role": "user", "content": build_user_prompt_step03_relevance_cn(abstract, interest_description)},
    ]
    rel_text = (llm.chat_text(messages_rel) or "").strip()
    is_relevant = rel_text.startswith("是") or rel_text.lower().startswith("yes")

    if sleep_s > 0:
        time.sleep(sleep_s)

    return {
        "paperID": paper_id,
        "fetched": meta,
        "analysis": {
            "gpt_summary": summary_obj,
            "is_relevant": is_relevant,
        },
        "runtime": {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "llm_provider": cfg.llm_provider,
            "llm_model": (cfg.zhipu.model if cfg.zhipu else cfg.openai_compat.model if cfg.openai_compat else ""),
        },
    }


def _read_master_rows(master_csv: Path) -> list[dict]:
    if not master_csv.exists():
        return []
    with master_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_master_rows(master_csv: Path, rows: list[dict]) -> None:
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    # 与 pipeline/update_paper_list.py 的 master schema 对齐
    fieldnames = [
        "paperID",
        "sources",
        "createDate",
        "base_analysis",
        "relevance",
        "download",
        "deep_analysis",
        "publish",
    ]
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--out_dir", default="storage/analysis/base")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--interest", default=os.getenv("INTEREST_DESCRIPTION", "3D场景表示、理解、智能"))
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    todo = [r for r in rows if (r.get("base_analysis") or "False").strip().lower() != "true"]
    random.shuffle(todo)

    # 创建共享 Client，复用延迟设置避免 429
    shared_client = arxiv.Client(delay_seconds=3, num_retries=3)

    done = 0
    for r in tqdm(todo, total=len(todo), desc="analyze_01_base", unit="paper"):
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        try:
            result = analyze_one(pid, args.interest, sleep_s=args.sleep, arxiv_client=shared_client)
            out_path = out_dir / f"{pid}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 更新 master：基础分析完成 + 相关性判断结果
            r["base_analysis"] = "True"
            r["relevance"] = "True" if result["analysis"]["is_relevant"] else "False"
            done += 1
            _write_master_rows(master_csv, rows)
            print(f"[OK] analyzed: {pid} -> {out_path}")
        except Exception as e:
            print(f"[ERR] analyze failed: {pid} [{type(e).__name__}] ; {e}")
            continue

    print(f"[DONE] analyzed={done} ; master_updated={master_csv}")


if __name__ == "__main__":
    main()