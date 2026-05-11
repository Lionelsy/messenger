import os
import csv
import json
import argparse
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
import arxiv
from tqdm import tqdm


def load_topics(topic_yml: str) -> List[Dict[str, str]]:
    """topic.yml -> task list: [{topic, subtopic, query}, ...]"""
    with open(topic_yml, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    tasks: List[Dict[str, str]] = []
    for topic, subtopics in (data or {}).items():
        if isinstance(subtopics, dict):
            for subtopic, query in subtopics.items():
                tasks.append({"topic": str(topic), "subtopic": str(subtopic), "query": str(query)})
        else:
            # 兼容旧格式：topic: "query"
            tasks.append({"topic": str(topic), "subtopic": "default", "query": str(subtopics)})

    return tasks


def _install_requests_timeout(client: arxiv.Client, timeout_s: float) -> None:
    """
    arxiv.Client 内部用 requests.Session.get(...) 且未传 timeout，可能导致某些请求“永久等待”。
    这里给 session.get 注入默认 timeout，避免卡死。
    """
    if timeout_s <= 0:
        return
    session = getattr(client, "_session", None)
    if session is None:
        return
    orig_get = session.get

    def _get_with_timeout(url, *args, **kwargs):
        kwargs.setdefault("timeout", timeout_s)
        return orig_get(url, *args, **kwargs)

    session.get = _get_with_timeout  # type: ignore[assignment]


def fetch_arxiv(
    query: str,
    max_results: int = 30,
    *,
    client: Optional[arxiv.Client] = None,
    max_attempts: int = 6,
    backoff_base_seconds: float = 10.0,
) -> List[arxiv.Result]:
    """
    抓取 arXiv 搜索结果。

    关键点：必须复用同一个 arxiv.Client，才能让其内置的 delay_seconds
    在不同 query 之间生效，避免触发 429。
    """
    if client is None:
        client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as e:
            last_err = e
            status = getattr(e, "status", None)
            if status != 429 or attempt >= max_attempts - 1:
                raise
            sleep_s = backoff_base_seconds * (2**attempt) + random.uniform(0.0, 1.0)
            print(
                f"[WARN] arXiv 429; sleep {sleep_s:.1f}s then retry ({attempt+1}/{max_attempts})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            if attempt >= max_attempts - 1:
                raise
            sleep_s = min(backoff_base_seconds * (2**attempt), 60.0) + random.uniform(0.0, 1.0)
            print(
                f"[WARN] arXiv error; sleep {sleep_s:.1f}s then retry ({attempt+1}/{max_attempts}) -> {e}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_s)

    # 理论上走不到这里（上面最后一次会 raise），留个兜底
    if last_err is not None:
        raise last_err
    return []


def run(
    topic_yml: str = "config/topic.yml",
    out_dir: str = "storage/fetch-arxiv",
    max_results: int = 30,
    include_extra_fields: bool = True,
    date_str: Optional[str] = None,
    http_timeout: float = 30.0,
) -> str:
    """
    读取 topic.yml -> 抓取 -> 输出 CSV
    include_extra_fields=True 时，CSV 会额外包含 topic/subtopic/title/url，便于后续去重与分析。
    同时将每篇论文的完整元数据缓存到 {out_dir}/meta/{paperID}.json，供 analyze 阶段直接复用。
    """
    tasks = load_topics(topic_yml)

    os.makedirs(out_dir, exist_ok=True)
    meta_dir = Path(out_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"arxiv_data_{today}.csv")

    # 用 dict 去重（同一篇可能出现在多个 subtopic），保留首次出现
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    # 复用同一个 Client，让其内置的 delay_seconds 贯穿整个任务列表
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=8, num_retries=3)
    _install_requests_timeout(client, http_timeout)

    for t in tqdm(tasks, total=len(tasks), desc="fetch_arxiv", unit="topic"):
        time.sleep(random.uniform(5.0, 10.0))
        print(
            f"[INFO] topic={t.get('topic','')} subtopic={t.get('subtopic','')} query={t.get('query','')}",
            file=sys.stderr,
            flush=True,
        )
        try:
            results = fetch_arxiv(t["query"], max_results=max_results, client=client)
        except Exception as e:
            msg = f"[WARN] skip query due to error: topic={t['topic']} subtopic={t['subtopic']} err={e}"
            print(msg, file=sys.stderr, flush=True)
            continue
        for r in results:
            paper_id = r.get_short_id()
            publish_date = r.published.date().strftime("%Y-%m-%d")

            if paper_id not in rows_by_id:
                row = {"id": paper_id, "publish_time": publish_date}
                if include_extra_fields:
                    row.update(
                        {
                            "topic": t["topic"],
                            "subtopic": t["subtopic"],
                            "title": (r.title or "").replace("\n", " ").strip(),
                            "paper_url": r.entry_id,
                        }
                    )
                rows_by_id[paper_id] = row

                # 缓存完整元数据到 JSON，供 analyze 阶段直接读取
                meta_path = meta_dir / f"{paper_id}.json"
                if not meta_path.exists():
                    meta_obj = {
                        "paperID": paper_id,
                        "title": (r.title or "").replace("\n", " ").strip(),
                        "abstract": (r.summary or "").replace("\n", " ").strip(),
                        "authors": [a.name for a in (r.authors or [])],
                        "published": r.published.isoformat(),
                        "updated": r.updated.isoformat(),
                        "arxiv_url": getattr(r, "entry_id", "") or "",
                        "pdf_url": getattr(r, "pdf_url", "") or "",
                        "categories": list(getattr(r, "categories", []) or []),
                    }
                    meta_path.write_text(
                        json.dumps(meta_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    # 写 CSV（用标准库 csv，替代 pandas）——你原逻辑是 df.to_csv(...) :contentReference[oaicite:8]{index=8}
    if include_extra_fields:
        fieldnames = ["id", "publish_time", "topic", "subtopic", "title", "paper_url"]
    else:
        fieldnames = ["id", "publish_time"]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_by_id.values():
            writer.writerow(row)

    print(f"[OK] wrote {len(rows_by_id)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="输出文件日期（YYYY-MM-DD），用于与调度器对齐")
    ap.add_argument("--topic_yml", default="config/topic.yml")
    ap.add_argument("--out_dir", default="storage/fetch-arxiv")
    ap.add_argument("--max_results", type=int, default=30)
    ap.add_argument("--http_timeout", type=float, default=30.0, help="单次 arXiv HTTP 请求超时（秒）")
    args = ap.parse_args()

    run(
        topic_yml=args.topic_yml,
        out_dir=args.out_dir,
        max_results=args.max_results,
        date_str=args.date,
        http_timeout=args.http_timeout,
    )
