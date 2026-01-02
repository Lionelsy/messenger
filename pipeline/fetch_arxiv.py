import os
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import yaml
import arxiv


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


def fetch_arxiv(query: str, max_results: int = 30) -> List[arxiv.Result]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return list(client.results(search))


def run(
    topic_yml: str = "config/topic.yml",
    out_dir: str = "storage/fetch-arxiv",
    max_results: int = 30,
    include_extra_fields: bool = True,
    date_str: Optional[str] = None,
) -> str:
    """
    读取 topic.yml -> 抓取 -> 输出 CSV
    include_extra_fields=True 时，CSV 会额外包含 topic/subtopic/title/url，便于后续去重与分析。
    """
    tasks = load_topics(topic_yml)

    os.makedirs(out_dir, exist_ok=True)
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"arxiv_data_{today}.csv")

    # 用 dict 去重（同一篇可能出现在多个 subtopic），保留首次出现
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    for t in tasks:
        results = fetch_arxiv(t["query"], max_results=max_results)
        for r in results:
            paper_id = r.get_short_id()            # 你原来用的就是它 :contentReference[oaicite:6]{index=6}
            publish_date = r.published.date().strftime("%Y-%m-%d")  # 你原来用 r.published.date() :contentReference[oaicite:7]{index=7}

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
    args = ap.parse_args()

    run(
        topic_yml=args.topic_yml,
        out_dir=args.out_dir,
        max_results=args.max_results,
        date_str=args.date,
    )
