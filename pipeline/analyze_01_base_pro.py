from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from tqdm import tqdm

import arxiv

from analyze_01_base import (
    analyze_one, fetch_arxiv_metadata,
    _read_master_rows, _write_master_rows,
)


def pre_fetch_metadata(todo_rows: list[dict], client: arxiv.Client, delay: float = 1.0) -> None:
    """
    串行预抓取所有缺失的 arXiv 元数据到本地缓存。
    在并行 LLM 分析之前调用，避免多线程同时打 arXiv API 触发 429。
    """
    meta_dir = _ROOT / "storage" / "fetch-arxiv" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for r in todo_rows:
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        if not (meta_dir / f"{pid}.json").exists():
            missing.append(pid)

    if not missing:
        print(f"[INFO] All {len(todo_rows)} papers have cached metadata.")
        return

    print(f"[INFO] Pre-fetching metadata for {len(missing)} papers (sequential)...")
    fetched, failed = 0, 0
    for pid in tqdm(missing, desc="pre_fetch_metadata", unit="paper"):
        try:
            meta = fetch_arxiv_metadata(pid, client=client)
            (meta_dir / f"{pid}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            fetched += 1
        except Exception as e:
            print(f"\n[WARN] pre-fetch failed for {pid}: {e}")
            failed += 1
        time.sleep(delay)

    print(f"[INFO] Pre-fetch done: {fetched} ok, {failed} failed.")


def process_task(row: dict, interest: str, out_dir: Path, sleep_s: float):
    """
    单个任务的工作函数：处理一篇论文并保存结果。
    arXiv 元数据从本地缓存读取（由 pre_fetch_metadata 预抓取），不再调用 arXiv API。
    """
    pid = (row.get("paperID") or "").strip()
    if not pid:
        return None, False

    try:
        result = analyze_one(pid, interest, sleep_s=sleep_s)

        out_path = out_dir / f"{pid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return pid, result["analysis"]["is_relevant"]
    except Exception as e:
        return pid, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--out_dir", default="storage/analysis/base")
    ap.add_argument("--sleep", type=float, default=0.1, help="每个线程在请求后的休眠时间")
    ap.add_argument("--workers", type=int, default=4, help="并发线程数")
    ap.add_argument("--interest", default=os.getenv("INTEREST_DESCRIPTION", "3D空间表示、理解、智能"))
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    # 筛选待处理任务
    todo_rows = [r for r in rows if (r.get("base_analysis") or "False").strip().lower() != "true"]
    random.shuffle(todo_rows)
    if not todo_rows:
        print("[INFO] No pending papers to analyze.")
        return

    print(f"[START] Total todo: {len(todo_rows)} using {args.workers} workers")

    # === 阶段 1：串行预抓取 arXiv 元数据 ===
    shared_client = arxiv.Client(delay_seconds=3, num_retries=3)
    pre_fetch_metadata(todo_rows, shared_client, delay=1.0)

    # === 阶段 2：并行 LLM 分析 ===
    print(f"[INFO] Starting parallel LLM analysis ({args.workers} workers)...")
    done_count = 0
    master_lock = Lock()
    row_map = { (r.get("paperID") or "").strip(): r for r in rows }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pid = {
            executor.submit(process_task, r, args.interest, out_dir, args.sleep): (r.get("paperID") or "").strip()
            for r in todo_rows
        }

        for future in tqdm(as_completed(future_to_pid), total=len(todo_rows), desc="Parallel Analysis"):
            pid = future_to_pid[future]
            try:
                res_pid, status = future.result()
                if isinstance(status, Exception):
                    print(f"\n[ERR] {pid} failed [{type(status).__name__}]: {status}")
                else:
                    with master_lock:
                        if res_pid in row_map:
                            row_map[res_pid]["base_analysis"] = "True"
                            row_map[res_pid]["relevance"] = "True" if status else "False"
                            done_count += 1
                            _write_master_rows(master_csv, rows)
            except Exception as e:
                print(f"\n[CRITICAL] Unexpected error for {pid}: {e}")

    print(f"\n[DONE] Successfully analyzed: {done_count} papers. Master CSV updated.")

if __name__ == "__main__":
    main()
