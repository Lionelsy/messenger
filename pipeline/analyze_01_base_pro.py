from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from tqdm import tqdm
# ... 保持原有的 import 不变 ...

# [此处保留你原有的 _parse_json_obj, fetch_arxiv_metadata, analyze_one 等函数]

from analyze_01_base import analyze_one, _read_master_rows, _write_master_rows

def process_task(row: dict, interest: str, out_dir: Path, sleep_s: float):
    """
    单个任务的工作函数：处理一篇论文并保存结果
    """
    pid = (row.get("paperID") or "").strip()
    if not pid:
        return None, False

    try:
        # 执行分析
        result = analyze_one(pid, interest, sleep_s=sleep_s)
        
        # 保存 JSON 文件
        out_path = out_dir / f"{pid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 返回结果用于更新 master 列表
        return pid, result["analysis"]["is_relevant"]
    except Exception as e:
        return pid, e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--out_dir", default="storage/analysis/base")
    ap.add_argument("--sleep", type=float, default=0.1, help="每个线程在请求后的休眠时间")
    ap.add_argument("--workers", type=int, default=10, help="并发线程数")
    ap.add_argument("--interest", default=os.getenv("INTEREST_DESCRIPTION", "3D场景表示、理解、智能"))
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
    if not todo_rows:
        print("[INFO] No pending papers to analyze.")
        return

    print(f"[START] Total todo: {len(todo_rows)} using {args.workers} workers")

    # 使用线程池执行
    done_count = 0
    # 将 rows 转换成 dict 以便根据 paperID 快速定位更新
    row_map = { (r.get("paperID") or "").strip(): r for r in rows }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_pid = {
            executor.submit(process_task, r, args.interest, out_dir, args.sleep): (r.get("paperID") or "").strip() 
            for r in todo_rows
        }

        # 使用 tqdm 监听任务完成情况
        for future in tqdm(as_completed(future_to_pid), total=len(todo_rows), desc="Parallel Analysis"):
            pid = future_to_pid[future]
            try:
                res_pid, status = future.result()
                if isinstance(status, Exception):
                    print(f"\n[ERR] {pid} failed: {status}")
                else:
                    # 更新内存中的 rows 数据
                    if res_pid in row_map:
                        row_map[res_pid]["base_analysis"] = "True"
                        row_map[res_pid]["relevance"] = "True" if status else "False"
                        done_count += 1
            except Exception as e:
                print(f"\n[CRITICAL] Unexpected error for {pid}: {e}")

    # 任务全部完成后，统一写入 CSV
    _write_master_rows(master_csv, rows)
    print(f"\n[DONE] Successfully analyzed: {done_count} papers. Master CSV updated.")

if __name__ == "__main__":
    main()