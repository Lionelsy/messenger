from __future__ import annotations
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tqdm import tqdm

# ... [保留原有的 _ROOT, get_ai_clients, REQUIRED_Q_KEYS, ALT_KEY_MAP 等所有工具函数] ...

from config.ai import get_ai_clients
from config.prompt import (
    SYSTEM_CN_JSON,
    build_user_prompt_step03_deep_cn,
    build_user_prompt_step03_deep_fix_cn,
)

from analyze_03_deep import _read_master_rows, _write_master_rows, _load_parsed_md, _parse_json_obj_relaxed, _deep_result_is_valid, _normalize_deep_result, _repair_to_required_json, _is_true, _load_existing_deep

def process_deep_task(
    row: dict, 
    args: argparse.Namespace, 
    llm: Any
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """
    单个论文的深度分析工作函数
    返回: (paperID, success_status, result_object_or_error)
    """
    pid = (row.get("paperID") or "").strip()
    parse_dir = Path(args.parse_dir)
    out_dir = Path(args.out_dir)

    try:
        parse_path = parse_dir / f"{pid}.json"
        if not parse_path.exists():
            return pid, False, f"missing parse file: {parse_path}"

        title, md = _load_parsed_md(parse_path, pid)
        md = md[: args.max_chars]

        # Step 1: LLM 深度解读
        messages = [
            {"role": "system", "content": SYSTEM_CN_JSON},
            {"role": "user", "content": build_user_prompt_step03_deep_cn(title, md)},
        ]
        out_text = llm.chat_text(messages, response_json=True)
        deep_obj = _parse_json_obj_relaxed(out_text)

        # Step 2: 格式验证与修正
        if _deep_result_is_valid(deep_obj):
            deep_obj = _normalize_deep_result(deep_obj)
        else:
            # 额外一步：格式修复
            deep_obj = _repair_to_required_json(llm, out_text)

        # Step 3: 写入结果文件
        out_path = out_dir / f"{pid}.json"
        payload = {
            "paperID": pid,
            "title": title,
            "deep_understanding": deep_obj,
            "meta": {
                "parse_path": str(parse_path),
                "ts": datetime.now().isoformat(timespec="seconds"),
            },
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # 模拟请求间隔
        if args.sleep > 0:
            time.sleep(args.sleep)
            
        return pid, True, deep_obj

    except Exception as e:
        return pid, False, str(e)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--parse_dir", default="storage/papers/parse")
    ap.add_argument("--out_dir", default="storage/analysis/deep")
    ap.add_argument("--max_chars", type=int, default=20000)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=5, help="并发线程数")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv empty")
        return

    # 1. 筛选待分析任务
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    todo_rows = []
    for r in rows:
        pid = (r.get("paperID") or "").strip()
        if not pid: continue
        # 满足深度分析的前置条件
        if (_is_true(r.get("base_analysis")) and 
            _is_true(r.get("relevance")) and 
            _is_true(r.get("download")) and 
            not _is_true(r.get("publish"))):
            
            # 检查是否已存在有效结果
            out_path = out_dir / f"{pid}.json"
            if _is_true(r.get("deep_analysis")) and _load_existing_deep(out_path):
                continue
            todo_rows.append(r)

    if args.limit > 0:
        todo_rows = todo_rows[: args.limit]

    if not todo_rows:
        print("[INFO] No papers need deep analysis.")
        return

    # 2. 初始化 AI 客户端 (线程安全)
    _cfg, llm, _ocr = get_ai_clients()

    # 3. 并发执行
    print(f"[START] Deep analyzing {len(todo_rows)} papers with {args.workers} workers...")
    
    # 用字典存储更新结果，最后统一同步到 rows
    results_map = {} 

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pid = {
            executor.submit(process_deep_task, r, args, llm): r["paperID"] 
            for r in todo_rows
        }

        for future in tqdm(as_completed(future_to_pid), total=len(todo_rows), desc="Deep Analysis"):
            pid = future_to_pid[future]
            try:
                res_pid, success, data = future.result()
                if success:
                    results_map[res_pid] = "True"
                else:
                    print(f"\n[ERR] {res_pid} failed: {data}")
            except Exception as e:
                print(f"\n[CRITICAL] {pid} crash: {e}")

    # 4. 更新 Master 数据并保存
    done_count = 0
    for r in rows:
        pid = r.get("paperID")
        if pid in results_map:
            r["deep_analysis"] = "True"
            done_count += 1

    _write_master_rows(master_csv, rows)
    print(f"[DONE] Processed {done_count} papers. Master CSV updated.")

if __name__ == "__main__":
    main()