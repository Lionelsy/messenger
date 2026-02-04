"""
对切合当前研究主题的论文进行深度分析，并保存分析结果到本地
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
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.ai import get_ai_clients
from config.prompt import (
    SYSTEM_CN_JSON,
    SYSTEM_CN_PLAIN,
    build_user_prompt_step03_deep_single_q_cn,
    build_user_prompt_step03_deep_fix_cn,
)


REQUIRED_Q_KEYS = [
    "这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？",
    "作者是如何思考并设计出这个方法的？是否有借鉴现有工作？",
    "这个方法的核心思想是什么？整体实现流程是怎样的？",
    "论文的关键创新点有哪些？相比之前的工作，有什么不同？",
    "如果要用一句话总结这篇论文的贡献，你会怎么说？",
]

ALT_KEY_MAP = {
    # 有些模型会把 5 个问题输出成“概念化 key”（英文/蛇形命名）
    "problem_and_importance": REQUIRED_Q_KEYS[0],
    "design_motivation_and_insights": REQUIRED_Q_KEYS[1],
    "core_idea_and_workflow": REQUIRED_Q_KEYS[2],
    "innovations_and_contributions": REQUIRED_Q_KEYS[3],
    "one_sentence_summary": REQUIRED_Q_KEYS[4],
}


MASTER_FIELDS = [
    "paperID",
    "sources",
    "createDate",
    "base_analysis",
    "relevance",
    "download",
    "deep_analysis",
    "publish",
]


def _read_master_rows(master_csv: Path) -> List[dict]:
    if not master_csv.exists():
        return []
    with master_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_master_rows(master_csv: Path, rows: List[dict]) -> None:
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in MASTER_FIELDS})


def _is_true(v: str) -> bool:
    return (v or "").strip().lower() == "true"


def _parse_json_obj_relaxed(text: str) -> Dict[str, Any]:
    """
    解析 LLM 输出为 JSON 对象：
    - 先直接 json.loads
    - 失败则从文本中提取第一个 {...} 再 json.loads
    """
    t = (text or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"cannot find json object in: {t[:200]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON is not an object")
    return obj


def _load_parsed_md(parse_path: Path, paper_id: str) -> Tuple[str, str]:
    """
    从 OCR 解析结果里拿 md_content，并尽量拿到标题。
    返回：(title, md_content)
    """
    data = json.loads(parse_path.read_text(encoding="utf-8"))
    results = data.get("results") or {}

    pid = (paper_id or "").strip()
    if pid and pid in results:
        item = results[pid] or {}
    else:
        # 兜底：取 results 的第一个 key
        if isinstance(results, dict) and results:
            _k = next(iter(results.keys()))
            item = results[_k] or {}
        else:
            raise ValueError("parse json missing results")

    md = (item.get("md_content") or "").strip()
    if not md:
        raise ValueError("parse json missing md_content")

    # 很粗糙的标题提取：取 markdown 第一行的 '# ' 标题
    title = ""
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return title, md


def _deep_result_is_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    if keys == set(REQUIRED_Q_KEYS):
        return True
    if keys == set(ALT_KEY_MAP.keys()):
        return True
    return False


def _normalize_deep_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一输出为 REQUIRED_Q_KEYS（中文问题做 key）。
    """
    keys = set(obj.keys())
    if keys == set(REQUIRED_Q_KEYS):
        return obj
    if keys == set(ALT_KEY_MAP.keys()):
        return {ALT_KEY_MAP[k]: obj.get(k, "unknown") for k in ALT_KEY_MAP.keys()}
    raise ValueError(f"deep json keys invalid: {list(obj.keys())[:10]}")


def _repair_to_required_json(llm, bad_output_text: str) -> Dict[str, Any]:
    """
    当模型没有按要求输出 key 时，追加一次“格式修正”请求，把内容强制转换为 REQUIRED_Q_KEYS。
    """
    fix_prompt = build_user_prompt_step03_deep_fix_cn(REQUIRED_Q_KEYS, bad_output_text)
    messages_fix = [
        {"role": "system", "content": SYSTEM_CN_JSON},
        {"role": "user", "content": fix_prompt},
    ]
    fixed_text = llm.chat_text(messages_fix, response_json=True)
    fixed_obj = _parse_json_obj_relaxed(fixed_text)
    if set(fixed_obj.keys()) != set(REQUIRED_Q_KEYS):
        raise ValueError(f"repair failed, keys={list(fixed_obj.keys())[:10]}")
    return fixed_obj


def _normalize_plain_answer(text: str) -> str:
    """
    规范化单题纯文本输出：
    - 去掉首尾空白
    - 去掉常见的前缀“回答：/答案：”
    - 空则返回 "unknown"
    """
    t = (text or "").strip()
    if not t:
        return "unknown"
    # 去掉一些模型爱加的标签
    t = re.sub(r"^(回答|答案|答复)\s*[:：]\s*", "", t)
    # 去掉包裹引号
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
        t = t[1:-1].strip()
    return t or "unknown"


def _load_existing_deep(out_path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not out_path.exists():
            return None
        data = json.loads(out_path.read_text(encoding="utf-8"))
        deep = data.get("deep_understanding")
        return deep if isinstance(deep, dict) else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--parse_dir", default="storage/papers/parse")
    ap.add_argument("--out_dir", default="storage/analysis/deep")
    ap.add_argument("--max_chars", type=int, default=20000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="0 表示不限制；否则只处理前 N 篇")
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    parse_dir = Path(args.parse_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    # 处理条件：
    # - base_analysis=True
    # - relevance=True
    # - download=True（意味着 PDF/OCR 已准备好）
    # - publish=False（未发布）
    # - deep_analysis!=True（尚未深度解读）
    todo = []
    for r in rows:
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        if not _is_true(r.get("base_analysis", "")):
            continue
        if not _is_true(r.get("relevance", "")):
            continue
        if not _is_true(r.get("download", "")):
            continue
        if _is_true(r.get("publish", "")):
            continue
        out_path = out_dir / f"{pid}.json"
        existing = _load_existing_deep(out_path)
        already_ok = existing is not None and _deep_result_is_valid(existing)
        if _is_true(r.get("deep_analysis", "")) and already_ok:
            continue
        todo.append(r)

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    if not todo:
        print("[DONE] no papers to deep analyze")
        return

    _cfg, llm, _ocr = get_ai_clients()

    done = 0
    for r in tqdm(todo, total=len(todo), desc="analyze_03_deep", unit="paper"):
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        try:
            parse_path = parse_dir / f"{pid}.json"
            if not parse_path.exists():
                raise FileNotFoundError(f"missing parse file: {parse_path}")

            title, md = _load_parsed_md(parse_path, pid)
            md = md[: args.max_chars]

            # 逐问题多次请求：每次只回答 1 个问题，最后组装成 REQUIRED_Q_KEYS 对应的 deep_obj
            deep_obj: Dict[str, Any] = {}
            for q in REQUIRED_Q_KEYS:
                messages = [
                    {"role": "system", "content": SYSTEM_CN_PLAIN},
                    {"role": "user", "content": build_user_prompt_step03_deep_single_q_cn(title, md, q)},
                ]
                out_text = llm.chat_text(messages)
                deep_obj[q] = _normalize_plain_answer(out_text)

                # 每次请求间隔（避免过快打满/触发限流）
                if args.sleep > 0:
                    time.sleep(args.sleep)

            # deep_obj 是程序端按 REQUIRED_Q_KEYS 构造的，理论上恒 valid；这里保留一个断言式兜底
            if not _deep_result_is_valid(deep_obj):
                raise ValueError("deep_obj keys invalid after per-question calls")

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

            # 只标记 deep_analysis，不动 publish
            r["deep_analysis"] = "True"
            done += 1
        except Exception as e:
            print(f"[ERR] {pid}: {e}")
            continue

    _write_master_rows(master_csv, rows)
    print(f"[DONE] deep_analyzed={done} ; master_updated={master_csv}")

if __name__ == "__main__":
    main()
