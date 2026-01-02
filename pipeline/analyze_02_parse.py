"""
对切合当前研究主题的论文进行下载，解析，并保存信息到本地
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
from tqdm import tqdm

# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.ai import get_ai_clients


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


def download_arxiv_pdf(paper_id: str, pdf_dir: Path, timeout: int = 120) -> Path:
    """
    下载 arXiv PDF 到本地：
    - 保存路径：{pdf_dir}/{paper_id}.pdf
    - 下载 URL：https://arxiv.org/pdf/{paper_id}.pdf （paper_id 通常包含 v1）
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pid = (paper_id or "").strip()
    if not pid:
        raise ValueError("paper_id is empty")

    url = f"https://arxiv.org/pdf/{pid}.pdf"
    out_path = pdf_dir / f"{pid}.pdf"

    headers = {"User-Agent": "messager-bot/0.1"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--pdf_dir", default="storage/papers/pdfs")
    ap.add_argument("--parse_dir", default="storage/papers/parse")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    pdf_dir = Path(args.pdf_dir)
    parse_dir = Path(args.parse_dir)
    parse_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    # 只处理：relevance=True 且「尚未同时具备 pdf + parse 结果」的论文
    todo = []
    for r in rows:
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        if not _is_true(r.get("relevance", "")):
            continue
        pdf_path = pdf_dir / f"{pid}.pdf"
        parse_path = parse_dir / f"{pid}.json"
        if pdf_path.exists() and parse_path.exists():
            continue
        todo.append(r)

    _cfg, _llm, ocr = get_ai_clients()

    done = 0
    for r in tqdm(todo, total=len(todo), desc="analyze_02_parse", unit="paper"):
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue

        try:
            pdf_path = pdf_dir / f"{pid}.pdf"
            parse_path = parse_dir / f"{pid}.json"

            # 1) 下载（如果缺失）
            if not pdf_path.exists():
                pdf_path = download_arxiv_pdf(pid, pdf_dir)
                r["download"] = "True"
            else:
                # 只要文件存在，就视为已下载（避免 CSV 状态滞后）
                r["download"] = "True"

            # 2) OCR 解析（如果缺失）
            if not parse_path.exists():
                parsed = ocr.ocr_pdf(str(pdf_path))
                with parse_path.open("w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

            done += 1
        except Exception as e:
            print(f"[ERR] {pid}: {e}")
            continue
        finally:
            if args.sleep > 0:
                time.sleep(args.sleep)

    _write_master_rows(master_csv, rows)
    print(f"[DONE] parsed={done} ; master_updated={master_csv}")


if __name__ == "__main__":
    main()