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


def _looks_like_pdf(path: Path) -> bool:
    """快速判定文件是否真 PDF（防止保存了 HTML reCAPTCHA 页面）"""
    try:
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False


def download_arxiv_pdf(
    paper_id: str,
    pdf_dir: Path,
    timeout: int = 600,
    max_retries: int = 3,
    sleep_base: float = 1.0,
) -> Path:
    """
    下载 arXiv PDF 到本地，并做真实性校验：
    - 优先用 export.arxiv.org，降低触发 reCAPTCHA 概率
    - 下载完成后检查文件头必须为 %PDF-
    - 若检测到 HTML/非 PDF：删除文件并重试
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pid = (paper_id or "").strip()
    if not pid:
        raise ValueError("paper_id is empty")

    out_path = pdf_dir / f"{pid}.pdf"

    # 更稳的顺序：export -> arxiv
    urls = [
        f"https://export.arxiv.org/pdf/{pid}.pdf",
        f"https://arxiv.org/pdf/{pid}.pdf",
    ]

    # UA 尽量像浏览器一点（很多站对奇怪 UA 更敏感）
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                # 若存在旧文件但不是真 PDF，先删掉
                if out_path.exists() and not _looks_like_pdf(out_path):
                    out_path.unlink(missing_ok=True)

                with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                    r.raise_for_status()

                    # content-type 不是强保证，但可提前预警
                    ctype = (r.headers.get("Content-Type") or "").lower()
                    # 有些情况下会返回 text/html（reCAPTCHA）
                    if "text/html" in ctype:
                        # 读一点点内容，帮助报错定位（不落盘）
                        sample = r.raw.read(256, decode_content=True)
                        raise RuntimeError(f"Got HTML instead of PDF from {url}: {sample[:120]!r}")

                    # 流式写入
                    with out_path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)

                # 关键：写完必须验证 PDF 魔数
                if not _looks_like_pdf(out_path):
                    # 可能是 HTML / challenge 页面被保存了
                    try:
                        # 额外取一点文本辅助定位（可选）
                        txt = out_path.read_bytes()[:512]
                    except Exception:
                        txt = b""
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded file is not a real PDF (%PDF- missing). "
                        f"url={url} head={txt[:120]!r}"
                    )

                return out_path

            except Exception as e:
                last_err = e
                # 换下一个 url
                continue

        # 本轮两个 url 都失败：退避重试
        if attempt < max_retries:
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download valid PDF for {pid} after retries: {last_err}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--pdf_dir", default="storage/papers/pdfs")
    ap.add_argument("--parse_dir", default="storage/papers/parse")
    ap.add_argument("--sleep", type=float, default=0.1)
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


            # 2) OCR 解析（如果缺失）
            if not parse_path.exists():
                parsed = ocr.ocr_pdf(str(pdf_path))
                with parse_path.open("w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
            
            # 3) 更新 master 记录
            r["download"] = "True"

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