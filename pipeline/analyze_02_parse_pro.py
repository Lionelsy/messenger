"""
对切合当前研究主题的论文进行下载，解析，并保存信息到本地
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    # 如果你想记录失败原因，可以把下面这列打开，并同时在写入处赋值
    # "download_error",
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


# -----------------------------
# Thread-local requests session
# -----------------------------
_tls = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


def download_arxiv_pdf(
    paper_id: str,
    pdf_dir: Path,
    timeout: int = 120,
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

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    last_err: Exception | None = None
    session = _get_session()

    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                # 若存在旧文件但不是真 PDF，先删掉
                if out_path.exists() and not _looks_like_pdf(out_path):
                    out_path.unlink(missing_ok=True)

                with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
                    r.raise_for_status()

                    ctype = (r.headers.get("Content-Type") or "").lower()
                    if "text/html" in ctype:
                        sample = r.raw.read(256, decode_content=True)
                        raise RuntimeError(f"Got HTML instead of PDF from {url}: {sample[:120]!r}")

                    with out_path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)

                if not _looks_like_pdf(out_path):
                    head = out_path.read_bytes()[:256] if out_path.exists() else b""
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded file is not a real PDF (%PDF- missing). url={url} head={head[:120]!r}"
                    )

                return out_path

            except Exception as e:
                last_err = e
                continue

        if attempt < max_retries:
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download valid PDF for {pid} after retries: {last_err}")


def _process_one_paper(pid: str, pdf_dir: Path, parse_dir: Path, ocr) -> Tuple[str, bool, str]:
    """
    单篇论文流水线（不分离下载和解析）：
    1) PDF 缺失或是假 -> 下载/重下并校验
    2) parse 缺失 -> OCR 解析并写 json
    """
    try:
        pdf_path = pdf_dir / f"{pid}.pdf"
        parse_path = parse_dir / f"{pid}.json"

        # 1) 下载（缺失 or 假 PDF）
        if (not pdf_path.exists()) or (not _looks_like_pdf(pdf_path)):
            pdf_path = download_arxiv_pdf(pid, pdf_dir)

        # 2) OCR 解析（缺失才解析）
        if not parse_path.exists():
            parsed = ocr.ocr_pdf(str(pdf_path))
            with parse_path.open("w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)

        return pid, True, "ok"
    except Exception as e:
        return pid, False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--pdf_dir", default="storage/papers/pdfs")
    ap.add_argument("--parse_dir", default="storage/papers/parse")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="paper-level threads (download+parse in one worker). Keep small to avoid reCAPTCHA / GPU contention.",
    )
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    pdf_dir = Path(args.pdf_dir)
    parse_dir = Path(args.parse_dir)
    parse_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    # 只处理：relevance=True 且「尚未同时具备 pdf + parse 结果」的论文
    todo_pids: List[str] = []
    pid_to_row: dict[str, dict] = {}

    for r in rows:
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        pid_to_row[pid] = r

        if not _is_true(r.get("relevance", "")):
            continue

        pdf_path = pdf_dir / f"{pid}.pdf"
        parse_path = parse_dir / f"{pid}.json"
        if pdf_path.exists() and parse_path.exists():
            continue

        todo_pids.append(pid)

    if not todo_pids:
        print("[DONE] nothing to do")
        return

    _cfg, _llm, ocr = get_ai_clients()

    ok_cnt = 0
    fail_cnt = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(_process_one_paper, pid, pdf_dir, parse_dir, ocr): pid
            for pid in todo_pids
        }

        for fut in tqdm(as_completed(futs), total=len(futs), desc="analyze_02_parse", unit="paper"):
            pid, ok, msg = fut.result()

            if ok:
                pid_to_row[pid]["download"] = "True"
                # pid_to_row[pid].pop("download_error", None)
                ok_cnt += 1
            else:
                pid_to_row[pid]["download"] = "False"
                # pid_to_row[pid]["download_error"] = msg[:300]
                fail_cnt += 1
                print(f"[ERR] {pid}: {msg}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    _write_master_rows(master_csv, rows)
    print(f"[DONE] ok={ok_cnt} fail={fail_cnt} ; master_updated={master_csv}")


if __name__ == "__main__":
    main()
