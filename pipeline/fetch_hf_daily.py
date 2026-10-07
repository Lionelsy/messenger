import os
import re
import csv
import argparse
import random
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional

ID_RE = re.compile(r"^/papers/(?P<id>\d{4}\.\d{5})$")

def fetch_hf_daily(date_str: str, timeout: int = 30, max_retries: int = 3) -> List[Dict[str, Any]]:
    url = f"https://huggingface.co/papers/date/{date_str}"
    headers = {"User-Agent": "paper-daily-bot/0.1"}

    # 指数退避重试：网络抖动/5xx 时重试，避免一次失败让 run_daily.sh（set -e）中断整天流水线
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = 2.0 * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"[WARN] HF 抓取失败（{e}）；重试 {attempt}/{max_retries} 后 {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
    else:
        # 重试耗尽：返回空（写出空 CSV），不阻塞当天其余阶段
        print(f"[WARN] HF 抓取重试耗尽（{last_err}）；本日跳过 HF 论文，其余阶段继续", flush=True)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    items = []
    seen = set()

    for a in soup.find_all("a", href=True):
        m = ID_RE.match(a["href"])
        if not m:
            continue
        arxiv_id = m.group("id")
        if arxiv_id in seen:
            continue
        # 每张卡片有多个 /papers/{id} 锚点：第一个是缩略图（无文本），标题锚点才有内容；
        # 跳过无文本锚点（不标记 seen），取第一个有文本的标题
        title = a.get_text(" ", strip=True)
        if not title:
            continue
        seen.add(arxiv_id)

        items.append(
            {
                "id": arxiv_id + "v1",
                "title": title,
                "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}v1",
                "publish_time": date_str,
            }
        )

    return items


def save_csv(items: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["id", "publish_time", "title", "hf_url", "arxiv_url"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in items:
            w.writerow(it)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="抓取日期（YYYY-MM-DD），用于与调度器对齐")
    ap.add_argument("--out_dir", default=os.path.join("storage", "fetch-hf-daily"))
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    items = fetch_hf_daily(date_str)
    out_path = os.path.join(args.out_dir, f"hf_papers_{date_str}.csv")
    save_csv(items, out_path)
    print(f"[OK] {len(items)} items -> {out_path}")
