import os
import re
import csv
import argparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any

ID_RE = re.compile(r"^/papers/(?P<id>\d{4}\.\d{5})$")

def fetch_hf_daily(date_str: str, timeout: int = 30) -> List[Dict[str, Any]]:
    url = f"https://huggingface.co/papers/date/{date_str}"
    headers = {"User-Agent": "paper-daily-bot/0.1"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

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
        seen.add(arxiv_id)

        title = a.get_text(" ", strip=True)
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
