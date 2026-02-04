from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Optional


# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_pubdate(s: str) -> Optional[datetime]:
    """
    解析 RSS 里的 pubDate（RFC822）：
    例如：Sat, 09 Nov 2024 00:26:14 +0800
    """
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.strptime(s, "%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rss_file", default="arxiv.rss")
    ap.add_argument("--days", type=int, default=14, help="删除多少天之前的条目")
    ap.add_argument("--now", default=None, help="统一的当前时间（RFC2822），用于与调度器对齐")
    args = ap.parse_args()

    rss_path = _ROOT / args.rss_file
    if not rss_path.exists():
        print("[DONE] rss file not found, skip")
        return

    tree = ET.parse(str(rss_path))
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        print("[DONE] invalid rss: missing channel")
        return

    now_dt = datetime.now(timezone.utc)
    if args.now:
        now_dt = datetime.strptime(args.now.strip(), "%a, %d %b %Y %H:%M:%S %z").astimezone(timezone.utc)

    time_limit = now_dt - timedelta(days=args.days)

    items = list(channel.findall("item"))
    removed = 0
    for item in items:
        pub = item.findtext("pubDate") or ""
        dt = _parse_pubdate(pub)
        if dt is None:
            continue
        if dt.astimezone(timezone.utc) < time_limit:
            channel.remove(item)
            removed += 1

    # 更新 lastBuildDate
    last = channel.find("lastBuildDate")
    if last is not None:
        last.text = now_dt.astimezone(timezone(timedelta(hours=8))).strftime("%a, %d %b %Y %H:%M:%S %z")

    tree.write(str(rss_path), encoding="utf-8", xml_declaration=True)
    print(f"[DONE] removed={removed} ; rss_updated={rss_path}")


if __name__ == "__main__":
    main()

