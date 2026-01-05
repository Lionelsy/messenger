from __future__ import annotations

import sys
import argparse
import csv
import json
import re
import html
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from xml.etree import ElementTree as ET
from xml.dom import minidom


# 允许把该文件当脚本运行：确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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

TZ_OFFSET = timezone(timedelta(hours=8))  # 东八区


def _is_true(v: str) -> bool:
    return (v or "").strip().lower() == "true"


def _clean_string(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, (str, bytes)):
        x = str(x)
    return re.sub(r"[\x00-\x1F\x7F]", "", x)


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


def _rfc2822(dt: datetime) -> str:
    return dt.astimezone(TZ_OFFSET).strftime("%a, %d %b %Y %H:%M:%S %z")


def _get_or_create_channel(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        channel = ET.SubElement(root, "channel")
    return channel


def _load_or_init_rss(
    rss_path: Path,
    feed_title: str,
    feed_link: str,
    feed_description: str,
    run_dt: datetime,
) -> Tuple[ET.ElementTree, ET.Element, set]:
    """
    返回：(tree, channel, existing_ids)
    existing_ids 用 <guid> 文本判重。
    """
    existing_ids: set = set()

    if rss_path.exists():
        tree = ET.parse(str(rss_path))
        channel = _get_or_create_channel(tree)
        for item in channel.findall("item"):
            guid = item.findtext("guid") or ""
            if guid:
                existing_ids.add(guid.strip())
        return tree, channel, existing_ids

    # 初始化最小 RSS 2.0
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = feed_title
    ET.SubElement(channel, "link").text = feed_link
    ET.SubElement(channel, "description").text = feed_description
    ET.SubElement(channel, "language").text = "zh-CN"
    ET.SubElement(channel, "lastBuildDate").text = _rfc2822(run_dt)
    tree = ET.ElementTree(rss)
    return tree, channel, existing_ids


def _set_last_build_date(channel: ET.Element, run_dt: datetime) -> None:
    node = channel.find("lastBuildDate")
    if node is None:
        node = ET.SubElement(channel, "lastBuildDate")
    node.text = _rfc2822(run_dt)


def _load_base_json(base_path: Path) -> Dict[str, Any]:
    return json.loads(base_path.read_text(encoding="utf-8"))


def _load_deep_json(deep_path: Path) -> Dict[str, Any]:
    return json.loads(deep_path.read_text(encoding="utf-8"))


def _build_description_html(base: Dict[str, Any], deep: Dict[str, Any], is_relevant: bool) -> str:
    fetched = base.get("fetched") or {}
    analysis = base.get("analysis") or {}
    gpt_summary = (analysis.get("gpt_summary") or {}) if isinstance(analysis, dict) else {}

    title = _clean_string(fetched.get("title"))
    arxiv_url = _clean_string(fetched.get("arxiv_url") or fetched.get("pdf_url"))
    published = _clean_string(fetched.get("published"))
    abstract = _clean_string(fetched.get("abstract"))

    deep_understanding = deep.get("deep_understanding") or {}
    has_deep = isinstance(deep_understanding, dict) and bool(deep_understanding)

    parts: list[str] = []

    # 对“已做深度解读且与研究主题相关”的条目加醒目标记
    if has_deep and is_relevant:
        parts.append("<p>⭐ 与研究主题相关</p>")
    # GPT 基础摘要
    if isinstance(gpt_summary, dict) and gpt_summary:
        parts.append("<h3>GPT 基础摘要</h3>")
        for k, v in gpt_summary.items():
            parts.append(
                f"<p><strong>{html.escape(_clean_string(k))}:</strong> {html.escape(_clean_string(v))}</p>"
            )

    # 深度解读
    if has_deep:
        parts.append("<h3>深度解读</h3>")
        for k, v in deep_understanding.items():
            parts.append(
                f"<p><strong>{html.escape(_clean_string(k))}:</strong> {html.escape(_clean_string(v))}</p>"
            )

    if abstract:
        parts.append("<h3>Abstract</h3>")
        parts.append(f"<p>{html.escape(abstract)}</p>")

    return "\n".join(parts)


def _add_item(
    channel: ET.Element,
    paper_id: str,
    base: Dict[str, Any],
    deep: Dict[str, Any],
    run_dt: datetime,
    is_relevant: bool,
) -> None:
    fetched = base.get("fetched") or {}
    title = _clean_string(fetched.get("title") or paper_id)
    link = _clean_string(fetched.get("arxiv_url") or fetched.get("pdf_url") or "")
    authors = fetched.get("authors") or []
    if isinstance(authors, list):
        author_text = _clean_string(", ".join([str(a) for a in authors]))
    else:
        author_text = _clean_string(authors)

    item = ET.Element("item")
    ET.SubElement(item, "guid").text = paper_id
    ET.SubElement(item, "title").text = title
    if link:
        ET.SubElement(item, "link").text = link
    if author_text:
        ET.SubElement(item, "author").text = author_text
    ET.SubElement(item, "pubDate").text = _rfc2822(run_dt)

    # 先写成普通文本（ElementTree 会自动转义）；最终写文件前会统一改成 CDATA
    desc = _build_description_html(base, deep, is_relevant=is_relevant)
    ET.SubElement(item, "description").text = desc

    # 插入到 channel 开头（紧跟在 metadata 后面）
    insert_pos = 0
    for i, child in enumerate(list(channel)):
        if child.tag == "item":
            insert_pos = i
            break
        insert_pos = i + 1
    channel.insert(insert_pos, item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    ap.add_argument("--base_dir", default="storage/analysis/base")
    ap.add_argument("--deep_dir", default="storage/analysis/deep")
    ap.add_argument("--rss_file", default="arxiv.rss")
    ap.add_argument("--feed_title", default="Arxiv论文推荐")
    ap.add_argument("--feed_link", default="https://arxiv.org/")
    ap.add_argument("--feed_description", default="Arxiv论文推荐")
    ap.add_argument("--run_pubdate", default=None, help="统一发布时间（RFC2822），用于与调度器对齐")
    ap.add_argument("--limit", type=int, default=0, help="0 表示不限制；否则只发布前 N 篇")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    master_csv = Path(args.master_csv)
    base_dir = Path(args.base_dir)
    deep_dir = Path(args.deep_dir)
    rss_path = _ROOT / args.rss_file

    rows = _read_master_rows(master_csv)
    if not rows:
        print(f"[WARN] master csv not found or empty: {master_csv}")
        return

    # 筛选：已完成基础分析（或深度分析）且尚未 publish
    todo = []
    for r in rows:
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue
        if not (_is_true(r.get("base_analysis", "")) or _is_true(r.get("deep_analysis", ""))):
            continue
        if _is_true(r.get("publish", "")):
            continue
        todo.append(r)

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    run_dt = datetime.now(TZ_OFFSET)
    if args.run_pubdate:
        # 例如：Wed, 31 Dec 2025 18:52:08 +0800
        run_dt = datetime.strptime(args.run_pubdate.strip(), "%a, %d %b %Y %H:%M:%S %z")

    tree, channel, existing_ids = _load_or_init_rss(
        rss_path,
        feed_title=args.feed_title,
        feed_link=args.feed_link,
        feed_description=args.feed_description,
        run_dt=run_dt,
    )

    will_publish = []
    for r in todo:
        pid = (r.get("paperID") or "").strip()
        if not pid or pid in existing_ids:
            continue
        base_path = base_dir / f"{pid}.json"
        if not base_path.exists():
            continue
        will_publish.append(pid)

    if args.dry_run:
        print(f"[DRY_RUN] will_publish={len(will_publish)}")
        for pid in will_publish[:50]:
            print("-", pid)
        return

    published_n = 0
    for r in todo:
        pid = (r.get("paperID") or "").strip()
        if not pid or pid in existing_ids:
            continue

        base_path = base_dir / f"{pid}.json"
        deep_path = deep_dir / f"{pid}.json"
        if not base_path.exists():
            continue

        base = _load_base_json(base_path)
        deep: Dict[str, Any] = {}
        if deep_path.exists():
            deep = _load_deep_json(deep_path)

        analysis = base.get("analysis") or {}
        base_is_relevant = bool(analysis.get("is_relevant")) if isinstance(analysis, dict) else False
        row_is_relevant = _is_true(r.get("relevance", ""))
        is_relevant = row_is_relevant or base_is_relevant

        _add_item(channel, pid, base, deep, run_dt=run_dt, is_relevant=is_relevant)
        existing_ids.add(pid)

        r["publish"] = "True"
        published_n += 1

    _set_last_build_date(channel, run_dt=run_dt)
    rss_path.parent.mkdir(parents=True, exist_ok=True)
    # 写出时把每个 <description> 包成 CDATA，让 RSS 阅读器按 HTML 渲染
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
    dom = minidom.parseString(xml_bytes)
    for node in dom.getElementsByTagName("description"):
        # 取当前文本内容（minidom 会把 &lt;h2&gt; 还原成 <h2>）
        txt = ""
        if node.firstChild is not None and node.firstChild.nodeType == node.firstChild.TEXT_NODE:
            txt = node.firstChild.data
            node.removeChild(node.firstChild)
        node.appendChild(dom.createCDATASection(txt))

    rss_path.parent.mkdir(parents=True, exist_ok=True)
    with rss_path.open("wb") as f:
        f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
        f.write(dom.documentElement.toxml(encoding="utf-8"))

    _write_master_rows(master_csv, rows)
    print(f"[DONE] rss_updated={rss_path} ; published={published_n} ; master_updated={master_csv}")


if __name__ == "__main__":
    main()

