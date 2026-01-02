import csv
import argparse
from pathlib import Path
from datetime import datetime


MASTER_FIELDS = [
    "paperID",
    "sources",
    "createDate",
    "base_analysis",    # 是否进行了基础分析
    "relevance",        # 是否相关
    "download",         # 是否下载了论文
    "deep_analysis",    # 是否进行了深度分析
    "publish",          # 是否发布了论文
]

DEFAULT_FALSE = "False"


def split_sources(s: str) -> set:
    return {x.strip() for x in (s or "").split(";") if x.strip()}


def join_sources(ss: set) -> str:
    return ";".join(sorted(ss))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_master(master_path: Path) -> dict[str, dict]:
    master: dict[str, dict] = {}
    for r in read_csv(master_path):
        pid = (r.get("paperID") or "").strip()
        if not pid:
            continue

        # 兼容旧字段名：analysis -> base_analysis
        base_analysis = (r.get("base_analysis") or r.get("analysis") or DEFAULT_FALSE).strip() or DEFAULT_FALSE
        deep_analysis = (r.get("deep_analysis") or DEFAULT_FALSE).strip() or DEFAULT_FALSE
        download = (r.get("download") or DEFAULT_FALSE).strip() or DEFAULT_FALSE
        publish = (r.get("publish") or DEFAULT_FALSE).strip() or DEFAULT_FALSE
        relevance = (r.get("relevance") or DEFAULT_FALSE).strip() or DEFAULT_FALSE

        master[pid] = {
            "paperID": pid,
            "sources": (r.get("sources") or "").strip(),
            "createDate": (r.get("createDate") or "").strip(),
            "base_analysis": base_analysis,
            "relevance": relevance,
            "download": download,
            "deep_analysis": deep_analysis,
            "publish": publish,
        }
    return master


def upsert(master: dict[str, dict], pid: str, source: str, today: str):
    pid = (pid or "").strip()
    if not pid:
        return

    if pid not in master:
        master[pid] = {
            "paperID": pid,
            "sources": source,
            "createDate": today,
            "base_analysis": DEFAULT_FALSE,
            "relevance": DEFAULT_FALSE,
            "download": DEFAULT_FALSE,
            "deep_analysis": DEFAULT_FALSE,
            "publish": DEFAULT_FALSE,
        }
    else:
        ss = split_sources(master[pid].get("sources", ""))
        ss.add(source)
        master[pid]["sources"] = join_sources(ss)
        # createDate / base_analysis / relevance / download / deep_analysis / publish 都不在这里动


def write_master(master_path: Path, master: dict[str, dict]):
    master_path.parent.mkdir(parents=True, exist_ok=True)
    items = [master[k] for k in sorted(master.keys(), reverse=True)]

    with master_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        w.writeheader()
        for row in items:
            w.writerow({k: row.get(k, "") for k in MASTER_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--arxiv_csv", default=None)
    ap.add_argument("--hf_csv", default=None)
    ap.add_argument("--master_csv", default="storage/papers_master.csv")
    args = ap.parse_args()

    date_str = args.date

    arxiv_csv = Path(args.arxiv_csv) if args.arxiv_csv else Path(
        f"storage/fetch-arxiv/arxiv_data_{date_str}.csv"
    )
    hf_csv = Path(args.hf_csv) if args.hf_csv else Path(
        f"storage/fetch-hf-daily/hf_papers_{date_str}.csv"
    )
    master_csv = Path(args.master_csv)

    master = load_master(master_csv)

    for r in read_csv(arxiv_csv):
        upsert(master, r.get("id", ""), "arxiv", date_str)

    for r in read_csv(hf_csv):
        upsert(master, r.get("id", ""), "hf", date_str)

    write_master(master_csv, master)

    print(f"[OK] merged -> {master_csv} ; rows={len(master)}")


if __name__ == "__main__":
    main()
