from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, TextIO


@dataclass(frozen=True)
class ScheduleConfig:
    # 周一=1 ... 周日=7
    weekdays: set[int] = frozenset({1, 2, 3, 4, 5, 6})
    hhmm: str = "14:01"
    poll_seconds: int = 10

TZ_OFFSET = timezone(timedelta(hours=8))  # 与 RSS 一致


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _python() -> str:
    return sys.executable


def _run_cmd(cmd: List[str], cwd: Path, logf: TextIO) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    logf.write(f"\n[RUN] {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
    logf.flush()

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=logf,
        stderr=logf,
    )
    if p.returncode != 0:
        raise RuntimeError(f"command failed (exit={p.returncode}): {' '.join(cmd)}")


def _today_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _lock_path(root: Path) -> Path:
    # 用文件锁避免同一时刻多实例重复跑（不会放到 git）
    return root / "storage" / ".run_daily.lock"


def _state_path(root: Path) -> Path:
    # 记录最近一次成功触发日期，避免在 14:01 这一分钟内重复触发
    return root / "storage" / ".run_daily.last_run"

def _logs_dir(root: Path) -> Path:
    return root / "storage" / "logs"


def _new_log_file(root: Path) -> Path:
    d = _logs_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return d / f"run_daily-{ts}.log"


def _acquire_lock(lock_file: Path) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL：如果文件已存在则失败
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"lock exists: {lock_file} (another run_daily may be running)")


def _release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)  # py3.8+; 若更老环境可换 try/except
    except Exception:
        pass


def _read_last_run(state_file: Path) -> str:
    try:
        return state_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_last_run(state_file: Path, date_str: str) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(date_str, encoding="utf-8")


def run_pipeline_once(root: Path) -> None:
    """
    按顺序执行你当前工程的每日流程脚本：
    1) 抓取 arXiv/hf
    2) 合并 master 表
    3) base 分析
    4) 下载 + OCR 解析
    5) 深度解读
    6) 发布 RSS 新条目
    7) 清理 RSS 旧条目
    """
    # 统一的“当日时间戳/日期”：一次触发内所有脚本都用它，避免前后不一致
    run_dt = datetime.now(TZ_OFFSET)
    date_str = run_dt.strftime("%Y-%m-%d")
    run_pubdate = run_dt.strftime("%a, %d %b %Y %H:%M:%S %z")

    scripts = [
        # 抓取阶段：用同一个 date 命名输出 CSV
        ([root / "pipeline" / "fetch_arxiv.py", "--date", date_str], "fetch_arxiv"),
        ([root / "pipeline" / "fetch_hf_daily.py", "--date", date_str], "fetch_hf_daily"),
        ([root / "pipeline" / "update_paper_list.py", "--date", date_str], "update_paper_list"),
        # 分析/下载/深度：内部会基于 master 状态推进（不强制 date）
        ([root / "pipeline" / "analyze_01_base.py"], "analyze_01_base"),
        ([root / "pipeline" / "analyze_02_parse.py"], "analyze_02_parse"),
        ([root / "pipeline" / "analyze_03_deep.py"], "analyze_03_deep"),
        # 发布阶段：统一 pubDate/lastBuildDate/删除阈值的“当前时间”
        ([root / "pipeline" / "publish_add_new_items.py", "--run_pubdate", run_pubdate], "publish_add_new_items"),
        ([root / "pipeline" / "publish_delete_old_items.py", "--now", run_pubdate], "publish_delete_old_items"),
        # 最后：把 RSS 推送到 git（独立脚本，避免把 git 逻辑写进 Python）
        (["bash", root / "scripts" / "publish_rss.sh"], "publish_rss"),
    ]

    for cmd_parts, _name in scripts:
        # cmd_parts 可能是 ["bash", Path(".../xx.sh")] 或 [Path(".../xx.py"), ...]
        if str(cmd_parts[0]) == "bash":
            sp = Path(cmd_parts[1])
        else:
            sp = Path(cmd_parts[0])
        if not sp.exists():
            raise FileNotFoundError(f"missing script: {sp}")

    log_path = _new_log_file(root)
    print(f"[LOG] {log_path}")
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[START] {datetime.now().isoformat(timespec='seconds')}\n")
        logf.write(f"[RUN_DT] {run_pubdate}\n")
        logf.write(f"[RUN_DATE] {date_str}\n")
        logf.write(f"[CWD] {root}\n")
        logf.write(f"[PY] {_python()}\n")
        logf.flush()

        for cmd_parts, _name in scripts:
            # 如果 cmd_parts 以 bash 开头，就按 bash 执行；否则默认用 python 执行
            if str(cmd_parts[0]) == "bash":
                _run_cmd([str(x) for x in cmd_parts], cwd=root, logf=logf)
            else:
                _run_cmd([_python(), *[str(x) for x in cmd_parts]], cwd=root, logf=logf)

        logf.write(f"\n[END] {datetime.now().isoformat(timespec='seconds')}\n")
        logf.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="14:01", help="触发时间，格式 HH:MM（默认 14:01）")
    ap.add_argument("--weekdays", default="1,2,3,4,5,6", help="周几触发：1=周一...7=周日（默认周一到周六）")
    ap.add_argument("--poll", type=int, default=10, help="轮询间隔秒数（默认 10）")
    ap.add_argument("--once", action="store_true", help="立刻跑一次后退出（不进入常驻调度）")
    args = ap.parse_args()

    root = _repo_root()
    cfg = ScheduleConfig(
        weekdays=set(int(x) for x in args.weekdays.split(",") if x.strip()),
        hhmm=args.time,
        poll_seconds=max(1, args.poll),
    )

    lock_file = _lock_path(root)
    state_file = _state_path(root)

    if args.once:
        _acquire_lock(lock_file)
        try:
            run_pipeline_once(root)
            _write_last_run(state_file, _today_key(datetime.now()))
        finally:
            _release_lock(lock_file)
        return

    print(f"[SCHED] root={root}")
    print(f"[SCHED] weekdays={sorted(cfg.weekdays)} time={cfg.hhmm} poll={cfg.poll_seconds}s")

    while True:
        now = datetime.now()
        # 周一=1...周日=7
        if now.isoweekday() in cfg.weekdays and now.strftime("%H:%M") == cfg.hhmm:
            today = _today_key(now)
            last = _read_last_run(state_file)
            if last != today:
                print(f"\n[TRIGGER] {today} {cfg.hhmm}")
                _acquire_lock(lock_file)
                try:
                    run_pipeline_once(root)
                    _write_last_run(state_file, today)
                except Exception as e:
                    print(f"[ERROR] pipeline failed: {e}")
                finally:
                    _release_lock(lock_file)

            # 避免在同一分钟内多次触发
            time.sleep(65)
            continue

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()