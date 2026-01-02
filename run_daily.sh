#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Paths & env
# -----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/storage/logs"
mkdir -p "$LOG_DIR"

# RSS 使用 UTC+8
export TZ="Asia/Shanghai"

# -----------------------------
# Shared timestamps (one run)
# -----------------------------
DATE_STR="$(date +%Y-%m-%d)"
RUN_PUBDATE="$(date +"%a, %d %b %Y %H:%M:%S %z")"
LOG_FILE="$LOG_DIR/run_pipeline-$(date +%Y%m%d-%H%M%S).log"

echo "[LOG] $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[START] $(date -Is)"
echo "[DATE]  $DATE_STR"
echo "[PUBDT] $RUN_PUBDATE"
echo "[ROOT]  $ROOT"

# -----------------------------
# Pipeline
# -----------------------------
python "$ROOT/pipeline/fetch_arxiv.py" --date "$DATE_STR"
python "$ROOT/pipeline/fetch_hf_daily.py" --date "$DATE_STR"
python "$ROOT/pipeline/update_paper_list.py" --date "$DATE_STR"

python "$ROOT/pipeline/analyze_01_base.py"
python "$ROOT/pipeline/analyze_02_parse.py"
python "$ROOT/pipeline/analyze_03_deep.py"

python "$ROOT/pipeline/publish_add_new_items.py" \
  --run_pubdate "$RUN_PUBDATE"

python "$ROOT/pipeline/publish_delete_old_items.py" \
  --now "$RUN_PUBDATE"

bash "$ROOT/scripts/publish_rss.sh"

echo "[END] $(date -Is)"
