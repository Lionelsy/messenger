#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/storage/logs"
mkdir -p "$LOG_DIR"

export TZ="Asia/Shanghai"

DATE_STR="$(date +%Y-%m-%d)"
RUN_PUBDATE="$(date +"%a, %d %b %Y %H:%M:%S %z")"
LOG_FILE="$LOG_DIR/run_pipeline-$(date +%Y%m%d-%H%M%S).log"

echo "[LOG] $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[START] $(date -Is)"
echo "[DATE]  $DATE_STR"
echo "[PUBDT] $RUN_PUBDATE"
echo "[ROOT]  $ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[FATAL] venv python not found or not executable: $PY"
  echo "[HINT] Did you create the uv venv in this project directory?"
  exit 1
fi

set -a; source .env; set +a

"$PY" -V
"$PY" -c "import sys; print('[INFO] exe:', sys.executable)"

"$PY" "$ROOT/pipeline/fetch_arxiv.py" --date "$DATE_STR"
"$PY" "$ROOT/pipeline/fetch_hf_daily.py" --date "$DATE_STR"
"$PY" "$ROOT/pipeline/update_paper_list.py" --date "$DATE_STR"

"$PY" "$ROOT/pipeline/analyze_01_base_pro.py"
"$PY" "$ROOT/pipeline/analyze_02_parse_pro.py"
"$PY" "$ROOT/pipeline/analyze_03_deep_pro.py"

"$PY" "$ROOT/pipeline/publish_add_new_items.py" --run_pubdate "$RUN_PUBDATE"
"$PY" "$ROOT/pipeline/publish_delete_old_items.py" --now "$RUN_PUBDATE"

bash "$ROOT/scripts/publish_rss.sh"

echo "[END] $(date -Is)"
