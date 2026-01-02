#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p storage/logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="storage/logs/bash_test-${TS}.log"

echo "[START] $(date -Is)" | tee -a "$LOG_FILE"
echo "[CWD] $ROOT_DIR" | tee -a "$LOG_FILE"
echo "[PY]  $(command -v python)" | tee -a "$LOG_FILE"
echo "[LOG] $LOG_FILE" | tee -a "$LOG_FILE"

run_step () {
  local name="$1"
  shift
  echo -e "\n[RUN] ${name}: $*" | tee -a "$LOG_FILE"
  "$@" >>"$LOG_FILE" 2>&1
  echo "[OK]  ${name}" | tee -a "$LOG_FILE"
}

# 如果你想在这里加载 .env（需要你自己先创建好 .env），取消下面两行注释：
set -a; source .env; set +a

run_step "fetch_arxiv"              python pipeline/fetch_arxiv.py
run_step "fetch_hf_daily"           python pipeline/fetch_hf_daily.py
run_step "update_paper_list"        python pipeline/update_paper_list.py
run_step "analyze_01_base"          python pipeline/analyze_01_base.py
run_step "analyze_02_parse"         python pipeline/analyze_02_parse.py
run_step "analyze_03_deep"          python pipeline/analyze_03_deep.py
run_step "publish_add_new_items"    python pipeline/publish_add_new_items.py
run_step "publish_delete_old_items" python pipeline/publish_delete_old_items.py

echo -e "\n[END] $(date -Is)" | tee -a "$LOG_FILE"