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

# 本地局域网服务（LLM/MinerU）请求不绕道系统代理（若 shell 设置了 http_proxy）
# 保留用户已有的大写和小写排除项
export NO_PROXY="192.168.10.2,127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}${no_proxy:+,$no_proxy}"
export no_proxy="$NO_PROXY"

# --- 服务预检：LLM/OCR 配置不可用时提前失败，避免整夜空转后静默零产出 ---
if [[ "${LLM_PROVIDER:-zhipu}" == "openai_compat" && -n "${LLM_BASE_URL:-}" ]]; then
  # /v1/models 在配置了 key 的端点上返回 401（需鉴权），因此带上 Bearer 头
  _LLM_AUTH=()
  if [[ -n "${LLM_API_KEY:-}" ]]; then
    _LLM_AUTH=(-H "Authorization: Bearer ${LLM_API_KEY}")
  fi
  if ! curl -sf --connect-timeout 10 --max-time 20 "${_LLM_AUTH[@]}" "${LLM_BASE_URL%/}/models" > /dev/null 2>&1; then
    echo "[FATAL] LLM 服务不可达: ${LLM_BASE_URL%/}/models（curl 失败）"
    echo "[HINT] 请先启动 LLM 服务（vLLM 等），或检查 LLM_BASE_URL 配置"
    exit 1
  fi
  echo "[INFO] LLM 预检通过: ${LLM_BASE_URL%/}/models"
fi
if [[ "${OCR_PROVIDER:-cloud}" == "cloud" && -z "${MINERU_KEY:-}" ]]; then
  echo "[FATAL] OCR_PROVIDER=cloud 但 MINERU_KEY 为空，请检查 .env"
  exit 1
fi

"$PY" -V
"$PY" -c "import sys; print('[INFO] exe:', sys.executable)"

"$PY" "$ROOT/pipeline/fetch_arxiv.py" --date "$DATE_STR"
"$PY" "$ROOT/pipeline/fetch_hf_daily.py" --date "$DATE_STR"
"$PY" "$ROOT/pipeline/update_paper_list.py" --date "$DATE_STR"

# curl -s http://192.168.31.125:5050/llm/start && sleep 600
"$PY" "$ROOT/pipeline/analyze_01_base_pro.py"
# curl -s http://192.168.31.125:5050/llm/stop && sleep 60

# curl -s http://192.168.31.125:5050/mineru/start && sleep 60
"$PY" "$ROOT/pipeline/analyze_02_parse_pro.py"
# curl -s http://192.168.31.125:5050/mineru/stop && sleep 60

# curl -s http://192.168.31.125:5050/llm/start && sleep 600
"$PY" "$ROOT/pipeline/analyze_03_deep_pro.py"
# curl -s http://192.168.31.125:5050/llm/stop && sleep 60

"$PY" "$ROOT/pipeline/publish_add_new_items.py" --run_pubdate "$RUN_PUBDATE"
"$PY" "$ROOT/pipeline/publish_delete_old_items.py" --now "$RUN_PUBDATE"

bash "$ROOT/scripts/publish_rss.sh"

echo "[END] $(date -Is)"
