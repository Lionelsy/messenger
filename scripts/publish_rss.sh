#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST_DIR="${ROOT_DIR}/messenger"

if [[ ! -d "${DEST_DIR}" ]]; then
  echo "[ERR] missing ${DEST_DIR}"
  exit 1
fi

HAS_CHANGES=false

for pair in "arxiv.rss:DailyPapers.rss" "SpatialAI.rss:SpatialAI.rss" "Huggingface.rss:Huggingface.rss"; do
  SRC="${ROOT_DIR}/${pair%%:*}"
  DST="${DEST_DIR}/${pair##*:}"

  if [[ ! -f "${SRC}" ]]; then
    echo "[WARN] missing ${SRC}, skip"
    continue
  fi

  echo "[STEP] copy ${SRC} -> ${DST}"
  cp -f "${SRC}" "${DST}"
  HAS_CHANGES=true
done

if [[ "${HAS_CHANGES}" != "true" ]]; then
  echo "[WARN] no rss files to push"
  exit 0
fi

echo "[STEP] git add/commit/push in ${DEST_DIR}"
cd "${DEST_DIR}"

git add DailyPapers.rss SpatialAI.rss Huggingface.rss

# 没有变更就直接退出（避免空提交报错）
if git diff --cached --quiet; then
  echo "[OK] no changes, skip commit/push"
  exit 0
fi

MSG="Update RSS feeds $(date +%F\ %T)"
git commit -m "${MSG}"
git push

echo "[OK] pushed"

