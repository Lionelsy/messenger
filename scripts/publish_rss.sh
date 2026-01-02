#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_RSS="${ROOT_DIR}/arxiv.rss"
DEST_DIR="${ROOT_DIR}/Messager"
DEST_RSS="${DEST_DIR}/arxiv.rss"

if [[ ! -f "${SRC_RSS}" ]]; then
  echo "[ERR] missing ${SRC_RSS}"
  exit 1
fi
if [[ ! -d "${DEST_DIR}" ]]; then
  echo "[ERR] missing ${DEST_DIR}"
  exit 1
fi

echo "[STEP] copy rss -> ${DEST_RSS}"
cp -f "${SRC_RSS}" "${DEST_RSS}"

echo "[STEP] git add/commit/push in ${DEST_DIR}"
cd "${DEST_DIR}"

git add arxiv.rss

# 没有变更就直接退出（避免空提交报错）
if git diff --cached --quiet; then
  echo "[OK] no changes, skip commit/push"
  exit 0
fi

MSG="Update arxiv.rss $(date +%F\ %T)"
git commit -m "${MSG}"
git push

echo "[OK] pushed"

