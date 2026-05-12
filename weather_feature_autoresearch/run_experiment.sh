#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DESC="${1:-trial $(date -u +%Y%m%dT%H%M%SZ)}"
if [ "${#}" -ge 1 ]; then
  shift
fi

PREPARE_ARGS=("$@")
mkdir -p logs
TS="logs/experiment_$(date -u +%Y%m%dT%H%M%SZ).log"

# 必须在 orphan 实验分支上运行（与 main/master 无共同祖先）。排障：WEATHER_AR_SKIP_ORPHAN_CHECK=1
if [ -z "${WEATHER_AR_SKIP_ORPHAN_CHECK:-}" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  for base in main master; do
    if git show-ref --verify --quiet "refs/heads/$base" 2>/dev/null; then
      if git merge-base HEAD "$base" >/dev/null 2>&1; then
        echo "weather_feature_autoresearch: 当前分支与 '$base' 存在共同祖先，不符合要求。" >&2
        echo "请先在 main 上执行: git checkout --orphan <唯一分支名>，再 git add -A && git commit，然后在本分支跑 trial。" >&2
        echo "（仅排障可设 WEATHER_AR_SKIP_ORPHAN_CHECK=1）" >&2
        exit 1
      fi
    fi
  done
fi

if git rev-parse --git-dir >/dev/null 2>&1; then
  git add dig.py 2>/dev/null || true
  if git diff --cached --quiet -- dig.py 2>/dev/null; then
    git commit --allow-empty -m "exp(weather-feature): ${DESC}"
  else
    git commit -m "exp(weather-feature): ${DESC}"
  fi
else
  echo "current directory is not a git repo; skip commit" >&2
fi

PYTHON="${PYTHON:-python}"
set +e
"$PYTHON" prepare.py --mode trial --log-file "$TS" "${PREPARE_ARGS[@]}"
EC=$?
set -e

"$PYTHON" append_results_all.py --log "$TS" --description "$DESC"
exit "$EC"
