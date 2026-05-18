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

# 须在 orphan 实验分支上运行（非 main/master，且与 main/master 无共同祖先）。排障：WEATHER_AR_SKIP_ORPHAN_CHECK=1
if [ -z "${WEATHER_AR_SKIP_ORPHAN_CHECK:-}" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  current="$(git branch --show-current 2>/dev/null || true)"
  case "$current" in
    main|master)
      echo "weather_feature_autoresearch: 当前在 '$current'，禁止在此跑 trial。" >&2
      echo "请先 checkout autoresearch（或上一实验分支），再按 program.md 创建 auto/<tag> orphan；" >&2
      echo "创建 orphan 前不要先 checkout main/master。" >&2
      echo "（仅排障可设 WEATHER_AR_SKIP_ORPHAN_CHECK=1）" >&2
      exit 1
      ;;
  esac
  for base in main master; do
    if git show-ref --verify --quiet "refs/heads/$base" 2>/dev/null; then
      if git merge-base HEAD "$base" >/dev/null 2>&1; then
        echo "weather_feature_autoresearch: 当前分支 '${current:-HEAD}' 与 '$base' 存在共同祖先，不符合 orphan 实验线。" >&2
        echo "请按 program.md Phase A：在 autoresearch（勿先切 main）上执行 git checkout --orphan auto/<tag>，再跑 baseline/trial。" >&2
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
