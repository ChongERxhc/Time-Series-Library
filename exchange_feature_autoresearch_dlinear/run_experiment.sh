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

if [ -z "${WEATHER_AR_SKIP_ORPHAN_CHECK:-}" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  current="$(git branch --show-current 2>/dev/null || true)"
  case "$current" in
    main|master)
      echo "exchange_feature_autoresearch_dlinear: 当前在 '$current'，禁止在此跑 trial。" >&2
      exit 1
      ;;
  esac
  for base in main master; do
    if git show-ref --verify --quiet "refs/heads/$base" 2>/dev/null; then
      if git merge-base HEAD "$base" >/dev/null 2>&1; then
        echo "exchange_feature_autoresearch_dlinear: 当前分支与 '$base' 存在共同祖先，不符合 orphan 实验线。" >&2
        exit 1
      fi
    fi
  done
fi

if git rev-parse --git-dir >/dev/null 2>&1; then
  git add dig.py 2>/dev/null || true
  if git diff --cached --quiet -- dig.py 2>/dev/null; then
    git commit --allow-empty -m "exp(exchange-dlinear): ${DESC}"
  else
    git commit -m "exp(exchange-dlinear): ${DESC}"
  fi
fi

PYTHON="${PYTHON:-python}"
set +e
"$PYTHON" prepare.py --mode trial --log-file "$TS" "${PREPARE_ARGS[@]}"
EC=$?
set -e

"$PYTHON" append_results_all.py --log "$TS" --description "$DESC"
exit $EC
