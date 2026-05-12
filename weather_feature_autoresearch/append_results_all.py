from __future__ import annotations

import argparse
import csv
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
RESULTS_FILE = WORK_DIR / "results_all.tsv"
DEFAULT_HORIZONS = [96, 192, 336, 720]


def parse_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


def git_output(args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(args, cwd=WORK_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def best_score_before(results_path: Path) -> float:
    if not results_path.exists():
        return float("-inf")
    best = float("-inf")
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score = parse_float(row.get("score"))
            if math.isfinite(score):
                best = max(best, score)
    return best


def ensure_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Append weather autoresearch result row.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE)
    parser.add_argument("--min-improve", type=float, default=0.001)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    args = parser.parse_args()

    parsed = parse_log(args.log)
    score = parse_float(parsed.get("score"))
    best_before = best_score_before(args.results)
    if not math.isfinite(score):
        status = "crash"
    elif score > best_before + args.min_improve:
        status = "keep"
    else:
        status = "discard"

    commit = git_output(["git", "rev-parse", "HEAD"])
    dirty = "1" if git_output(["git", "status", "--porcelain"]) else "0"

    fieldnames = [
        "utc_iso",
        "commit",
        "feature_set_name",
        "input_dim",
        "eval_dims",
        "score",
        "best_score_before",
        "status",
        "positive_horizons",
        "avg_mse_improve",
        "avg_mae_improve",
    ]
    for horizon in args.horizons:
        fieldnames.extend([
            f"mse_{horizon}",
            f"mae_{horizon}",
            f"mse_improve_{horizon}",
            f"mae_improve_{horizon}",
        ])
    fieldnames.extend(["dirty", "description", "log_file", "generated_data"])

    row = {
        "utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "feature_set_name": parsed.get("feature_set_name", ""),
        "input_dim": parsed.get("input_dim", ""),
        "eval_dims": parsed.get("eval_dims", ""),
        "score": parsed.get("score", ""),
        "best_score_before": "" if best_before == float("-inf") else f"{best_before:.10g}",
        "status": status,
        "positive_horizons": parsed.get("positive_horizons", ""),
        "avg_mse_improve": parsed.get("avg_mse_improve", ""),
        "avg_mae_improve": parsed.get("avg_mae_improve", ""),
        "dirty": dirty,
        "description": args.description,
        "log_file": str(args.log),
        "generated_data": parsed.get("generated_data", ""),
    }
    for horizon in args.horizons:
        for key in ("mse", "mae", "mse_improve", "mae_improve"):
            row[f"{key}_{horizon}"] = parsed.get(f"{key}_{horizon}", "")

    ensure_header(args.results, fieldnames)
    with args.results.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)

    print(f"appended: {args.results}")
    print(f"status: {status}")
    print(f"score: {row['score']}")


if __name__ == "__main__":
    main()
