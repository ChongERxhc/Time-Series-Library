from __future__ import annotations

import argparse
import csv
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
RESULTS_FILE = WORK_DIR / "results_all.tsv"
DEFAULT_HORIZON = 96

RESULT_FIELDNAMES = [
    "utc_iso",
    "commit",
    "feature_set_name",
    "feature_fp",
    "n_features",
    "input_dim",
    "eval_dims",
    "pred_len",
    "baseline_val_mse",
    "val_mse",
    "val_improve",
    "score",
    "best_score_before",
    "status",
    "test_mse",
    "test_mae",
    "dirty",
    "description",
    "log_file",
    "staging_csv",
]


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
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = f.readline().strip().split("\t")
        if existing != fieldnames:
            raise RuntimeError(
                f"{path} has legacy columns; rename or archive it before using the new schema. "
                f"Expected header starting with: {fieldnames[:6]}..."
            )
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Append weather autoresearch result row.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE)
    parser.add_argument(
        "--min-improve",
        type=float,
        default=0.0,
        help="Minimum score margin over historical best to mark keep (default 0: any strict improvement).",
    )
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

    row = {
        "utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "feature_set_name": parsed.get("feature_set_name", ""),
        "feature_fp": parsed.get("feature_fp", ""),
        "n_features": parsed.get("n_features", ""),
        "input_dim": parsed.get("input_dim", ""),
        "eval_dims": parsed.get("eval_dims", ""),
        "pred_len": parsed.get("pred_len", str(DEFAULT_HORIZON)),
        "baseline_val_mse": parsed.get("baseline_val_mse", ""),
        "val_mse": parsed.get("val_mse", ""),
        "val_improve": parsed.get("val_improve", ""),
        "score": parsed.get("score", ""),
        "best_score_before": "" if best_before == float("-inf") else f"{best_before:.10g}",
        "status": status,
        "test_mse": parsed.get("test_mse", ""),
        "test_mae": parsed.get("test_mae", ""),
        "dirty": dirty,
        "description": args.description,
        "log_file": str(args.log),
        "staging_csv": parsed.get("staging_csv", ""),
    }

    ensure_header(args.results, RESULT_FIELDNAMES)
    with args.results.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, delimiter="\t")
        writer.writerow(row)

    print(f"appended: {args.results}")
    print(f"status: {status}")
    print(f"score: {row['score']}")


if __name__ == "__main__":
    main()
