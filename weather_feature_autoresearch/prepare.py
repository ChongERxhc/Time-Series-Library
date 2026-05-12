from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "weather"
GENERATED_DIR = WORK_DIR / "generated_data"
BASELINES_FILE = WORK_DIR / "baselines.json"
SENTINEL_THRESHOLD = -1000
ORIGINAL_DIMS = 21
DEFAULT_HORIZONS = [96, 192, 336, 720]


class _TeeTextStream:
    """Mirror writes to multiple text streams (e.g. console + session log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


def resolve_log_file_path(raw: str | None, mode: str) -> Path | None:
    if raw is None:
        return None
    if raw.strip().lower() == "auto":
        log_dir = WORK_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return log_dir / f"{mode}_{ts}.log"
    return Path(raw).expanduser().resolve()


def clean_sensor_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    numeric_cols = [col for col in clean_df.columns if col != "date"]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df.loc[clean_df[col] <= SENTINEL_THRESHOLD, col] = np.nan
        clean_df[col] = clean_df[col].ffill().bfill()
    return clean_df


def load_dig_module():
    dig_path = WORK_DIR / "dig.py"
    spec = importlib.util.spec_from_file_location("weather_feature_dig", dig_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {dig_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_features(raw_df: pd.DataFrame, features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("compute_features must return a pandas DataFrame")
    if len(features) != len(raw_df):
        raise ValueError(f"feature row count mismatch: {len(features)} != {len(raw_df)}")
    if features.empty:
        raise ValueError("feature DataFrame is empty")
    if len(features.columns) > 64:
        raise ValueError("too many features; keep each trial focused")
    bad_names = [col for col in features.columns if not col.startswith("llm_")]
    if bad_names:
        raise ValueError(f"feature names must start with llm_: {bad_names}")
    overlap = set(features.columns).intersection(set(raw_df.columns))
    if overlap:
        raise ValueError(f"feature columns overlap raw columns: {sorted(overlap)}")
    if features.columns.duplicated().any():
        raise ValueError("duplicated feature columns")

    numeric = features.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"features contain NaN or non-numeric values: {bad_cols}")
    arr = numeric.to_numpy(dtype="float64")
    if not np.isfinite(arr).all():
        raise ValueError("features contain inf values")


def feature_fingerprint(feature_set_name: str, columns: list[str]) -> str:
    payload = json.dumps({"name": feature_set_name, "columns": columns}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def build_trial_dataset() -> tuple[str, Path, int]:
    dig = load_dig_module()
    feature_set_name = getattr(dig, "FEATURE_SET_NAME", "unnamed_feature_set")
    raw_df = pd.read_csv(DATA_DIR / "weather.csv")
    clean_df = clean_sensor_sentinels(raw_df)
    features = dig.compute_features(raw_df=raw_df.copy(), clean_df=clean_df)
    validate_features(raw_df, features)

    features = features.reset_index(drop=True)
    trial_df = pd.concat([raw_df.reset_index(drop=True), features], axis=1)
    input_dim = len([col for col in trial_df.columns if col != "date"])

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    fp = feature_fingerprint(feature_set_name, list(features.columns))
    output_path = GENERATED_DIR / f"{feature_set_name}_{fp}.csv"
    trial_df.to_csv(output_path, index=False)
    return feature_set_name, output_path, input_dim


def run_tslib_experiment(
    *,
    data_path: str,
    model_id: str,
    input_dim: int,
    pred_len: int,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    eval_dims: int,
    model: str,
    cuda_visible_devices: str,
    des: str,
) -> tuple[float, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(DATA_DIR) + "/",
        "--data_path",
        data_path,
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        "custom",
        "--features",
        "M",
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--e_layers",
        "3",
        "--d_layers",
        "1",
        "--factor",
        "3",
        "--enc_in",
        str(input_dim),
        "--dec_in",
        str(input_dim),
        "--c_out",
        str(input_dim),
        "--d_model",
        "512",
        "--d_ff",
        "512",
        "--des",
        des,
        "--train_epochs",
        str(train_epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--itr",
        "1",
    ]
    if eval_dims > 0:
        cmd.extend(["--eval_dims", str(eval_dims)])

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"TSLib run failed for pred_len={pred_len} with code {proc.returncode}")

    match = re.search(r"mse:([^,\s]+),\s*mae:([^,\s]+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Cannot parse mse/mae for pred_len={pred_len}")
    return float(match.group(1)), float(match.group(2))


def load_baselines(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_baselines(path: Path, baselines: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
        f.write("\n")


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype="float64")))


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def run_baseline(args: argparse.Namespace) -> None:
    baselines = load_baselines(args.baselines)
    for horizon in args.horizons:
        mse, mae = run_tslib_experiment(
            data_path="weather.csv",
            model_id=f"weather_{args.model}_baseline_pl{horizon}_ep{args.train_epochs}",
            input_dim=ORIGINAL_DIMS,
            pred_len=horizon,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_dims=0,
            model=args.model,
            cuda_visible_devices=args.cuda_visible_devices,
            des=f"AutoResearchBaseline_pl{horizon}",
        )
        baselines[str(horizon)] = {"mse": mse, "mae": mae}
        print_kv(f"baseline_mse_{horizon}", mse)
        print_kv(f"baseline_mae_{horizon}", mae)
    save_baselines(args.baselines, baselines)
    print_kv("baseline_file", args.baselines)


def run_trial(args: argparse.Namespace) -> None:
    baselines = load_baselines(args.baselines)
    missing = [h for h in args.horizons if str(h) not in baselines]
    if missing:
        raise RuntimeError(f"missing baselines for horizons {missing}; run --mode baseline first")

    feature_set_name, dataset_path, input_dim = build_trial_dataset()
    rel_data_path = os.path.relpath(dataset_path, DATA_DIR)
    improvements: list[float] = []
    mae_improvements: list[float] = []

    print_kv("feature_set_name", feature_set_name)
    print_kv("generated_data", dataset_path)
    print_kv("input_dim", input_dim)
    print_kv("eval_dims", ORIGINAL_DIMS)

    for horizon in args.horizons:
        mse, mae = run_tslib_experiment(
            data_path=rel_data_path,
            model_id=f"weather_{args.model}_{feature_set_name}_pl{horizon}_ep{args.train_epochs}",
            input_dim=input_dim,
            pred_len=horizon,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_dims=ORIGINAL_DIMS,
            model=args.model,
            cuda_visible_devices=args.cuda_visible_devices,
            des=f"AutoResearchTrial_{feature_set_name}_pl{horizon}",
        )
        base = baselines[str(horizon)]
        mse_improve = (base["mse"] - mse) / base["mse"]
        mae_improve = (base["mae"] - mae) / base["mae"]
        improvements.append(float(mse_improve))
        mae_improvements.append(float(mae_improve))

        print_kv(f"mse_{horizon}", mse)
        print_kv(f"mae_{horizon}", mae)
        print_kv(f"mse_improve_{horizon}", mse_improve)
        print_kv(f"mae_improve_{horizon}", mae_improve)

    score = median(improvements)
    avg_mse_improve = float(np.mean(improvements))
    avg_mae_improve = float(np.mean(mae_improvements))
    positive_horizons = int(sum(v > 0 for v in improvements))

    print_kv("score", score)
    print_kv("avg_mse_improve", avg_mse_improve)
    print_kv("avg_mae_improve", avg_mae_improve)
    print_kv("positive_horizons", positive_horizons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather feature autoresearch runner.")
    parser.add_argument("--mode", choices=["baseline", "trial"], default="trial")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--baselines", type=Path, default=BASELINES_FILE)
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write full session stdout/stderr to this UTF-8 file (still prints to console). "
        "Use the literal value 'auto' for logs/<mode>_<UTC>.log under weather_feature_autoresearch/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = resolve_log_file_path(args.log_file, args.mode)
    log_fp = None
    saved_out, saved_err = sys.stdout, sys.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("w", encoding="utf-8")
        log_fp.write("# weather_feature_autoresearch session log\n")
        log_fp.write(f"# utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        log_fp.write(f"# mode: {args.mode}\n")
        log_fp.write("# ---\n")
        log_fp.flush()
        sys.stdout = _TeeTextStream(saved_out, log_fp)
        sys.stderr = _TeeTextStream(saved_err, log_fp)
        print_kv("session_log", str(log_path))
    try:
        if args.mode == "baseline":
            run_baseline(args)
        else:
            run_trial(args)
    finally:
        if log_fp is not None:
            sys.stdout = saved_out
            sys.stderr = saved_err
            log_fp.close()
            print(f"session_log: {log_path}", file=saved_err)


if __name__ == "__main__":
    main()
