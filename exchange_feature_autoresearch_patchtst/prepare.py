from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "exchange_rate"
RAW_CSV = "exchange_rate.csv"
DATASET_PREFIX = "exchange"
DATA_FREQ = "d"
GENERATED_DIR = WORK_DIR / "generated_data"
ARCHIVE_DIR = GENERATED_DIR / "archive"
STAGING_CSV = GENERATED_DIR / "_staging.csv"
BASELINES_FILE = WORK_DIR / "baselines.json"
SENTINEL_THRESHOLD = -1000
ORIGINAL_DIMS = 8
DEFAULT_HORIZON = 96
DEFAULT_HORIZONS = [DEFAULT_HORIZON]
DEFAULT_MODEL = "PatchTST"
DEFAULT_TRAIN_EPOCHS = 10

MODEL_RUN_KWARGS: dict[str, dict[str, int]] = {
    "DLinear": {
        "e_layers": 2,
        "n_heads": 8,
        "d_model": 512,
        "d_ff": 2048,
        "d_layers": 1,
        "factor": 3,
    },
    "PatchTST": {
        "e_layers": 2,
        "n_heads": 4,
        "d_model": 512,
        "d_ff": 2048,
        "d_layers": 1,
        "factor": 3,
    },
    "iTransformer": {
        "e_layers": 3,
        "n_heads": 8,
        "d_model": 512,
        "d_ff": 512,
        "d_layers": 1,
        "factor": 3,
    },
}


def get_model_run_kwargs(model: str) -> dict[str, int]:
    if model not in MODEL_RUN_KWARGS:
        raise ValueError(
            f"unsupported model {model!r}; add an entry to MODEL_RUN_KWARGS in prepare.py"
        )
    return dict(MODEL_RUN_KWARGS[model])

EPOCH_VALI_RE = re.compile(
    r"Epoch:\s*\d+,\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*[\d.eE+-]+\s*"
    r"Vali Loss:\s*([\d.eE+-]+)\s*Test Loss:",
)
TEST_METRICS_RE = re.compile(r"mse:([^,\s]+),\s*mae:([^,\s]+)")


@dataclass
class TrialBuildResult:
    feature_set_name: str
    feature_fp: str
    n_features: int
    input_dim: int
    staging_path: Path
    rel_data_path: str


class _TeeTextStream:
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
    spec = importlib.util.spec_from_file_location("exchange_feature_dig", dig_path)
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


def build_trial_dataset_staging() -> TrialBuildResult:
    dig = load_dig_module()
    feature_set_name = getattr(dig, "FEATURE_SET_NAME", "unnamed_feature_set")
    raw_df = pd.read_csv(DATA_DIR / RAW_CSV)
    clean_df = clean_sensor_sentinels(raw_df)
    features = dig.compute_features(raw_df=raw_df.copy(), clean_df=clean_df)
    validate_features(raw_df, features)
    features = features.reset_index(drop=True)
    trial_df = pd.concat([raw_df.reset_index(drop=True), features], axis=1)
    input_dim = len([col for col in trial_df.columns if col != "date"])
    fp = feature_fingerprint(feature_set_name, list(features.columns))
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(STAGING_CSV, index=False)
    rel_data_path = os.path.relpath(STAGING_CSV, DATA_DIR)
    return TrialBuildResult(
        feature_set_name=feature_set_name,
        feature_fp=fp,
        n_features=len(features.columns),
        input_dim=input_dim,
        staging_path=STAGING_CSV,
        rel_data_path=rel_data_path,
    )


def remove_staging(staging_path: Path) -> None:
    if staging_path.exists():
        staging_path.unlink()


def archive_staging(meta: TrialBuildResult) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{meta.feature_set_name}_{meta.feature_fp}.csv"
    shutil.copy2(meta.staging_path, dest)
    return dest


def parse_best_val_mse(stdout: str) -> float:
    losses = [float(m.group(1)) for m in EPOCH_VALI_RE.finditer(stdout)]
    if not losses:
        raise RuntimeError(
            "Cannot parse Vali Loss from TSLib log; expected Epoch lines with Vali Loss"
        )
    return float(min(losses))


def parse_test_metrics(stdout: str) -> tuple[float | None, float | None]:
    match = TEST_METRICS_RE.search(stdout)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


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
    model_run_kwargs: dict[str, int],
) -> str:
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
        "--freq",
        DATA_FREQ,
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--e_layers",
        str(model_run_kwargs["e_layers"]),
        "--d_layers",
        str(model_run_kwargs["d_layers"]),
        "--factor",
        str(model_run_kwargs["factor"]),
        "--n_heads",
        str(model_run_kwargs["n_heads"]),
        "--enc_in",
        str(input_dim),
        "--dec_in",
        str(input_dim),
        "--c_out",
        str(input_dim),
        "--d_model",
        str(model_run_kwargs["d_model"]),
        "--d_ff",
        str(model_run_kwargs["d_ff"]),
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
    return proc.stdout


def load_baselines(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "val_mse" in entry:
            out[key] = entry
        elif "mse" in entry:
            raise RuntimeError(
                f"baselines.json[{key}] uses legacy test 'mse'; rerun baseline"
            )
        else:
            raise RuntimeError(f"invalid baseline entry for horizon {key}: {entry}")
    return out


def save_baselines(path: Path, baselines: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
        f.write("\n")


def baseline_val_mse(baselines: dict[str, dict[str, float]], horizon: int) -> float:
    key = str(horizon)
    if key not in baselines:
        raise KeyError(key)
    return float(baselines[key]["val_mse"])


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def run_horizon_experiment(
    args: argparse.Namespace,
    *,
    horizon: int,
    data_path: str,
    input_dim: int,
    eval_dims: int,
    model_id: str,
    des: str,
    model_run_kwargs: dict[str, int],
) -> tuple[float, float | None, float | None]:
    stdout = run_tslib_experiment(
        data_path=data_path,
        model_id=model_id,
        input_dim=input_dim,
        pred_len=horizon,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_dims=eval_dims,
        model=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
        des=des,
        model_run_kwargs=model_run_kwargs,
    )
    val_mse = parse_best_val_mse(stdout)
    test_mse, test_mae = parse_test_metrics(stdout)
    return val_mse, test_mse, test_mae


def run_baseline(args: argparse.Namespace) -> None:
    model_run_kwargs = get_model_run_kwargs(args.model)
    baselines: dict[str, dict[str, float]] = {}
    for horizon in args.horizons:
        val_mse, test_mse, test_mae = run_horizon_experiment(
            args,
            horizon=horizon,
            data_path=RAW_CSV,
            input_dim=ORIGINAL_DIMS,
            eval_dims=0,
            model_id=f"{DATASET_PREFIX}_{args.model}_baseline_pl{horizon}_ep{args.train_epochs}",
            des=f"AutoResearchBaseline_pl{horizon}",
            model_run_kwargs=model_run_kwargs,
        )
        entry: dict[str, float] = {"val_mse": val_mse}
        if test_mse is not None:
            entry["test_mse"] = test_mse
        if test_mae is not None:
            entry["test_mae"] = test_mae
        baselines[str(horizon)] = entry
        print_kv(f"baseline_val_mse_{horizon}", val_mse)
        if test_mse is not None:
            print_kv(f"baseline_test_mse_{horizon}", test_mse)
        if test_mae is not None:
            print_kv(f"baseline_test_mae_{horizon}", test_mae)
    save_baselines(args.baselines, baselines)
    print_kv("baseline_file", args.baselines)
    print_kv("pred_len", args.horizons[0] if len(args.horizons) == 1 else args.horizons)


def run_trial(args: argparse.Namespace) -> None:
    baselines = load_baselines(args.baselines)
    horizon = args.horizons[0]
    if len(args.horizons) != 1:
        raise RuntimeError("trial mode expects a single horizon for scoring")
    missing = [h for h in args.horizons if str(h) not in baselines]
    if missing:
        raise RuntimeError(f"missing baselines for horizons {missing}; run --mode baseline first")
    model_run_kwargs = get_model_run_kwargs(args.model)
    meta = build_trial_dataset_staging()
    print_kv("model", args.model)
    print_kv("feature_set_name", meta.feature_set_name)
    print_kv("feature_fp", meta.feature_fp)
    print_kv("n_features", meta.n_features)
    print_kv("input_dim", meta.input_dim)
    print_kv("eval_dims", ORIGINAL_DIMS)
    print_kv("staging_csv", meta.rel_data_path)
    print_kv("pred_len", horizon)
    try:
        val_mse, test_mse, test_mae = run_horizon_experiment(
            args,
            horizon=horizon,
            data_path=meta.rel_data_path,
            input_dim=meta.input_dim,
            eval_dims=ORIGINAL_DIMS,
            model_id=f"{DATASET_PREFIX}_{args.model}_{meta.feature_set_name}_pl{horizon}_ep{args.train_epochs}",
            des=f"AutoResearchTrial_{meta.feature_set_name}_pl{horizon}",
            model_run_kwargs=model_run_kwargs,
        )
    finally:
        if not args.keep_staging:
            remove_staging(meta.staging_path)
    base_val = baseline_val_mse(baselines, horizon)
    val_improve = (base_val - val_mse) / base_val
    score = float(val_improve)
    print_kv("baseline_val_mse", base_val)
    print_kv("val_mse", val_mse)
    print_kv("val_improve", val_improve)
    if test_mse is not None:
        print_kv("test_mse", test_mse)
    if test_mae is not None:
        print_kv("test_mae", test_mae)
    print_kv("score", score)
    if args.keep_staging and meta.staging_path.exists():
        archived = archive_staging(meta)
        print_kv("archived_csv", os.path.relpath(archived, WORK_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exchange feature autoresearch runner (PatchTST line)."
    )
    parser.add_argument("--mode", choices=["baseline", "trial"], default="trial")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--baselines", type=Path, default=BASELINES_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--log-file", default=None, metavar="PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = resolve_log_file_path(args.log_file, args.mode)
    log_fp = None
    saved_out, saved_err = sys.stdout, sys.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("w", encoding="utf-8")
        log_fp.write("# exchange_feature_autoresearch_patchtst session log\n")
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
