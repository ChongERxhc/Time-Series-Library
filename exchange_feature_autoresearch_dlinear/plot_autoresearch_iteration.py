#!/usr/bin/env python3
"""绘制 AutoResearch 迭代曲线：val / test 分图（独立 y 轴尺度）+ baseline 水平线。

用法（仓库根或本目录）:
    conda activate tslib
    python exchange_feature_autoresearch_dlinear/plot_autoresearch_iteration.py

默认输出两张图（由 --output 推导）:
    figures/autoresearch_iteration_val.png
    figures/autoresearch_iteration_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = WORK_DIR / "results_all.tsv"
DEFAULT_BASELINES = WORK_DIR / "baselines.json"
DEFAULT_OUTPUT = WORK_DIR / "figures" / "autoresearch_iteration.png"
DEFAULT_HORIZON = 96


def load_baselines(path: Path, horizon: int) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    entry = raw[str(horizon)]
    out = {"val_mse": float(entry["val_mse"])}
    if "test_mse" in entry:
        out["test_mse"] = float(entry["test_mse"])
    if "test_mae" in entry:
        out["test_mae"] = float(entry["test_mae"])
    return out


def load_trials(path: Path, baseline_val_mse: float, rtol: float = 0.02) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["status"].fillna("") != "crash"].copy()
    for col in ("val_mse", "test_mse", "baseline_val_mse", "score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["val_mse"])
    df = df[np.isfinite(df["val_mse"])]

    # 排除 baseline 口径不一致的 trial（如误用 DLinear baseline）
    if "baseline_val_mse" in df.columns and np.isfinite(baseline_val_mse):
        mask = np.isfinite(df["baseline_val_mse"])
        close = np.abs(df.loc[mask, "baseline_val_mse"] - baseline_val_mse) <= rtol * baseline_val_mse
        df = df[~mask | close]

    df = df.sort_values("utc_iso").reset_index(drop=True)
    df["iteration"] = np.arange(1, len(df) + 1)
    df["val_mse_cummin"] = df["val_mse"].cummin()
    if "test_mse" in df.columns:
        df["test_mse_cummin"] = df["test_mse"].cummin()
    return df


def _output_paths(base: Path) -> tuple[Path, Path]:
    """autoresearch_iteration.png -> _val.png / _test.png"""
    stem = base.stem
    if stem.endswith("_val") or stem.endswith("_test"):
        stem = stem.rsplit("_", 1)[0]
    return base.with_name(f"{stem}_val{base.suffix}"), base.with_name(f"{stem}_test{base.suffix}")


def _ylim_for_series(*arrays: np.ndarray, baseline: float | None = None, pad_ratio: float = 0.06):
    vals = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if baseline is not None and np.isfinite(baseline):
        vals.append(np.array([baseline]))
    if not vals:
        return None
    allv = np.concatenate(vals)
    lo, hi = float(allv.min()), float(allv.max())
    span = max(hi - lo, 1e-6)
    return lo - pad_ratio * span, hi + pad_ratio * span


def _plot_score_panel(ax, df: pd.DataFrame) -> None:
    x = df["iteration"].to_numpy()
    score = df["score"].to_numpy()
    colors = np.where(score > 0, "#2ca02c", "#d62728")
    ax.bar(x, score, color=colors, alpha=0.65, width=0.85, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("score\n(val improve)")
    ax.grid(True, alpha=0.25, axis="y")


def plot_iteration(
    df: pd.DataFrame,
    baselines: dict[str, float],
    *,
    model_name: str,
    output_val: Path,
    output_test: Path,
    dpi: int = 150,
) -> None:
    output_val.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
    })

    x = df["iteration"].to_numpy()
    val = df["val_mse"].to_numpy()
    val_best = df["val_mse_cummin"].to_numpy()
    test = df["test_mse"].to_numpy() if "test_mse" in df.columns else None
    test_best = df["test_mse_cummin"].to_numpy() if "test_mse_cummin" in df.columns else None

    b_val = baselines["val_mse"]
    b_test = baselines.get("test_mse")

    status = df["status"].fillna("unknown")
    keep_mask = status == "keep"
    discard_mask = status == "discard"

    has_score = "score" in df.columns and df["score"].notna().any()

    # ========== Val 图 ==========
    if has_score:
        fig_val, (ax_val, ax_score) = plt.subplots(
            2, 1, figsize=(12, 7), height_ratios=[3, 1], sharex=True
        )
    else:
        fig_val, ax_val = plt.subplots(1, 1, figsize=(12, 5))
        ax_score = None

    ax_val.axhline(
        b_val, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.9,
        label=f"Baseline val MSE ({model_name}, no LLM feats) = {b_val:.4f}",
    )
    ax_val.plot(x, val, color="#9ecae1", linewidth=1.0, alpha=0.95, zorder=2, label="Per-trial val MSE")
    ax_val.plot(
        x, val_best, color="#1f77b4", linewidth=2.4, zorder=4,
        label="Cumulative best val MSE (AutoResearch objective)",
    )
    ax_val.scatter(
        x[keep_mask], val[keep_mask], c="#2ca02c", s=40, edgecolors="white",
        linewidths=0.5, zorder=5, label="keep",
    )
    ax_val.scatter(
        x[discard_mask], val[discard_mask], c="#bdbdbd", s=24, alpha=0.85,
        zorder=4, label="discard",
    )
    ax_val.set_ylabel("Val MSE (scaled space)")
    ax_val.set_title(
        f"Validation MSE over AutoResearch iterations\n"
        f"{model_name}, pred_len={DEFAULT_HORIZON}, n={len(df)} trials"
    )
    ax_val.grid(True, alpha=0.25)
    ax_val.legend(loc="upper right", framealpha=0.92)
    ylim_val = _ylim_for_series(val, val_best, baseline=b_val)
    if ylim_val:
        ax_val.set_ylim(ylim_val)

    if ax_score is not None:
        _plot_score_panel(ax_score, df)
        ax_score.set_xlabel("Iteration (chronological)")
    else:
        ax_val.set_xlabel("Iteration (chronological)")

    fig_val.tight_layout()
    fig_val.savefig(output_val, bbox_inches="tight")
    plt.close(fig_val)
    print(f"saved: {output_val}")

    # ========== Test 图 ==========
    if test is None or not np.isfinite(test).any():
        print("skip test figure: no test_mse in results")
        return

    fig_test, ax_test = plt.subplots(1, 1, figsize=(12, 5))
    valid = np.isfinite(test)

    if b_test is not None and np.isfinite(b_test):
        ax_test.axhline(
            b_test, color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.9,
            label=f"Baseline test MSE ({model_name}, no LLM feats) = {b_test:.4f}",
        )
    ax_test.plot(
        x[valid], test[valid], color="#fdd0a2", linewidth=1.0, alpha=0.95, zorder=2,
        label="Per-trial test MSE (not used for feature selection)",
    )
    if test_best is not None and np.isfinite(test_best).any():
        ax_test.plot(
            x, test_best, color="#d62728", linewidth=2.4, linestyle="-", zorder=4,
            label="Cumulative best test MSE (post-hoc)",
        )
    ax_test.scatter(
        x[keep_mask & valid], test[keep_mask & valid], c="#2ca02c", s=40, edgecolors="white",
        linewidths=0.5, zorder=5, label="keep (val-best round)",
    )
    ax_test.scatter(
        x[discard_mask & valid], test[discard_mask & valid], c="#bdbdbd", s=24, alpha=0.85,
        zorder=4, label="discard",
    )
    ax_test.set_ylabel("Test MSE (scaled space)")
    ax_test.set_xlabel("Iteration (chronological)")
    ax_test.set_title(
        f"Test MSE over AutoResearch iterations (generalization)\n"
        f"{model_name}, pred_len={DEFAULT_HORIZON} — may worsen while val improves"
    )
    ax_test.grid(True, alpha=0.25)
    ax_test.legend(loc="upper right", framealpha=0.92)
    ylim_test = _ylim_for_series(test[valid], test_best[valid] if test_best is not None else test[valid], baseline=b_test)
    if ylim_test:
        ax_test.set_ylim(ylim_test)

    fig_test.tight_layout()
    fig_test.savefig(output_test, bbox_inches="tight")
    plt.close(fig_test)
    print(f"saved: {output_test}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot autoresearch val/test iteration curves.")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--model", default="DLinear", help="Label in plot title")
    p.add_argument("--baseline-rtol", type=float, default=0.02,
                   help="Drop trials whose baseline_val_mse differs from baselines.json")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baselines = load_baselines(args.baselines, args.horizon)
    df = load_trials(args.results, baselines["val_mse"], rtol=args.baseline_rtol)
    if df.empty:
        raise SystemExit("No valid trials to plot.")
    out_val, out_test = _output_paths(args.output)
    plot_iteration(
        df, baselines, model_name=args.model, output_val=out_val, output_test=out_test, dpi=args.dpi
    )

    n_keep = int((df["status"] == "keep").sum())
    best_val_idx = int(df["val_mse"].idxmin()) + 1
    print(f"trials: {len(df)}, keep: {n_keep}, best val @ iter {best_val_idx}: {df['val_mse'].min():.6f}")
    if df["test_mse"].notna().any():
        best_test_idx = int(df["test_mse"].idxmin()) + 1
        print(f"best test @ iter {best_test_idx}: {df['test_mse'].min():.6f} (baseline test: {baselines.get('test_mse')})")


if __name__ == "__main__":
    main()
