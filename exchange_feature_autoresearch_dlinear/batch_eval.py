#!/usr/bin/env python3
"""批量评估所有 score>0 的特征集在测试集上的表现。

用法:
    conda activate tslib
    python batch_eval.py [--min-score 0.0] [--top-n N] [--model DLinear]
                         [--pred-len 96] [--train-epochs 10] [--cuda-visible-devices 0]
                         [--output batch_eval_report.json] [--dry-run] [--report-only]

--report-only: 只从 results_all.tsv 提取现有 test 指标生成汇总报告，不重新运行测试。
--dry-run:     打印将要测试的特征集列表及 checkpoint 状态，不实际运行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# 将 prepare.py 所在目录加入路径以便复用函数
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import prepare as prep

ROOT = prep.ROOT
WORK_DIR = prep.WORK_DIR
DATA_DIR = prep.DATA_DIR
STAGING_CSV = prep.STAGING_CSV
ORIGINAL_DIMS = prep.ORIGINAL_DIMS

TEST_METRICS_RE = re.compile(r"mse:([^,\s]+),\s*mae:([^,\s]+)")


def parse_test_output(stdout: str) -> tuple[float | None, float | None]:
    match = TEST_METRICS_RE.search(stdout)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def build_setting(
    model_id: str,
    model: str,
    pred_len: int,
    model_run_kwargs: dict[str, int],
    des: str = "",
) -> str:
    """构造与 run.py 完全一致的 setting 字符串（ii=0）。"""
    return (
        f"long_term_forecast_{model_id}_{model}_custom_ftM_sl96_ll48_pl{pred_len}_"
        f"dm{model_run_kwargs['d_model']}_nh{model_run_kwargs['n_heads']}_"
        f"el{model_run_kwargs['e_layers']}_dl{model_run_kwargs['d_layers']}_"
        f"df{model_run_kwargs['d_ff']}_"
        f"expand2_dc4_fc{model_run_kwargs['factor']}_ebtimeF_dtTrue_{des}_0"
    )


def checkpoint_exists(setting: str) -> bool:
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    return ckpt.exists()


def git_checkout_dig(commit: str) -> None:
    cmd = ["git", "checkout", commit, "--", str(WORK_DIR / "dig.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)


def git_restore_dig() -> None:
    """将 dig.py 恢复到 HEAD（撤销可能的 checkout 修改）。"""
    cmd = ["git", "checkout", "HEAD", "--", str(WORK_DIR / "dig.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_test_inference(
    *,
    data_path: str,
    model_id: str,
    input_dim: int,
    pred_len: int,
    eval_dims: int,
    model: str,
    cuda_visible_devices: str,
    des: str,
    model_run_kwargs: dict[str, int],
) -> tuple[float | None, float | None, str]:
    """调用 run.py --is_training 0 进行测试并解析指标。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "0",
        "--root_path", str(DATA_DIR) + "/",
        "--data_path", data_path,
        "--model_id", model_id,
        "--model", model,
        "--data", "custom",
        "--features", "M",
        "--freq", prep.DATA_FREQ,
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", str(pred_len),
        "--e_layers", str(model_run_kwargs["e_layers"]),
        "--d_layers", str(model_run_kwargs["d_layers"]),
        "--factor", str(model_run_kwargs["factor"]),
        "--n_heads", str(model_run_kwargs["n_heads"]),
        "--enc_in", str(input_dim),
        "--dec_in", str(input_dim),
        "--c_out", str(input_dim),
        "--d_model", str(model_run_kwargs["d_model"]),
        "--d_ff", str(model_run_kwargs["d_ff"]),
        "--des", des,
        "--itr", "1",
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
    test_mse, test_mae = parse_test_output(proc.stdout)
    return test_mse, test_mae, proc.stdout


def load_candidates(
    tsv_path: Path,
    min_score: float,
    top_n: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["score"] > min_score]
    df = df[df["status"] != "crash"]
    # 按 feature_set_name 去重，保留得分最高的那条记录
    df = df.sort_values("score", ascending=False).drop_duplicates("feature_set_name", keep="first")
    # 如果有同名但不同 commit 的情况，取最后训练的那个（也就是最高分）
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate score>0 feature sets on test set.")
    parser.add_argument("--results-tsv", type=Path, default=WORK_DIR / "results_all.tsv")
    parser.add_argument("--min-score", type=float, default=0.0, help="只评估 score 大于此值的特征集")
    parser.add_argument("--top-n", type=int, default=None, help="只评估得分最高的前 N 个（默认全部）")
    parser.add_argument("--model", default=prep.DEFAULT_MODEL, help="训练时使用的模型名")
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-epochs", type=int, default=prep.DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--output", type=Path, default=WORK_DIR / "batch_eval_report.json")
    parser.add_argument("--dry-run", action="store_true", help="仅打印待测列表和 checkpoint 状态")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="直接从 results_all.tsv 提取已有的 test 指标生成报告，不重新跑测试",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每条测试的完整 stdout")
    args = parser.parse_args()
    model_run_kwargs = prep.get_model_run_kwargs(args.model)

    df = load_candidates(args.results_tsv, args.min_score, args.top_n)
    total = len(df)
    print(f"候选特征集数量: {total} (score > {args.min_score}, status != crash)")
    if total == 0:
        print("无候选特征集，退出。")
        return

    if args.report_only:
        report = []
        for _, row in df.iterrows():
            report.append({
                "feature_set_name": row["feature_set_name"],
                "commit": row["commit"],
                "n_features": int(row["n_features"]) if not pd.isna(row["n_features"]) else None,
                "input_dim": int(row["input_dim"]) if not pd.isna(row["input_dim"]) else None,
                "pred_len": int(row["pred_len"]) if not pd.isna(row["pred_len"]) else args.pred_len,
                "score": float(row["score"]),
                "val_mse": float(row["val_mse"]) if not pd.isna(row["val_mse"]) else None,
                "test_mse": float(row["test_mse"]) if not pd.isna(row["test_mse"]) else None,
                "test_mae": float(row["test_mae"]) if not pd.isna(row["test_mae"]) else None,
                "description": row["description"] if not pd.isna(row["description"]) else "",
                "status": row["status"] if not pd.isna(row["status"]) else "",
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"汇总报告已保存至: {args.output}")
        return

    # dry-run / 预检查
    print("\n预检查 checkpoint 状态:")
    missing_ckpts = 0
    for _, row in df.iterrows():
        fs = row["feature_set_name"]
        pred_len = int(row["pred_len"]) if not pd.isna(row["pred_len"]) else args.pred_len
        model_id = f"weather_{args.model}_{fs}_pl{pred_len}_ep{args.train_epochs}"
        des = f"AutoResearchTrial_{fs}_pl{pred_len}"
        setting = build_setting(model_id, args.model, pred_len, model_run_kwargs, des=des)
        ok = checkpoint_exists(setting)
        if not ok:
            missing_ckpts += 1
        print(f"  [{'OK' if ok else 'MISSING'}] {fs} (score={row['score']:.6f})")
    if missing_ckpts:
        print(f"\n警告: {missing_ckpts}/{total} 个特征集缺少 checkpoint，测试将失败。")
        if not args.dry_run:
            cont = input("是否继续? [y/N]: ")
            if cont.lower() not in ("y", "yes"):
                print("已取消。")
                return
    if args.dry_run:
        print("\ndry-run 结束，未执行任何测试。")
        return

    # 批量测试
    report = []
    original_dig_commit: str | None = None
    try:
        # 记录当前 dig.py 的 commit 以便最后恢复
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        original_dig_commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        pass

    for idx, row in df.iterrows():
        commit = row["commit"]
        fs = row["feature_set_name"]
        n_features = int(row["n_features"]) if not pd.isna(row["n_features"]) else None
        input_dim = int(row["input_dim"]) if not pd.isna(row["input_dim"]) else None
        eval_dims = int(row["eval_dims"]) if not pd.isna(row["eval_dims"]) else ORIGINAL_DIMS
        pred_len = int(row["pred_len"]) if not pd.isna(row["pred_len"]) else args.pred_len
        score = float(row["score"])
        old_test_mse = float(row["test_mse"]) if not pd.isna(row["test_mse"]) else None
        old_test_mae = float(row["test_mae"]) if not pd.isna(row["test_mae"]) else None
        description = row["description"] if not pd.isna(row["description"]) else ""

        print(f"\n[{idx + 1}/{total}] {fs} | commit={commit} | score={score:.6f}")

        # 1) 恢复 dig.py 到对应 commit
        try:
            git_checkout_dig(commit)
        except subprocess.CalledProcessError as e:
            print(f"  [SKIP] git checkout 失败: {e}")
            report.append({
                "feature_set_name": fs,
                "commit": commit,
                "score": score,
                "old_test_mse": old_test_mse,
                "old_test_mae": old_test_mae,
                "test_mse": None,
                "test_mae": None,
                "status": "checkout_failed",
                "description": description,
            })
            continue

        # 2) 生成 staging CSV
        try:
            meta = prep.build_trial_dataset_staging()
            # 校验 feature_set_name 一致性
            if meta.feature_set_name != fs:
                print(f"  [WARN] feature_set_name 不匹配: TSV={fs}, dig.py={meta.feature_set_name}")
            data_path = meta.rel_data_path
            input_dim = meta.input_dim
            n_features = meta.n_features
        except Exception as e:
            print(f"  [SKIP] 生成 staging 失败: {e}")
            report.append({
                "feature_set_name": fs,
                "commit": commit,
                "score": score,
                "old_test_mse": old_test_mse,
                "old_test_mae": old_test_mae,
                "test_mse": None,
                "test_mae": None,
                "status": "staging_failed",
                "description": description,
            })
            continue

        # 3) 构造 model_id / des / setting
        model_id = f"weather_{args.model}_{fs}_pl{pred_len}_ep{args.train_epochs}"
        des = f"AutoResearchTrial_{fs}_pl{pred_len}"
        setting = build_setting(model_id, args.model, pred_len, model_run_kwargs, des=des)

        if not checkpoint_exists(setting):
            print(f"  [SKIP] checkpoint 不存在: {setting}")
            report.append({
                "feature_set_name": fs,
                "commit": commit,
                "score": score,
                "old_test_mse": old_test_mse,
                "old_test_mae": old_test_mae,
                "test_mse": None,
                "test_mae": None,
                "status": "checkpoint_missing",
                "description": description,
            })
            continue

        # 4) 运行测试
        try:
            test_mse, test_mae, stdout = run_test_inference(
                data_path=data_path,
                model_id=model_id,
                input_dim=input_dim,
                pred_len=pred_len,
                eval_dims=eval_dims,
                model=args.model,
                cuda_visible_devices=args.cuda_visible_devices,
                des=des,
                model_run_kwargs=model_run_kwargs,
            )
            if args.verbose:
                print(stdout)
            if test_mse is None:
                print(f"  [WARN] 未能从 stdout 解析 test metrics")
                status = "parse_failed"
            else:
                print(f"  test_mse={test_mse:.7f} test_mae={test_mae:.7f}")
                status = "ok"
        except Exception as e:
            print(f"  [FAIL] 测试运行异常: {e}")
            test_mse, test_mae, stdout = None, None, ""
            status = "exception"

        report.append({
            "feature_set_name": fs,
            "commit": commit,
            "n_features": n_features,
            "input_dim": input_dim,
            "pred_len": pred_len,
            "score": score,
            "val_mse": float(row["val_mse"]) if not pd.isna(row["val_mse"]) else None,
            "old_test_mse": old_test_mse,
            "old_test_mae": old_test_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "status": status,
            "description": description,
        })

        # 5) 清理 staging
        prep.remove_staging(meta.staging_path)

    # 恢复 dig.py
    print("\n恢复 dig.py ...")
    try:
        git_restore_dig()
        print("已恢复。")
    except subprocess.CalledProcessError as e:
        print(f"恢复 dig.py 失败: {e}")

    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存至: {args.output}")

    # 简单汇总
    ok_count = sum(1 for r in report if r["status"] == "ok")
    print(f"成功: {ok_count}/{len(report)}")


if __name__ == "__main__":
    main()
