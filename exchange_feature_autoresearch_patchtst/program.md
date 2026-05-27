# Exchange Feature AutoResearch — PatchTST 实验线

子目录：`exchange_feature_autoresearch_patchtst/`。数据：`dataset/exchange_rate/exchange_rate.csv`（8 维，日频）。  
默认模型：**PatchTST**，`train_epochs=10`，`pred_len=96`。  
Git 分支建议：`auto/20260528-exchange-patchtst`。Worktree：`/data/nishome/xuhaochen/tslib-wt/exchange-patchtst`。

## Phase A

```bash
conda activate tslib
cd /data/nishome/xuhaochen/tslib-wt/exchange-patchtst
# 若在 worktree 内，仓库根即当前目录

python exchange_feature_autoresearch_patchtst/prepare.py --mode baseline --log-file auto
```

## Phase B

```bash
bash exchange_feature_autoresearch_patchtst/run_experiment.sh "本轮说明" --cuda-visible-devices 0
```

`dig.py` 起点：`calendar_dow_month_v1`（周/月周期）。`results_all.tsv` 勿 commit。
