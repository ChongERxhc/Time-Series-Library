# Exchange Feature AutoResearch — iTransformer 实验线（TSLib 汇率长期预测特征挖掘）

本目录用于在 **Time-Series-Library（TSLib）** 的 **Exchange Rate** 长期预测任务上，以 **可审计、可重复** 的方式迭代挖掘派生特征。**禁止跳过 Phase A**；**Phase A 结束前不得开跑 trial**；用户确认进入 **Phase B** 后应 **持续多轮实验循环**，除非用户明确要求停止。

工作目录约定：本线推荐在 **独立 git worktree** 中运行（与 Weather 线并行互不 `checkout` 冲突）：

```text
/data/nishome/xuhaochen/tslib-wt/exchange-itransformer
```

仓库根即该 worktree 目录（内含完整 TSLib 与 `dataset/`）。子目录：`exchange_feature_autoresearch/`（本文件所在目录）。

与 `exchange_feature_autoresearch_patchtst/`（PatchTST）、`exchange_feature_autoresearch_dlinear/`（DLinear）及全部 **Weather** 实验线 **独立**：各自 `baselines.json`、`results_all.tsv`，**互不引用**对方 baseline。`model_id` 使用前缀 **`exchange_`**，与 `weather_*` checkpoint 区分。

当前 worktree 建议分支：**`auto/20260528-exchange-itransformer`**（orphan，与 `main`/`master` 无共同祖先）。

---

## 一、环境与依赖（硬规则）

1. **必须在 `tslib` conda 环境中运行**：

   ```bash
   conda activate tslib
   cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer
   ```

2. **Python**：优先使用当前已激活环境中的 `python`；若需显式指定：

   ```bash
   export PYTHON="$(which python)"
   ```

3. **GPU**：`prepare.py` 默认 `CUDA_VISIBLE_DEVICES=0`。并行跑多条实验线时，**每条线指定不同卡**：

   ```bash
   python exchange_feature_autoresearch/prepare.py ... --cuda-visible-devices 1
   ```

4. **不要安装新依赖**：仅使用 TSLib 与 `tslib` 环境已有包（`pandas`、`numpy` 等）。

5. **checkpoint 共享**：所有 worktree 共用同一仓库的 `checkpoints/` 目录；靠 **`exchange_` + 模型名 + 特征名** 的 `model_id` 区分，禁止两条进程同时写同一 `setting`。

---

## 二、Setup

> 下面内容可直接复制给 agent，作为启动 prompt（**Exchange + iTransformer** 实验线；先 setup，再开跑 trial）。

你必须按「两阶段」执行，禁止跳步。

### Phase A（仅 setup，不得开始特征挖掘 / trial）

#### 1) 实验分支与 worktree

- 本线 worktree 路径：`/data/nishome/xuhaochen/tslib-wt/exchange-itransformer`
- 分支名：`auto/20260528-exchange-itransformer`；标签格式：`YYYYMMDD-exchange-itransformer`（例：`20260528-exchange-itransformer`）。
- **已初始化 orphan 的 worktree**：若 `git merge-base HEAD main` 无输出，可直接 Phase A baseline，**不必**重复 orphan。
- **新开会话需新建 orphan**（在**非** `main`/`master` 的分支上，勿先 `checkout main`）：

```bash
cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer
git branch --show-current   # 不得为 main / master
git checkout --orphan auto/<tag>
git reset
git add .
git commit -m "chore: init auto/<tag> root commit"
```

- Phase A / Phase B 的 baseline 与 trial 均在本次 **`auto/<tag>`** orphan 分支上进行。

#### 2) 历史可见性约束（硬规则）

- **禁止**：`git log --all`、`git reflog`、按历史 hash 的 `git show`。
- **只允许**：`git log --oneline -n 20`（仅当前实验分支）。

#### 3) 阅读并确认边界

- `exchange_feature_autoresearch/dig.py`（**唯一可改**：`FEATURE_SET_NAME` + `compute_features`）
- `prepare.py`、`run_experiment.sh`、`append_results_all.py`（固定框架；除非用户要求修 bug，否则不改）

#### 4) 运行环境检查

- 使用 `tslib` conda 环境（见第一节）。
- 生成 baseline（建议落盘日志；默认 `train_epochs=10`，仅 `pred_len=96`）：

```bash
cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer
python exchange_feature_autoresearch/prepare.py --mode baseline --log-file auto
```

- 日志路径：`exchange_feature_autoresearch/logs/baseline_<UTC>.log`。
- 确认 `exchange_feature_autoresearch/baselines.json` 在 `"96"` 下含 **`val_mse`**（旧版仅 test `mse` 须重跑 baseline）。

#### 5) 审计文件规则

- `exchange_feature_autoresearch/results_all.tsv`：每次 trial 运行都要追加一行，**禁止 commit**。

#### 6) setup 完成后先汇报，等待用户确认

- 必须输出：当前分支名、最近 5 条 commit、`baselines.json` 是否存在及 `96` 的 baseline **val_mse** 摘要。
- 明确询问：是否进入正式实验循环？
- 在用户未明确回复「开始/继续」前：**禁止**修改 `dig.py` 中的特征、**禁止**运行 `bash exchange_feature_autoresearch/run_experiment.sh`。

### Phase B（仅在用户确认后执行）

#### 7) 进入实验循环

- 每轮最小单元：`bash exchange_feature_autoresearch/run_experiment.sh "本轮说明"`（可选 `--train-epochs` 等，同 `prepare.py`，默认 10；默认仅 horizon 96）。
- 每轮必须留下：独立 commit（若有 git）+ `exchange_feature_autoresearch/logs/experiment_<UTC>.log` + `results_all.tsv` 新增一行。
- 已在 Phase B 内时：应持续多轮尝试，勿在每轮结束再次询问「是否继续」；勿擅自中断整条循环（除非用户明确要求停止）；单次崩溃可修复后重跑该轮；长期无 score 提升应换思路。
- 仅排障可设 `WEATHER_AR_SKIP_ORPHAN_CHECK=1` 跳过 `run_experiment.sh` 的 orphan 校验（平时不要用）。

**补充说明（与脚本一致）**

- 会话日志：`exchange_feature_autoresearch/logs/`（`mkdir -p` 自动创建）。
- `run_experiment.sh` 会拒绝在 `main`/`master` 上运行，并校验与 `main`/`master` **无共同祖先**；非 git 目录会跳过 commit 与该校验，仍会写日志与 `results_all.tsv`。
- 若 `Permission denied`：使用 `bash exchange_feature_autoresearch/run_experiment.sh "..."`，或对脚本 `chmod +x`。
- 并行实验：Weather 三线在 `tslib-wt/weather-*`，Exchange 三线在 `tslib-wt/exchange-*`，各 `cd` 各目录，各配 `--cuda-visible-devices`。

---

## 三、数据与列含义（`exchange_rate.csv`）

数据文件：`dataset/exchange_rate/exchange_rate.csv`  
时间列：`date`（**日频**，格式如 `1990/1/1 0:00`）。  
其后 **8 个数值列**（与 TSLib `custom` + `enc_in=8` 一致）：

| 列名 | 含义（简要） |
|------|--------------|
| `0` | 汇率序列 0（数据集原始列名，无货币名 metadata） |
| `1` | 汇率序列 1 |
| `2` | 汇率序列 2 |
| `3` | 汇率序列 3 |
| `4` | 汇率序列 4 |
| `5` | 汇率序列 5 |
| `6` | 汇率序列 6 |
| `OT` | 目标列（与其它 TSLib Exchange 实验一致，作为 `OT` 通道） |

**规模**：约 7.5k 日度样本（1990 年起）。  
**特征工程建议**：日历周期（周/月/季）、滞后收益率、跨币种价差或比值；避免照搬 Weather 的物理风场特征。  
**本框架约定**：写入训练 CSV 的 **原始 8 维** 与 `exchange_rate.csv` **完全一致**；`clean_df` 用于 `compute_features` 内计算 `llm_*`（哨兵阈值与 Weather 相同，日频汇率通常无 `-9999`）。

**`dig.py` 起点（本线）**：`calendar_dow_month_v1`（星期与月份 sin/cos，共 4 列 `llm_*`）。

---

## 四、核心目标与 score

**开发阶段固定 `pred_len = 96`**（每轮 trial 只训练一次）。**优化目标为验证集**，不用测试集做特征选择。

从 TSLib 训练日志中解析每个 epoch 的 `Vali Loss`（验证集 MSE），取 **最小值** 作为 `trial_val_mse`（与 EarlyStopping 按 val 存 checkpoint 一致）：

```text
val_improve = (baseline_val_mse - trial_val_mse) / baseline_val_mse
score       = val_improve
```

- `score > 0`：相对 **本线** baseline，验证集 MSE 变好。
- `score < 0`：验证集变差。

日志末尾的 `test_mse` / `test_mae` **仅作审计参考**，**不参与** `score` 与 keep/discard。

`append_results_all.py` 根据历史最优 `score` 判定 `keep` / `discard`（见第八节）。

---

## 五、固定实验口径（与 `prepare.py` 一致）

| 项 | 值 |
|------|-----|
| 任务 | `long_term_forecast` |
| 数据 | `dataset/exchange_rate/exchange_rate.csv`（baseline）；trial 写入 **`generated_data/_staging.csv`**（覆盖写，默认跑完删除） |
| `data` | `custom` |
| `freq` | **`d`**（日频） |
| `features` | `M` |
| `seq_len` | 96 |
| `label_len` | 48 |
| `pred_len` | 默认 **96**（trial 评分只用第一个 horizon） |
| 模型 | 默认 **`iTransformer`**（`e_layers=3`，`n_heads=8`，`d_ff=512`；与 Weather iTransformer 线口径一致）；**改模型后须重跑 baseline** |
| `e_layers/d_layers/factor/n_heads` | 3 / 1 / 3 / 8 |
| `d_model/d_ff` | 512 / 512 |
| `train_epochs` | 默认 **10** |
| `model_id` 前缀 | **`exchange_`** |
| trial 输入维 | `8 + 特征数` |
| trial 评估维 | `--eval_dims 8`（只优化与度量 **原始 8 维**） |

---

## 六、文件职责（agent 边界）

| 文件 | 是否可改 | 说明 |
|------|----------|------|
| `dig.py` | **仅此处可改** | `FEATURE_SET_NAME` + `compute_features(raw_df, clean_df)` |
| `prepare.py` | **禁止** | 生成数据、调 `run.py`、解析指标、打印 `score` 等键值日志 |
| `run_experiment.sh` | **禁止**（除非修 bug） | orphan 校验 → commit（若有 git）→ `prepare.py --mode trial` → `append_results_all.py` |
| `append_results_all.py` | **禁止**（除非修 bug） | 解析日志、追加 `results_all.tsv` |
| `baselines.json` | 由 `prepare --mode baseline` 写入 | 各 horizon 的 **`val_mse`**（及可选 `test_mse` 审计） |
| `results_all.tsv` | **只追加、不删改历史行** | 主账；**不要 commit** |
| `generated_data/_staging.csv` | 每轮 trial 临时写入 | 默认训练后删除；`--keep-staging` 保留并复制到 `archive/` |
| `logs/*.log` | 自动生成 | baseline / trial 会话完整输出 |

---

## 七、`dig.py` 编写契约

1. **必须实现**：

   ```python
   FEATURE_SET_NAME = "唯一英文标识，用于文件名与结果表"

   def compute_features(raw_df, clean_df) -> pd.DataFrame:
       ...
   ```

2. **`raw_df`**：原始读入，**不得原地修改**影响落盘列。

3. **`clean_df`**：与 `raw_df` 同索引；数值列已做哨兵剔除与 `ffill/bfill`，**仅用于**计算 `llm_*`。

4. **返回**：行数与 `raw_df` 完全一致；列仅为新增特征；列名 **全部** 以 `llm_` 开头；不得与 `raw_df` 列名重复；无 NaN、无 inf。

5. **每轮特征数量**：建议 **3–12** 个。

6. **严禁未来泄露**：

   - 禁止 `shift(-k)`、禁止 `rolling(center=True)`、禁止用未来标签构造特征。
   - 允许 `shift(k)`（k>0）、过去窗口 `rolling(..., min_periods=1)`、当前时刻跨列组合。

7. **Exchange 特有**：优先日历/滞后/价差类特征；不要复制 Weather 气象物理特征除非有明确金融含义。

---

## 八、Keep / Discard（`append_results_all.py`）

从日志中读取 `score:` 行，与 `results_all.tsv` 中历史最优 `score` 比较：

```text
若 score 无法解析 → status = crash
若 score > best_score_before + min_improve → status = keep
否则 → status = discard
```

默认 **`min_improve = 0`**。若需更保守：

```bash
python exchange_feature_autoresearch/append_results_all.py --log ... --min-improve 0.001
```

---

## 九、命令速查

**Baseline（首次、改 horizon/epoch/model 后重做）**：

```bash
conda activate tslib
cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer

python exchange_feature_autoresearch/prepare.py \
  --mode baseline \
  --log-file auto \
  --cuda-visible-devices 0
```

**一轮 trial（先改 `dig.py`）**：

```bash
cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer
bash exchange_feature_autoresearch/run_experiment.sh "本轮描述" --cuda-visible-devices 0
```

**查看账本**：

```bash
column -t -s $'\t' exchange_feature_autoresearch/results_all.tsv | less -S
```

**日志中抽取关键行**：

```bash
grep -E "^(feature_set_name|feature_fp|n_features|staging_csv|baseline_val_mse|val_mse|val_improve|score|test_mse):" \
  exchange_feature_autoresearch/logs/experiment_*.log | tail -n 50
```

**查看 worktree 列表**（在任意克隆目录执行）：

```bash
git worktree list
```

---

## 十、给 agent 的一行摘要

激活 **tslib** → **`cd /data/nishome/xuhaochen/tslib-wt/exchange-itransformer`**（分支 **`auto/20260528-exchange-itransformer`**）→ **iTransformer baseline**（`train_epochs=10`）→ **`baselines.json` 含 val_mse@96** → **`results_all.tsv` 勿 commit** → 汇报并等 **「开始/继续」** → **Phase B**：`run_experiment.sh`，只改 **`dig.py`**；**score = 验证集相对本线 baseline @ pl=96**；**`model_id` 前缀 `exchange_`**；**`generated_data/` 默认空**。
