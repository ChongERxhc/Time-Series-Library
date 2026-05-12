# Weather Feature AutoResearch（TSLib 天气长期预测特征挖掘）

本目录用于在 **Time-Series-Library（TSLib）** 的 `weather` 长期预测任务上，以 **可审计、可重复** 的方式迭代挖掘派生特征。**禁止跳过 Phase A**；**Phase A 结束前不得开跑 trial**；用户确认进入 **Phase B** 后应 **持续多轮实验循环**，除非用户明确要求停止。

工作目录约定：所有命令均在 **仓库根** 执行：

```text
/data/nishome/xuhaochen/Time-Series-Library
```

子目录：`weather_feature_autoresearch/`（本文件所在目录）。

---

## 一、环境与依赖（硬规则）

1. **必须在 `tslib` conda 环境中运行**（与 TSLib 官方 README 一致）：

   ```bash
   conda activate tslib
   cd /data/nishome/xuhaochen/Time-Series-Library
   ```

2. **Python**：优先使用当前已激活环境中的 `python`。若需显式指定：

   ```bash
   export PYTHON="$(which python)"
   ```

3. **GPU**：`prepare.py` 默认 `CUDA_VISIBLE_DEVICES=0`。多卡或指定卡时传入：

   ```bash
   python weather_feature_autoresearch/prepare.py ... --cuda-visible-devices 1
   ```

4. **不要安装新依赖**：仅使用 TSLib 与 `tslib` 环境已有包（`pandas`、`numpy` 等）。

---

## 二、Setup

下面这段可直接作为给 agent 的启动 prompt（weather-tslib 专用，先 setup 后开跑）：

```text
你必须按“两阶段”执行，禁止跳步：

Phase A（仅 setup，不得开始特征挖掘 / trial）：

1) 先创建当天实验分支（orphan）
- 标签用 YYYYMMDD-weather（如 20260512-weather），分支名：autoresearch/<tag>
- 在仓库根目录（Time-Series-Library）执行：
  git checkout --orphan autoresearch/<tag>
  git reset
  git add .
  git commit -m "chore: init autoresearch/<tag> root commit"

2) 历史可见性约束（硬规则）
- 禁止：git log --all、git reflog、按历史 hash 的 git show
- 只允许：git log --oneline -n 20（仅当前实验分支）

3) 阅读并确认边界
- weather_feature_autoresearch/dig.py（唯一编辑区：FEATURE_SET_NAME + compute_features）
- prepare.py、run_experiment.sh、append_results_all.py（固定框架与记录口径，除非用户明确要求修 bug，否则不改）
- 本轮不扩展 LOO/OFO 等消融子流程

4) 运行环境检查
- 使用 tslib conda 环境（conda activate tslib；仓库根见上文路径）
- 生成 baseline（建议同时落盘日志）：`python weather_feature_autoresearch/prepare.py --mode baseline --horizons 96 192 336 720 --train-epochs 10 --log-file auto`（日志在 `weather_feature_autoresearch/logs/baseline_<UTC>.log`）
- 确认 `weather_feature_autoresearch/baselines.json` 各 `pred_len` 下存在 `mse`/`mae`；终端或日志中各 horizon 有 MSE 等可核对输出

5) 审计文件规则
- weather_feature_autoresearch/results_all.tsv：每次 trial 运行都要追加一行，禁止 commit

6) setup 完成后先汇报，等待用户确认
- 必须输出 setup 检查结果（当前分支名、最近 5 条 commit、baselines.json 是否存在及各 horizon baseline MSE 摘要）。
- 明确询问：是否进入正式实验循环？
- 在用户未明确回复「开始/继续」前：禁止修改 dig.py 中的特征、禁止运行 bash weather_feature_autoresearch/run_experiment.sh。

Phase B（仅在用户确认后执行）：

7) 进入实验循环
- 每轮最小单元：bash weather_feature_autoresearch/run_experiment.sh "本轮说明"（可选：--horizons 96 192 336 720 --train-epochs 10 等，同 prepare.py）
- 每轮必须留下：独立 commit（若有 git）+ `weather_feature_autoresearch/logs/experiment_<UTC>.log`（完整会话输出，由 `prepare.py --log-file` 写入）+ `results_all.tsv` 新增一行
- 已在 Phase B 内时：应持续多轮尝试，勿在每轮结束再次询问「是否继续」；勿擅自中断整条循环（除非用户明确要求停止）；单次崩溃可修复后重跑该轮；长期无 score 提升应换思路而非死磕同一组特征
- 仅排障可设 WEATHER_AR_SKIP_ORPHAN_CHECK=1 跳过 run_experiment.sh 的 orphan 校验（平时不要用）
```

**补充说明（与脚本一致）**：会话日志统一落在 **`weather_feature_autoresearch/logs/`**（`mkdir -p` 自动创建）。`run_experiment.sh` 会校验当前分支与 `main`/`master` **无共同祖先**（等价于在 orphan 实验线上）；非 git 目录会跳过 commit 与该校验，仍会写日志与 `results_all.tsv`。每次**新会话**应回到 `main` 再按 1) 新建 `autoresearch/<tag>`。若 `run_experiment.sh` 报 `Permission denied`，改用 `bash weather_feature_autoresearch/run_experiment.sh "..."`，或对脚本 `chmod +x`。

---

## 三、数据与列含义（`weather.csv`）

数据文件：`dataset/weather/weather.csv`  
时间列：`date`（约 10 分钟间隔）。  
其后 **21 个数值列**为原始变量（与 TSLib `custom` + `enc_in=21` 一致）。列名与常见含义如下（与 Jena 风格气象站字段一致；若与论文表述略有出入，以列名与单位为准）：

| 列名 | 含义（简要） |
|------|----------------|
| `p (mbar)` | 气压 |
| `T (degC)` | 气温 |
| `Tpot (K)` | 位温 |
| `Tdew (degC)` | 露点温度 |
| `rh (%)` | 相对湿度 |
| `VPmax (mbar)` | 饱和水汽压（或最大可承受水汽压，依数据定义） |
| `VPact (mbar)` | 实际水汽压 |
| `VPdef (mbar)` | 饱和差 / 水汽压差（VPmax 与 VPact 相关） |
| `sh (g/kg)` | 比湿 |
| `H2OC (mmol/mol)` | 水汽摩尔浓度 |
| `rho (g/m**3)` | 空气密度 |
| `wv (m/s)` | 风速 |
| `max. wv (m/s)` | 风速最大值（窗口内） |
| `wd (deg)` | 风向（度） |
| `rain (mm)` | 降雨量 |
| `raining (s)` | 降雨持续时间等 |
| `SWDR (W/m)` | 短波向下辐射（编码显示可能为乱码，实为 W/m² 量级） |
| `PAR (mol/m/s)` | 光合有效辐射 |
| `max. PAR (mol/m/s)` | PAR 最大值 |
| `Tlog (degC)` | 与温度相关的变换量（数据集中已给出） |
| `OT` | 目标或综合输出变量（本数据集中为数值列；具体定义以数据集说明为准） |

**异常值**：部分列（常见为 `wv (m/s)`、`OT`）可能出现 **`-9999`** 等传感器哨兵值。  
**本框架约定**：写入训练 CSV 的 **原始 21 维** 与 `weather.csv` **完全一致**（含哨兵）；`clean_df` 仅用于在 `compute_features` 内计算 `llm_*`，避免派生列被 `-9999` 污染。

---

## 四、核心目标与 score

对每个 horizon `h ∈ {96,192,336,720}`：

```text
mse_improve_h = (baseline_mse_h - trial_mse_h) / baseline_mse_h
```

**主分数**：

```text
score = median(mse_improve_96, mse_improve_192, mse_improve_336, mse_improve_720)
```

- `score > 0`：多数 horizon 上 MSE 有改善倾向。
- `score < 0`：整体变差倾向。

`append_results_all.py` 根据历史最优判定 `keep` / `discard`（见第八节）。

---

## 五、固定实验口径（与 `prepare.py` 一致）

| 项 | 值 |
|----|-----|
| 任务 | `long_term_forecast` |
| 数据 | `dataset/weather/weather.csv`（baseline）；trial 为 `generated_data/` 下生成 CSV |
| `data` | `custom` |
| `features` | `M` |
| `seq_len` | 96 |
| `label_len` | 48 |
| `pred_len` | 默认 96/192/336/720（可由 CLI 覆盖） |
| 模型 | 默认 **`DLinear`**（更快；可用 `--model` 覆盖；**改模型后须重跑 baseline**） |
| `e_layers/d_layers/factor` | 3 / 1 / 3 |
| `d_model/d_ff` | 512 / 512 |
| trial 输入维 | `21 + 特征数` |
| trial 评估维 | `--eval_dims 21`（只优化与度量 **原始 21 维**） |

---

## 六、文件职责（agent 边界）

| 文件 | 是否可改 | 说明 |
|------|-----------|------|
| `dig.py` | **仅此处可改** | `FEATURE_SET_NAME` + `compute_features(raw_df, clean_df)` |
| `prepare.py` | **禁止** | 生成数据、调 `run.py`、解析 `mse/mae`、打印 `score` 等键值日志 |
| `run_experiment.sh` | **禁止**（除非修 bug） | 校验 **orphan 分支**（与 `main`/`master` 无共同祖先）→ commit（若有 git）→ `prepare.py --mode trial` → `append_results_all.py` |
| `append_results_all.py` | **禁止**（除非修 bug） | 解析日志、追加 `results_all.tsv` |
| `baselines.json` | 由 `prepare --mode baseline` 写入 | baseline 的 mse/mae |
| `results_all.tsv` | **只追加、不删改历史行** | 主账；**不要 commit** 到 git（若团队有约定） |
| `generated_data/*.csv` | 自动生成 | 每轮 trial 的增强表 |
| `logs/*.log` | 自动生成 | `prepare.py --log-file …` 写入；trial 经 `run_experiment.sh` 为 `experiment_<UTC>.log`，baseline 常用 `baseline_<UTC>.log`（`--log-file auto`） |

---

## 七、`dig.py` 编写契约

1. **必须实现**：

   ```python
   FEATURE_SET_NAME = "唯一英文标识，用于文件名与结果表"

   def compute_features(raw_df, clean_df) -> pd.DataFrame:
       ...
   ```

2. **`raw_df`**：原始读入，**不得原地修改**影响落盘列（若需中间变量请拷贝）。

3. **`clean_df`**：与 `raw_df` 同索引；数值列已做哨兵剔除与 `ffill/bfill`，**仅用于**计算 `llm_*`。

4. **返回**：行数与 `raw_df` 完全一致；列仅为新增特征；列名 **全部** 以 `llm_` 开头；不得与 `raw_df` 列名重复；无 NaN、无 inf。

5. **每轮特征数量**：建议 **3–12** 个；过多易噪声与过拟合。

6. **严禁未来泄露**（与旧版一致）：

   - 禁止 `shift(-k)`、禁止 `rolling(center=True)`、禁止用未来标签构造特征。
   - 允许 `shift(k)`（k>0）、过去窗口 `rolling(..., min_periods=1)`、当前时刻跨列代数组合。

---

## 八、Keep / Discard（`append_results_all.py`）

从日志中读取 `score:` 行，与 `results_all.tsv` 中历史最优 `score` 比较：

```text
若 score 无法解析 → status = crash
若 score > best_score_before + min_improve → status = keep
否则 → status = discard
```

默认 `min_improve = 0.001`（0.1%）。覆盖示例：

```bash
python weather_feature_autoresearch/append_results_all.py --log ... --min-improve 0.002
```

---

## 九、命令速查

**Baseline（首次或改 horizon/epoch 后重做）**：

```bash
conda activate tslib
cd /data/nishome/xuhaochen/Time-Series-Library

python weather_feature_autoresearch/prepare.py \
  --mode baseline \
  --horizons 96 192 336 720 \
  --train-epochs 10 \
  --log-file auto
```

Baseline 的完整训练输出会写入 `weather_feature_autoresearch/logs/baseline_<UTC>.log`（由 `--log-file auto` 决定时间戳）。

**一轮 trial（先改 `dig.py`）**：有 git 时须已在 **orphan 分支**（与 `main`/`master` 无共同祖先），否则 `run_experiment.sh` 会退出并报错。脚本会为每轮自动传入 `--log-file`，完整会话写入 `weather_feature_autoresearch/logs/experiment_<UTC>.log`。

```bash
bash weather_feature_autoresearch/run_experiment.sh "本轮描述" \
  --horizons 96 192 336 720 \
  --train-epochs 10
```

**查看账本**：

```bash
column -t -s $'\t' weather_feature_autoresearch/results_all.tsv | less -S
```

**日志中抽取关键行**：

```bash
grep -E "^(feature_set_name|generated_data|input_dim|eval_dims|mse_|mse_improve_|score|positive_horizons|avg_mse_improve):" \
  weather_feature_autoresearch/logs/experiment_*.log | tail -n 50
```

---

## 十、给 agent 的一行摘要

激活 **tslib**，仓库根先完成 **Phase A**：`main` → **`git checkout --orphan autoresearch/<YYYYMMDD-weather>`** → `git reset` → `git add .` → 首 commit；**git log** 遵守硬规则；**baseline** 写好 **`baselines.json`**；**`results_all.tsv` 勿 commit**；汇报后 **等用户「开始/继续」** 再进 **Phase B**。Phase B 内 **`bash weather_feature_autoresearch/run_experiment.sh`**、只改 **`dig.py`**，每轮 **commit + 日志 + TSV 一行**，**持续多轮、勿每轮再问是否继续**；目标 **`score = median(各 horizon mse_improve)`**；不做消融子流程。
