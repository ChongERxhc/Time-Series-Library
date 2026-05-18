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

1) 创建当天 orphan 实验分支（每次新会话一条 orphan 线）
- 标签用 YYYYMMDD-weather（如 20260518-weather），分支名：`auto/<tag>`
- **硬规则：创建 orphan 之前不要切换分支**——禁止先 `git checkout main`、`git checkout master` 或其它非实验分支；在**当前所在分支**（通常为 `autoresearch` 或上一轮 `auto/<旧tag>`）上**直接**执行 `--orphan`，不要绕道 `main`。
- **创建前检查**（在仓库根目录）：
  - `git branch --show-current` 不得为 `main` / `master`。
  - 若当前在 `main`/`master`：先 `git checkout autoresearch`（或上一实验 orphan 分支），**再**创建新 orphan；**禁止**从 `main` 上执行 `git checkout --orphan`。
- **创建命令**（确认不在 `main`/`master` 后）：
  ```bash
  git checkout --orphan auto/<tag>
  git reset
  git add .
  git commit -m "chore: init auto/<tag> root commit"
  ```
- Phase A / Phase B 的 baseline 与 trial 均在本次 **`auto/<tag>`** orphan 分支上进行（与 `main`/`master` 无共同祖先）。

2) 历史可见性约束（硬规则）
- 禁止：git log --all、git reflog、按历史 hash 的 git show
- 只允许：git log --oneline -n 20（仅当前实验分支）

3) 阅读并确认边界
- weather_feature_autoresearch/dig.py（唯一编辑区：FEATURE_SET_NAME + compute_features）
- prepare.py、run_experiment.sh、append_results_all.py（固定框架与记录口径，除非用户明确要求修 bug，否则不改）

4) 运行环境检查
- 使用 tslib conda 环境（conda activate tslib；仓库根见上文路径）
- 生成 baseline（建议同时落盘日志）：`python weather_feature_autoresearch/prepare.py --mode baseline --train-epochs 10 --log-file auto`（默认仅 `pred_len=96`；日志在 `weather_feature_autoresearch/logs/baseline_<UTC>.log`）
- 确认 `weather_feature_autoresearch/baselines.json` 在 `"96"` 下存在 **`val_mse`**（旧版 test `mse` 需重跑 baseline）

5) 审计文件规则
- weather_feature_autoresearch/results_all.tsv：每次 trial 运行都要追加一行，禁止 commit

6) setup 完成后先汇报，等待用户确认
- 必须输出 setup 检查结果（当前分支名、最近 5 条 commit、baselines.json 是否存在及 `96` 的 baseline **val_mse** 摘要）。
- 明确询问：是否进入正式实验循环？
- 在用户未明确回复「开始/继续」前：禁止修改 dig.py 中的特征、禁止运行 bash weather_feature_autoresearch/run_experiment.sh。

Phase B（仅在用户确认后执行）：

7) 进入实验循环
- 每轮最小单元：bash weather_feature_autoresearch/run_experiment.sh "本轮说明"（可选：--train-epochs 10 等，同 prepare.py；默认仅 horizon 96）
- 每轮必须留下：独立 commit（若有 git）+ `weather_feature_autoresearch/logs/experiment_<UTC>.log`（完整会话输出，由 `prepare.py --log-file` 写入）+ `results_all.tsv` 新增一行
- 已在 Phase B 内时：应持续多轮尝试，勿在每轮结束再次询问「是否继续」；勿擅自中断整条循环（除非用户明确要求停止）；单次崩溃可修复后重跑该轮；长期无 score 提升应换思路而非死磕同一组特征
- 仅排障可设 WEATHER_AR_SKIP_ORPHAN_CHECK=1 跳过 run_experiment.sh 的 orphan 校验（平时不要用）
```

**补充说明（与脚本一致）**：会话日志统一落在 **`weather_feature_autoresearch/logs/`**（`mkdir -p` 自动创建）。`run_experiment.sh` 会拒绝在 `main`/`master` 上运行，并校验当前分支与 `main`/`master` **无共同祖先**（须在 `auto/<tag>` 等 orphan 实验线上）；非 git 目录会跳过 commit 与该校验，仍会写日志与 `results_all.tsv`。每次**新会话**：在 **`autoresearch`（或当前非 main 分支）上**按 1) 创建 `auto/<tag>`，**不要**先 `checkout main`。若 `run_experiment.sh` 报 `Permission denied`，改用 `bash weather_feature_autoresearch/run_experiment.sh "..."`，或对脚本 `chmod +x`。

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

**开发阶段固定 `pred_len = 96`**（每轮 trial 只训练一次）。**优化目标为验证集**，不用测试集做特征选择。

从 TSLib 训练日志中解析每个 epoch 的 `Vali Loss`（验证集 MSE），取 **最小值** 作为 `trial_val_mse`（与 EarlyStopping 按 val 存 checkpoint 一致）：

```text
val_improve = (baseline_val_mse - trial_val_mse) / baseline_val_mse
score       = val_improve
```

- `score > 0`：相对 baseline，验证集 MSE 变好。
- `score < 0`：验证集变差。

日志末尾的 `test_mse` / `test_mae` **仅作审计参考**，**不参与** `score` 与 keep/discard。

`append_results_all.py` 根据历史最优 `score` 判定 `keep` / `discard`（见第八节）。**测试集**仅在日后对终选特征集做一次性终评（本框架暂不自动跑 final 模式）。

---

## 五、固定实验口径（与 `prepare.py` 一致）

| 项 | 值 |
|----|-----|
| 任务 | `long_term_forecast` |
| 数据 | `dataset/weather/weather.csv`（baseline）；trial 写入 **`generated_data/_staging.csv`**（覆盖写，默认跑完删除） |
| `data` | `custom` |
| `features` | `M` |
| `seq_len` | 96 |
| `label_len` | 48 |
| `pred_len` | 默认 **96**（`--horizons` 可覆盖；trial 评分只用第一个 horizon） |
| 模型 | 默认 **`iTransformer`**（跨变量交互，更适合加列特征；可用 `--model DLinear` 等覆盖；**改模型后须重跑 baseline**） |
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
| `baselines.json` | 由 `prepare --mode baseline` 写入 | 各 horizon 的 **`val_mse`**（及可选 `test_mse` 审计） |
| `results_all.tsv` | **只追加、不删改历史行** | 主账；**不要 commit** 到 git；**列 schema 已更新**，旧表需归档后重建 |
| `generated_data/_staging.csv` | 每轮 trial 临时写入 | 默认训练后删除；`--keep-staging` 保留并复制到 `archive/` |
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

默认 **`min_improve = 0`**：只要 `score` 严格高于历史最优即 `keep`，不要求额外相对提升幅度。若日后需要更保守，可显式加大阈值，例如：

```bash
python weather_feature_autoresearch/append_results_all.py --log ... --min-improve 0.001
```

---

## 九、命令速查

**Baseline（首次、升级框架后、或改 horizon/epoch/model 后重做）**：

```bash
conda activate tslib
cd /data/nishome/xuhaochen/Time-Series-Library

python weather_feature_autoresearch/prepare.py \
  --mode baseline \
  --train-epochs 10 \
  --log-file auto
```

Baseline 的完整训练输出会写入 `weather_feature_autoresearch/logs/baseline_<UTC>.log`（由 `--log-file auto` 决定时间戳）。

**一轮 trial（先改 `dig.py`）**：有 git 时须已在 **orphan 分支**（与 `main`/`master` 无共同祖先），否则 `run_experiment.sh` 会退出并报错。脚本会为每轮自动传入 `--log-file`，完整会话写入 `weather_feature_autoresearch/logs/experiment_<UTC>.log`。

```bash
bash weather_feature_autoresearch/run_experiment.sh "本轮描述" \
  --train-epochs 10
```

**查看账本**：

```bash
column -t -s $'\t' weather_feature_autoresearch/results_all.tsv | less -S
```

**日志中抽取关键行**：

```bash
grep -E "^(feature_set_name|feature_fp|n_features|staging_csv|baseline_val_mse|val_mse|val_improve|score|test_mse):" \
  weather_feature_autoresearch/logs/experiment_*.log | tail -n 50
```

---

## 十、给 agent 的一行摘要

激活 **tslib**，仓库根先完成 **Phase A**：在 **`autoresearch`（勿先切 `main`）** 上 **`git checkout --orphan auto/<YYYYMMDD-weather>`** → 首 commit → baseline；**git log** 遵守硬规则；**`baselines.json` 含 val_mse@96**；**`results_all.tsv` 勿 commit**；汇报后等 **「开始/继续」** 再 **Phase B**（在 **`auto/<tag>`** 上 `run_experiment.sh`，只改 **`dig.py`**）；**score = 验证集相对 baseline 改善 @ pl=96**；**`_staging.csv` 不落盘堆积**。
