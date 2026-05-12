# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言偏好

使用中文与用户互动。

## 项目概述

Time-Series-Library (TSLib) 是一个深度时间序列分析库，支持五种主流任务：

- **长期预测** (long_term_forecast)
- **短期预测** (short_term_forecast)
- **缺失值填充** (imputation)
- **异常检测** (anomaly_detection)
- **分类** (classification)
- **零样本预测** (zero_shot_forecast)

## 环境配置

```bash
conda create -n tslib python=3.11
conda activate tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 运行规则

**重要**：运行任何 Python 实验前，必须先激活 tslib 环境：

```bash
conda activate tslib
```

## 常用命令

### 运行实验

```bash
python -u run.py --task_name <任务名> --is_training 1 --model <模型名> --data <数据集> --root_path <数据路径>
```

### 快速测试（1个epoch）

```bash
# 长期预测
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test --model DLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 2

# 分类
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model TimesNet --data UEA --train_epochs 1 --num_workers 0
```

### 使用脚本运行

```bash
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
bash ./scripts/classification/TimesNet.sh
```

## 架构

```
run.py              # 统一入口，解析参数并分发任务到对应的Exp类
exp/                # 实验管道
  exp_basic.py      # 基类，自动扫描models/目录加载模型
  exp_long_term_forecasting.py
  exp_short_term_forecasting.py
  exp_imputation.py
  exp_anomaly_detection.py
  exp_classification.py
  exp_zero_shot_forecasting.py
models/             # 所有模型实现（自动被发现）
  TimesNet.py, TimeMixer.py, iTransformer.py, etc.
layers/             # 可复用组件：注意力、卷积、Embedding等
data_provider/      # 数据加载
  data_factory.py   # 根据任务返回正确的DataLoader
  data_loader.py    # 滑动窗口逻辑
  uea.py, m4.py     # 特定数据集解析
utils/              # 工具：metrics.py, tools.py, augmentation.py
scripts/            # Bash脚本，用于复现实验结果
```

## 关键设计

1. **模型自动发现**：`Exp_Basic` 在初始化时扫描 `models/` 目录，所有 `.py` 文件（除 `__init__.py`）都会被自动加载。模型类名应为 `Model` 或与文件名相同。

2. **任务分发**：通过 `--task_name` 参数，`run.py` 动态导入对应的 `Exp_*` 类并执行训练/测试。

3. **数据格式**：

   - 长期预测/填充/异常检测：`./data/ETT-small/ETTh1.csv` 格式（时间戳 + 多变量）
   - 分类：UEA格式，依赖 `data_provider/uea.py`
   - 短期预测：M4格式，依赖 `data_provider/m4.py`

4. **新增模型**：在 `models/` 目录下添加新模型文件，遵循现有接口（`class Model`），即可通过 `--model <文件名>` 直接使用。

5. **GPU选择**：默认使用 CUDA，通过 `--gpu 0` 指定GPU，`--use_multi_gpu` 启用多卡。

## 依赖

主要依赖见 `requirements.txt`。部分模型有额外依赖：

- Mamba: 需要 `mamba_ssm`
- Moirai: 需要 `uni2ts`
- Chronos/TimesFM/TiRex等LTSM: 需要对应的预训练权重和库
