# GreenMate B06 — 传统机器学习闭环（环境注意力预测）

> 计科 NOVA 竞赛 · GreenMate 番茄生长环境认知系统 · B06 阶段

基于 B02 历史环境时序数据，用「最近 3 小时（6 个采样点）」窗口，预测未来 1~2 小时
的 `next_window_attention`（`NORMAL` / `WATCH`）。完成一套**可复现、可保存重载、
可解释**的传统机器学习闭环；**禁止深度学习 / YOLO 等大模型**。

技术栈：`Python + pandas + scikit-learn`。

## 重要：运行环境

本阶段依赖 pandas / scikit-learn，而它们**尚无 Python 3.14 预编译 wheel**，因此
必须用 **Python 3.12** 运行（其余 B02~B05 仍用 3.14 不受影响）：

```powershell
py -3.12 b06_train.py        # 全流程：训练 + 评估 + 保存 + 推理
py -3.12 test_b06.py         # 三组测试用例
py -3.12 b06_windowing.py    # 单独窗口化/重新生成样本
```

## 交付资产

```
b06_ml.py                      # 核心：窗口化/特征/标签/切分/训练/评估/保存重载/推理/日志
b06_windowing.py               # 窗口化处理脚本（步骤 1）
b06_train.py                   # 训练评估脚本（步骤 2~6 全流程）
test_b06.py                    # 三组测试用例
b06_windows.csv                # 生成的窗口化样本数据集
b06_model.joblib               # 保存的模型（含版本/特征列表/窗口配置）
b06_eval_report.md             # 评估报告（含混淆矩阵/基线对比/错误分析）
experiment_logs/B06_run_log.txt# 运行日志（与 S1~S5 格式统一）
B06_README.md
```

## 六步开发流程

| 步骤 | 实现 | 载体 |
|---|---|---|
| 1 窗口化 | 输入 6 点(3h) → 预测 2~4 点(1~2h)，window_size 可改 | `build_windows()` |
| 2 特征+标签 | current/mean/delta/std/slope/range ×3 通道 =18 维；标签 NORMAL/WATCH | `extract_features()` / `compute_label()` |
| 3 时间切分 | 前 5 天训练 / 后 2 天测试，边界样本丢弃，时间不重叠 | `time_split()` |
| 4 训练+评估 | 多数类基线 vs 随机森林；混淆矩阵/F1/召回率 + 错误分析 | `train_models()` / `evaluate()` / `error_analysis()` |
| 5 保存+重载 | 保存模型/特征列表/窗口配置；冷加载不重训练 | `save_bundle()` / `load_bundle()` |
| 6 新窗口推理 | 最新 6 点构造特征 → 预测 + model_version，标记 MODEL_OUTPUT | `predict_window()` / `infer_latest()` |

## 标签规则（可解释，覆盖 B02 全部 5 类扰动）

`next_window_attention = WATCH` 当未来窗口均值满足任一条件：

| 条件 | 阈值 | 对应扰动 |
|---|---|---|
| 温度 | >30℃ 或 <18℃ | 高温热害 / 低温授粉不良 |
| 湿度 | >80% 或 <55% | 高湿 / 干燥 |
| 光照 | <5000lux 或 >30000lux | 弱光 / 异常强光 |

正常基线（温 26℃ / 湿 65% / 光 12000lux）远离所有阈值，标签边界干净。

## 防数据泄露（关键要求）

- 特征 **只来自输入窗口（过去 6 点）**：`current / mean / delta / std / slope / range`；
- **未来原始值、sim_scenario、B03 天气、B04 知识库一律不进入特征**；
- 标签基于未来窗口单独计算，与特征侧时间隔离（标签本身依赖未来是预测目标，不构成泄露）。

## 三组测试用例

- **TestA 标准时序样本**：加载模型 → 对最新 6 点构造特征 → 推理，输出 `predicted_attention` 与 `model_version`。
- **TestB 错误边界样本**：定位假阳性/假阴性，给出可解释归因（区分「预测提前量不足」与「扰动尾部」）。
- **TestC 模型重载测试**：保存模型后，用**全新子进程冷加载**（模拟完全重启）→ 推理 → 校验版本一致且不现场重训练。

## 实测结果（可复现）

- 训练集 234（NORMAL 169 / WATCH 65），测试集 89（NORMAL 72 / WATCH 17），边界丢弃 6。
- F1(macro)：基线 0.4472 → 随机森林 **0.7110**；F1(WATCH)：0 → **0.5000**。
- 混淆矩阵：`[[71 1],[11 6]]`——6 个真 WATCH 命中，11 个假阴性多属「预测提前量不足」（扰动出现在未来 1 小时而输入窗口仍正常），1 个假阳性属「扰动尾部」。

## 核心特征（可解释）

`temperature_current`、`light_current`、`humidity_current` 为最重要的 3 个特征——
它们直接编码了输入窗口最后时刻的环境状态，与注意力判定高度相关。