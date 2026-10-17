"""GreenMate B06 · 训练评估脚本（开发步骤 2~6 全流程）。

跑完「窗口化 → 特征/标签 → 时间切分 → 训练+朴素基线对比 → 评估+错误分析 →
模型保存 → 最新窗口推理」；产出并落盘：窗口 CSV、模型 joblib、评估报告 md、
运行日志 run_log。模型保存后可直接重载推理，不会现场重训练。

用法（用 Python 3.12 运行）：
    py -3.12 b06_train.py
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import b06_ml as ml


def main() -> int:
    run_id = f"run-B06-{datetime.datetime.now().strftime('%H%M%S')}"
    print("=" * 72)
    print("GreenMate B06 · 传统机器学习闭环（训练 + 评估 + 保存）")
    print("=" * 72)

    # 1 窗口化
    samples = ml.load_samples()
    dataset = ml.build_windows(samples, ml.WINDOW_SIZE, ml.FORECAST_HORIZON)
    dataset.to_csv(ml.DATASET_PATH, index=False, encoding="utf-8-sig")
    print(f"[1] 窗口样本：{len(dataset)} 条 → {ml.DATASET_PATH}")

    # 2 特征/标签（已内建于 build_windows）
    print(f"[2] 特征数 {len(ml.FEATURE_COLS)}（仅输入窗口），标签 next_window_attention")

    # 3 时间切分
    split = ml.time_split(samples, dataset)
    train_df, test_df = split["train"], split["test"]
    train_dist = ml.class_distribution(train_df)
    test_dist = ml.class_distribution(test_df)
    print(f"[3] 切分时间 {split['split_time']}；训练集 {len(train_df)}（{train_dist}）、"
          f"测试集 {len(test_df)}（{test_dist}）、边界丢弃 {split['boundary_count']}")

    if train_dist.get("NORMAL", 0) == 0 or train_dist.get("WATCH", 0) == 0:
        print("[!] 训练集缺少某个类别，无法训练有意义的模型 → 中止")
        return 1

    # 4 训练 + 评估
    baseline, model = ml.train_models(train_df)
    base_result = ml.evaluate(baseline, test_df)
    model_result = ml.evaluate(model, test_df)
    errors = ml.error_analysis(model, test_df, model_result)
    importances = ml.feature_importance(model)
    print(f"[4] 基线 F1={base_result['f1_macro']:.4f} → 模型 F1={model_result['f1_macro']:.4f}")
    print(f"    混淆矩阵(模型):\n{np.array2string(model_result['cm'])}")
    print(f"    错误样本 {len(errors)} 个")

    # 5 保存
    window_config = {"window_size": ml.WINDOW_SIZE, "horizon": ml.FORECAST_HORIZON}
    ml.save_bundle(model, ml.FEATURE_COLS, window_config, metrics=model_result)
    print(f"[5] 模型已保存 → {ml.MODEL_PATH}（版本 {ml.MODEL_VERSION}）")

    # 6 最新窗口推理（模拟冷加载，验证可重载）
    fresh_bundle = ml.load_bundle(ml.MODEL_PATH)
    infer = ml.infer_latest(fresh_bundle)
    print(f"[6] 最新窗口推理：predicted_attention={infer['predicted_attention']}，"
          f"model_version={infer['model_version']}")

    # 报告 + 日志
    report_lines = _build_report(split, train_dist, test_dist, base_result,
                                 model_result, errors, importances, infer)
    with open(ml.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log = ml.format_run_log(run_id, split, train_dist, test_dist, base_result,
                            model_result, errors, infer, importances)
    ml.write_log(log)
    print(f"[+] 评估报告 → {ml.REPORT_PATH}")
    print(f"[+] 运行日志 → {ml.LOG_PATH}")
    print("\n" + "\n".join(report_lines))
    return 0


def _build_report(split, train_dist, test_dist, base_result, model_result,
                  errors, importances, infer) -> list:
    lines = [
        "# GreenMate B06 评估报告 · 传统机器学习闭环",
        "",
        f"模型版本: {ml.MODEL_VERSION}",
        f"窗口: 输入 {ml.WINDOW_SIZE} 点(3h) → 预测 {ml.FORECAST_HORIZON} 点(1h)",
        f"切分时间: {split['split_time']}（前 5 天训练 / 后 2 天测试，边界丢弃 {split['boundary_count']}）",
        "",
        "## 数据分布",
        f"- 训练集 {train_dist}",
        f"- 测试集 {test_dist}",
        "",
        "## 朴素基线 vs 随机森林（测试集）",
        "| 指标 | 基线 | 模型 |",
        "|------|------|------|",
        f"| F1(macro) | {base_result['f1_macro']:.4f} | {model_result['f1_macro']:.4f} |",
        f"| F1(WATCH) | {base_result['f1_watch']:.4f} | {model_result['f1_watch']:.4f} |",
        f"| Recall(WATCH) | {base_result['recall_watch']:.4f} | {model_result['recall_watch']:.4f} |",
        "",
        "## 混淆矩阵（模型）",
        "",
        f"| 实际\\预测 | NORMAL | WATCH |",
        f"|-----------|--------|-------|",
        f"| NORMAL | {model_result['cm'][0,0]} | {model_result['cm'][0,1]} |",
        f"| WATCH | {model_result['cm'][1,0]} | {model_result['cm'][1,1]} |",
        "",
        "## 核心特征重要性",
        "| 特征 | 重要性 |",
        "|------|--------|",
    ]
    for it in importances[:8]:
        lines.append(f"| {it['feature']} | {it['importance']} |")
    lines += ["", "## 错误样本分析"]
    if not errors:
        lines.append("（无错分样本，需进一步排查数据泄露）")
    for e in errors:
        lines.append(f"- [{e['kind']}] {e['window_id']}（实际 {e['actual']} → 预测 {e['predicted']}）：{e['explanation']}")
    lines += [
        "",
        "## 最新窗口推理",
        f"- predicted_attention = {infer['predicted_attention']}",
        f"- model_version = {infer['model_version']}",
        f"- 概率 = {infer['probability']}",
    ]
    return lines


if __name__ == "__main__":
    sys.exit(main())