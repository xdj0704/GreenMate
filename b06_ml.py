"""GreenMate B06 — 传统机器学习闭环（核心模块）。

业务目标：基于 B02 历史环境时序数据，用最近 3 小时（6 个采样点）窗口，预测未来
1~2 小时的 next_window_attention（NORMAL / WATCH）。完成一套可复现、可保存重载、
可解释的传统机器学习闭环。禁止深度学习 / YOLO 等大模型。

技术栈：Python + pandas + scikit-learn（本文件被 b06_windowing.py / b06_train.py /
test_b06.py 复用，不直接作为 CLI 入口）。

开发步骤落实：
  1 窗口化       build_windows()
  2 特征+标签    extract_features() / compute_label()（严格无未来值泄露）
  3 时间切分     time_split()（前 5 天训练 / 后 2 天测试，边界样本丢弃）
  4 训练+评估    train_models() / evaluate() / error_analysis()
  5 保存+重载    save_bundle() / load_bundle()
  6 新窗口推理   predict_window() / infer_latest()

注意：本项目其余阶段（B02~B05）运行在 Python 3.14，但由于 pandas / scikit-learn
尚无 3.14 预编译 wheel，本阶段必须用 Python 3.12 运行：
    "C:\\Program Files\\Python312\\python.exe" b06_train.py
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
)

# ── 配置（可复现的关键元信息）─────────────────────────────────────────────────
WINDOW_SIZE = 6            # 输入窗口点数（3 小时 @ 30min）
FORECAST_HORIZON = 2       # 预测未来点数（1 小时），可 2~4（1~2 小时）
TRAIN_DAYS = 5             # 前 5 天训练，后 2 天测试（7 天数据）
MODEL_VERSION = "GreenMate-B06-RF-v1.0"

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greenmate.db")
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b06_windows.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b06_model.joblib")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b06_eval_report.md")
LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiment_logs", "B06_run_log.txt"
)

CHANNELS = ["temperature", "humidity", "light"]

# 标签判定阈值（可解释的「注意」边界，覆盖 B02 全部 5 类扰动、并对齐 B04 知识库适宜区间）
# 正常基线：温 26℃ / 湿 65% / 光 12000lux；扰动段：高温高湿、低温、干燥+强光、弱光、干燥。
ATTENTION_RULES = {
    "temperature_high": 30.0,     # 超上限（高温热害）
    "temperature_low": 18.0,      # 低于下限（低温授粉不良）
    "humidity_high": 80.0,        # 空气过湿
    "humidity_low": 55.0,         # 空气过干（正常 65%，留裕量）
    "light_low": 5000.0,          # 弱光（光合不足）
    "light_high": 30000.0,        # 异常强光（正常 ~12000lux，3 万以上为异常）
}


def _feature_names() -> List[str]:
    """特征列（固定顺序，模型保存/重载依赖此顺序一致）。"""
    cols: List[str] = []
    for c in CHANNELS:
        cols += [f"{c}_current", f"{c}_mean", f"{c}_delta",
                 f"{c}_std", f"{c}_slope", f"{c}_range"]
    return cols


FEATURE_COLS: List[str] = _feature_names()


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_samples(db_path: str = DB_PATH) -> pd.DataFrame:
    """从 B02 的 greenmate.db 读取环境时序，按时间戳升序返回。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, temperature, humidity, light "
        "FROM environment_samples ORDER BY timestamp ASC"
    )
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["timestamp", "temperature", "humidity", "light"])
    for c in CHANNELS:
        df[c] = df[c].astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 2：特征 + 标签（严格无数据泄露）
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(win_df: pd.DataFrame) -> Dict[str, float]:
    """从「输入窗口（仅过去）」提取特征。

    只允许使用 window 内已发生的观测值；**未来原始值 / sim_scenario / 天气 /
    知识库信息一律不进入特征**，防止数据泄露。
    """
    feat: Dict[str, float] = {}
    for c in CHANNELS:
        s = win_df[c].astype(float).reset_index(drop=True)
        n = len(s)
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, s.values, 1)[0]) if n >= 2 else 0.0
        feat[f"{c}_current"] = float(s.iloc[-1])
        feat[f"{c}_mean"] = float(s.mean())
        feat[f"{c}_delta"] = float(s.iloc[-1] - s.iloc[0])
        feat[f"{c}_std"] = float(s.std(ddof=0))
        feat[f"{c}_slope"] = slope
        feat[f"{c}_range"] = float(s.max() - s.min())
    return feat


def compute_label(future_records: List[Dict[str, float]]) -> str:
    """基于「未来窗口」判定 next_window_attention（NORMAL / WATCH）。

    标签本身必须依赖未来（这是预测目标，不构成泄露）；特征侧绝不使用未来值，
    二者严格分开。
    """
    if not future_records:
        return "NORMAL"
    n = len(future_records)
    avg_t = sum(r["temperature"] for r in future_records) / n
    avg_h = sum(r["humidity"] for r in future_records) / n
    avg_l = sum(r["light"] for r in future_records) / n
    if avg_t > ATTENTION_RULES["temperature_high"] or avg_t < ATTENTION_RULES["temperature_low"]:
        return "WATCH"
    if avg_h > ATTENTION_RULES["humidity_high"] or avg_h < ATTENTION_RULES["humidity_low"]:
        return "WATCH"
    if avg_l < ATTENTION_RULES["light_low"] or avg_l > ATTENTION_RULES["light_high"]:
        return "WATCH"
    return "NORMAL"


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 1：窗口化
# ══════════════════════════════════════════════════════════════════════════════
def build_windows(sample_df: pd.DataFrame,
                  window_size: int = WINDOW_SIZE,
                  horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """按时间戳排序后滑窗：输入 window_size 个点，标签取未来 horizon 个点。"""
    df = sample_df.sort_values("timestamp").reset_index(drop=True)
    records: List[Dict[str, Any]] = []
    n = len(df)
    for i in range(n - window_size - horizon + 1):
        win = df.iloc[i:i + window_size]
        fut = df.iloc[i + window_size:i + window_size + horizon]
        feat = extract_features(win)
        label = compute_label(fut.to_dict("records"))
        rec: Dict[str, Any] = {
            "window_id": f"w{i:04d}",
            "window_start": win["timestamp"].iloc[0],
            "window_end": win["timestamp"].iloc[-1],
            "label_start": fut["timestamp"].iloc[0],
            "label_end": fut["timestamp"].iloc[-1],
            "label": label,
        }
        rec.update(feat)
        records.append(rec)
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 3：按时间切分（前 5 天训练 / 后 2 天测试，边界样本丢弃）
# ══════════════════════════════════════════════════════════════════════════════
def time_split(sample_df: pd.DataFrame, dataset_df: pd.DataFrame,
               train_days: int = TRAIN_DAYS) -> Dict[str, Any]:
    """按时间切分，丢弃跨边界的样本，保证训练/测试时间不重叠。"""
    train_points = train_days * 48  # 每天 48 个 30min 采样点
    split_time = sample_df["timestamp"].iloc[train_points]

    df = dataset_df.copy()
    df["set"] = "boundary"
    # 整段（输入窗口 + 标签未来）都在训练期 → train
    df.loc[df["label_end"] <= split_time, "set"] = "train"
    # 整段都在测试期 → test
    df.loc[df["window_start"] >= split_time, "set"] = "test"

    train_df = df[df["set"] == "train"].drop(columns="set").reset_index(drop=True)
    test_df = df[df["set"] == "test"].drop(columns="set").reset_index(drop=True)
    n_boundary = int((df["set"] == "boundary").sum())
    return {"train": train_df, "test": test_df, "boundary_count": n_boundary,
            "split_time": split_time}


def class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {"NORMAL": 0, "WATCH": 0}
    return df["label"].value_counts().to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 4：训练 + 评估
# ══════════════════════════════════════════════════════════════════════════════
def train_models(train_df: pd.DataFrame):
    """朴素基线（多数类）+ 传统机器学习模型（随机森林，非深度学习）。"""
    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X, y)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced", n_jobs=1,
    )
    model.fit(X, y)
    return baseline, model


def evaluate(model, df: pd.DataFrame) -> Dict[str, Any]:
    X = df[FEATURE_COLS].values
    y_true = df["label"].values
    y_pred = model.predict(X)
    labels = ["NORMAL", "WATCH"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_watch = f1_score(y_true, y_pred, labels=labels, pos_label="WATCH", zero_division=0)
    rec_watch = recall_score(y_true, y_pred, labels=labels, pos_label="WATCH", zero_division=0)
    return {
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "cm": cm,
        "f1_macro": f1_macro,
        "recall_macro": rec_macro,
        "f1_watch": f1_watch,
        "recall_watch": rec_watch,
    }


def error_analysis(model, test_df: pd.DataFrame, result: Dict[str, Any],
                   limit: int = 3) -> List[Dict[str, Any]]:
    """找出假阳性/假阴性样本并给出可解释的归因（步骤 4 的错误分析）。"""
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    out: List[Dict[str, Any]] = []
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        row = test_df.iloc[idx]
        out.append({
            "window_id": row["window_id"],
            "window_end": row["window_end"],
            "actual": t,
            "predicted": p,
            "kind": "假阴性(FN)" if t == "WATCH" else "假阳性(FP)",
            "temp_last": round(float(row["temperature_current"]), 1),
            "hum_last": round(float(row["humidity_current"]), 1),
            "light_last": round(float(row["light_current"]), 1),
            "explanation": _explain_error(t, p, row),
        })
        if len(out) >= limit:
            break
    return out


def _explain_error(actual: str, predicted: str, row: pd.Series) -> str:
    t = float(row["temperature_current"])
    h = float(row["humidity_current"])
    l = float(row["light_current"])
    end_dev: List[str] = []
    if t > ATTENTION_RULES["temperature_high"]:
        end_dev.append(f"高温{t:.1f}℃")
    if t < ATTENTION_RULES["temperature_low"]:
        end_dev.append(f"低温{t:.1f}℃")
    if h > ATTENTION_RULES["humidity_high"]:
        end_dev.append(f"高湿{h:.0f}%")
    if h < ATTENTION_RULES["humidity_low"]:
        end_dev.append(f"干燥{h:.0f}%")
    if l < ATTENTION_RULES["light_low"]:
        end_dev.append(f"弱光{l:.0f}lux")
    if l > ATTENTION_RULES["light_high"]:
        end_dev.append(f"强光{l:.0f}lux")

    if actual == "WATCH" and predicted == "NORMAL":  # 假阴性 FN
        if end_dev:
            return (
                f"输入窗口末端已现扰动迹象（{'、'.join(end_dev)}），模型未据此升级为 WATCH，"
                "属对弱扰动的召回不足（假阴性）——特征仅用过去窗口，非数据泄露。"
            )
        return (
            f"输入窗口末端仍处正常带（温{t:.1f}℃/湿{h:.0f}%/光{l:.0f}lux），"
            "扰动发生在未来 1 小时——模型仅见过去 6 点，无法提前看到窗口外突变，"
            "属「预测提前量不足」的固有假阴性，非数据泄露。"
        )
    # 假阳性 FP
    dev = "、".join(end_dev) if end_dev else "轻微波动"
    return (
        f"输入窗口末端已见扰动迹象（{dev}），但未来 1 小时向正常回归，"
        "模型基于当前异常特征过度预警，属「扰动尾部」造成的假阳性。"
    )


def feature_importance(model, feature_cols=None) -> List[Dict[str, Any]]:
    cols = feature_cols or FEATURE_COLS
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    return [{"feature": cols[i], "importance": round(float(imp[i]), 4)} for i in order]


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 5：模型保存 / 重载
# ══════════════════════════════════════════════════════════════════════════════
def save_bundle(model, feature_cols: List[str], window_config: Dict[str, int],
                metrics: Optional[Dict[str, Any]] = None,
                path: str = MODEL_PATH) -> str:
    bundle = {
        "model_version": MODEL_VERSION,
        "model": model,
        "feature_cols": feature_cols,
        "window_config": window_config,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {},
    }
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str = MODEL_PATH) -> Dict[str, Any]:
    """重载模型（完全重启后的冷加载，不在现场重训练）。"""
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 6：新窗口推理
# ══════════════════════════════════════════════════════════════════════════════
def predict_window(bundle: Dict[str, Any], win_df: pd.DataFrame) -> Dict[str, Any]:
    """对任意「输入窗口」执行推理，返回 predicted_attention + model_version。"""
    feat = extract_features(win_df)
    X = pd.DataFrame([feat])[bundle["feature_cols"]]
    pred = bundle["model"].predict(X.values)[0]
    proba = bundle["model"].predict_proba(X.values)[0]
    classes = list(bundle["model"].classes_)
    return {
        "source": "MODEL_OUTPUT",
        "predicted_attention": str(pred),
        "model_version": bundle["model_version"],
        "probability": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }


def infer_latest(bundle: Dict[str, Any], db_path: str = DB_PATH,
                 window_size: int | None = None) -> Dict[str, Any]:
    """取最新 window_size 个采样点构造特征并推理（步骤 6）。"""
    ws = window_size or bundle["window_config"].get("window_size", WINDOW_SIZE)
    samples = load_samples(db_path).tail(ws).reset_index(drop=True)
    result = predict_window(bundle, samples)
    result["window_start"] = samples["timestamp"].iloc[0]
    result["window_end"] = samples["timestamp"].iloc[-1]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 运行日志（与 S1~S5 run_log 格式统一，供案例复现）
# ══════════════════════════════════════════════════════════════════════════════
def format_run_log(run_id: str, split_info: Dict[str, Any],
                   train_dist: Dict[str, int], test_dist: Dict[str, int],
                   baseline_result: Dict[str, Any], model_result: Dict[str, Any],
                   errors: List[Dict[str, Any]], infer_result: Optional[Dict[str, Any]],
                   importances: List[Dict[str, Any]]) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _cm_table(cm) -> str:
        lines = [
            "| 实际\\预测 | NORMAL | WATCH |",
            "|-----------|--------|-------|",
            f"| NORMAL | {int(cm[0, 0])} | {int(cm[0, 1])} |",
            f"| WATCH  | {int(cm[1, 0])} | {int(cm[1, 1])} |",
        ]
        return "\n".join(lines)

    lines = [
        "# GreenMate B06 传统机器学习闭环 · 运行日志",
        "",
        f"run_id: {run_id}",
        f"运行时间: {ts}",
        f"模型: 随机森林 RandomForestClassifier（传统机器学习，非深度学习）",
        f"模型版本: {MODEL_VERSION}",
        f"窗口配置: 输入 {WINDOW_SIZE} 点（3h）→ 预测 {FORECAST_HORIZON} 点（1h）→ next_window_attention",
        "",
        "## 数据集切分（前 5 天训练 / 后 2 天测试，时间不重叠）",
        "",
        f"切分时间: {split_info['split_time']}",
        f"边界丢弃样本: {split_info['boundary_count']}",
        f"训练集: {train_dist}",
        f"测试集: {test_dist}",
        "",
        "## 朴素基线 vs 模型（测试集）",
        "",
        "| 指标 | 朴素基线(多数类) | 随机森林 |",
        "|------|----------------|----------|",
        f"| F1(macro) | {baseline_result['f1_macro']:.4f} | {model_result['f1_macro']:.4f} |",
        f"| Recall(macro) | {baseline_result['recall_macro']:.4f} | {model_result['recall_macro']:.4f} |",
        f"| F1(WATCH) | {baseline_result['f1_watch']:.4f} | {model_result['f1_watch']:.4f} |",
        f"| Recall(WATCH) | {baseline_result['recall_watch']:.4f} | {model_result['recall_watch']:.4f} |",
        "",
        "## 混淆矩阵（测试集 · 模型）",
        "",
        _cm_table(model_result["cm"]),
        "",
        "## 核心特征重要性（可解释性）",
        "",
        "| 特征 | 重要性 |",
        "|------|--------|",
    ]
    for it in importances[:8]:
        lines.append(f"| {it['feature']} | {it['importance']} |")
    lines += [
        "",
        "## 错误样本分析（假阳性/假阴性归因）",
        "",
    ]
    if not errors:
        lines.append("（测试集无错分样本；应进一步核对数据泄露风险）")
    for e in errors:
        lines.append(f"- [{e['kind']}] {e['window_id']}：实际 {e['actual']} / 预测 {e['predicted']}，"
                     f"窗口末端 温{e['temp_last']}℃/湿{e['hum_last']}%/光{e['light_last']}lux。"
                     f"解释：{e['explanation']}")
    lines += [
        "",
        "## 最新窗口推理（MODEL_OUTPUT，独立于 B03 天气 / B04 知识库）",
        "",
    ]
    if infer_result:
        lines.append(f"- predicted_attention: {infer_result['predicted_attention']}")
        lines.append(f"- model_version: {infer_result['model_version']}")
        lines.append(f"- 概率: {infer_result['probability']}")
        lines.append(f"- 窗口: {infer_result['window_start']} ~ {infer_result['window_end']}")
    else:
        lines.append("- （未执行推理）")
    lines += ["", "---"]
    return "\n".join(lines)


def write_log(log: str, path: str = LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(log)