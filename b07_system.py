"""GreenMate B07 — 系统组装与产品化实践（System Assembly & Productization）。

定位：B07 不新增任何底层能力，只把 B01~B06 已交付的单项能力组装为一条
「可运行、可理解、可追溯」的完整链路，并输出产品化视图。

复用资产（全部为既有代码，本文件只做编排与呈现）：
  - B01  `greenmate.scenarios`   S1~S5 五类场景定义（时序 / 阈值 / 预期结论）
  - B01  `greenmate.model`       连续状态认知（趋势统计 trend_stats）
  - B02  `b02_data_memory`       环境历史 / 当前观测（SQLite）
  - B03  `weather`               外部天气预报（Open-Meteo PUBLIC）
  - B04  `b04_rag`               番茄知识库 RAG 检索证据
  - B05  `b05_tools` + `b05_agent` 工具注册 / 真实执行
  - B06  `b06_ml`                传统 ML 模型 MODEL_OUTPUT（可选，依赖 3.12）

固定执行顺序（B07 硬约束）：
    ① 工具结果 → ② 证据记录 → ③ 结构化判断 → ④ LLM 自然语言解释

用法：
    python b07_system.py                       # 全场景 S1~S5，生成报告 / JSON / 日志
    python b07_system.py --location 南京        # 指定外部天气查询地点
    python b07_system.py --no-llm              # 跳过 LLM 调用（降级为模板解释）

产物（相对本文件目录，即 E:\bo1_project 根目录）：
    B07_系统报告.html        产品化视图（运行状态 / 过程 / 证据 / 对照结果）
    b07_judgments.json       五场景结构化判断记录（可对比校验）
    B07_run_log.txt          完整可追溯运行日志
"""

from __future__ import annotations

import datetime
import html
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Windows GBK 终端兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── 复用既有资产（B01 / B05）──────────────────────────────────────────────────
from greenmate import scenarios                 # B01：S1~S5 场景定义
from greenmate.model import trend_stats        # B01：连续状态认知
from b05_tools import ToolRegistry             # B05：工具注册与真实执行

BASE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE, "B07_系统报告.html")
JSON_PATH = os.path.join(BASE, "b07_judgments.json")
LOG_PATH = os.path.join(BASE, "B07_run_log.txt")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

# 场景使用的知识检索问题（用于④专业知识证据，真实调用 B04 RAG）
RAG_QUERY = "番茄花期最适温度与湿度范围"

# ── B06 可选导入（pandas/scikit-learn 仅 3.12 有预编译 wheel；缺失时⑤降级）──
try:
    import b06_ml as ml
    _ML_IMPORT_OK = True
except Exception as _ml_exc:  # pragma: no cover - 环境缺失
    ml = None
    _ML_IMPORT_OK = False
    _ML_IMPORT_ERR = str(_ml_exc)


# ══════════════════════════════════════════════════════════════════════════════
# 固定顺序 stage 标记（正文与 HTML 共用）
# ══════════════════════════════════════════════════════════════════════════════
STAGE_NAMES = ["工具结果", "证据记录", "结构化判断", "LLM自然语言解释"]

# 五类证据编号 → 中文名（沿用 B01 的严格分类）
EVIDENCE_TYPES = ["①历史观测数据", "②当前实时观测", "③外部天气证据",
                  "④专业知识证据", "⑤模型预测MODEL_OUTPUT"]


def _llm_available() -> bool:
    key = LLM_API_KEY or ""
    return bool(key) and len(key) > 10 and "your" not in key.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 数据容器
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class EvidenceRecord:
    """一条规范化证据记录（stage ② 产物）。"""
    type: str                  # 五类证据编号中文名
    source: str                # 来源描述
    status: str                # ok / partial / unavailable / error
    summary: str               # 证据内容摘要（真实数据）
    citation: str = ""         # 引用（RAG 证据带来源）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredJudgment:
    """结构化判断（stage ③ 产物），确定性、可对比。"""
    scenario_id: str
    current_state: str         # 正常稳定 / 渐进漂移 / 突变异常 / 数据缺失 / 证据不足
    attention_level: str       # 正常 / 中等关注 / 高关注 / 观察中
    evidence_integrity: str    # 完整 / 不完整
    degraded: bool             # 是否发生可信降级
    degradation_reason: str    # 降级原因（未降级则为空）
    trend: Dict[str, Any]      # 连续状态认知指标（slope / r / delta / monotonic）
    breaches: List[str]        # 越限参数列表
    limits: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioRun:
    """单场景一次完整四阶段运行结果。"""
    scenario_id: str
    scenario_name: str
    declared_state: str          # 场景声明结论（对照基准）
    declared_level: str
    tools: Dict[str, Any]        # stage ① 工具结果
    evidence: List[EvidenceRecord]  # stage ② 证据记录
    judgment: StructuredJudgment     # stage ③ 结构化判断
    explanation: str             # stage ④ LLM 解释
    judge_limitations: str       # 判断局限（来自场景定义）
    match_with_declared: bool    # 计算结论是否与场景声明一致（对照）

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Stage ① 工具结果 —— 真实执行（复用 B05 工具注册表 + B06 模型）
# ══════════════════════════════════════════════════════════════════════════════
def collect_tool_results(location: str) -> Dict[str, Any]:
    """真实调用各能力，返回结构化工具结果；每个工具独立降级，失败不中断。"""
    registry = ToolRegistry()
    tools: Dict[str, Any] = {}

    # B02：当前环境 / 历史时序
    tools["get_current_environment"] = registry.execute("get_current_environment", {})
    tools["get_environment_history"] = registry.execute("get_environment_history", {"hours": 24})
    # B03：外部天气
    tools["get_weather_forecast"] = registry.execute("get_weather_forecast", {"location": location})
    # B04：知识检索
    tools["search_tomato_knowledge"] = registry.execute(
        "search_tomato_knowledge", {"query": RAG_QUERY, "top_k": 3},
    )
    # B06：模型预测（可选，缺失时明确标注）
    tools["model_prediction"] = _run_model_inference()
    return tools


def _run_model_inference() -> Dict[str, Any]:
    """B06 模型推理（可选）。依赖 3.12 环境；缺失则显式降级为不可用。"""
    if not _ML_IMPORT_OK:
        return {"status": "unavailable",
                "error": f"B06 依赖不可用（pandas/scikit-learn 需 Python 3.12）：{_ML_IMPORT_ERR}"}
    if not (os.path.exists(ml.MODEL_PATH) and os.path.exists(ml.DB_PATH)):
        return {"status": "unavailable",
                "error": "未找到 B06 模型文件 b06_model.joblib 或数据库 greenmate.db"}
    try:
        bundle = ml.load_bundle(ml.MODEL_PATH)
        infer = ml.infer_latest(bundle)
        infer["status"] = "ok"
        return infer
    except Exception as exc:  # 模型重载/推理异常 → 显式降级
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage ② 证据记录 —— 场景证据可用性 + 真实内容注入
# ══════════════════════════════════════════════════════════════════════════════
def build_evidence(sc: Dict[str, Any], tools: Dict[str, Any]) -> List[EvidenceRecord]:
    """把场景证据可用性（S4 缺数据 / S5 缺 RAG）与真实工具内容合并。

    场景 evidence_list 声明了五类证据各自的 status（这是 5 类情境的"设定"），
    ①/②/④ 尊重该设定；③ 外部天气、⑤ 模型预测则注入最新真实工具结果。
    """
    # 从场景时序计算连续状态指标（真实计算）
    series = sc["timeseries"]
    temps = [float(r[1]) for r in series]
    hums = [float(r[2]) for r in series]
    lights = [float(r[3]) for r in series]
    co2s = [float(r[4]) for r in series]
    t = trend_stats(temps)

    # 场景原始证据（按类型建立索引，保留其 status / source）
    declared = {e["source_type"]: e for e in sc["evidence_list"]}

    records: List[EvidenceRecord] = []
    for etype in EVIDENCE_TYPES:
        decl = declared.get(etype, {})
        status = decl.get("status", "ok")
        source = decl.get("source", "—")

        if etype == "①历史观测数据":
            summary = (f"共 {len(temps)} 个观测点，温度 {temps[0]:.1f}→{temps[-1]:.1f}°C"
                       f"（{t['delta']:+.1f}°C），湿度 {hums[0]:.0f}%→{hums[-1]:.0f}%，"
                       f"线性相关 r={t['r']:.3f}，{'单调' if t['monotonic'] else '非单调'}")
            citation = ""
        elif etype == "②当前实时观测":
            summary = f"温度 {temps[-1]:.1f}°C，湿度 {hums[-1]:.0f}%RH，光照 {lights[-1]:.0f}lux，CO₂ {co2s[-1]:.0f}ppm"
            citation = ""
        elif etype == "③外部天气证据":
            w = tools.get("get_weather_forecast", {})
            if w.get("status") == "ok":
                status = "ok"
                source = "Open-Meteo 天气预报 API（PUBLIC，未来预报，尚未发生）"
                summary = (f"{w.get('location')} 预报 {w.get('forecast_time')}：室外温度 "
                           f"{w.get('temperature')}°C，湿度 {w.get('relative_humidity')}%，"
                           f"降水概率 {w.get('precipitation_probability')}%")
            else:
                status = "error"
                summary = f"天气预报获取失败：{w.get('error')}"
            citation = ""
        elif etype == "④专业知识证据":
            r = tools.get("search_tomato_knowledge", {})
            if status == "ok" and r.get("status") == "ok" and r.get("results"):
                top = r["results"][0]
                status = "ok"
                source = "番茄知识库 RAG（TF-IDF 检索，先验检索）"
                summary = top.get("text", "")[:80]
                citation = top.get("citation", "")
            elif status != "ok":
                # S5：尊重场景设定（RAG 不可用 → 证据不足）
                summary = "RAG 检索不可用，未获取花期适宜阈值（温度/湿度上下限未知）"
            else:
                status = "unavailable"
                summary = f"知识检索无结果：{r.get('error', '未命中')}"
            citation = citation
        elif etype == "⑤模型预测MODEL_OUTPUT":
            m = tools.get("model_prediction", {})
            if status == "ok" and m.get("status") == "ok":
                summary = (f"predicted_attention={m.get('predicted_attention')}，"
                           f"概率 {json.dumps(m.get('probability', {}), ensure_ascii=False)}，"
                           f"模型版本 {m.get('model_version')}")
                source = f"传统机器学习模型（{m.get('model_version', 'B06')}，仅辅助，不覆盖实测）"
            elif status != "ok":
                summary = "模型输出不可用（历史数据不足或依赖缺失）"
            else:
                status = "unavailable"
                summary = f"模型推理不可用：{m.get('error', '未配置')}"
            citation = ""

        records.append(EvidenceRecord(type=etype, source=source, status=status,
                                      summary=summary, citation=citation))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Stage ③ 结构化判断 —— 确定性规则（可对比校验）
# ══════════════════════════════════════════════════════════════════════════════
def _is_abrupt(vals: List[float]) -> bool:
    """突变检测：末步跳变显著大于历史步长（对应 S3 突发异常 vs S2 渐进漂移）。"""
    if len(vals) < 3:
        return False
    steps = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    last_step = steps[-1]
    prev = steps[:-1]
    prev_mean = sum(prev) / len(prev) if prev else 0.0
    prev_max = max(prev) if prev else 0.0
    return last_step > 1.0 and (last_step >= 3 * max(prev_mean, 0.15) or last_step - prev_max > 0.8)


def make_judgment(sc: Dict[str, Any], evidence: List[EvidenceRecord]) -> StructuredJudgment:
    """确定性结构化判断：连续状态认知 + 多源证据组织 + 可信降级。

    判断顺序固定（对应 B07 的可信降级优先序）：
      1) 历史观测不可用 / 数据点不足  → 数据缺失（可信降级）
      2) 专业知识不可用 / 阈值缺失     → 证据不足（可信降级）
      3) 阈值可得且数据完整            → 正常稳定 / 渐进漂移 / 突变异常
    """
    series = sc["timeseries"]
    temps = [float(r[1]) for r in series]
    hums = [float(r[2]) for r in series]
    lights = [float(r[3]) for r in series]
    t = trend_stats(temps)
    limits = sc.get("limits")

    status_map = {e.type: e.status for e in evidence}
    hist_status = status_map.get("①历史观测数据", "ok")
    rag_status = status_map.get("④专业知识证据", "ok")

    trend = {
        "points": len(temps),
        "t_start": round(temps[0], 1),
        "t_end": round(temps[-1], 1),
        "delta": round(t["delta"], 2),
        "rate_per_hour": round(t["rate_per_hour"], 3),
        "r": round(t["r"], 3),
        "monotonic": t["monotonic"],
    }

    ok_count = sum(1 for s in status_map.values() if s == "ok")
    integrity = "完整" if ok_count == 5 else "不完整"

    degraded = False
    degradation_reason = ""
    breaches: List[str] = []

    # ① 数据缺失（历史观测不可用 / 数据点不足以拟合趋势）
    if hist_status != "ok" or len(temps) < 4:
        state, level = "数据缺失", "观察中"
        degraded = True
        degradation_reason = "历史观测不足，无法可靠拟合趋势，结论降级为观察中"
    # ② 证据不足（知识阈值缺失）
    elif rag_status != "ok" or not limits:
        state, level = "证据不足", "观察中"
        degraded = True
        degradation_reason = "专业知识/阈值不可用，越限无法量化，结论退回人工评估"
    # ③ 阈值与数据齐备 → 连续状态认知
    else:
        margin_t = temps[-1] - limits.get("temp", 1e9)
        margin_h = hums[-1] - limits.get("hum", 1e9)
        if limits.get("temp") is not None and margin_t > 0:
            breaches.append(f"温度+{margin_t:.1f}°C")
        if limits.get("hum") is not None and margin_h > 0:
            breaches.append(f"湿度+{margin_h:.0f}%")

        if not breaches:
            state, level = "正常稳定", "正常"
        elif _is_abrupt(temps):
            state, level = "突变异常", "高关注"
        else:
            state, level = "渐进漂移", "中等关注"

    return StructuredJudgment(
        scenario_id=sc["id"],
        current_state=state,
        attention_level=level,
        evidence_integrity=integrity,
        degraded=degraded,
        degradation_reason=degradation_reason,
        trend=trend,
        breaches=breaches,
        limits=limits,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage ④ LLM 自然语言解释 —— 严格基于判断依据与限制，不新增事实
# ══════════════════════════════════════════════════════════════════════════════
def explain(sc: Dict[str, Any], judgment: StructuredJudgment,
            evidence: List[EvidenceRecord], use_llm: bool = True) -> str:
    """基于结构化判断 + 证据 + 局限生成自然语言解释。"""
    if not (use_llm and _llm_available()):
        return _fallback_explanation(sc, judgment)

    # 构造给 LLM 的输入：只给判断依据与限制，要求严格据此解读，禁止补造事实
    evidence_lines = "\n".join(
        f"- {e.type}（{e.status}）：{e.summary}" + (f"　引用：{e.citation}" if e.citation else "")
        for e in evidence
    )
    judgment_json = json.dumps(judgment.to_dict(), ensure_ascii=False, indent=2)
    prompt = (
        "你是 GreenMate 番茄温室环境分析助手。以下是系统已完成的结构化判断，"
        "请只依据给定的判断依据与限制，用一段中文自然语言向用户解释结果，"
        "严格做到：\n"
        "1) 只陈述给定判断中的事实（状态、关注等级、趋势、证据完整性、越限项）；\n"
        "2) 明确系统限制与不确定性，不额外发挥、不补造数据、不给出病害诊断或农业处置建议；\n"
        "3) 若发生可信降级，说明降级原因及其对结论置信度的影响。\n\n"
        f"【场景】{sc['id']} · {sc['name']}\n"
        f"【结构化判断】\n{judgment_json}\n\n"
        f"【证据记录】\n{evidence_lines}\n\n"
        f"【判断局限】\n{sc.get('judge_limitations', '无')}\n\n"
        "请输出自然语言解释（150~250 字）。"
    )
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是可信环境分析助手，只基于给定事实解释，禁止编造。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # LLM 异常 → 降级为模板解释
        return _fallback_explanation(sc, judgment, note=f"LLM调用失败：{exc}")


def _fallback_explanation(sc: Dict[str, Any], judgment: StructuredJudgment,
                          note: str = "LLM未配置") -> str:
    """无 LLM 时的确定性模板解释（严格基于判断字段，不编造）。"""
    tr = judgment.trend
    return (
        f"（{note}，以下为基于结构化判断的模板解释）\n"
        f"场景 {sc['id']}（{sc['name']}）判断为「{judgment.current_state}」，"
        f"关注等级「{judgment.attention_level}」，证据完整性「{judgment.evidence_integrity}」。"
        f"温度 {tr['t_start']}→{tr['t_end']}°C（{tr['delta']:+.2f}°C，速率 {tr['rate_per_hour']:+.3f}°C/h，"
        f"相关 r={tr['r']:.3f}）。"
        + (f"越限项：{'、'.join(judgment.breaches)}。" if judgment.breaches else "无越限项。")
        + (f"已发生可信降级：{judgment.degradation_reason}。" if judgment.degraded else "未发生可信降级。")
        + f"判断局限：{sc.get('judge_limitations', '无')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 编排：单场景四阶段运行 + 全场景对照
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(sc: Dict[str, Any], tools: Dict[str, Any], use_llm: bool) -> ScenarioRun:
    """按固定顺序执行一次场景的四阶段链路。"""
    evidence = build_evidence(sc, tools)          # ② 证据记录
    judgment = make_judgment(sc, evidence)        # ③ 结构化判断
    explanation = explain(sc, judgment, evidence, use_llm=use_llm)  # ④ LLM 解释

    declared_state = sc.get("current_state", "")
    declared_level = sc.get("attention_level", "")
    match = (judgment.current_state == declared_state)

    return ScenarioRun(
        scenario_id=sc["id"],
        scenario_name=sc["name"],
        declared_state=declared_state,
        declared_level=declared_level,
        tools=tools,                              # ① 工具结果（共享）
        evidence=evidence,
        judgment=judgment,
        explanation=explanation,
        judge_limitations=sc.get("judge_limitations", ""),
        match_with_declared=match,
    )


def run_all(location: str, use_llm: bool) -> List[ScenarioRun]:
    """S1~S5 全场景：工具结果只采集一次（共享外部证据），逐场景跑四阶段。"""
    tools = collect_tool_results(location)        # ① 工具结果（真实执行一次）
    runs = []
    for sid in scenarios.all_scenario_ids():
        runs.append(run_scenario(scenarios.get_scenario(sid), tools, use_llm))
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# 输出：结构化判断 JSON / 运行日志 / 产品化 HTML
# ══════════════════════════════════════════════════════════════════════════════
def _write_json(runs: List[ScenarioRun], location: str) -> None:
    payload = {
        "system": "GreenMate B07 系统组装",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "fixed_order": STAGE_NAMES,
        "runs": [r.to_dict() for r in runs],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_log(runs: List[ScenarioRun], tools: Dict[str, Any], location: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GreenMate B07 · 系统组装运行日志",
        "",
        f"生成时间: {ts}",
        f"天气查询地点: {location}",
        f"固定执行顺序: {' → '.join(STAGE_NAMES)}",
        "",
        "## ① 工具结果（真实执行）",
        "",
    ]
    for name, result in tools.items():
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(result, ensure_ascii=False, default=str))
        lines.append("```")
        lines.append("")

    for r in runs:
        lines += [
            f"## 场景 {r.scenario_id} · {r.scenario_name}",
            "",
            "### ② 证据记录",
            "",
            "| 证据类型 | 状态 | 来源 | 摘要 |",
            "|----------|------|------|------|",
        ]
        for e in r.evidence:
            lines.append(f"| {e.type} | {e.status} | {e.source} | {e.summary} |")
        lines += [
            "",
            "### ③ 结构化判断（JSON）",
            "",
            "```json",
            json.dumps(r.judgment.to_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
            "### ④ LLM 自然语言解释",
            "",
            r.explanation,
            "",
            f"> 场景声明结论：{r.declared_state} / {r.declared_level}；"
            f"计算结论：{r.judgment.current_state} / {r.judgment.attention_level}；"
            f"对照：{'✅ 一致' if r.match_with_declared else '❌ 偏差'}",
            "",
            "---",
            "",
        ]

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── HTML 产品化视图 ──────────────────────────────────────────────────────────
_LEVEL_CLASS = {"正常": "ok", "低关注": "ok", "中等关注": "warn", "高关注": "alert", "观察中": "watch"}
_STATUS_BADGE = {"ok": ("可用", "ok"), "partial": ("部分", "part"),
                 "unavailable": ("不可用", "un"), "error": ("错误", "un")}

_CSS = """
:root{--green:#2e7d32;--green-dark:#1b5e20;--green-soft:#e8f2e8;--ink:#22301f;
--muted:#6b7a66;--bg:#f2f6f1;--line:#dfe7dc;}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
background:var(--bg);color:var(--ink);line-height:1.65}
.topbar{background:linear-gradient(135deg,var(--green-dark),var(--green));color:#fff;padding:18px 24px}
.brand{font-size:22px;font-weight:700}
.sub{font-size:13px;opacity:.92;margin-top:4px}
main{max-width:1100px;margin:0 auto;padding:20px 22px 40px}
section{background:#fff;border-radius:10px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:18px}
h2{margin:0 0 14px;font-size:17px;color:var(--green-dark)}
h2::before{content:"▌";color:var(--green);margin-right:6px}
h3{font-size:15px;color:var(--green-dark);margin:18px 0 8px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{background:var(--green-soft);color:var(--green-dark);text-align:left;padding:8px 10px;border-bottom:2px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.st-ok{background:#e5f6e8;color:#15803d}.st-part{background:#fdf3e0;color:#b45309}
.st-un{background:#fde8e8;color:#b91c1c}
.hero{display:flex;gap:26px;flex-wrap:wrap;border-left:6px solid var(--green);margin-bottom:18px}
.hero .item .lb{font-size:12px;color:var(--muted)}
.hero .item .v{font-size:18px;font-weight:700}
.ok{color:#15803d}.warn{color:#b45309}.watch{color:#ea580c}.alert{color:#dc2626}
.chips{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}
.chip{background:var(--green-soft);border-radius:8px;padding:7px 11px;font-size:12.5px}
pre{background:#f6f8f5;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px}
details{margin:10px 0}summary{cursor:pointer;color:var(--green-dark);font-weight:600}
blockquote{margin:10px 0 0;padding:10px 14px;background:var(--green-soft);border-left:4px solid var(--green);border-radius:6px;font-size:14px}
footer{text-align:center;color:var(--muted);font-size:12.5px;padding-bottom:26px}
"""


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _render_html(runs: List[ScenarioRun], tools: Dict[str, Any], location: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 对照表
    compare_rows = []
    for r in runs:
        j = r.judgment
        cls = _LEVEL_CLASS.get(j.attention_level, "ok")
        match_txt = "✅ 一致" if r.match_with_declared else "❌ 偏差"
        compare_rows.append(
            f"<tr><td><b>{r.scenario_id}</b> {_esc(r.scenario_name)}</td>"
            f"<td>{_esc(r.declared_state)}</td>"
            f"<td class='{cls}'><b>{_esc(j.current_state)}</b></td>"
            f"<td class='{cls}'>{_esc(j.attention_level)}</td>"
            f"<td>{_esc(j.evidence_integrity)}</td>"
            f"<td>{_esc(j.trend['r'])} / {_esc(j.trend['rate_per_hour'])}</td>"
            f"<td>{match_txt}</td></tr>"
        )

    # 场景详情
    detail_blocks = []
    for r in runs:
        j = r.judgment
        ev_rows = "".join(
            f"<tr><td>{_esc(e.type)}</td>"
            f"<td><span class='pill st-{_STATUS_BADGE.get(e.status,('?','ok'))[1]}'>"
            f"{_STATUS_BADGE.get(e.status,('?','ok'))[0]}</span></td>"
            f"<td>{_esc(e.source)}</td><td>{_esc(e.summary)}</td>"
            f"<td>{_esc(e.citation)}</td></tr>"
            for e in r.evidence
        )
        cls = _LEVEL_CLASS.get(j.attention_level, "ok")
        degrade = (f"<blockquote>⚠ 可信降级：{_esc(j.degradation_reason)}</blockquote>"
                   if j.degraded else "")
        detail_blocks.append(f"""
<section>
  <h2>{r.scenario_id} · {_esc(r.scenario_name)}</h2>
  <div class="hero"><span class="item"><span class="lb">状态判断</span><br>
    <span class="v {cls}">{_esc(j.current_state)}</span></span>
    <span class="item"><span class="lb">关注等级</span><br>
    <span class="v {cls}">{_esc(j.attention_level)}</span></span>
    <span class="item"><span class="lb">证据完整度</span><br>
    <span class="v">{_esc(j.evidence_integrity)}</span></span>
    <span class="item"><span class="lb">趋势 / r</span><br>
    <span class="v">{_esc(j.trend['rate_per_hour'])}°C/h · r={_esc(j.trend['r'])}</span></span></div>
  {degrade}
  <h3>② 证据记录（五类证据，严格区分）</h3>
  <table><tr><th>证据类型</th><th>状态</th><th>来源</th><th>内容</th><th>引用</th></tr>{ev_rows}</table>
  <h3>③ 结构化判断（JSON，可对比校验）</h3>
  <pre>{_esc(json.dumps(j.to_dict(), ensure_ascii=False, indent=2))}</pre>
  <h3>④ LLM 自然语言解释</h3>
  <p>{_esc(r.explanation)}</p>
  <blockquote>场景声明结论：{_esc(r.declared_state)} / {_esc(r.declared_level)}　→　
  对照结果：{'✅ 一致' if r.match_with_declared else '❌ 偏差'}</blockquote>
</section>""")

    # 工具结果摘要
    tool_blocks = []
    for name, result in tools.items():
        st = result.get("status", "?")
        tool_blocks.append(
            f"<details><summary>{_esc(name)}　"
            f"<span class='pill st-{_STATUS_BADGE.get(st,('?','ok'))[1]}'>{_ESC_STATUS(st)}</span>"
            f"</summary><pre>{_esc(json.dumps(result, ensure_ascii=False, default=str))}</pre></details>"
        )

    boundary_rows = [
        ("环境状态识别", "温度/湿度/光照时序的状态、趋势、越限", "病害诊断、长势判断、产量预测"),
        ("外部天气", "未来 24h 预报作为扰动风险证据（明确未发生）", "将预报当作已发生事实"),
        ("知识检索", "仅返回知识库片段并附引用，领域限于已注册资料", "超范围常识补造、无来源结论"),
        ("模型预测", "仅 next_window_attention 辅助证据，不覆盖实测", "覆盖实测、替代规则判断"),
        ("多源证据", "五类证据严格区分、分歧显式表达、可信降级", "强行合并矛盾证据、虚构补全"),
    ]
    boundary_table = "".join(
        f"<tr><td><b>{_esc(a)}</b></td><td>{_esc(b)}</td><td>{_esc(c)}</td></tr>"
        for a, b, c in boundary_rows
    )

    all_tool = "".join(tool_blocks)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreenMate B07 · 系统组装与产品化</title><style>{_CSS}</style></head>
<body>
<header class="topbar">
  <div class="brand">GreenMate 连续状态认知工作台 · B07 系统组装</div>
  <div class="sub">生成时间：{ts}　·　天气查询地点：{_esc(location)}　·　固定顺序：{' → '.join(STAGE_NAMES)}　·　复用资产：B01~B06</div>
</header>
<main>

<section>
  <h2>五场景对照结果（结构化判断，可对比校验）</h2>
  <table><tr><th>场景</th><th>声明结论</th><th>计算结论</th><th>关注等级</th>
  <th>证据完整度</th><th>趋势 r / 速率</th><th>对照</th></tr>{''.join(compare_rows)}</table>
</section>

<section>
  <h2>① 工具结果（真实执行，可追溯）</h2>
  {all_tool}
</section>

{''.join(detail_blocks)}

<section>
  <h2>系统能力边界（第四遍 · 边界判断）</h2>
  <table><tr><th>能力</th><th>系统能做的（边界内）</th><th>系统明确不做的（边界外）</th></tr>{boundary_table}</table>
</section>

</main>
<footer>GreenMate B07 · 复用 B01~B06 资产，零新增底层能力 · 演示数据，判断输出仅供参考</footer>
</body></html>"""


def _ESC_STATUS(st: str) -> str:
    return {"ok": "可用", "partial": "部分", "unavailable": "不可用", "error": "错误"}.get(st, str(st))


def write_html(runs: List[ScenarioRun], tools: Dict[str, Any], location: str) -> None:
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(_render_html(runs, tools, location))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = sys.argv[1:]
    location = "南京"
    use_llm = True

    if "--location" in args:
        i = args.index("--location")
        if i + 1 < len(args):
            location = args[i + 1]
    if "--no-llm" in args:
        use_llm = False

    print("=" * 72)
    print("GreenMate B07 · 系统组装与产品化实践")
    print(f"固定执行顺序: {' → '.join(STAGE_NAMES)}")
    print(f"天气地点: {location}   LLM: {'启用' if use_llm else '跳过（模板解释）'}")
    print("=" * 72)

    runs = run_all(location, use_llm)
    tools = runs[0].tools if runs else {}

    _write_json(runs, location)
    _write_log(runs, tools, location)
    write_html(runs, tools, location)

    # 控制台对照摘要
    print("\n五场景对照（声明 → 计算）：")
    for r in runs:
        j = r.judgment
        mark = "✅" if r.match_with_declared else "❌"
        print(f"  {mark} {r.scenario_id} {r.scenario_name}："
              f"{r.declared_state} → {j.current_state}"
              f"（{j.attention_level}，证据{j.evidence_integrity}）")

    print(f"\n[+] 结构化判断记录 → {JSON_PATH}")
    print(f"[+] 产品化视图     → {HTML_PATH}")
    print(f"[+] 运行日志       → {LOG_PATH}")


if __name__ == "__main__":
    main()