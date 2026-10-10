"""渲染层：把场景定义 + 计算上下文渲染为四大视图与自然语言输出。"""

import json

from greenmate.model import trend_stats


def _fmt_1(v):
    return f"{v:.1f}"


def build_context(series):
    """从时序数据计算渲染上下文（数值已格式化为字符串）。"""
    times = [r[0] for r in series]
    temps = [r[1] for r in series]
    hums = [r[2] for r in series]
    lights = [r[3] for r in series]
    co2s = [r[4] for r in series]

    t = trend_stats(temps)
    hrs = len(temps) - 1

    def f1(v):
        return f"{v:.1f}"

    def f0(v):
        return f"{v:.0f}"

    return {
        "t_start": f1(t["start"]),
        "t_end": f1(t["end"]),
        "t_hist_end": f1(temps[-2]),
        "h_start": f0(hums[0]),
        "h_end": f0(hums[-1]),
        "h_hist_end": f0(hums[-2]),
        "co2_start": str(co2s[0]),
        "co2_end": str(co2s[-1]),
        "co2_hist_end": str(co2s[-2]),
        "light_start": str(lights[0]),
        "light_end": str(lights[-1]),
        "light_hist_end": str(lights[-2]),
        "t_rate": f"{(temps[-1] - temps[0]) / hrs:+.2f}",
        "h_rate": f"{(hums[-1] - hums[0]) / hrs:+.2f}",
        "r_t": f"{t['r']:.3f}",
        "t_delta_full": f"{t['delta']:+.1f}",
        "t_delta_hist": f"{temps[-2] - temps[0]:+.1f}",
        "h_delta_full": f"{hums[-1] - hums[0]:+.0f}",
        "h_delta_hist": f"{hums[-2] - hums[0]:+.0f}",
        "times": times,
        "temps": temps,
    }


def render_meta(sc):
    return "\n".join([
        "### 1 运行元信息",
        "",
        "```",
        f"run_id: {sc['run_id']}",
        f"analysis_time: {sc['analysis_time']}",
        f"location:       {sc['location']}",
        f"当前激活场景:  {sc['id']}（{sc['name']}）",
        f"引擎版本:      {sc['engine_version']}",
        "```",
        "",
    ])


def render_input(sc, ctx):
    series = sc["timeseries"]
    lines = [
        "输入数据（引擎自动生成）：",
        "",
        "| 时刻 | 温度(°C) | 湿度(%RH) | 光照(lux) | CO₂(ppm) |",
        "|------|---------|----------|----------|----------|",
    ]
    for idx, (tm, tp, hm, lt, c2) in enumerate(series):
        if idx == len(series) - 1:
            lines.append(f"| **{tm}（当前）** | **{tp}** | **{hm}** | **{lt}** | **{c2}** |")
        else:
            lines.append(f"| {tm} | {tp} | {hm} | {lt} | {c2} |")
    lines += [
        "",
        f"外部天气预报：{sc['weather_text']}",
        f"RAG检索：{sc['rag']['source']} — {sc['rag_summary']}",
        f"MODEL_OUTPUT：{sc['model_output_line']}",
        "",
    ]
    return "\n".join(lines)


def build_judgment_doc(sc, ctx):
    """构建结构化判断文档（占位符已填充），供 Markdown 渲染与 Web 层复用。"""
    rendered = []
    for e in sc["evidence_list"]:
        d = dict(e)
        d["result"] = e["result"].format(**ctx)
        rendered.append(d)

    return {
        "current_state": sc["current_state"],
        "attention_level": sc["attention_level"],
        "trend_description": sc["trend_description"].format(**ctx),
        "short_term_forecast": sc["short_term_forecast"],
        "evidence_integrity": sc["evidence_integrity"],
        "judge_limitations": sc["judge_limitations"],
        "evidence_list": rendered,
    }


def render_judgment(sc, ctx):
    body = json.dumps(build_judgment_doc(sc, ctx), ensure_ascii=False, indent=2)
    return "### 2 结构化判断\n\n```json\n" + body + "\n```\n"


def render_view_a(sc, ctx):
    va = sc["view_a"]
    lines = [
        "#### 视图A：状态总览【现在怎样？】",
        "",
        "```",
        "┌─────────────────────────────────────────────────────┐",
        f"│  {sc['run_id']} · {sc['location']} · {sc['analysis_time']}   │",
        f"│  状态：{sc['current_state']}        关注等级：{sc['attention_icon']} {sc['attention_level']}          │",
        "├──────────────┬──────────────────────────────────────┤",
        "│  参数        │  当前值趋势        状态      │",
        "├──────────────┼──────────────────────────────────────┤",
    ]
    for name, value_tpl, trend_tpl, status in va["rows"]:
        value = value_tpl.format(**ctx)
        trend = trend_tpl.format(**ctx)
        lines.append(f"│  {name}        │  {value}         {trend}   {status} │")
    lines += [
        "├──────────────┴──────────────────────────────────────┤",
        f"│  短时预测：{va['short_forecast']}                │",
        f"│  明日风险：{va['risk']}            │",
        f"│  证据完整度：{va['integrity']}               │",
        "└─────────────────────────────────────────────────────┘",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_view_b(sc, ctx):
    vb = sc["view_b"]
    times = "   ".join(ctx["times"])
    temps_line = "   ".join(f"{t:.1f}" for t in ctx["temps"])
    lines = [
        "#### 视图B：状态演化【为什么变成这样？】",
        "",
        vb.get("intro", "这是一段持续6小时的双参数同向单调漂移，演化路径清晰："),
        "",
        "```",
        "温度演化轨迹（°C）",
        "时间   " + times,
        "温度   " + temps_line,
        "",
        " " + vb["limit_line"],
        "```",
        "",
        "**关键演化节点：**",
        "",
    ]
    for n in vb["nodes"]:
        lines.append(f"- {n}")
    lines += [
        "",
        f"**漂移驱动分析**：{vb['driver']}",
        "",
        f"**{vb['significance_title']}**：{vb['significance']}",
        "",
    ]
    return "\n".join(lines)


def render_view_c(sc, ctx):
    vc = sc["view_c"]
    lines = [
        "#### 视图C：证据空间【凭什么这样判断？】",
        "",
        "**▌支持“" + vc["support_title"] + "”的证据**",
        "",
        "| 证据 | 内容 | 支持力度 |",
        "|------|------|---------|",
    ]
    for label, content, strength in vc["support"]:
        lines.append(f"| {label} | {content} | {strength} |")
    lines += [
        "",
        "**▌反向提示证据（缓和因素）**",
        "",
        "| 证据 | 内容 |",
        "|------|------|",
    ]
    for label, content in vc["mitigate"]:
        lines.append(f"| {label} | {content} |")
    lines += [
        "",
        "**▌证据不足项**",
        "",
    ]
    for g in vc["gaps"]:
        lines.append(f"- {g}")
    lines += [
        "",
        f"**▌无分歧项**：{vc['no_dispute']}",
        "",
    ]
    return "\n".join(lines)


def render_view_d(sc, ctx):
    vd = sc["view_d"]
    lines = [
        "#### 视图D：场景回放与对比【判断是否可验证？】",
        "",
        f"**本次输入条件（{sc['run_id']}）：**",
        "",
    ]
    for c in vd["conditions"]:
        lines.append(f"- {c}")
    lines += [
        "",
        "**复现验证方案：**",
        "",
        "```",
    ]
    for item in vd["verifications"]:
        title, inp, expect = item[0], item[1], item[2]
        goal = item[3] if len(item) > 3 else ""
        lines.append(title)
        lines.append(f"  {inp}")
        lines.append(f"  {expect}")
        if goal:
            lines.append(f"  → {goal}")
        lines.append("")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_narrative(sc, ctx):
    return "\n".join([
        "### 4 自然语言综合解释",
        "",
        f"基于 `{sc['run_id']}` 结构化判断：",
        "",
        sc["narrative_body"],
        "",
        f"> {sc['narrative_closing']}",
        "",
    ])


def render_all(sc, ctx):
    parts = [
        "# GreenMate 连续状态认知工作台 · 实况演示",
        "",
        render_meta(sc),
        render_input(sc, ctx),
        render_judgment(sc, ctx),
        "### 3 四大视图",
        "",
        render_view_a(sc, ctx),
        render_view_b(sc, ctx),
        render_view_c(sc, ctx),
        render_view_d(sc, ctx),
        render_narrative(sc, ctx),
    ]
    return "\n".join(parts)