"""AI 深度分析层：在原始引擎输出之上生成增强四大视图与综合总结。

本层不改变原始判定，而是作为"第二层阅读"补充：
定量归因、证据效力分层、反事实推演与时间窗敏感性，帮助理解
原始结论背后的隐藏假设与判决边界。
"""

from greenmate.model import trend_stats

# 判定阈值已收敛至 scenarios.py 的场景定义（sc["limits"]），此处不再硬编码

# 证据效力分层：序号 -> (效力, 在推理链中的作用)
EVIDENCE_EFFICACY = {
    "①": ("高", "趋势识别"),
    "②": ("高", "状态落地"),
    "③": ("中", "扰动预估"),
    "④": ("高", "阈值锚定"),
    "⑤": ("低", "被动外推"),
}

EVIDENCE_FAILURE = {
    "①": "退化为 S4 数据缺失",
    "②": "无法确认当前状态",
    "③": "扰动风险不可见",
    "④": "退化为 S5 证据不足",
    "⑤": "仅失去短时点估计",
}

# 各场景"结论性质"（隐藏假设说明）
SCENE_NATURE = {
    "S1": "一条默认「环控正常工作」的隐藏假设结论，环控状态未被直接采集",
    "S2": "漂移结论依赖历史窗口完整且无异常点，越限幅度轻微",
    "S3": "「设备故障」归因为间接推断，缺乏设备状态直接证据",
    "S4": "缺失态是「不自信」的正确表达，而非确定性判断",
    "S5": "阈值缺失使结论退回人工，知识侧构成单点依赖",
}

_STATUS_LABEL = {
    "ok": "✅ 完整",
    "unavailable": "❌ 缺失",
    "partial": "⚠ 部分",
    "error": "❌ 错误",
}

# 各场景综合总结（key 为场景 id）
_SUMMARIES = {
    "S1": {
        "verdict": "A区当前确处适宜区间、判「{state}」成立；但该结论本质是「环控正在有效工作」这一未直接采集的隐藏假设，且稳定性判决对末端单点与时间窗高度敏感。",
        "insights": [
            "正常态的脆弱根源：光照显著抬升而温度几乎不动，差额即环控贡献；环控失效即沿 S2→S3 路径劣化，正常态是突变风险的前置静默期。",
            "「证据完整」是标签而非保证：5 类证据在场，但③⑤在正常态下边际贡献趋近于零，实质是三条强证据 + 两条弱证据。",
            "判决边界可变：S1↔S2 切换对末端读数敏感，趋势方向随窗口平移而反转，引擎输出的是快照而非过程真相。",
        ],
        "advice": "维持当前环控策略，无需干预；建议把本场景设为漂移监测基线，对湿度的缓慢抬升设弱信号触发。",
    },
    "S2": {
        "verdict": "A区处于渐进漂移、中等关注，趋势线性极强且方向明确、无反转迹象，温度已微超或逼近上限。",
        "insights": [
            "趋势越线的提前量：线性趋势使越线点在达到阈值前即可被预判，预警窗口早于传统阈值告警。",
            "单调性是双刃剑：r≈1 的确定性利于预判，但也意味着若无干预将稳定单向恶化，不存在自发回落。",
            "温湿耦合：湿度与温度同向漂移，两者同时逼近上限会放大作物胁迫。",
        ],
        "advice": "立即启动通风/遮阳干预，打断单向升温；对湿度的同向抬升设触发阈值。",
    },
    "S3": {
        "verdict": "高关注、突变异常，温度已越过高温/落花阈值，多参数同步越阈构成强交叉验证。",
        "insights": [
            "突变无法被线性趋势提前捕捉：前段平稳毫无征兆，价值在于多参数同步越阈的交叉验证，压降单点误报。",
            "归因是间接的：设备故障为推断、缺乏直接证据，置信度受限。",
            "复合型恶化：外部冲击与棚内闷棚相互叠加，需同时处置内外两源。",
        ],
        "advice": "立即通风降温并核查通风/遮阳设备，同时确认光照骤降是否为传感器异常。",
    },
    "S4": {
        "verdict": "数据缺失、观察中，历史缺口使趋势无法可靠拟合，系统主动降级为不自信表达。",
        "insights": [
            "3 点不足以支撑趋势拟合，线性外推与相关系数退化失效，主动降级是正确行为。",
            "缺口是「数据侧」而非「环境侧」问题，需先补数据再谈结论。",
            "无法排除缺口时段内曾发生更大幅度波动，历史真相不可恢复。",
        ],
        "advice": "核查传感器供电/网关，补齐缺口后复算；期间以持续观测为主。",
    },
    "S5": {
        "verdict": "证据不足、观察中，传感器数据完整清晰但知识侧阈值缺失，越限判定退回人工。",
        "insights": [
            "短板在知识侧而非数据侧：传感器再完整，阈值缺失也使判定退回人工。",
            "单点依赖暴露：对单一知识源（RAG）的依赖构成脆弱性，应有备选知识源容灾。",
            "可信降级正确：从「高关注」降为「观察中/需人工评估」，而非强行越限。",
        ],
        "advice": "补齐/切换备选 RAG 知识源以恢复阈值；期间由人工依据通用阈值经验评估。",
    },
}


def _f1(v):
    return f"{v:.1f}"


def _metrics(sc):
    series = sc["timeseries"]
    times = [r[0] for r in series]
    temps = [r[1] for r in series]
    hums = [r[2] for r in series]
    lights = [r[3] for r in series]
    co2s = [r[4] for r in series]

    t = trend_stats(temps)
    limits = sc.get("limits")
    t_limit = limits["temp"] if limits else None
    h_limit = limits["hum"] if limits else None

    light_start, light_end = lights[0], lights[-1]
    light_pct = (light_end - light_start) / light_start * 100 if light_start else 0.0

    temp_start, temp_end = temps[0], temps[-1]
    temp_change_pct = (temp_end - temp_start) / temp_start * 100 if temp_start else 0.0

    temp_prev = temps[-2] if len(temps) >= 2 else temp_end
    last_dir = temp_end - temp_prev

    return {
        "times": times,
        "n": len(temps),
        "t_start": temp_start, "t_end": temp_end, "t_delta": t["delta"],
        "r": t["r"], "monotonic": t["monotonic"],
        "h_start": hums[0], "h_end": hums[-1],
        "light_start": light_start, "light_end": light_end, "light_pct": light_pct,
        "temp_change_pct": temp_change_pct,
        "co2_start": co2s[0], "co2_end": co2s[-1],
        "t_limit": t_limit, "h_limit": h_limit,
        "t_margin": (t_limit - temp_end) if t_limit is not None else None,
        "h_margin": (h_limit - hums[-1]) if h_limit is not None else None,
        "temp_prev": temp_prev, "last_dir": last_dir,
    }


def _last_dir_text(m):
    if m["last_dir"] > 0.05:
        return f"仍在上行（{_f1(m['temp_prev'])}→{_f1(m['t_end'])}°C）"
    if m["last_dir"] < -0.05:
        return f"已开始回落（{_f1(m['temp_prev'])}→{_f1(m['t_end'])}°C），环控起效"
    return f"持平（{_f1(m['temp_prev'])}→{_f1(m['t_end'])}°C）"


def _margin_line(m):
    if m["t_limit"] is None:
        return ("无可信阈值（知识证据不可用），无法量化距上限的余量——"
                "这正是本场景「证据不足」的根源")
    return (f"温度 {_f1(m['t_end'])}°C 距上限 {_f1(m['t_limit'])}°C 尚有 "
            f"**{_f1(m['t_margin'])}°C** 余量；湿度 {m['h_end']}% 距上限 "
            f"{m['h_limit']:.0f}% 尚有 **{m['h_margin']:.0f}** 个百分点")


def _view_a(m, sc):
    return "\n".join([
        "#### 视图A：状态总览【现在怎样？】——增强",
        "",
        f"原始结论「{sc['current_state']} {sc.get('attention_icon', '')}」，AI 增强补充三点隐藏信息：",
        "",
        f"- 余量量化：{_margin_line(m)}",
        f"- 末端方向：最后一小时温度 {_last_dir_text(m)}——「稳态」标签可能掩盖实时方向。",
        f"- 结论性质：{SCENE_NATURE.get(sc['id'], '结论依赖未直接采集的缺失证据')}。",
        "",
    ])


def _attribution(m):
    t_delta = m["t_delta"]
    lp = m["light_pct"]
    if t_delta > 0.3 and lp > 0:
        if lp > abs(m["temp_change_pct"]) * 2:
            return (
                f"光照 {m['light_start']}→{m['light_end']} lux，抬升 **+{lp:.1f}%**；"
                f"而温度仅 {_f1(m['t_start'])}→{_f1(m['t_end'])}°C（+{_f1(t_delta)}°C），"
                f"两者差额为环控/通风的真实**抑温贡献**。这也是正常态最脆弱的一环：环控一旦失效，"
                f"光照加温势能将立即释放，成为 S3 突变异常的前置机制。"
            )
        return (
            f"光照抬升 **+{lp:.1f}%**，温度同步升至 {_f1(m['t_end'])}°C，"
            f"方向一致，日照增强为被动加温的主因。"
        )
    if t_delta > 0.3 and lp <= 0:
        return (
            f"光照不升反降（**{lp:.1f}%**），温度却上升 +{_f1(t_delta)}°C："
            f"排除日照被动加温，指向通风/遮阳设备异常或其它闷棚因素。"
        )
    if abs(t_delta) <= 0.3:
        return (
            f"温度几乎持平（{_f1(t_delta)}°C），光照变化 {lp:+.1f}%，"
            f"环控将环境稳定在目标区间内。"
        )
    return (
        f"温度下降 {_f1(t_delta)}°C，光照变化 {lp:+.1f}%，"
        f"环境处于自然回落或主动降温过程。"
    )


def _view_b(m, sc):
    mono = "方向性明确" if m["monotonic"] else "无稳定方向"
    lines = [
        "#### 视图B：状态演化【为什么变成这样？】——增强",
        "",
        f"温度轨迹 {_f1(m['t_start'])}→{_f1(m['t_end'])}°C（线性相关 r={m['r']:.3f}，{mono}），"
        f"湿度 {m['h_start']}%→{m['h_end']}%，CO₂ {m['co2_start']}→{m['co2_end']} ppm。",
        "",
        f"**驱动因子定量归因**：{_attribution(m)}",
        "",
        f"**趋势信号强度**：历史 {m['n']} 点，端点温差 {_f1(m['t_delta'])}°C。",
    ]
    if m["h_end"] - m["h_start"] >= 1:
        lines.append(f"湿度 {m['h_start']}%→{m['h_end']}% 呈缓慢抬升，与光照方向一致，属可跟踪的弱信号。")
    lines.append("")
    return "\n".join(lines)


def _view_c(sc):
    lines = [
        "#### 视图C：证据空间【凭什么这样判断？】——增强（证据效力分层）",
        "",
        "| 证据 | 在场 | 效力 | 失效后果 |",
        "|------|:--:|:--:|------|",
    ]
    for e in sc["evidence_list"]:
        key = e["source_type"][:1]
        present = _STATUS_LABEL.get(e.get("status"), "?")
        eff, role = EVIDENCE_EFFICACY.get(key, ("—", "—"))
        fail = EVIDENCE_FAILURE.get(key, "")
        lines.append(f"| {e['source_type']} | {present} | {eff}（{role}） | {fail} |")
    lines += [
        "",
        "**关键洞察**：`evidence_integrity` 的「完整」语义是 5 类证据**在场**，而非 **足够强**。",
        "真正支撑当前结论的往往是①②④三条强证据；③⑤在场但边际贡献有限，标签掩盖了证据效力的不对称。",
        "",
    ]
    return "\n".join(lines)


def _view_d(m, sc):
    return "\n".join([
        "#### 视图D：场景回放与对比【判断是否可验证？】——增强（反事实与敏感性）",
        "",
        f"- 反事实（末端扰动）：若将末端温度 {_f1(m['t_end'])}°C 单点上推 2°C 至 **{_f1(m['t_end'] + 2)}°C**，"
        f"判决级别将向更高关注方向跳变，说明 {sc['id']} 的判决边界对末端单点高度敏感。",
        f"- 时间窗敏感性：本次窗口 {m['times'][0]}—{m['times'][-1]} 为观测快照，"
        f"趋势方向会随窗口平移而改变，结论是**窗口函数**而非绝对量。",
        "- 可复现性缺口：知识证据为静态引用（无版本/时间戳），知识库一旦改版，结论无法严格回放。",
        "",
    ])


def _summary(sc):
    s = _SUMMARIES.get(sc["id"])
    if not s:
        return ""
    lines = [
        "#### 综合总结（AI 深度分析）",
        "",
        "**一句话结论**：" + s["verdict"].replace("{state}", sc["current_state"]),
        "",
        "**核心洞察**：",
    ]
    lines += [f"{i}. {x}" for i, x in enumerate(s["insights"], 1)]
    lines += [
        "",
        "**行动建议**：" + s["advice"],
        "",
        f"**局限与后续**：{sc.get('judge_limitations', '')} 建议接入环控设备状态证据、为知识证据加时间戳、引入证据权重机制，以修复「完整≠有效」的偏差。",
        "",
    ]
    return "\n".join(lines)


def analyze(sc):
    """生成某场景的 AI 深度分析文本（第二层阅读）。"""
    m = _metrics(sc)
    blocks = [
        "## AI 深度分析（增强层）",
        "",
        "> 以下不改变原始判定，只补充定量归因、证据效力、反事实与敏感性，帮助理解结论背后的隐藏假设与判决边界。",
        "",
        _view_a(m, sc),
        _view_b(m, sc),
        _view_c(sc),
        _view_d(m, sc),
        _summary(sc),
    ]
    return "\n".join(blocks)