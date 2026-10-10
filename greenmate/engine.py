"""运行器：编排一次场景运行的完整流程。"""

from greenmate import scenarios, render, ai_analysis


def run(scenario_id):
    sc = scenarios.get_scenario(scenario_id)
    if sc is None:
        return _stub_message(scenario_id)

    ctx = render.build_context(sc["timeseries"])
    raw = render.render_all(sc, ctx)
    return raw + "\n" + ai_analysis.analyze(sc)


def _stub_message(scenario_id):
    sid = scenario_id.upper()
    known = scenarios.all_scenario_ids()
    name = scenarios.SCENARIO_NAMES.get(sid, "未知")
    lines = [
        "# GreenMate 连续状态认知工作台 · 实况演示",
        "",
        f"当前请求场景：{sid}（{name}）",
        "",
        "该场景尚未定义，请先在 `greenmate/scenarios.py` 中补充其输入数据、外部证据与判断规则。",
        "",
        f"已登记场景：{'、'.join(known)}（5类场景均已实现）",
        "",
    ]
    return "\n".join(lines)