"""GreenMate B05 · 工具智能体测试脚本。

三组测试用例（严格对齐 B05 规范）：
  TestA【单工具任务】       查询最近 24 小时温室环境历史 → 真实调用 history 工具
  TestB【多工具任务】       本地环境 + 外部天气预报 → 评估风险（≥2 个工具协同）
  TestC【工具失败测试】     错误 location / 超时 / 空输入 → 明确失败证据，不伪造成功

运行：
    python test_b05.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b05_tools import ToolRegistry        # noqa: E402
from b05_agent import ToolAgent, format_run_log, write_log  # noqa: E402
from weather import fetch_weather          # noqa: E402


# ── 固定任务（步骤 1：先确保业务函数可独立运行；这里走智能体真实执行）────────
TASK_HISTORY = "请查询最近24小时番茄温室的环境历史记录，并总结温度和湿度的变化趋势。"
TASK_RISK = (
    "请先获取番茄温室当前环境数据，再获取南京的外部天气预报，"
    "综合两者评估番茄当前生长风险（低/中/高）。"
)
TASK_INVALID = "请查询地点「不存在的城市XYZ_98765」的天气预报，并告诉我结果。"


def _find(calls, name):
    return [c for c in calls if c.name == name]


def _test_a(agent: ToolAgent) -> bool:
    print("=" * 72)
    print("TestA【单工具任务】查询最近 24 小时温室环境历史")
    print("=" * 72)
    print(f"任务：{TASK_HISTORY}\n")

    outcome = agent.run(TASK_HISTORY)
    calls = outcome["calls"]
    history_calls = _find(calls, "get_environment_history")

    print(f"\n[回答] {outcome['final_answer']}\n")

    ok = (
        len(history_calls) == 1
        and history_calls[0].status == "ok"
        and history_calls[0].result.get("count", 0) > 0
    )
    print(f"判定：工具 get_environment_history 真实调用 {len(history_calls)} 次、"
          f"返回 {history_calls[0].result.get('count', 0) if history_calls else 0} 条 → "
          f"{'PASS ✅' if ok else 'FAIL ❌'}")
    print()
    return ok


def _test_b(agent: ToolAgent) -> bool:
    print("=" * 72)
    print("TestB【多工具任务】本地环境 + 外部天气 → 风险评估（≥2 工具协同）")
    print("=" * 72)
    print(f"任务：{TASK_RISK}\n")

    outcome = agent.run(TASK_RISK)
    calls = outcome["calls"]
    names = [c.name for c in calls]
    distinct = set(names)

    cur = _find(calls, "get_current_environment")
    weather = _find(calls, "get_weather_forecast")

    print(f"\n[调用序列] {' → '.join(names) if names else '（无）'}")
    print(f"[回答] {outcome['final_answer']}\n")

    cur_ok = bool(cur) and cur[0].status == "ok"
    weather_called = bool(weather)
    weather_ok = weather_called and weather[0].status == "ok"

    ok = len(distinct) >= 2 and cur_ok and weather_called
    print(f"判定：调用 {len(distinct)} 个不同工具（{sorted(distinct)}）、"
          f"当前环境{'✅' if cur_ok else '❌'}、天气{'已调用' + ('✅ok' if weather_ok else '❌error') if weather_called else '未调用'} → "
          f"{'PASS ✅' if ok else 'FAIL ❌'}")
    if weather_called and not weather_ok:
        print("     （注：天气工具被真实调用但返回 error，可能为网络/地点原因；"
              "智能体应如实说明，不伪造成功）")
    print()

    # 把 TestB 的多工具完整调用链路落盘为 B05_run_log.txt（案例复现原始证据）
    write_log(format_run_log(agent, outcome, TASK_RISK))
    print(f"[B05] 已落盘完整工具调用日志：experiment_logs/B05_run_log.txt\n")
    return ok


def _test_c(_agent: ToolAgent) -> bool:
    print("=" * 72)
    print("TestC【工具失败测试】错误 location / 超时 / 空输入 → 明确失败证据")
    print("=" * 72)

    registry = ToolRegistry()
    checks = []

    # ① 缺少必填参数 location
    r = registry.execute("get_weather_forecast", {})
    checks.append(("缺少必填参数 location", r))
    print(f"\n▶ 缺少必填参数 location：status={r['status']}  error={r.get('error')}")

    # ② 错误 location（地理编码失败）
    r = registry.execute("get_weather_forecast", {"location": "不存在的城市XYZ_98765"})
    checks.append(("错误 location 参数", r))
    print(f"▶ 错误 location 参数：status={r['status']}  error={r.get('error')}")

    # ③ 空输入 query
    r = registry.execute("search_tomato_knowledge", {"query": ""})
    checks.append(("空输入 query", r))
    print(f"▶ 空输入 query：status={r['status']}  error={r.get('error')}")

    # ④ 超时（直接调 weather.fetch_weather 注入 timeout=0.001s）
    r = fetch_weather("南京", timeout=0.001)
    checks.append(("超时（timeout=0.001s）", r))
    print(f"▶ 超时（timeout=0.001s）：status={r.get('status')}  error={r.get('error')}")

    # 判定：全部必须 status=error 且携带明确 error，绝不能出现 ok（伪造成功）
    all_failed = all(r.get("status") == "error" and r.get("error") for _, r in checks)
    print("\n判定：4 个失败场景全部返回 status=error 且含错误说明 → "
          f"{'PASS ✅' if all_failed else 'FAIL ❌'}")

    # —— 补充：智能体层面的失败任务演示（证据展示，不作硬性 gate）——
    print("\n" + "-" * 72)
    print("补充：智能体处理「错误 location」任务（验证不伪造成功）")
    print("-" * 72)
    print(f"任务：{TASK_INVALID}\n")
    outcome = _agent.run(TASK_INVALID)
    wc = _find(outcome["calls"], "get_weather_forecast")
    print(f"\n[回答] {outcome['final_answer']}\n")
    if wc:
        print(f"天气工具状态：{wc[0].status}  error={wc[0].error or '—'}")
        print("（若 status=error，说明系统如实返回失败，未伪造成功）")
    print()
    return all_failed


def main() -> int:
    agent = ToolAgent()
    print("GreenMate B05 · 工具智能体 · 工具注册与契约")
    print(f"已注册工具：{agent.registry.list_names()}\n")
    for name in agent.registry.list_names():
        t = agent.registry.get(name)
        print(f"  - {name}  必填参数={t.required}")
    print()

    results = {
        "TestA 单工具任务": _test_a(agent),
        "TestB 多工具任务": _test_b(agent),
        "TestC 工具失败": _test_c(agent),
    }

    print("=" * 72)
    print("测试汇总")
    print("=" * 72)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS ✅' if ok else 'FAIL ❌'}")
        all_ok = all_ok and ok
    print(f"\n总结：{'全部通过 ✅' if all_ok else '存在失败项 ❌'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())