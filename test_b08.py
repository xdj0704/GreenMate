"""GreenMate B08 · 系统回归测试脚本（第四遍：搭建固定测试用例）。

B08 不新增底层能力，本脚本只做「冻结行为的可复现断言」，守护核心能力不损坏：
  R1~R5  直接调用 b07_system.make_judgment（确定性规则），断言 S1~S5 冻结基线；
  R6     断言可信降级优先序（①历史缺失 应优先于 ④知识缺失）；
  R7     断言突变检测 _is_abrupt 的边界行为。

运行（无需网络 / 无需 LLM / 无需 B06 模型，纯确定性，Python 3.8+ 即可）：
    python test_b08.py

冻结基线（GreenMate v1.0-B08，见 B08_验证方案.md §三、§七）：
    S1 正常稳定 / 正常      S2 渐进漂移 / 中等关注   S3 突变异常 / 高关注
    S4 数据缺失 / 观察中     S5 证据不足 / 观察中
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b07_system import EvidenceRecord, _is_abrupt, make_judgment  # noqa: E402
from greenmate import scenarios                                   # noqa: E402

# 冻结基线：scenario_id -> (current_state, attention_level, degraded, breaches)
BASELINE = {
    "S1": ("正常稳定", "正常", False, []),
    "S2": ("渐进漂移", "中等关注", False, ["温度+0.2°C"]),
    "S3": ("突变异常", "高关注", False, ["温度+0.9°C", "湿度+4%"]),
    "S4": ("数据缺失", "观察中", True, []),
    "S5": ("证据不足", "观察中", True, []),
}

# 五类证据编号（沿用 B01/B07 的固定分类，保证 status 注入顺序正确）
_EVIDENCE_TYPES = ["①历史观测数据", "②当前实时观测", "③外部天气证据",
                   "④专业知识证据", "⑤模型预测MODEL_OUTPUT"]


def _evidence_from_scenario(sc):
    """从场景声明（evidence_list 的 status/source/citation）构造证据记录。

    只取 status/source/citation 这类「场景设定」，跳过 ③天气/⑤模型 的真实网络调用，
    使回归测试完全离线、确定性。"""
    declared = {e["source_type"]: e for e in sc["evidence_list"]}
    records = []
    for etype in _EVIDENCE_TYPES:
        e = declared.get(etype, {})
        records.append(EvidenceRecord(
            type=etype,
            source=e.get("source", ""),
            status=e.get("status", "ok"),
            summary="",
            citation=e.get("citation") or "",
        ))
    return records


def _evidence(status_map):
    """按给定五类 status 构造证据（用于降级优先序 / 边界测试）。"""
    return [EvidenceRecord(type=t, source="", status=status_map.get(t, "ok"),
                           summary="", citation="") for t in _EVIDENCE_TYPES]


def test_scenario_judgments() -> bool:
    """R1~R5：S1~S5 结构化判断 == 冻结基线。"""
    print("=" * 72)
    print("R1~R5【场景判断基线】S1~S5 结构化判断 == 冻结基线")
    print("=" * 72)
    all_ok = True
    for sid in scenarios.all_scenario_ids():
        sc = scenarios.get_scenario(sid)
        j = make_judgment(sc, _evidence_from_scenario(sc))
        want_state, want_level, want_degraded, want_breaches = BASELINE[sid]
        ok = (
            j.current_state == want_state
            and j.attention_level == want_level
            and j.degraded == want_degraded
            and j.breaches == want_breaches
        )
        all_ok = all_ok and ok
        mark = "PASS ✅" if ok else "FAIL ❌"
        print(f"  {mark} {sid} {sc['name']}：{j.current_state}/{j.attention_level}"
              f"（degraded={j.degraded}，越限={j.breaches or '无'}）")
        if not ok:
            print(f"        期望 {want_state}/{want_level}，degraded={want_degraded}，"
                  f"breaches={want_breaches}")
    print(f"判定 → {'PASS ✅' if all_ok else 'FAIL ❌'}\n")
    return all_ok


def test_degradation_priority() -> bool:
    """R6：可信降级优先序——①历史缺失 应优先于 ④知识缺失。"""
    print("=" * 72)
    print("R6【可信降级优先序】①历史缺失 > ④知识缺失")
    print("=" * 72)
    sc = scenarios.get_scenario("S2")  # 历史完整、limits 存在，仅切换证据 status

    # 情形 A：①partial + ④unavailable → 数据缺失（① 优先命中）
    jA = make_judgment(sc, _evidence({"①历史观测数据": "partial",
                                      "④专业知识证据": "unavailable"}))
    okA = jA.current_state == "数据缺失" and jA.degraded
    print(f"  A ①partial+④unavailable → {jA.current_state}（degraded={jA.degraded}）"
          f"  → {'PASS ✅' if okA else 'FAIL ❌'}")

    # 情形 B：①ok + ④unavailable → 证据不足（④ 命中）
    jB = make_judgment(sc, _evidence({"④专业知识证据": "unavailable"}))
    okB = jB.current_state == "证据不足" and jB.degraded
    print(f"  B ①ok+④unavailable      → {jB.current_state}（degraded={jB.degraded}）"
          f"  → {'PASS ✅' if okB else 'FAIL ❌'}")

    ok = okA and okB
    print(f"判定 → {'PASS ✅' if ok else 'FAIL ❌'}\n")
    return ok


def test_is_abrupt_boundary() -> bool:
    """R7：突变检测 _is_abrupt 的边界行为。"""
    print("=" * 72)
    print("R7【突变检测边界】渐进序列 / 突变序列 / 末步恰为阈值")
    print("=" * 72)
    gradual = [23.6, 24.0, 24.5, 25.0, 25.4, 25.8, 26.2]  # S2：单调但末步小
    abrupt = [24.0, 24.2, 24.4, 24.6, 24.8, 26.0, 28.9]   # S3：末步 +2.9
    edge = [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 25.0]     # 末步恰 =1.0，不触发

    checks = [
        ("渐进漂移序列 → 非突变", not _is_abrupt(gradual)),
        ("突变序列     → 突变", _is_abrupt(abrupt)),
        ("末步=1.0°C   → 非突变（严格 >1.0 才触发）", not _is_abrupt(edge)),
    ]
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  {'PASS ✅' if ok else 'FAIL ❌'}  {name}")
    print(f"判定 → {'PASS ✅' if all_ok else 'FAIL ❌'}\n")
    return all_ok


def main() -> int:
    print("GreenMate B08 · 系统回归测试（冻结基线 v1.0-B08）\n")
    results = {
        "R1~R5 场景判断基线": test_scenario_judgments(),
        "R6 可信降级优先序": test_degradation_priority(),
        "R7 突变检测边界": test_is_abrupt_boundary(),
    }
    print("=" * 72)
    print("测试汇总")
    print("=" * 72)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS ✅' if ok else 'FAIL ❌'}")
        all_ok = all_ok and ok
    print(f"\n总结：{'全部通过 ✅' if all_ok else '存在失败项 ❌'}")
    print("（R8 模块级回归请另跑：python test_b04.py / python test_b05.py / py -3.12 test_b06.py）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
