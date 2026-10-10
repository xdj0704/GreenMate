"""GreenMate 连续状态认知引擎 CLI。

用法：
    python main.py S2      # 运行单个场景并打印到控制台
    python main.py ALL     # 一键遍历 S1-S5，导出单个 Markdown 报告
    python main.py web     # 启动 Web 界面（http://127.0.0.1:8765）
"""

import sys

from greenmate.engine import run
from greenmate import scenarios

REPORT_PATH = "GreenMate全场景报告.md"


def run_all():
    ids = scenarios.all_scenario_ids()
    sections = []
    for sid in ids:
        section = run(sid)
        lines = section.split("\n")
        lines[0] = f"# 场景{sid}（{scenarios.SCENARIO_NAMES[sid]}）· GreenMate 连续状态认知工作台"
        sections.append("\n".join(lines))

    index = "\n".join(
        f"- **{sid}**：{scenarios.SCENARIO_NAMES[sid]}"
        for sid in ids
    )
    header = "\n".join([
        "# GreenMate 连续状态认知工作台 · 全场景报告（S1-S5）",
        "",
        "本报告由引擎一键遍历生成，覆盖五类场景：",
        "",
        index,
        "",
        "---",
        "",
    ])

    report = header + "\n\n---\n\n".join(sections) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"已生成全场景报告：{REPORT_PATH}")
    print(f"包含场景：{'、'.join(ids)}（共 {len(ids)} 个）")
    print(f"报告行数：{len(report.splitlines())}")


def main():
    scenario_id = sys.argv[1].upper() if len(sys.argv) > 1 else "S2"
    if scenario_id == "ALL":
        run_all()
    elif scenario_id == "WEB":
        from greenmate import web
        web.serve(open_browser="--no-browser" not in sys.argv)
    else:
        print(run(scenario_id))


if __name__ == "__main__":
    main()
