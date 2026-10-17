"""GreenMate B05 — 工具智能体（Tool Agent）。

核心：让 LLM 自主选择并调用已封装工具，形成「调用 → 执行 → 返回 → 继续」闭环。
不硬编码任何工具调用顺序；工具是否调用、调用哪些、传什么参数、调用几次，全部由
LLM 依据任务与工具返回结果动态决定。

实现要点（严格对齐 B05 规范）：
  - 使用 OpenAI function-calling（tools 参数）发起多轮工具调用循环；
  - 每一轮：解析 LLM 返回的 tool_call → 真实执行 Python 函数 → 把结果回填 →
    继续下一轮，直到 LLM 不再产生 tool_call，输出最终回答；
  - 完整记录每次工具调用（名称 / 入参 / 返回结果 / 调用序列 / 错误信息），
    并把「假调用（嘴上说）」与「真实执行」严格区分——只有 LLM 返回的 tool_call
    才触发真实执行，其余文字声明一律不计为工具调用。

用法：
    python b05_agent.py "查询最近24小时温室环境历史"
    python b05_agent.py --tool-log       # 只打印上一次运行的完整调用日志
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

from b05_tools import ToolRegistry

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
B05_MAX_ROUNDS = int(os.getenv("B05_MAX_ROUNDS", "6"))  # 多轮工具调用上限

LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiment_logs", "B05_run_log.txt"
)

SYSTEM_PROMPT = (
    "你是 GreenMate 番茄温室环境分析助手。你可以调用工具获取真实数据。\n"
    "必须遵守：\n"
    "1) 只能基于工具返回的真实数据作答，禁止编造任何数据，禁止声称执行了未执行的工具；\n"
    "2) 按需自主选择工具与参数，可用单个或多个工具，顺序由你决定，不要硬编码；\n"
    "3) 工具返回 status='error' 时，必须如实说明失败原因与缺失证据，不得伪造成功；\n"
    "4) 用中文回答，在结论中标注数据来源工具名称。"
)


# ══════════════════════════════════════════════════════════════════════════════
# ToolCallRecord —— 单次工具调用的完整证据（区分真实执行 vs 假调用）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ToolCallRecord:
    seq: int                     # 调用序号（全局递增，反映调用序列）
    name: str                    # 工具名
    arguments: Dict[str, Any]    # 入参
    status: str                  # ok / error
    result: Optional[Dict[str, Any]] = None  # 真实返回结果
    error: str = ""              # 错误信息
    started_at: str = ""         # 起始时间
    elapsed_ms: int = 0          # 耗时毫秒

    def result_summary(self, max_items: int = 3) -> str:
        """为日志生成精简但真实的结果摘要。"""
        if self.status != "ok" or self.result is None:
            return f"error: {self.error}"
        r = dict(self.result)
        # 大列表截断，保留 count + 前若干条，仍体现「真实数据」
        for key in ("records", "results"):
            if key in r and isinstance(r[key], list):
                total = len(r[key])
                r[key] = r[key][:max_items]
                r[f"{key}_shown"] = f"{total} 条中展示前 {min(total, max_items)} 条"
        return json.dumps(r, ensure_ascii=False, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# ToolAgent —— 多轮工具调用循环
# ══════════════════════════════════════════════════════════════════════════════
class ToolAgent:
    VERSION = "GreenMate Tool-Agent v1.0 (B05)"

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.calls: List[ToolCallRecord] = []
        self._client: Optional[OpenAI] = None
        if self._llm_available():
            self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    @staticmethod
    def _llm_available() -> bool:
        return bool(LLM_API_KEY) and "your" not in LLM_API_KEY.lower() and "在这里" not in LLM_API_KEY

    def run(self, task: str, max_rounds: int = B05_MAX_ROUNDS) -> Dict[str, Any]:
        """执行一次智能体任务，返回 {final_answer, calls, rounds, run_id}。"""
        self.calls = []
        if self._client is None:
            return {
                "final_answer": "[LLM未配置] 请在 .env 中填入有效 LLM_API_KEY 后重试。"
                                "（步骤 1 自检与工具级失败测试不依赖 LLM，可直接运行）",
                "calls": [], "rounds": 0,
            }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        final_answer = ""

        for _round in range(1, max_rounds + 1):
            resp = self._client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                tools=self.registry.openai_schemas(),
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1600,
            )
            msg = resp.choices[0].message

            # 无 tool_call → 最终回答，结束循环
            if not getattr(msg, "tool_calls", None):
                final_answer = (msg.content or "").strip()
                break

            # 有 tool_call → 真实执行（只有这里才触发真实函数调用）
            tool_calls = msg.tool_calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {"_raw": tc.function.arguments}
                    result = {"status": "error", "error": f"参数 JSON 解析失败：{exc}"}
                else:
                    result = self._execute(name, arguments)

                # 记录证据 + 回填给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        else:
            final_answer = "[智能体] 达到最大工具调用轮次上限，停止循环。"

        return {
            "final_answer": final_answer,
            "calls": self.calls,
            "rounds": _round,
            "run_id": self._new_run_id(),
        }

    def _execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """真实执行单次工具调用，并记录完整证据。"""
        seq = len(self.calls) + 1
        started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.perf_counter()
        result = self.registry.execute(name, arguments)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        rec = ToolCallRecord(
            seq=seq,
            name=name,
            arguments=arguments,
            status=result.get("status", "error"),
            result=result,
            error=result.get("error", ""),
            started_at=started_at,
            elapsed_ms=elapsed_ms,
        )
        self.calls.append(rec)
        print(f"[B05] ▶ 工具调用 #{seq} {name}  入参={json.dumps(arguments, ensure_ascii=False)}"
              f"  → {'✅ ok' if rec.status == 'ok' else '❌ error'}"
              + ("" if rec.status == "ok" else f"（{rec.error}）"))
        return result

    @staticmethod
    def _new_run_id() -> str:
        return f"run-B05-{datetime.datetime.now().strftime('%H%M%S')}"


# ══════════════════════════════════════════════════════════════════════════════
# 日志输出（与 S1~S5 run_log 保持一致，供案例复现）
# ══════════════════════════════════════════════════════════════════════════════
def format_run_log(agent: ToolAgent, outcome: Dict[str, Any], task: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = outcome.get("run_id", "")
    lines = [
        "# GreenMate B05 工具智能体 · 运行日志",
        "",
        f"run_id: {run_id}",
        f"运行时间: {ts}",
        f"模块版本: {agent.VERSION}",
        f"模型: {LLM_MODEL_NAME}",
        f"任务: {task}",
        f"工具轮次: {outcome.get('rounds', 0)}　真实工具调用次数: {len(outcome.get('calls', []))}",
        "",
        "## 工具调用序列（真实执行，非模拟）",
        "",
    ]

    calls = outcome.get("calls", [])
    if not calls:
        lines.append("（本次未产生任何 tool_call → 无真实工具调用）")
    else:
        lines.append("| seq | 工具名 | 入参 | 状态 | 错误信息 | 耗时(ms) |")
        lines.append("|-----|--------|------|------|----------|----------|")
        for c in calls:
            args_str = json.dumps(c.arguments, ensure_ascii=False)
            err = c.error or "—"
            lines.append(f"| {c.seq} | {c.name} | {args_str} | {c.status} | {err} | {c.elapsed_ms} |")
    lines += ["", "## 各工具真实返回结果", ""]

    for c in calls:
        lines.append(f"### [{c.seq}] {c.name}")
        lines.append("```json")
        lines.append(c.result_summary())
        lines.append("```")
        lines.append("")

    lines += [
        "## 最终回答",
        "",
        outcome.get("final_answer", ""),
        "",
        "---",
    ]
    return "\n".join(lines)


def write_log(log: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(log)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "--tool-log":
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print("[B05] 尚未生成运行日志，请先执行一次任务")
        return

    task = " ".join(args)
    agent = ToolAgent()
    print(f"[B05] ▶ 任务：{task}")
    outcome = agent.run(task)
    log = format_run_log(agent, outcome, task)
    write_log(log)
    print("\n" + log)


if __name__ == "__main__":
    main()