"""GreenMate B05 — 工具注册模块（Tool Registry）。

本模块把 B02 / B03 / B04 已写好的 4 个标准业务函数封装为「工具」，定义统一
工具契约（name / description / parameters / required），并提供真实的执行入口。
智能体（b05_agent.py）只允许通过本模块调用工具，保证「调用即真实执行」，杜绝
假调用。

4 个基础工具（复用既有业务函数）：
  1. get_current_environment()           ← b02_data_memory.get_current_environment
  2. get_environment_history(hours=24)   ← b02_data_memory.get_environment_history
  3. get_weather_forecast(location)      ← weather.fetch_weather
  4. search_tomato_knowledge(query)      ← b04_rag.GreenMateRAG.retrieve

用法（开发步骤第 1 步：不经过 LLM，单独先跑每个业务函数）：
    python b05_tools.py              # 自检：直接执行 4 个工具
    python b05_tools.py --schema     # 打印工具契约 JSON
"""

from __future__ import annotations

import datetime
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

# 复用既有业务函数（B02 / B03 / B04）
from b02_data_memory import get_current_environment, get_environment_history
from weather import fetch_weather
from b04_rag import GreenMateRAG

# Windows GBK 终端兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# 基础工具真实实现（每个都能独立运行，返回统一 dict：status=ok/error）
# ══════════════════════════════════════════════════════════════════════════════

def _tool_get_current_environment() -> Dict[str, Any]:
    """工具 1：获取温室当前最新环境数据（本地 SIMULATED）。"""
    row = get_current_environment()
    if not row:
        return {"status": "error", "error": "环境数据库暂无采样记录（greenmate.db 为空）"}
    timestamp, source_type, temperature, humidity, light = row
    return {
        "status": "ok",
        "provider": "本地 SIMULATED 模拟传感器",
        "timestamp": timestamp,
        "source_type": source_type,
        "temperature": temperature,
        "humidity": humidity,
        "light": light,
    }


def _tool_get_environment_history(hours: int = 24) -> Dict[str, Any]:
    """工具 2：获取最近 N 小时环境历史时序（本地 SIMULATED）。"""
    # 参数校验（对应开发步骤第 6 步：参数错误需明确提示）
    if not isinstance(hours, int) or isinstance(hours, bool) or hours <= 0:
        return {"status": "error",
                "error": f"参数错误：hours 必须为正整数，收到 {hours!r}"}

    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=hours)
    rows = get_environment_history(start.isoformat(), end.isoformat())
    if not rows:
        return {"status": "error",
                "error": f"最近 {hours} 小时内无历史记录（时间窗 {start.isoformat()} ~ {end.isoformat()}）"}

    records = [
        {"timestamp": r[0], "source_type": r[1], "temperature": r[2],
         "humidity": r[3], "light": r[4]}
        for r in rows
    ]
    return {
        "status": "ok",
        "provider": "本地 SIMULATED 模拟传感器",
        "hours": hours,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "count": len(records),
        "records": records,
    }


def _tool_get_weather_forecast(location: str) -> Dict[str, Any]:
    """工具 3：获取指定地点外部天气预报（PUBLIC · Open-Meteo）。

    直接透传 weather.fetch_weather 的失败信息：错误地点 / 超时 / HTTP 错误都会
    反映在 status='error' 与 error 字段里，不伪造成功。
    """
    return fetch_weather(location)


# RAG 单例：仅在首次调用 search_tomato_knowledge 时加载知识库并建索引
_rag_instance: Optional[GreenMateRAG] = None


def _get_rag() -> GreenMateRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GreenMateRAG()
        _rag_instance.load_documents_from_dir()
        _rag_instance.build()
    return _rag_instance


def _tool_search_tomato_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """工具 4：番茄知识库 RAG 检索（先验检索，不调 LLM）。"""
    if not query or not str(query).strip():
        return {"status": "error", "error": "参数错误：query 为空，无法检索"}
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return {"status": "error", "error": f"参数错误：top_k 必须为正整数，收到 {top_k!r}"}

    rag = _get_rag()
    evidence = rag.retrieve(str(query).strip(), top_k=top_k)
    return {
        "status": "ok",
        "provider": "番茄知识库 RAG（TF-IDF 检索）",
        "query": str(query).strip(),
        "count": len(evidence),
        "results": [e.to_dict() for e in evidence],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 工具契约定义
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Tool:
    """一个已注册的工具：契约 + 真实执行函数。"""
    name: str
    description: str
    parameters: Dict[str, Any]           # JSON Schema（含 required）
    required: List[str]                  # 必填参数名（契约显式列出）
    func: Callable[..., Dict[str, Any]]


TOOL_SPECS: List[Tool] = [
    Tool(
        name="get_current_environment",
        description=(
            "获取番茄温室当前最新一条环境数据（温度、湿度、光照）。"
            "来源为本地 SIMULATED 模拟传感器，无参数。"
        ),
        parameters={"type": "object", "properties": {}},
        required=[],
        func=_tool_get_current_environment,
    ),
    Tool(
        name="get_environment_history",
        description=(
            "获取最近 N 小时番茄温室环境历史时序记录（温度、湿度、光照）。"
            "用于观察趋势、回溯环境变化。来源为本地 SIMULATED 模拟传感器。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "回溯小时数，例如 24 表示最近 24 小时",
                },
            },
        },
        required=["hours"],
        func=_tool_get_environment_history,
    ),
    Tool(
        name="get_weather_forecast",
        description=(
            "获取指定地点的外部天气预报（室外温度、湿度、降水概率、云量、短波辐射）。"
            "来源为 Open-Meteo PUBLIC 数据，与本地 SIMULATED 数据严格区分。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市/地点名称，例如 '南京'、'北京'",
                },
            },
        },
        required=["location"],
        func=_tool_get_weather_forecast,
    ),
    Tool(
        name="search_tomato_knowledge",
        description=(
            "在番茄种植知识库中检索与问题相关的片段（RAG，TF-IDF 检索，不调用 LLM）。"
            "返回 Top-K 片段及其可信来源引用，用于回答番茄环境/栽培知识问题。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询词/问题，例如 '番茄花期最适温度'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，默认 3",
                },
            },
        },
        required=["query"],
        func=_tool_search_tomato_knowledge,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 工具注册表 + 真实执行
# ══════════════════════════════════════════════════════════════════════════════
class ToolRegistry:
    """工具的注册与执行入口。

    执行时会做三层校验，失败均返回 status='error' 的 dict（不抛异常给智能体）：
      ① 工具名是否存在；
      ② 必填参数是否缺失 / 类型是否合法；
      ③ 业务函数是否抛异常。
    """

    def __init__(self, specs: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in (specs or TOOL_SPECS):
            self._tools[t.name] = t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """真实执行一个工具，返回统一结构 dict（含 status）。

        params 为已解析的自变量字典；缺失必填参数 / 工具不存在 / 业务异常都在
        返回值中体现为 status='error' 与明确的 error 说明（对应步骤 6：失败显式）。
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error",
                    "error": f"工具不存在：'{name}'，可用工具 {self.list_names()}"}

        # ② 必填参数校验
        missing = [k for k in tool.required if k not in params or params[k] in (None, "")]
        if missing:
            return {"status": "error",
                    "error": f"参数校验失败：缺少必填参数 {missing}（工具 {name}）"}

        # ③ 真实执行业务函数
        try:
            result = tool.func(**{k: v for k, v in params.items() if k in tool.parameters.get("properties", {})})
            if not isinstance(result, dict):
                result = {"status": "ok", "data": result}
            return result
        except Exception as exc:  # 业务函数异常（如超时、网络错误）真实上抛→记录
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def openai_schemas(self) -> List[Dict[str, Any]]:
        """输出 OpenAI function-calling 所需的 tools 列表。"""
        schemas = []
        for t in self._tools.values():
            params = dict(t.parameters)
            params.setdefault("type", "object")
            if "required" not in params and t.required:
                params["required"] = t.required
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": params,
                },
            })
        return schemas


# ── 步骤 1 自检：不经过 LLM，直接跑 4 个工具 ─────────────────────────────────
def run_selftest(registry: Optional[ToolRegistry] = None) -> Dict[str, Any]:
    registry = registry or ToolRegistry()
    print("=" * 72)
    print("GreenMate B05 · 步骤 1 自检（不经过 LLM，直接执行 4 个业务函数）")
    print("=" * 72)

    cases = [
        ("get_current_environment", {}),
        ("get_environment_history", {"hours": 24}),
        ("get_weather_forecast", {"location": "南京"}),
        ("search_tomato_knowledge", {"query": "番茄花期最适温度", "top_k": 3}),
    ]
    summary = {}
    for name, params in cases:
        print(f"\n▶ 工具 {name}  入参 {params}")
        result = registry.execute(name, params)
        summary[name] = result.get("status")
        _print_result(result)
    print("\n" + "=" * 72)
    for k, v in summary.items():
        print(f"  {k}: {'✅ ok' if v == 'ok' else '❌ ' + str(v)}")
    print("=" * 72)
    return summary


def _print_result(result: Dict[str, Any]) -> None:
    if result.get("status") == "ok":
        # 精简打印，避免大段 records 刷屏
        keys_to_show = {k: v for k, v in result.items() if k not in ("records", "results", "data")}
        if "records" in result:
            keys_to_show["records(前3条)"] = result["records"][:3]
        if "results" in result:
            keys_to_show["results(前1条)"] = [
                {k: r[k] for k in ("source_id", "section", "score") if k in r}
                for r in result["results"][:1]
            ]
        print(f"  ✅ {json.dumps(keys_to_show, ensure_ascii=False, default=str)[:400]}")
    else:
        print(f"  ❌ 失败：{result.get('error')}")


def print_schema() -> None:
    registry = ToolRegistry()
    print(json.dumps(registry.openai_schemas(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--schema" in sys.argv:
        print_schema()
    else:
        run_selftest()