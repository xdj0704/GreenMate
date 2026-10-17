# GreenMate B05 — 工具智能体（Tool Agent）

> 计科 NOVA 竞赛 · GreenMate 番茄生长环境认知系统 · B05 阶段

本阶段目标：让 **LLM 自主选择并调用已封装工具**，形成「调用 → 执行 → 返回 → 继续」
闭环；**不硬编码任何工具调用顺序**。

## 交付资产

```
b05_tools.py                    # 工具注册模块（4 个工具契约 + 真实执行 + 自检）
b05_agent.py                    # 工具智能体（LLM 多轮工具调用循环 + 证据日志）
test_b05.py                     # 三组测试用例（TestA/TestB/TestC）
experiment_logs/B05_run_log.txt # 完整工具调用日志（可复现原始证据）
.env.example                    # 环境变量模板（新增 B05 配置）
B05_README.md
```

## 4 个基础工具（复用 B02/B03/B04 业务函数）

| 工具名 | 复用来源 | 契约参数 | 数据来源 |
|---|---|---|---|
| `get_current_environment` | `b02_data_memory.get_current_environment` | 无 | 本地 SIMULATED |
| `get_environment_history` | `b02_data_memory.get_environment_history` | `hours:int` | 本地 SIMULATED |
| `get_weather_forecast` | `weather.fetch_weather` | `location:str` | Open-Meteo PUBLIC |
| `search_tomato_knowledge` | `b04_rag.GreenMateRAG.retrieve` | `query:str`,`top_k:int` | 番茄知识库 RAG |

每个工具契约 = `name / description / parameters / required`，由 `ToolRegistry` 统一注册，
并把 `parameters` 转为 OpenAI function-calling 的 `tools` 结构。

## 六步开发流程（对齐规范）

| 步骤 | 实现 | 载体 |
|---|---|---|
| 1 单独先跑业务函数 | 不经 LLM，直接执行 4 个工具 | `python b05_tools.py` |
| 2 工具定义 | name/description/parameters/required | `TOOL_SPECS` |
| 3 单工具调用链路 | LLM 输出 tool_call → 解析参数 → 真实执行 → result | `ToolAgent.run` |
| 4 多工具任务 | LLM 自选工具/参数/顺序，代码不写死流程 | `ToolAgent.run` 多轮循环 |
| 5 MCP 外部工具（可选拔高） | 未强制；已预留 registry 扩展点 | 见「扩展 MCP」 |
| 6 失败验证与证据 | 参数错误/超时/空输入 → 明确 error，不伪造成功 | `ToolRegistry.execute` |

## 核心结构体

- `Tool`：工具契约（`name/description/parameters/required/func`）。
- `ToolCallRecord`：单次工具调用证据，记录 `seq/name/arguments/status/result/error/started_at/elapsed_ms`，
  是最小可复现证据单元，用于区分「真实执行」与「假调用」。
- `ToolRegistry`：工具注册与执行入口，执行时做「存在性 + 必填参数 + 业务异常」三层校验，失败统一返回 `status='error'`。

## 真实 vs 假调用

只有 LLM 返回的 `tool_call`（function-calling 响应）才会触发真实 Python 函数执行；
LLM 在文字里声称「我调用了某工具」而不产生 `tool_call` 的，一律不计入调用日志。
每次调用都记录真实入参、真实返回结果与耗时。

## 运行

```bash
# 无任何第三方向量库依赖；LLM 智能体需 .env DeepSeek Key，工具自检无需 Key
python b05_tools.py                 # 步骤 1 自检：不经 LLM，直接跑 4 个工具
python b05_tools.py --schema        # 打印工具契约 JSON
python b05_agent.py "查询最近24小时温室环境历史"
python b05_agent.py --tool-log      # 查看上一次完整工具调用日志

python test_b05.py                  # 三组测试用例
```

## 三组测试用例说明

- **TestA【单工具任务】**：查询最近 24 小时历史 → 真实调用 `get_environment_history` 且返回 count>0。
- **TestB【多工具任务】**：当前环境 + 南京天气预报 → 风险评估，需 ≥2 个不同工具协同。
- **TestC【工具失败测试】**：错误 location、缺少必填参数、空 query、超时 → 全部返回 `status='error'` 且带明确 error，不伪造成功。

## 扩展 MCP（可选，不强制 B05 基础验收）

`ToolRegistry` 与 `Tool` 均为开放结构，接入外部 MCP 工具的路径：

1. 用 `Tool(name=..., parameters=..., func=...)` 注册一个 MCP 工具；
2. `func` 内部调用 MCP 客户端并把返回规整为 `{"status": "ok"/"error", ...}`；
3. `ToolAgent` 会自动把该工具暴露给 LLM 并纳入「调用→执行→返回→继续」循环，
   失败同样走 `status='error'` 证据链。

## 边界说明

- 本地环境数据为 `SIMULATED` 模拟传感器，外部天气为 `PUBLIC` 真实预报，二者严格区分。
- 天气工具依赖网络；网络不可用时返回 `status='error'`（超时/连接错误），智能体必须如实说明失败，不得伪造。
- 本模块只做「数据获取 + 环境层面分析」，不输出病害诊断结论。