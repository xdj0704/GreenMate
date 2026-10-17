# GreenMate B04 — RAG 检索增强生成模块

> 计科 NOVA 竞赛 · GreenMate 番茄生长环境认知系统 · B04 阶段

本阶段目标：实现**可信资料检索**。回答只能来自知识库文档；资料不足时明确输出
**【资料不足】**，禁止模型编造答案与虚假引用。

## 交付资产

```
b04_rag.py                     # RAG 模块（source_register + retrieval_evidence + 检索 + 回答 + 日志）
knowledge_base/
├── doc_001_番茄栽培技术手册.txt      # 温度/湿度/光照参数（国家农业数字图书馆-2024）
├── doc_002_番茄温室环境调控指南.txt  # 温湿度调控与补光（中国农业出版社-2023）
└── doc_003_番茄土壤与水分管理手册.txt # 土壤pH/水分/灌溉（省农业技术推广总站-2024）
test_b04.py                    # 三组测试用例（TestA/TestB/TestC）
experiment_logs/B04_run_log.txt # 运行生成的原始证据日志（可复现）
.env.example                   # 环境变量模板（复用 B03 的 LLM 配置）
B04_README.md
```

## 六步开发流程（对齐规范）

| 步骤 | 实现 | 载体 |
|------|------|------|
| 1 固定知识问题 | 3 个必须依赖知识库回答的问题 | `test_b04.py` 顶部 |
| 2 导入可信资料 | 校验 provider / date / scope，缺一即标记不可信 | `SourceRegister` |
| 3 切片+元数据 | chunk 保留 source_id / section / page，可反向定位 | `chunk_document` |
| 4 先验检索（不调 LLM） | `retrieve(query) -> Top-K` | `LightVectorIndex` |
| 5 回答+引用 | query+Top-K+元数据送 LLM，附真实引用 | `GreenMateRAG.answer` |
| 6 无来源降级 | 超范围问题输出「资料不足」，不编造 | `_no_source` |

## 关键结构体

- `SourceDocument`：注册资料，含 `source_id / title / provider / date / scope / text`。
- `RetrievalEvidence`：检索证据，记录 `query / retrieved_at / source_id / chunk_id /
  section / page / score / text / citation`，是最小可复现证据单元。
- `chunk_id` 形如 `doc_001#§3.1 温度#p1#0`，可唯一反向定位回原始文档对应节与页。

## 向量库方案（轻量）

纯 Python 标准库实现 **TF-IDF + 余弦相似度**，零第三方向量库依赖：
- 中文按「单字 + 双字 bigram」切分特征，英文/数字按整词；
- IDF 在建立索引时训练，查询词未登录时采用默认低频权重；
- `retrieve` 返回相似度降序的 Top-K，分数可解释、可复现。

## 运行

```bash
# 无需安装任何第三方库；LLM 回答可选（复用 B03 的 .env DeepSeek 配置）
python b04_rag.py --build              # 构建知识库索引
python b04_rag.py --source             # 查看已注册资料与可信度
python b04_rag.py "番茄花期最适温度是多少？"

# 跑三组测试用例
python test_b04.py
```

LLM 未配置时，`answer` 自动降级为「片段摘录 + 真实引用」，仅复述知识库原文，
不生成新事实；因此 `test_b04.py` 在无 API Key 环境也能完整复现证据。

## 三个固定知识问题

1. 番茄开花结果期最适宜的昼夜温度范围和空气湿度范围是多少？
2. 番茄温室应如何调控温度和空气湿度、光照不足时如何补光？
3. 番茄适宜的土壤酸碱度 pH 值和土壤水分管理要点是什么？

## 无来源问题（TestC）

- 番茄果实中维生素 C 的具体含量是多少毫克每 100 克？
  → 知识库只覆盖环境/土壤参数，不含营养含量，系统输出「资料不足，无法确认」。

## 边界说明

- 样例知识库文档为**教学用示例资料**，`provider/date/scope` 用于演示可信度校验，
  不构成真实出版引用担保。
- 本模块只做「资料检索 + 引用式回答」，不输出病害诊断结论。