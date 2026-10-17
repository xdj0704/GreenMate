"""GreenMate B04 — RAG 检索增强生成模块。

业务目标：实现可信资料检索，回答只能来自知识库文档；资料不足时明确输出
【资料不足】，禁止模型编造答案与虚假引用。

严格遵循计科 NOVA B04 规范的 6 步开发流程：
  1. 固定知识问题（由 test_b04.py / 知识库文档承担）
  2. 导入可信资料（source_register 校验 provider / date / scope）
  3. 文本切片 + 元数据（chunk 保留 source_id / section / page，可反向定位）
  4. 先验检索（retrieve 不调用 LLM，返回 Top-K 片段供人工核验）
  5. 回答 + 引用（query + Top-K + 元数据送 LLM，回答附真实引用）
  6. 无来源结果与证据（超出知识库范围 → 「资料不足，无法确认」）

向量库采用轻量方案：纯标准库 TF-IDF + 余弦相似度，零第三方向量库依赖。
LLM 回答层复用 B03 的 DeepSeek 配置，API Key 缺失时降级为「片段摘录 + 引用」，
保证在任何环境下都能复现检索证据。

用法：
    python b04_rag.py --build            # 构建知识库索引（幂等）
    python b04_rag.py "番茄花期最适温度是多少？"      # 检索 + 回答
    python b04_rag.py --source           # 列出已注册资料
"""

from __future__ import annotations

import datetime
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ── 读取环境变量（复用 B03 的 LLM 配置）────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv 非必需，缺失时直接读系统环境变量
    pass

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))          # 检索返回的 Top-K 片段数
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))  # 无来源判定阈值

# 知识库目录与样例文档（provider/date/scope 附在文件头元数据块）
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiment_logs", "B04_run_log.txt"
)


# ══════════════════════════════════════════════════════════════════════════════
# retrieval_evidence —— 检索证据结构体
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RetrievalEvidence:
    """单条检索证据：记录查询词、检索来源、片段、元数据、相似度与时间。

    是本模块的「原始证据」最小单元，可完整复现一次检索命中了什么、从哪里来。
    """

    query: str                 # 查询词
    retrieved_at: str          # 检索时间
    source_id: str             # 来源文档 id（可反向定位）
    chunk_id: str              # 片段唯一 id
    section: str               # 节标题
    page: int                  # 页号（节内块序号，供引文定位）
    score: float               # 相似度分数
    text: str                  # 片段正文
    citation: str = ""         # 规范引用：《title》§section，provider-date

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceDocument:
    """一份已注册的可信资料。"""

    source_id: str
    title: str
    provider: str
    date: str
    scope: str
    text: str
    trust_issues: List[str] = field(default_factory=list)

    @property
    def trust_ok(self) -> bool:
        return len(self.trust_issues) == 0

    @property
    def citation_base(self) -> str:
        return f"《{self.title}》，{self.provider}-{self.date}"


# ══════════════════════════════════════════════════════════════════════════════
# source_register —— 资料注册管理
# ══════════════════════════════════════════════════════════════════════════════
class SourceRegister:
    """注册并校验可信资料。

    校验规则：provider / date / scope 三者不可缺失，否则标记为不可信（但不清除，
    由上层决定是否继续使用）。这对应开发步骤第 2 步「校验资料可信度」。
    """

    REQUIRED = ("provider", "date", "scope")

    def __init__(self) -> None:
        self._docs: Dict[str, SourceDocument] = {}

    def register(self, doc: SourceDocument) -> SourceDocument:
        issues: List[str] = []
        for key in self.REQUIRED:
            if not getattr(doc, key, "").strip():
                issues.append(f"缺少元数据字段：{key}")
        if not doc.title.strip():
            issues.append("缺少元数据字段：title")
        if not doc.text.strip():
            issues.append("文档正文为空")
        doc.trust_issues = issues
        self._docs[doc.source_id] = doc
        return doc

    def list_sources(self) -> List[SourceDocument]:
        return list(self._docs.values())

    def get(self, source_id: str) -> Optional[SourceDocument]:
        return self._docs.get(source_id)


# ══════════════════════════════════════════════════════════════════════════════
# 轻量向量检索 —— 纯标准库 TF-IDF + 余弦相似度
# ══════════════════════════════════════════════════════════════════════════════
def _tokenize(text: str) -> List[str]:
    """切分特征：中文按单字 + 双字 bigram，英文/数字按整词。

    中文没有天然空格，采用字符 n-gram 是最轻量且可解释的切分方式，兼顾
    短查询与长片段的召回。
    """
    tokens: List[str] = []
    for seg in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
            tokens.append(seg)
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
            tokens.extend(list(seg))
        else:
            tokens.append(seg)
    return tokens


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    section: str
    page: int
    text: str
    citation_base: str = ""


class LightVectorIndex:
    """轻量向量索引：对每个 chunk 构建 TF-IDF 向量，查询时按余弦相似度排序。"""

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._idf: Dict[str, float] = {}
        self._vectors: List[Dict[str, float]] = []

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def build_index(self) -> None:
        """训练 IDF 并生成每个 chunk 的加权向量。"""
        n = len(self.chunks)
        tokenized = [_tokenize(c.text) for c in self.chunks]
        df: Counter = Counter()
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        self._idf = {
            t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()
        }
        self._vectors = []
        for toks in tokenized:
            tf = Counter(toks)
            vec = {t: cnt * self._idf.get(t, 1.0) for t, cnt in tf.items()}
            self._vectors.append(vec)

    def _query_vec(self, query: str) -> Dict[str, float]:
        tf = Counter(_tokenize(query))
        # 查询词使用已训练的 idf，未知词取默认 idf（全文档集意义上的低频词）
        default_idf = math.log((len(self.chunks) + 1) / 1) + 1.0
        return {t: cnt * self._idf.get(t, default_idf) for t, cnt in tf.items()}

    def search(self, query: str, top_k: int) -> List[Tuple[float, Chunk]]:
        qv = self._query_vec(query)
        scored = [(round(_cosine(qv, vec), 4), self.chunks[i])
                  for i, vec in enumerate(self._vectors)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 文本切片 + 元数据
# ══════════════════════════════════════════════════════════════════════════════
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")


def chunk_document(doc: SourceDocument, chunk_size: int = 180) -> List[Chunk]:
    """把文档正文按节（`##` 标题）切分，再按字符长度切块。

    每块保留 source_id / section / page 元数据，chunk_id 形如
    `{source_id}#{section}#p{page}#{i}`，可反向定位回原始文档对应节。
    """
    lines = doc.text.splitlines()
    chunks: List[Chunk] = []
    section = "正文"
    buffer: List[str] = []
    page = 1

    def flush() -> None:
        nonlocal page
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                cid = f"{doc.source_id}#{section}#p{page}#{len(chunks)}"
                chunks.append(Chunk(
                    chunk_id=cid, source_id=doc.source_id, section=section,
                    page=page, text=text, citation_base=doc.citation_base,
                ))
            buffer.clear()

    idx = 0
    for raw in lines:
        m = _SECTION_RE.match(raw.strip())
        if m:
            flush()
            section = m.group(1).strip()
            page = 1
            continue
        if not raw.strip():
            continue
        buffer.append(raw.strip())
        if sum(len(s) for s in buffer) >= chunk_size:
            flush()
            page += 1
        idx += 1
    flush()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 主模块 —— 注册 → 切片 → 索引 → 检索 → 回答 → 日志
# ══════════════════════════════════════════════════════════════════════════════
class GreenMateRAG:
    VERSION = "GreenMate-RAG v1.0 (B04)"

    def __init__(self) -> None:
        self.register = SourceRegister()
        self.index = LightVectorIndex()

    # ── 步骤 2：导入可信资料 ────────────────────────────────────────────────
    def load_documents_from_dir(self, directory: str = KNOWLEDGE_DIR) -> int:
        """从知识库目录加载 `*.txt` 样例文档。

        文件头使用如下元数据块（`key: value` 行）：
            # source_id: doc_001
            # title: 番茄栽培技术手册（节选）
            # provider: 国家农业数字图书馆
            # date: 2024
            # scope: 番茄花期/结果期温湿度与光照环境参数
        之后的正文用 `## ` 行分节。
        """
        if not os.path.isdir(directory):
            return 0
        loaded = 0
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".txt"):
                continue
            meta, body = self._parse_doc_file(os.path.join(directory, name))
            doc = SourceDocument(
                source_id=meta.get("source_id", name[:-4]),
                title=meta.get("title", name[:-4]),
                provider=meta.get("provider", ""),
                date=meta.get("date", ""),
                scope=meta.get("scope", ""),
                text=body,
            )
            self.register.register(doc)
            loaded += 1
        return loaded

    @staticmethod
    def _parse_doc_file(path: str) -> Tuple[dict, str]:
        meta: Dict[str, str] = {}
        body: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                mm = re.match(r"^#\s*([\w_]+)\s*:\s*(.+?)\s*$", line)
                if mm and mm.group(1) in ("source_id", "title", "provider", "date", "scope"):
                    meta[mm.group(1)] = mm.group(2)
                else:
                    body.append(raw)
        return meta, "\n".join(body)

    # ── 步骤 2 + 3：校验并建立索引 ─────────────────────────────────────────
    def build(self) -> int:
        """把已注册资料切片并构建轻量向量索引，返回 chunk 总数。"""
        self.index = LightVectorIndex()
        for doc in self.register.list_sources():
            if not doc.trust_ok:
                # 校验未通过的资料不进入索引，但保留在注册表供排查
                continue
            for chunk in chunk_document(doc):
                self.index.add(chunk)
        self.index.build_index()
        return len(self.index.chunks)

    # ── 步骤 4：先验检索（不调用 LLM）──────────────────────────────────────
    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[RetrievalEvidence]:
        """输入 query，返回 Top-K 片段（检索证据），全程不调用 LLM。"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hits = self.index.search(query, top_k)
        evidence: List[RetrievalEvidence] = []
        for score, chunk in hits:
            doc = self.register.get(chunk.source_id)
            if doc is not None:
                citation = f"《{doc.title}》{chunk.section}，{doc.provider}-{doc.date}，第{chunk.page}页"
            else:
                citation = f"{chunk.citation_base} {chunk.section}，第{chunk.page}页"
            evidence.append(RetrievalEvidence(
                query=query,
                retrieved_at=now,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                page=chunk.page,
                score=score,
                text=chunk.text,
                citation=citation,
            ))
        return evidence

    # ── 步骤 5 + 6：回答 + 引用 / 无来源降级 ───────────────────────────────
    def answer(self, query: str, evidence: List[RetrievalEvidence]) -> str:
        """把 query + Top-K + 元数据送 LLM；无证据或分数过低时输出【资料不足】。"""
        usable = [e for e in evidence if e.score > RAG_MIN_SCORE]
        if not usable:
            return self._no_source(query, evidence)

        context = self._render_context(usable)
        if not self._llm_available():
            return self._fallback_answer(query, usable)

        # 有 LLM 时，严格限定模型只能依据片段回答并附引用
        prompt = (
            "你是GreenMate番茄环境分析助手。请严格依据下方提供的知识库片段回答，"
            "只做环境层面信息说明，不输出病害诊断结论。\n\n"
            f"【问题】\n{query}\n\n"
            f"【知识库片段 Top-K】\n{context}\n\n"
            "回答要求：\n"
            "1. 回答只能来自上述片段，禁止使用片段之外的知识或常识编造。\n"
            "2. 每一条陈述后标注引用编号 [1][2]...，结尾列出引用来源（对应下面给出的 citation）。\n"
            "3. 若片段不足以回答问题，只回复：资料不足，无法确认。"
        )
        try:
            from openai import OpenAI
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是可信资料问答助手，禁止编造。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # LLM 异常时降级，不影响证据复现
            return self._fallback_answer(query, usable, note=f"LLM调用失败：{exc}")

    @staticmethod
    def _render_context(usable: List[RetrievalEvidence]) -> str:
        lines: List[str] = []
        for i, e in enumerate(usable, 1):
            lines.append(f"[{i}] {e.text}")
            lines.append(f"    引用：{e.citation}")
        return "\n".join(lines)

    def _fallback_answer(self, query: str, usable: List[RetrievalEvidence],
                         note: str = "LLM未配置") -> str:
        """无 LLM Key 时的降级回答：仅摘录最相关片段并附真实引用，不生成新事实。"""
        top = usable[0]
        lines = [f"（{note}，以下为知识库片段摘录，非模型生成）"]
        for i, e in enumerate(usable, 1):
            lines.append(f"[{i}] {e.text}\n    引用：{e.citation}")
        lines.append("回答：仅可确认上述片段内容，未做额外推断。")
        return "\n".join(lines)

    @staticmethod
    def _no_source(query: str, evidence: List[RetrievalEvidence]) -> str:
        """步骤 6：无来源结果与证据。"""
        return (
            "【资料不足】无法确认。\n"
            f"查询「{query}」在已注册知识库中未检索到相关片段（Top-K 最高相似度 "
            f"{max((e.score for e in evidence), default=0.0)}，低于阈值 {RAG_MIN_SCORE}），"
            "系统拒绝编造答案。"
        )

    @staticmethod
    def _llm_available() -> bool:
        return bool(LLM_API_KEY) and "your" not in LLM_API_KEY.lower()

    # ── 日志输出（与 S1~S5 run_log 格式保持一致，用于案例复现）──────────────
    def format_run_log(self, run_id: str, query: str,
                       evidence: List[RetrievalEvidence],
                       answer: str, stage: str = "answer") -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# GreenMate B04 RAG · 检索增强生成日志",
            "",
            f"run_id: {run_id}",
            f"检索时间: {ts}",
            f"模块版本: {self.VERSION}",
            f"查询: {query}",
            f"阶段: {stage}",
            "",
            "## 检索证据（Top-K，先验检索，未调用 LLM）",
            "",
        ]
        if not evidence:
            lines += ["（无检索结果）", ""]
        else:
            lines += [
                "| rank | score | source_id | section | page | 片段 |",
                "|------|-------|-----------|---------|------|------|",
            ]
            for i, e in enumerate(evidence, 1):
                snippet = e.text.replace("\n", " ")[:60]
                lines.append(
                    f"| {i} | {e.score} | {e.source_id} | {e.section} | {e.page} | {snippet} |"
                )
            lines.append("")

        lines += [
            "## 引用",
            "",
        ]
        if evidence:
            for i, e in enumerate(evidence, 1):
                lines.append(f"- [{i}] {e.citation}")
        else:
            lines.append("- （无引用）")
        lines += [
            "",
            "## 回答",
            "",
            answer,
            "",
            "---",
        ]
        return "\n".join(lines)

    def run(self, query: str, run_id: Optional[str] = None,
            stage: str = "answer") -> str:
        """一次完整查询：检索 → 回答 → 生成日志（并落盘）。"""
        evidence = self.retrieve(query)
        if stage == "retrieve":
            answer_text = "（先验检索阶段，暂不生成回答，待人工核验相关性）"
        else:
            answer_text = self.answer(query, evidence)
        if run_id is None:
            run_id = f"run-B04-{datetime.datetime.now().strftime('%H%M%S')}"
        log = self.format_run_log(run_id, query, evidence, answer_text, stage)
        self._write_log(log)
        return log

    @staticmethod
    def _write_log(log: str) -> None:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(log)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _list_sources(rag: GreenMateRAG) -> None:
    docs = rag.register.list_sources()
    if not docs:
        print("[B04] 尚未加载知识库，请先把文档放入 knowledge_base/ 目录")
        return
    for d in docs:
        ok = "✅" if d.trust_ok else "❌"
        print(f"[{ok}] {d.source_id} | {d.title} | {d.provider} | {d.date} | {d.scope}")
        if d.trust_issues:
            for issue in d.trust_issues:
                print(f"      - {issue}")


def main() -> None:
    import sys
    rag = GreenMateRAG()
    n = rag.load_documents_from_dir()
    print(f"[B04] 已加载 {n} 份资料 → 建立索引")
    chunk_count = rag.build()
    print(f"[B04] 索引构建完成，共 {chunk_count} 个片段")

    args = sys.argv[1:]
    if not args or args[0] == "--build":
        print("[B04] 构建完成。查询用法：python b04_rag.py \"你的问题\"")
        return
    if args[0] == "--source":
        _list_sources(rag)
        return

    query = " ".join(args)
    log = rag.run(query)
    print(log)


if __name__ == "__main__":
    main()