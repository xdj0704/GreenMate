"""GreenMate B04 · RAG 模块测试脚本。

三组测试用例（严格对齐 B04 规范）：
  TestA【资料支持】   问题在知识库内 → 返回 Top-K 片段、答案、可溯源引用
  TestB【先看检索】   先输出检索到的 Top-K 片段供人工核验，再生成回答
  TestC【No-source】  问题超出知识库范围 → 输出「资料不足」，禁止编造

运行：
    python test_b04.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b04_rag import GreenMateRAG, RAG_MIN_SCORE  # noqa: E402


# ── 固定知识问题（步骤 1：必须依赖番茄知识库回答）────────────────────────────
Q_TEMP_HUM = "番茄开花结果期最适宜的昼夜温度范围和空气湿度范围是多少？"
Q_CTRL = "番茄温室应如何调控温度和空气湿度、光照不足时如何补光？"
Q_SOIL = "番茄适宜的土壤酸碱度pH值和土壤水分管理要点是什么？"

# ── 超出知识库范围的问题（用于 TestC）───────────────────────────────────────
Q_NO_SOURCE = "番茄果实中维生素C的具体含量是多少毫克每100克？"


def _print_evidence(evidence):
    """与 run_log 一致的检索证据表，便于人工核验。"""
    if not evidence:
        print("    （无检索结果）")
        return
    print(f"{'rank':<5}{'score':<8}{'source_id':<10}{'section':<14}{'page':<6}片段")
    for i, e in enumerate(evidence, 1):
        snippet = e.text.replace("\n", " ")[:48]
        print(f"{i:<5}{e.score:<8}{e.source_id:<10}{e.section:<14}{e.page:<6}{snippet}")
    print()


def _test_a(rag: GreenMateRAG) -> bool:
    print("=" * 70)
    print("TestA【资料支持】问题在知识库内 → Top-K 片段 + 答案 + 可溯源引用")
    print("=" * 70)
    query = Q_TEMP_HUM
    print(f"查询：{query}\n")
    evidence = rag.retrieve(query)
    print("[检索证据 Top-K]")
    _print_evidence(evidence)
    answer = rag.answer(query, evidence)
    print(f"[回答]\n{answer}\n")

    ok = (
        len(evidence) > 0
        and evidence[0].score > RAG_MIN_SCORE
        and "资料不足" not in answer
        and "《" in answer  # 回答附带《title》引用，可溯源
    )
    print(f"TestA 判定：{'PASS ✅' if ok else 'FAIL ❌'}")
    print()
    return ok


def _test_b(rag: GreenMateRAG) -> bool:
    print("=" * 70)
    print("TestB【先看检索】先输出 Top-K 片段人工核验，再生成回答")
    print("=" * 70)
    query = Q_CTRL
    print(f"查询：{query}\n")

    # 第一步：仅检索，不生成回答（等待人工核验）
    evidence = rag.retrieve(query)
    print("[第 1 步 · 先验检索结果（未调用 LLM，请人工核验相关性）]")
    _print_evidence(evidence)

    revelant = any(e.score > RAG_MIN_SCORE for e in evidence)
    print(f"人工核验：检索到 {len(evidence)} 条，{'存在相关片段' if revelant else '无相关片段'}")
    print()

    # 第二步：核验通过后再生成回答
    answer = rag.answer(query, evidence)
    print("[第 2 步 · 生成回答]")
    print(answer)
    print()

    ok = revelant and "《" in answer
    print(f"TestB 判定：{'PASS ✅' if ok else 'FAIL ❌'}")
    print()
    return ok


def _test_c(rag: GreenMateRAG) -> bool:
    print("=" * 70)
    print("TestC【No-source】问题超出知识库范围 → 资料不足，禁止编造")
    print("=" * 70)
    query = Q_NO_SOURCE
    print(f"查询：{query}\n")

    evidence = rag.retrieve(query)
    print("[检索证据 Top-K]")
    _print_evidence(evidence)

    answer = rag.answer(query, evidence)
    print(f"[回答]\n{answer}\n")

    ok = "资料不足" in answer and "无法确认" in answer
    print(f"TestC 判定：{'PASS ✅' if ok else 'FAIL ❌'}")
    print()
    return ok


def main() -> int:
    rag = GreenMateRAG()
    n_docs = rag.load_documents_from_dir()
    n_chunks = rag.build()

    print("GreenMate B04 · RAG 检索增强生成模块 · 测试")
    print(f"已注册资料：{n_docs} 份；索引片段：{n_chunks} 个\n")
    for d in rag.register.list_sources():
        ok = "✅" if d.trust_ok else "❌"
        print(f"  [{ok}] {d.source_id} | {d.provider} | {d.date} | {d.scope}")
    print()

    results = {
        "TestA 资料支持": _test_a(rag),
        "TestB 先看检索": _test_b(rag),
        "TestC 无资料": _test_c(rag),
    }

    print("=" * 70)
    print("测试汇总")
    print("=" * 70)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS ✅' if ok else 'FAIL ❌'}")
        all_ok = all_ok and ok
    print(f"\n总结：{'全部通过 ✅' if all_ok else '存在失败项 ❌'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())