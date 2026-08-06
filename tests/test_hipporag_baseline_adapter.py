from fedcond_grag.baselines.graph_rag.hipporag_local import (
    _chunk_to_doc,
    _gold_answers,
    _gold_docs,
    _install_precomputed_openie,
    _title_key,
)


def test_chunk_to_doc_matches_hipporag_title_body_format():
    doc, title = _chunk_to_doc("12:Move (1970 film): Move is an American comedy film.")

    assert title == "Move (1970 film)"
    assert doc == "Move (1970 film)\nMove is an American comedy film."


def test_gold_docs_use_local_doc_when_supporting_title_is_present():
    local_doc = "Move (1970 film)\nFull local passage text."
    sample = {
        "evidence": [
            ["Move (1970 film)", ["Gold sentence subset."]],
            ["Missing title", ["This remains a gold-only doc."]],
        ]
    }

    docs = _gold_docs(sample, {_title_key("Move (1970 film)"): local_doc})

    assert docs[0] == local_doc
    assert docs[1] == "Missing title\nThis remains a gold-only doc."


def test_gold_answers_include_aliases_without_duplicates():
    sample = {"answer": "Alice", "answer_aliases": ["Alicia", "Alice"]}

    assert _gold_answers(sample) == ["Alice", "Alicia"]


def test_gold_docs_support_hipporag_context_format():
    sample = {
        "supporting_facts": [["Title A", 0]],
        "context": [["Title A", ["Sentence one.", "Sentence two."]], ["Title B", ["No."]]],
    }

    assert _gold_docs(sample, {}) == ["Title A\nSentence one. Sentence two."]


def test_precomputed_openie_target_uses_transformers_model_name(tmp_path):
    src = tmp_path / "openie.json"
    src.write_text('{"docs": []}', encoding="utf-8")

    _install_precomputed_openie(
        source=src,
        save_dir=tmp_path / "client_0",
        llm_name="Transformers/Qwen/Qwen2.5-7B-Instruct",
        docs=[],
    )

    assert (tmp_path / "client_0" / "openie_results_ner_Qwen_Qwen2.5-7B-Instruct.json").exists()
