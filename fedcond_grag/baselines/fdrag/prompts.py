"""FD-RAG LLM prompts.

Both prompts reproduce the *intent* of the ones described in ``fd-rag.pdf``
Tables 9 and 10 (the paper's full text is only given schematically). Slots
line up 1:1 with the spec:

- QA Memory Generation (Table 9): {example}, {type}, {extracted_facts}, {text}.
- RAG / Cognizer (Table 10):       {context}, {document}, {question}.

Kept in one module (per §6 suggested layout) so any prompt tweak is one file
edit.
"""

from __future__ import annotations

QA_GEN_EXAMPLE = """Question: Who directed the 2011 romantic comedy \"Vaada Poda Nanbargal\"?
Answer: Manikai"""


QA_GEN_PROMPT = """You are an information system generating retrieval-oriented QA memory questions grounded ONLY in the provided atomic facts and original text. Do NOT use external knowledge.

Rules:
- Write a specific, unambiguous {type} question.
- The answer must reflect the original content and reuse its exact expressions where possible.
- Encourage compositional reasoning across the given facts when the question type allows.
- Output STRICTLY two lines, in this format:
{example}

Extracted facts:
{extracted_facts}

Original text:
{text}
"""


RAG_PROMPT = """You are an intelligent assistant. Use the reference Q&A pairs and context documents below to answer the final question. Output ONLY the final answer, with no explanation.

Reference Q&A pairs:
{context}

Context documents:
{document}

Question: {question}
Answer:"""


def build_qa_prompt(*, qtype: str, facts: list[tuple[str, str]], text: str) -> str:
    facts_str = "\n".join(f"- {span} [{label}]" for span, label in facts) or "- (none)"
    return QA_GEN_PROMPT.format(
        example=QA_GEN_EXAMPLE, type=qtype, extracted_facts=facts_str, text=text
    )


def build_rag_prompt(*, question: str, ref_qa: list[tuple[str, str]], evidence: list[str]) -> str:
    ctx = "\n".join(f"Q: {q}\nA: {a}" for q, a in ref_qa) or "(none)"
    doc = "\n\n".join(evidence) or "(none)"
    return RAG_PROMPT.format(context=ctx, document=doc, question=question)
