"""DGRAG prompt templates (spec §5).

All 7 prompts from the spec in one place. The §9.1 short-answer addendum
is appended to PROMPT_RAG_ANSWER when `short_answer=True` — required for
HotPotQA/2WikiMQA/MuSiQue ACC/F1 evaluation.

Keep the "say so explicitly" clause in PROMPT_RAG_ANSWER: Confidence
Detection (step ⑤) depends on that language being present in the generated
candidates.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Phase A
# ------------------------------------------------------------------

PROMPT_EXTRACT = """Given a text document, identify all entities and all relationships among them.

For each entity output a JSON object with keys:
  "entity_name" (string, canonical, capitalized),
  "entity_type" (one of: person, place, event, object, organization, category, concept),
  "entity_description" (a complete description of its attributes and activities based only on the text).

For each relationship between two identified entities output a JSON object with keys:
  "source_entity", "target_entity",
  "relationship_keyword" (short high-level phrase, e.g. "is library for", "contrasts with"),
  "relationship_description" (why they are related).

Return a single JSON object:
{{"entities": [...], "relations": [...]}}

Use only information present in the text. Do not invent.

Text:
{text}
"""

PROMPT_EXTRACT_CONTINUE = """You previously extracted entities and relations from the text below.
Review your extraction and identify any entities or relationships you may have missed.
Return ONLY the newly found items (do not repeat already-found ones) as a JSON object:
{{"entities": [...], "relations": [...]}}

Prior extraction:
{prior}

Text:
{text}
"""

PROMPT_SUMMARIZE_SUBGRAPH = """You are given the entities and relationships of one subgraph of a knowledge graph.
Write a concise summary (<= {max_tokens} words) capturing: the subject area(s) covered,
the main entities, and the key relationships among them.
Describe what kind of knowledge this subgraph contains, not fine-grained details.
Do not invent information.

{kg_text}
"""

# ------------------------------------------------------------------
# Phase B
# ------------------------------------------------------------------

PROMPT_KEYWORDS = """Extract keywords from the query at two levels. Return JSON only, no explanation:
{{"high_level_keywords": [...], "low_level_keywords": [...]}}
high_level_keywords: overarching concepts, themes, subject areas.
low_level_keywords: specific entities, names, concrete details.

Query: {query}
"""

_RAG_BODY = """Answer the query using only the knowledge provided below.
If the provided knowledge does not contain the answer, say so explicitly and state
that the information is insufficient — do not guess.

---Knowledge Base---
Entities:
{entities}

Relationships:
{relationships}

Sources:
{chunks}
---

Query: {query}
Answer:"""

_RAG_SHORT_ADDENDUM = " Answer with the shortest span that answers the question. Output only the answer, no explanation."

PROMPT_RAG_ANSWER = _RAG_BODY
PROMPT_RAG_ANSWER_SHORT = _RAG_BODY + _RAG_SHORT_ADDENDUM

PROMPT_CONFIDENCE = """Below are {b} candidate answers generated for the same query.
Decide whether they express a lack of confidence or a lack of information
(e.g. "insufficient information", "I don't know", "the provided data does not
contain", "need more details").
Return JSON only: {{"confident": true|false, "reason": "<short>"}}

Query: {query}
Answers: {answers}
"""

PROMPT_SEMANTIC_CONSISTENCY = """Below are {b} candidate answers to the same query. Judge whether their core claims
and assertions are mutually consistent — whether they convey a coherent, non-
contradictory message. Ignore differences in wording, length, and formatting.
Return JSON only: {{"consistency": <float 0..1>, "reason": "<short>"}}

Query: {query}
Answers: {answers}
"""

PROMPT_SELECT_BEST = """Below are {b} candidate answers to the same query, judged mutually consistent.
Select the single best answer: most complete, most accurate, and clearest.
Return JSON only: {{"index": <int 0-based>, "reason": "<short>"}}

Query: {query}
Answers: {answers}
"""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_extract_prompt(text: str) -> str:
    return PROMPT_EXTRACT.format(text=text)


def build_extract_continue_prompt(text: str, prior: str) -> str:
    return PROMPT_EXTRACT_CONTINUE.format(text=text, prior=prior)


def build_summarize_prompt(kg_text: str, max_tokens: int = 200) -> str:
    return PROMPT_SUMMARIZE_SUBGRAPH.format(kg_text=kg_text, max_tokens=max_tokens)


def build_keywords_prompt(query: str) -> str:
    return PROMPT_KEYWORDS.format(query=query)


def build_rag_prompt(
    query: str,
    entities_text: str,
    relationships_text: str,
    chunks_text: str,
    short_answer: bool = True,
) -> str:
    tmpl = PROMPT_RAG_ANSWER_SHORT if short_answer else PROMPT_RAG_ANSWER
    return tmpl.format(
        query=query,
        entities=entities_text or "(none)",
        relationships=relationships_text or "(none)",
        chunks=chunks_text or "(none)",
    )


def build_confidence_prompt(query: str, answers: list[str]) -> str:
    answers_fmt = "\n".join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))
    return PROMPT_CONFIDENCE.format(query=query, answers=answers_fmt, b=len(answers))


def build_semantic_consistency_prompt(query: str, answers: list[str]) -> str:
    answers_fmt = "\n".join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))
    return PROMPT_SEMANTIC_CONSISTENCY.format(query=query, answers=answers_fmt, b=len(answers))


def build_select_best_prompt(query: str, answers: list[str]) -> str:
    answers_fmt = "\n".join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))
    return PROMPT_SELECT_BEST.format(query=query, answers=answers_fmt, b=len(answers))
