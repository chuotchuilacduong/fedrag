"""SpaCy NER with multiprocessing support for large-scale passage indexing."""

from __future__ import annotations

import os
from collections import defaultdict
from multiprocessing import Pool


# ---------------------------------------------------------------------------
# Worker (runs in child process — no shared state with parent)
# ---------------------------------------------------------------------------

def _ner_worker(args: tuple) -> tuple[dict, dict]:
    """Process a chunk of passages in an isolated subprocess."""
    model_name, hids, texts = args
    import spacy

    # Keep only ner + sentencizer (disable heavy parser/tagger)
    disabled = ["tok2vec", "tagger", "parser", "senter",
                "attribute_ruler", "lemmatizer"]
    nlp = spacy.load(model_name, disable=disabled, exclude=disabled)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)

    p_to_ents: dict[str, list[str]] = {}
    s_to_ents: dict[str, list[str]] = defaultdict(list)

    for hid, doc in zip(hids, nlp.pipe(texts, batch_size=64)):
        unique: set[str] = set()
        for ent in doc.ents:
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in s_to_ents[sent_text]:
                s_to_ents[sent_text].append(ent_text)
            unique.add(ent_text)
        p_to_ents[hid] = list(unique)

    return p_to_ents, dict(s_to_ents)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SpacyNER:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        # Validate model exists (fail fast)
        import spacy as _spacy
        _spacy.load(spacy_model, disable=["parser"])

    def batch_ner(
        self,
        hash_id_to_passage: dict[str, str],
        max_workers: int | None = None,
    ) -> tuple[dict, dict]:
        """Run NER on all passages using a multiprocessing pool."""
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 16)

        hids = list(hash_id_to_passage.keys())
        texts = [hash_id_to_passage[h] for h in hids]
        n = len(hids)

        if n == 0:
            return {}, {}

        # Split into per-worker chunks
        n_workers = min(max_workers, n)
        chunk_size = max(1, (n + n_workers - 1) // n_workers)
        chunks = [
            (self.spacy_model, hids[i:i+chunk_size], texts[i:i+chunk_size])
            for i in range(0, n, chunk_size)
        ]

        passage_to_ents: dict[str, list[str]] = {}
        sentence_to_ents: dict[str, list[str]] = defaultdict(list)

        if n_workers == 1:
            # Single-process path (avoids fork overhead for tiny inputs)
            p, s = _ner_worker(chunks[0])
            passage_to_ents.update(p)
            for sent, ents in s.items():
                for e in ents:
                    if e not in sentence_to_ents[sent]:
                        sentence_to_ents[sent].append(e)
        else:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_ner_worker, chunks)
            for p, s in results:
                passage_to_ents.update(p)
                for sent, ents in s.items():
                    for e in ents:
                        if e not in sentence_to_ents[sent]:
                            sentence_to_ents[sent].append(e)

        return passage_to_ents, dict(sentence_to_ents)

    def question_ner(self, question: str) -> set[str]:
        import spacy
        nlp = spacy.load(self.spacy_model, disable=["parser"])
        doc = nlp(question)
        return {
            ent.text.lower()
            for ent in doc.ents
            if ent.label_ not in ("ORDINAL", "CARDINAL")
        }
