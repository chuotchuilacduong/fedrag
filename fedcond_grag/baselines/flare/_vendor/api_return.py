"""Vendored from jzbjyb/FLARE's src/templates.py (MIT licensed, see
LICENSE in this directory) -- the `ApiReturn` class (+ its `Sentence`
helper), extracted verbatim. See VENDORED.md for what's vendored here vs.
hand-written in ../client_runner.py, and why.

This class is FLARE's actual algorithmic contribution used by this
baseline: given a generated span with per-token probabilities, truncate it
at a sentence boundary and mask out low-confidence tokens to form a
retrieval query (`use_as_query`, `mask_method='simple'` -- the setting used
in FLARE's own published 2wikihop config).
"""

from typing import List
from collections import namedtuple

import spacy
import tiktoken

Sentence = namedtuple('Sentence', 'text start_char end_char')


class ApiReturn:
    EOS = '<|endoftext|>'
    spacy_nlp = spacy.load('en_core_web_sm')
    # NOTE (fedrag fork): upstream defaulted to 'nltk' (PunktSentenceTokenizer),
    # which needs the separate `nltk` package + a downloaded punkt corpus.
    # spacy's sentencizer (already a dependency + already downloaded for
    # Stage A's NER) does the same job here; 'nltk' is left unimplemented
    # below since nothing in this baseline selects it.
    use_sentencizer = 'spacy'
    min_sent_len = 5

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = None,
        probs: List[float] = None,
        offsets: List[int] = None,
        finish_reason: str = 'stop',
        model: str = None,
        skip_len: int = 0,
    ):
        self.model = model
        self.prompt = prompt
        self.text = text

        self.tokens = tokens
        self.probs = probs
        self.offsets = offsets
        if self.has_tokens:
            assert len(tokens) == len(probs) == len(offsets)

        self.finish_reason = finish_reason
        if self.finish_reason is None:
            self.finish_reason = 'stop'  # TODO: a bug from openai?

        if skip_len:  # skip `skip_len` chars at the beginning
            self.text = self.text[skip_len:]
            if self.has_tokens:
                i = 0
                for i, off in enumerate(self.offsets):
                    if off == skip_len:
                        break
                    elif off > skip_len:  # the previous token span across the boundary
                        i = i - 1
                        assert i >= 0
                        break
                self.tokens = self.tokens[i:]
                self.probs = self.probs[i:]
                self.offsets = self.offsets[i:]

    @property
    def has_tokens(self):
        return self.tokens is not None

    @property
    def token_probs(self):
        if self.has_tokens:
            return self.probs
        else:
            return []

    @property
    def num_tokens(self):
        if self.has_tokens:
            return len(self.tokens)
        else:
            return len(tiktoken.encoding_for_model(self.model).encode(self.text))

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens

    @property
    def is_empty(self):
        return len(self.text.strip()) == 0

    @classmethod
    def get_sent(cls, text: str, position: str = 'begin'):
        if cls.use_sentencizer == 'spacy':
            sents = list(cls.spacy_nlp(text).sents)
        else:
            raise NotImplementedError
        if position == 'begin':
            break_at = len(text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= cls.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break
            return text[:break_at], break_at
        if position == 'end':
            break_at = 0
            for i in range(len(sents)):
                sent = sents[len(sents) - i - 1]
                if len(text) - sent.start_char >= cls.min_sent_len:  # TODO: argument
                    break_at = sent.start_char
                    break
            return text[break_at:], break_at
        raise NotImplementedError

    def truncate_at_prob(self, low: float):
        assert self.has_tokens, 'not supported'

        if self.num_tokens <= 1:
            return self

        break_point = self.num_tokens
        for i in range(self.num_tokens):
            t, p, o = self.tokens[i], self.probs[i], self.offsets[i]
            if p <= low:
                break_point = i
                break
        if break_point == 0 and self.num_tokens > 0:  # avoid deadlock
            break_point = 1

        while break_point < self.num_tokens:  # truncation
            assert break_point > 0
            keep = self.offsets[break_point] - len(self.prompt)
            if keep <= 0:
                break_point += 1
                continue

            self.text = self.text[:keep]
            self.tokens = self.tokens[:break_point]
            self.probs = self.probs[:break_point]
            self.offsets = self.offsets[:break_point]
            self.finish_reason = 'boundary'
            break

        return self

    def truncate_at_boundary(self, unit: str = 'sentence'):
        if self.num_tokens <= 1:
            return self

        if unit == 'sentence':
            if self.use_sentencizer == 'spacy':
                sents = list(self.spacy_nlp(self.text).sents)
            else:
                raise NotImplementedError
            break_at = len(self.text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= self.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break

            if break_at > 0 and break_at < len(self.text):  # truncation
                if self.has_tokens:
                    i = 0
                    for i in range(self.num_tokens):
                        if self.offsets[i] - len(self.prompt) >= break_at:
                            break_at = self.offsets[i] - len(self.prompt)
                            break
                    assert i > 0
                    self.tokens = self.tokens[:i]
                    self.probs = self.probs[:i]
                    self.offsets = self.offsets[:i]
                assert break_at > 0
                self.text = self.text[:break_at]
                self.finish_reason = 'boundary'
        else:
            raise NotImplementedError
        return self

    def truncate_at_substring(self, substr: str):
        position = self.text.find(substr)
        if position == -1:
            return
        self.text = self.text[:position]
        if self.has_tokens:
            i = 0
            for i, off in enumerate(self.offsets):
                if off - len(self.prompt) == position:
                    break
                elif off - len(self.prompt) > position:  # the previous token span across the boundary
                    i = i - 1
                    assert i >= 0
                    break
            self.tokens = self.tokens[:i]
            self.probs = self.probs[:i]
            self.offsets = self.offsets[:i]

    def use_as_query(
        self,
        low_prob: float = None,
        mask_prob: float = None,
        mask_method: str = 'simple',
        n_gen_char_in_prompt: int = 0,
        api_key: str = None,
    ):
        if not low_prob and not mask_prob:
            return self.text
        assert self.has_tokens, 'not supported'
        if low_prob:
            ok = False
            for p in self.probs:
                if p <= low_prob:
                    ok = True
                    break
            if not ok:
                return ''
        if mask_prob:
            if mask_method == 'simple':
                keep = [(t if p > mask_prob else ' ') for t, p in zip(self.tokens, self.probs)]
                keep = ''.join(keep).strip()
                return keep
            else:
                # NOTE (fedrag fork): upstream's 'wholeterm-decontextualize' /
                # 'wholeterm-askquestion' variants call back into
                # CtxPrompt.get_queries_from_text_for_retrieval() (few-shot
                # dataset machinery this baseline doesn't vendor -- see
                # VENDORED.md). Not reachable here: client_runner.py only
                # ever passes mask_method='simple', matching FLARE's own
                # published 2wikihop config.
                raise NotImplementedError(f"mask_method={mask_method!r} not supported by this vendored subset")
        else:
            return self.text
