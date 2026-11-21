"""Knowledge-base retrieval: section-level chunks, BM25, optional dense embeddings fused with RRF.

BM25 is pure Python and has no dependency, so it fits the 512 MB free Render instance. Dense
retrieval (``fastembed``, installed with the ``embeddings`` extra) is fused with BM25 by
reciprocal rank fusion when enabled.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Passage:
    """One section of a knowledge-base document."""

    passage_id: str
    file_name: str
    title: str
    section: str
    category: str
    version: str
    text: str

    def as_context(self) -> str:
        return (
            f"[{self.file_name} | {self.title} | section: {self.section} | "
            f"version {self.version}]\n{self.text}"
        )


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def chunk_document(doc: dict[str, Any]) -> list[Passage]:
    """Split a Markdown document on ``##`` headings; the text before the first one is the intro."""
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for line in str(doc["content"]).splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not line.startswith("# "):
            sections[-1][1].append(line)
    passages = []
    for i, (heading, lines) in enumerate(sections):
        text = "\n".join(lines).strip()
        if not text:
            continue
        passages.append(
            Passage(
                passage_id=f"{doc['file_name']}#{i}",
                file_name=str(doc["file_name"]),
                title=str(doc["title"]),
                section=heading,
                category=str(doc["category"]),
                version=str(doc["version"]),
                text=text,
            )
        )
    return passages


class BM25:
    def __init__(self, corpus: Sequence[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens) for tokens in corpus]
        self.lengths = [len(tokens) for tokens in corpus]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tokens in corpus:
            df.update(set(tokens))
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for term in query:
                tf = counts.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    s += self.idf[term] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Small ONNX sentence-embedding model (``pip install hypercare-agents[embeddings]``)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415 - optional dependency

        self._model = TextEmbedding(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(list(texts))]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class KnowledgeIndex:
    def __init__(self, passages: Sequence[Passage], embedder: Embedder | None = None) -> None:
        self.passages = list(passages)
        self._bm25 = BM25([tokenize(f"{p.title} {p.section} {p.text}") for p in self.passages])
        self._embedder = embedder
        self._vectors = (
            embedder.embed([f"{p.title}. {p.section}. {p.text}" for p in self.passages])
            if embedder
            else None
        )

    @classmethod
    def from_documents(
        cls, docs: Sequence[dict[str, Any]], embedder: Embedder | None = None
    ) -> KnowledgeIndex:
        return cls([p for d in docs for p in chunk_document(d)], embedder)

    @property
    def hybrid(self) -> bool:
        return self._vectors is not None

    def search(self, query: str, k: int = 4, file_name: str | None = None) -> list[Hit]:
        candidates = [
            i for i, p in enumerate(self.passages) if file_name is None or p.file_name == file_name
        ]
        if not candidates:
            return []
        bm25 = self._bm25.scores(tokenize(query))
        bm25_rank = sorted(
            (i for i in candidates if bm25[i] > 0), key=lambda i: bm25[i], reverse=True
        )
        if self._vectors is None or self._embedder is None:
            return [Hit(self.passages[i], round(bm25[i], 4)) for i in bm25_rank[:k]]

        q = self._embedder.embed([query])[0]
        dense = {i: _cosine(q, self._vectors[i]) for i in candidates}
        dense_rank = sorted(candidates, key=lambda i: dense[i], reverse=True)
        fused = reciprocal_rank_fusion([bm25_rank, dense_rank])
        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [Hit(self.passages[i], round(fused[i], 4)) for i in best]

    def first_passage(self, file_name: str) -> Passage | None:
        return next((p for p in self.passages if p.file_name == file_name), None)

    def search_many(self, queries: Sequence[str], k: int = 4) -> list[Hit]:
        """Union of the hits of several queries, best score per passage, top ``k`` overall."""
        best: dict[str, Hit] = {}
        for query in queries:
            for hit in self.search(query, k=k):
                current = best.get(hit.passage.passage_id)
                if current is None or hit.score > current.score:
                    best[hit.passage.passage_id] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
