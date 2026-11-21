from __future__ import annotations

from collections.abc import Sequence

from hypercare.retrieval import KnowledgeIndex, chunk_document
from hypercare.retrieval.index import reciprocal_rank_fusion, tokenize

DOC = {
    "file_name": "a.md",
    "title": "Doc A",
    "category": "X",
    "version": "1.0",
    "content": "# Doc A\nIntro text.\n\n## First\nalpha beta\n\n## Empty\n\n## Second\ngamma delta",
}


def test_chunk_on_level_two_headings() -> None:
    passages = chunk_document(DOC)
    assert [p.section for p in passages] == ["Introduction", "First", "Second"]
    assert passages[1].text == "alpha beta"
    assert "a.md" in passages[1].as_context()


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("How is the BAF applied?") == ["baf", "applied"]


def test_search_finds_the_right_section(index: KnowledgeIndex) -> None:
    hits = index.search("hazardous cargo in a shipper-owned container SOC line", k=2)
    assert hits[0].passage.file_name == "rules_special_cargo.md"
    assert hits[0].passage.section == "Combining HAZ and SOC"


def test_search_filtered_by_file(index: KnowledgeIndex) -> None:
    hits = index.search("surcharge", k=5, file_name="release_notes_3_2.md")
    assert hits
    assert {h.passage.file_name for h in hits} == {"release_notes_3_2.md"}
    assert index.search("anything", file_name="missing.md") == []


def test_search_many_deduplicates(index: KnowledgeIndex) -> None:
    hits = index.search_many(["draft price line", "draft line confirm"], k=3)
    ids = [h.passage.passage_id for h in hits]
    assert len(ids) == len(set(ids))


def test_rrf() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]])
    assert max(fused, key=lambda i: fused[i]) == 2


class KeywordEmbedder:
    """Two-dimensional fake embedding: (mentions 'gamma', mentions 'alpha')."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float("gamma" in t), float("alpha" in t) + 0.01] for t in texts]


def test_hybrid_search_uses_embeddings() -> None:
    index = KnowledgeIndex.from_documents([DOC], KeywordEmbedder())
    assert index.hybrid
    hits = index.search("gamma", k=1)
    assert hits[0].passage.section == "Second"
