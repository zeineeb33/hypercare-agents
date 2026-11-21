"""Knowledge agent: plan search queries, retrieve passages, answer strictly from them."""

from __future__ import annotations

from hypercare.agents.schemas import (
    KnowledgeAnswerOutput,
    KnowledgeResult,
    PassageRef,
    SearchPlanOutput,
)
from hypercare.llm.base import LLMClient, LLMError, Usage
from hypercare.retrieval import Hit, KnowledgeIndex

PLAN_SYSTEM = """\
You write search queries for the knowledge base of the hypercare support team of Northwind Shipping.
The knowledge base documents TideRate (the pricing platform: price lists, price lines, surcharges,
inland tariffs) and QuotePort (the quoting portal used by sales).

Given a support ticket, return one to three short keyword queries, one per distinct question in the
ticket. Use domain vocabulary (surcharge names, line statuses, quote validity, inland charges, special
cargo) rather than customer names, amounts or reference numbers."""

ANSWER_SYSTEM = """\
You are the knowledge agent of the hypercare support team for TideRate and QuotePort at Northwind
Shipping. Answer the support question using only the documentation passages provided.

- Cite in sources_used the file name of every document you rely on.
- Mention a document's version when it matters.
- If the passages do not answer the question, or answer only part of it, set insufficient_evidence to
  true and state what is missing. Do not fill gaps with general knowledge.
- Keep the answer under 150 words."""


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(h.passage.as_context() for h in hits)


class KnowledgeAgent:
    def __init__(self, llm: LLMClient, index: KnowledgeIndex, top_k: int = 4) -> None:
        self.llm = llm
        self.index = index
        self.top_k = top_k

    def plan(self, question: str) -> tuple[list[str], Usage]:
        """Ask the model for search queries; fall back to the raw question if it fails."""
        try:
            result = self.llm.structured(
                task="knowledge.plan",
                system=PLAN_SYSTEM,
                user=f"Support ticket:\n{question}",
                schema=SearchPlanOutput,
            )
        except LLMError:
            return [], Usage()
        queries = [q.strip() for q in result.output.queries if q.strip()][:3]
        return queries, result.usage

    def retrieve(
        self, question: str, queries: list[str], related_doc: str | None = None
    ) -> list[Hit]:
        hits = self.index.search_many([question, *queries], k=self.top_k)
        if related_doc and related_doc not in {h.passage.file_name for h in hits}:
            pinned = self.index.search(question, k=1, file_name=related_doc)
            first = self.index.first_passage(related_doc)
            if not pinned and first is not None:  # the ticket names it: include it anyway
                pinned = [Hit(first, 0.0)]
            hits = (pinned + hits)[: self.top_k]
        return hits

    def answer(self, question: str, hits: list[Hit]) -> tuple[KnowledgeAnswerOutput, Usage]:
        """Raises LLMError when the model fails."""
        result = self.llm.structured(
            task="knowledge.answer",
            system=ANSWER_SYSTEM,
            user=f"DOCUMENTATION PASSAGES:\n{format_passages(hits)}\n\nQUESTION:\n{question}",
            schema=KnowledgeAnswerOutput,
        )
        return result.output, result.usage

    def run(self, question: str, related_doc: str | None = None) -> tuple[KnowledgeResult, Usage]:
        queries, usage = self.plan(question)
        hits = self.retrieve(question, queries, related_doc)
        result = KnowledgeResult(
            question=question,
            queries=queries,
            passages=[
                PassageRef(
                    file_name=h.passage.file_name,
                    section=h.passage.section,
                    version=h.passage.version,
                    score=h.score,
                )
                for h in hits
            ],
        )
        if not hits:
            result.answer = KnowledgeAnswerOutput(
                answer="No documentation passage matches this question.",
                sources_used=[],
                confidence="LOW",
                insufficient_evidence=True,
            )
            return result, usage
        try:
            answer, answer_usage = self.answer(question, hits)
        except LLMError as e:
            result.error = str(e)
            return result, usage
        result.answer = answer
        return result, usage + answer_usage
