from __future__ import annotations

import asyncio
import json

import pytest

from eval.run_eval import (
    add_coherence_scores,
    judge_coherence,
    retrieval_metrics,
    score_question,
    summarise,
    to_markdown,
)


class Judge:
    model_name = "test-judge"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[dict[str, str]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.messages = messages
        return self.response


def test_coherence_judge_normalizes_score_and_uses_clear_factuality_boundary():
    judge = Judge('{"score": 5, "rationale": "Clear and logically ordered."}')

    score, rationale = asyncio.run(judge_coherence(judge, "Question", "Answer [1]"))

    assert score == 1.0
    assert rationale == "Clear and logically ordered."
    assert "Do not judge factual correctness" in judge.messages[0]["content"]
    assert json.loads(judge.messages[1]["content"])["answer"] == "Answer"


def test_coherence_judge_accepts_json_markdown_fence():
    judge = Judge('```json\n{"score": 4, "rationale": "Clear."}\n```')

    score, rationale = asyncio.run(judge_coherence(judge, "Question", "Answer"))

    assert score == 0.75
    assert rationale == "Clear."


def test_coherence_judge_retries_truncated_json_with_compact_prompt():
    class SequenceJudge:
        model_name = "test-judge"

        def __init__(self) -> None:
            self.responses = ['{"score": 3, "rationale": "truncated', '{"score": 4, "rationale": "Clear and ordered."}']
            self.messages: list[list[dict[str, str]]] = []

        async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            self.messages.append(messages)
            return self.responses.pop(0)

    judge = SequenceJudge()

    score, rationale = asyncio.run(judge_coherence(judge, "Question", "Answer"))

    assert score == 0.75 and rationale == "Clear and ordered."
    assert len(judge.messages) == 2
    assert "compact JSON" in judge.messages[1][0]["content"]


@pytest.mark.parametrize(
    "response",
    ['{"score": 0}', '{"score": 6}', '{"score": true}', "not json"],
)
def test_coherence_judge_rejects_invalid_scores(response: str):
    with pytest.raises(ValueError, match="Coherence judge"):
        asyncio.run(judge_coherence(Judge(response), "Question", "Answer"))


def test_add_coherence_scores_skips_refusals():
    judge = Judge('{"score": 3, "rationale": "Understandable."}')
    rows = [
        {"answered": True, "question": "Q", "answer": "A", "coherence_score": None},
        {"answered": False, "question": "Q2", "answer": "Not available", "coherence_score": None},
    ]

    asyncio.run(add_coherence_scores(judge, rows))

    assert rows[0]["coherence_score"] == 0.5
    assert rows[1]["coherence_score"] is None
    assert len(judge.messages) == 2


def test_summary_includes_coherence_hallucination_and_unsupported_claim_metrics():
    rows = [
        {
            "expected_answerable": True,
            "passed": True,
            "answered": True,
            "grounded": True,
            "citation_ok": True,
            "latency_ms": 100.0,
            "coherence_score": 0.75,
            "unsupported_claims": 0,
        },
        {
            "expected_answerable": False,
            "passed": False,
            "answered": True,
            "grounded": False,
            "citation_ok": None,
            "latency_ms": 200.0,
            "coherence_score": 0.25,
            "unsupported_claims": 2,
        },
    ]

    summary = summarise(rows, {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0})

    assert summary["coherence"] == 0.5
    assert summary["hallucinated_out_of_doc"] == 1
    assert summary["hallucination_rate"] == 1.0
    assert summary["unsupported_claims"] == 2


def test_refused_answer_is_excluded_from_citation_accuracy_but_not_coverage():
    rows = [
        {"expected_answerable": True, "passed": True, "answered": True, "grounded": True, "citation_ok": True, "citations": ["Policy.pdf p.1"], "latency_ms": 20, "coherence_score": None, "unsupported_claims": 0},
        {"expected_answerable": True, "passed": False, "answered": False, "grounded": None, "citation_ok": None, "citations": [], "latency_ms": 10, "coherence_score": None, "unsupported_claims": None},
    ]

    summary = summarise(rows, None)

    assert summary["citation_accuracy"] == 1.0
    assert summary["citation_coverage"] == 0.5


def test_question_metrics_use_gold_citations_verification_and_retrieved_text():
    question = {
        "id": "D1",
        "category": "direct",
        "question": "How many annual leave days?",
        "answerable": True,
        "must": ["24"],
    }
    response = {
        "answer": "Employees receive 24 annual leave days. [1]",
        "answered": True,
        "citations": [{"document": "Policy.pdf", "page": 2}],
        "sources": [{"snippet": "Employees receive 24 annual leave days", "score": 0.82}],
        "verification": {"ok": True, "unsupported_sentences": []},
        "confidence": "high",
        "timings": {"total_ms": 45.0},
    }

    row = score_question(question, response, {("Policy.pdf", 2)})

    assert row["citation_accuracy"] is True
    assert row["answer_relevance"] > 0
    assert row["context_relevance"] > 0
    assert row["retrieval_similarity"] == 0.82
    assert row["hallucination"] is False


def test_retrieval_metrics_average_fraction_of_gold_pages_found():
    class Retriever:
        def retrieve(self, question, **kwargs):
            class Hit:
                def __init__(self, filename, page):
                    self.filename, self.page = filename, page

            return type("Result", (), {"candidates": [Hit("Policy.pdf", 2), Hit("Policy.pdf", 3)]})()

    questions = [{"question": "Q", "_gold": [type("Gold", (), {"document": "Policy.pdf", "page": 2})(), type("Gold", (), {"document": "Policy.pdf", "page": 4})()]}]

    metrics = retrieval_metrics(Retriever(), questions, ["DOC001"])

    assert metrics["recall_at_5"] == 0.5
    assert metrics["recall_at_10"] == 0.5
    assert metrics["mrr"] == 1.0


def test_markdown_report_includes_question_level_quality_metrics():
    row = {
        "id": "D1", "category": "direct", "passed": True, "grounded": True,
        "answer_relevance": 0.75, "context_relevance": 0.5, "citation_accuracy": True,
        "hallucination": False, "retrieval_similarity": 0.82, "coherence_score": 1.0,
        "citations": ["Policy.pdf p.1"], "confidence": "high", "latency_ms": 12.0,
        "answered": True, "question": "Q", "answer": "A",
    }
    report = to_markdown(
        {},
        [row],
        None,
        {"mode": "test", "llm": "fake", "embedder": "fake", "reranker": "lexical", "timestamp": "now"},
    )

    assert "Answer rel." in report and "Context rel." in report
    assert "Citation acc." in report and "Retrieval similarity" in report
    assert "0.75" in report and "0.82" in report
