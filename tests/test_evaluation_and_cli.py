from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypercare.config import get_settings
from hypercare.evaluation import EvalCase, load_cases, mentions, run_evaluation, to_markdown
from hypercare.llm.fake import FakeLLM
from hypercare.retrieval import KnowledgeIndex


def test_mentions() -> None:
    assert mentions("Total USD 2,150 per box", "2150")
    assert mentions("refreshed every 6 hours", "six hours|6 hours")
    assert not mentions("nothing here", "BAF")


def test_cases_file_is_valid() -> None:
    cases = load_cases()
    assert len(cases) == 27
    assert {d for c in cases for d in c.expected} == {
        "ANSWER",
        "ESCALATE",
        "ASK_CLARIFICATION",
        "DETECT_KB_GAP",
    }


def test_run_evaluation(conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM) -> None:
    cases = [
        EvalCase(
            ticket_id=3,
            expected=["ANSWER"],
            sources=["rules_all_in_vs_base_rate.md"],
            mention=["1850", "2150"],
        ),
        EvalCase(ticket_id=23, expected=["ASK_CLARIFICATION"]),
        EvalCase(ticket_id=9, expected=["DETECT_KB_GAP"]),
        EvalCase(ticket_id=999, expected=["ANSWER"]),
    ]
    report = run_evaluation(conn, llm, index, model="fake", cases=cases, workers=2)
    assert report.cases == 4
    assert report.decision_accuracy == 0.75
    assert report.errors == 1
    assert report.source_accuracy == 1.0
    assert report.fact_accuracy == 1.0
    md = to_markdown(report)
    assert "Decision accuracy | 75%" in md


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HYPERCARE_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("HYPERCARE_MODE", "replay")
    monkeypatch.setenv("HYPERCARE_REPLAY_PATH", str(tmp_path / "replays.json"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_seed_run_review(cli_env: Path) -> None:
    from hypercare.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["seed"])
    assert result.exit_code == 0, result.output
    assert "tickets" in result.output

    # A clarification needs no LLM call, so it works with an empty replay file.
    result = runner.invoke(app, ["run", "23"])
    assert result.exit_code == 0, result.output
    assert "ASK_CLARIFICATION" in result.output

    result = runner.invoke(app, ["run", "23", "--json"])
    assert json.loads(result.output)["final"]["decision"] == "ASK_CLARIFICATION"

    # A ticket that needs the model escalates cleanly when the replay is missing.
    result = runner.invoke(app, ["run", "3"])
    assert result.exit_code == 0, result.output
    assert "ESCALATE" in result.output

    result = runner.invoke(app, ["review", "1", "--reviewer", "me"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_eval_single_ticket(cli_env: Path) -> None:
    from hypercare.cli import app

    out = cli_env / "reports"
    result = CliRunner().invoke(
        app, ["eval", "--ticket", "23", "--out", str(out), "--workers", "1"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "evaluation.json").read_text())
    assert report["decision_accuracy"] == 1.0
