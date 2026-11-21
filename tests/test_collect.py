from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypercare.agents.schemas import SearchPlanOutput
from hypercare.config import get_settings
from hypercare.llm import LLMUnavailableError
from hypercare.llm.collect import AnswerImportError, CollectingClient, import_answers
from hypercare.llm.replay import ReplayClient


def test_collecting_client_records_then_blocks(tmp_path: Path) -> None:
    client = CollectingClient(tmp_path / "replays.json")
    client.start_run()
    with pytest.raises(LLMUnavailableError):
        client.structured(task="a", system="s", user="u1", schema=SearchPlanOutput)
    # Later prompts of the same run depend on the missing answer: not collected.
    with pytest.raises(LLMUnavailableError, match="waiting"):
        client.structured(task="b", system="s", user="u2", schema=SearchPlanOutput)
    assert [v["task"] for v in client.pending.values()] == ["a"]
    client.start_run()
    with pytest.raises(LLMUnavailableError):
        client.structured(task="b", system="s", user="u2", schema=SearchPlanOutput)
    assert len(client.pending) == 2


def test_import_validates_and_stores(tmp_path: Path) -> None:
    replay = tmp_path / "replays.json"
    client = CollectingClient(replay)
    with pytest.raises(LLMUnavailableError):
        client.structured(task="knowledge.plan", system="s", user="u", schema=SearchPlanOutput)
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(client.pending))
    key = next(iter(client.pending))
    answers = tmp_path / "answers.json"

    answers.write_text(json.dumps({key: {"wrong": 1}}))
    with pytest.raises(AnswerImportError, match="does not match"):
        import_answers(pending, answers, replay, model_label="test")
    answers.write_text(json.dumps({"other:key": {"queries": []}}))
    with pytest.raises(AnswerImportError, match="not pending"):
        import_answers(pending, answers, replay, model_label="test")

    answers.write_text(json.dumps({key: {"queries": ["surcharges"]}}))
    assert import_answers(pending, answers, replay, model_label="test") == 1
    result = ReplayClient(replay).structured(
        task="knowledge.plan", system="s", user="u", schema=SearchPlanOutput
    )
    assert result.output.queries == ["surcharges"]
    assert result.usage.model == "test"


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HYPERCARE_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("HYPERCARE_MODE", "replay")
    monkeypatch.setenv("HYPERCARE_REPLAY_PATH", str(tmp_path / "replays.json"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_prompts_round_trip(cli_env: Path) -> None:
    from hypercare.cli import app

    runner = CliRunner()
    pending = cli_env / "pending.json"
    result = runner.invoke(app, ["prompts", "export", "--out", str(pending)])
    assert result.exit_code == 0, result.output
    data = json.loads(pending.read_text())
    assert {v["task"] for v in data.values()} == {"knowledge.plan"}
    assert len(data) == 28  # 30 tickets, minus the 2 clarifications that need no model

    result = runner.invoke(app, ["prompts", "show", str(pending), "--task", "knowledge.plan"])
    assert "SYSTEM for knowledge.plan" in result.output

    answers = cli_env / "answers.json"
    answers.write_text(json.dumps({k: {"queries": ["x"]} for k in data}))
    result = runner.invoke(app, ["prompts", "import", str(answers), "--pending", str(pending)])
    assert result.exit_code == 0, result.output
    assert "28 answer(s)" in result.output

    answers.write_text(json.dumps({"nope": {}}))
    result = runner.invoke(app, ["prompts", "import", str(answers), "--pending", str(pending)])
    assert result.exit_code == 1


def test_committed_replays_cover_the_whole_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every LLM call of the demo (30 tickets + self-healing loop) has a recorded answer."""
    from hypercare.cli import app

    monkeypatch.setenv("HYPERCARE_MODE", "replay")
    get_settings.cache_clear()
    try:
        out = tmp_path / "pending.json"
        result = CliRunner().invoke(app, ["prompts", "export", "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert json.loads(out.read_text()) == {}
        assert "1 approved" in result.output
    finally:
        get_settings.cache_clear()
