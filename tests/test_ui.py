from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hypercare.config import PACKAGE_DIR, get_settings

APP = str(PACKAGE_DIR / "ui" / "streamlit_app.py")


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HYPERCARE_DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("HYPERCARE_MODE", "replay")
    monkeypatch.setenv("HYPERCARE_REPLAY_PATH", str(tmp_path / "replays.json"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_app_renders_and_runs_a_ticket(app_env: None) -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert "Hypercare Agents" in at.title[0].value
    assert any("Replay mode" in i.value for i in at.info)

    at.selectbox[0].set_value(23).run()
    next(b for b in at.button if "Run the agents" in b.label).click().run()
    assert not at.exception
    assert any("ASK_CLARIFICATION" in m.value for m in at.markdown)


def test_knowledge_search_and_maintenance_in_replay(app_env: None) -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].set_value("inland pre-carriage").run()
    assert not at.exception
    assert any("guide_inland_tariff.md" in e.label for e in at.expander)
    # Without recorded answers the maintenance agent reports the problem instead of crashing.
    next(b for b in at.button if "maintenance agent" in b.label).click().run()
    assert not at.exception
