from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import anthropic
import pytest
from pydantic import BaseModel

from hypercare.config import Settings
from hypercare.llm import LLMError, LLMUnavailableError, RecordingClient, ReplayClient, build_llm
from hypercare.llm.anthropic_client import AnthropicClient
from hypercare.llm.base import Usage, estimate_cost, prompt_key
from hypercare.llm.fake import FakeLLM


class Out(BaseModel):
    answer: str


def test_cost_estimate() -> None:
    assert estimate_cost("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.0)
    assert estimate_cost("unknown-model", 10, 10) == 0.0


def test_usage_addition() -> None:
    total = Usage("m", 1, 2, 3, 4, 0.5) + Usage("", 1, 1, 0, 0, 0.25)
    assert (total.model, total.input_tokens, total.cost_usd) == ("m", 2, 0.75)


def test_record_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "replays.json"
    live = FakeLLM().on("t", {"answer": "hello"})
    RecordingClient(live, path).structured(task="t", system="s", user="u", schema=Out)
    data = json.loads(path.read_text())
    assert list(data) == [prompt_key("t", "s", "u")]

    replay = ReplayClient(path)
    assert len(replay) == 1
    assert replay.structured(task="t", system="s", user="u", schema=Out).output.answer == "hello"
    with pytest.raises(LLMUnavailableError, match="live mode"):
        replay.structured(task="t", system="s", user="other", schema=Out)


def test_replay_entry_with_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "replays.json"
    path.write_text(json.dumps({prompt_key("t", "s", "u"): {"output": {"x": 1}, "usage": {}}}))
    with pytest.raises(LLMUnavailableError, match="schema"):
        ReplayClient(path).structured(task="t", system="s", user="u", schema=Out)


def test_build_llm_modes(tmp_path: Path) -> None:
    replay = build_llm(Settings(mode="replay", replay_path=tmp_path / "r.json"))
    assert isinstance(replay, ReplayClient)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm(Settings(mode="live", _env_file=None, anthropic_api_key=None))  # type: ignore[call-arg]
    live = build_llm(
        Settings(mode="live", ANTHROPIC_API_KEY="k", replay_path=tmp_path / "r.json"),  # type: ignore[call-arg]
        record=True,
    )
    assert isinstance(live, RecordingClient)


# ---------------------------------------------------------------- Anthropic client vs a stub server


class Stub:
    def __init__(self) -> None:
        self.reply: dict[str, Any] = {}
        self.status = 200
        self.requests: list[dict[str, Any]] = []


@pytest.fixture
def stub() -> Iterator[tuple[Stub, str]]:
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state.requests.append({"body": body, "beta": self.headers.get("anthropic-beta")})
            data = json.dumps(state.reply).encode()
            self.send_response(state.status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def message(text: str, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        },
    }


def client(base_url: str) -> AnthropicClient:
    sdk = anthropic.Anthropic(api_key="test", base_url=base_url, max_retries=0)
    return AnthropicClient(model="claude-opus-5-5", effort="high", client=sdk)


def test_anthropic_client_request_and_parse(stub: tuple[Stub, str]) -> None:
    state, url = stub
    state.reply = message(json.dumps({"answer": "ok"}))
    result = client(url).structured(task="t", system="sys", user="hi", schema=Out)
    assert result.output.answer == "ok"
    assert result.usage.cache_read_tokens == 500
    assert result.usage.cost_usd == pytest.approx((1000 * 4 + 100 * 20 + 500 * 0.2) / 1e6)
    req = state.requests[0]
    assert req["beta"] == "server-side-fallback-2026-07-01"
    assert req["body"]["fallbacks"] == "default"
    assert req["body"]["output_config"]["effort"] == "high"
    assert req["body"]["output_config"]["format"]["type"] == "json_schema"
    assert req["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    ("status", "reply", "match"),
    [
        (200, message("", stop="refusal"), "declined|invalid structured output"),
        (200, message('{"answer": "o', stop="max_tokens"), "truncated|invalid structured output"),
        (
            401,
            {"type": "error", "error": {"type": "authentication_error", "message": "bad"}},
            "authentication",
        ),
        (
            400,
            {"type": "error", "error": {"type": "invalid_request_error", "message": "nope"}},
            "API error 400",
        ),
    ],
)
def test_anthropic_client_errors(
    stub: tuple[Stub, str], status: int, reply: dict[str, Any], match: str
) -> None:
    state, url = stub
    state.status, state.reply = status, reply
    with pytest.raises(LLMError, match=match):
        client(url).structured(task="t", system="s", user="u", schema=Out)


def test_anthropic_client_connection_error() -> None:
    with pytest.raises(LLMError, match="network"):
        client("http://127.0.0.1:9").structured(task="t", system="s", user="u", schema=Out)
