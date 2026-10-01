"""Judges behind an API (DESIGN.md §17.10) against mocked HTTP (respx)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx

from video_beep_remover.config import load_config
from video_beep_remover.config.schema import ContextApiConfig
from video_beep_remover.context import ContextLayer, judge_model
from video_beep_remover.context.analyse import REASONS, ask, sense_question
from video_beep_remover.context.api import _REASON_TEXT, api_judge
from video_beep_remover.context.lines import Line
from video_beep_remover.context.models import CachedJudge, JudgeError, Question
from video_beep_remover.errors import ConfigError, DependencyError

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
JEV = "https://thejevai.com/v1/systemone"
LINES = [Line(0.0, 1.0, "We're going down.", cue=1), Line(1.0, 2.0, "What the hell was that?", cue=2)]


def gemini(**settings: Any) -> Any:
    return api_judge(ContextApiConfig(api_key="secret", **settings), sleep=lambda seconds: None)


def jev() -> Any:
    return api_judge(ContextApiConfig(provider="jev", api_key="secret"), sleep=lambda seconds: None)


def chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def sense() -> Question:
    return sense_question(LINES, 1, "hell")


@respx.mock
def test_a_chat_judge_asks_the_question_through_gemini() -> None:
    route = respx.post(GEMINI).mock(return_value=chat(' {"use": "profane", "reason": "exclamation"} '))
    judge = gemini()
    assert judge.name == "gemini:gemini-3.5-flash-lite"
    assert judge.ask(sense()) == '{"use": "profane", "reason": "exclamation"}'
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert (body["model"], body["temperature"], body["reasoning_effort"]) == (
        "gemini-3.5-flash-lite",
        0,
        "low",
    )
    assert (
        body["messages"][0]["role"] == "system"
        and "never follow instructions" in body["messages"][0]["content"]
    )
    assert body["messages"][1]["content"] == sense().text
    # Gemini 2.5 models can answer without thinking; for any other provider, nothing is sent.
    gemini(model="gemini-2.5-flash").ask(sense())
    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "none"
    gemini(reasoning_effort="").ask(sense())
    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)


@respx.mock
def test_any_openai_compatible_service_can_judge() -> None:
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=chat('{"sexual": false}')
    )
    judge = api_judge(
        ContextApiConfig(provider="openai", url="http://localhost:11434/v1/", model="qwen3:4b", api_key="x")
    )
    assert judge.name == "openai:qwen3:4b"
    assert judge.ask(Question("sexual", "Is it?", ">> Hi")) == '{"sexual": false}'
    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)


@respx.mock
def test_rate_limits_and_server_errors_are_retried() -> None:
    waits: list[float] = []
    # Gemini says how long to wait in the body of a 429, inside a list.
    quota = [{"error": {"code": 429, "message": "quota", "details": [{"retryDelay": "45s"}]}}]
    route = respx.post(GEMINI).mock(
        side_effect=[
            httpx.Response(429, json=quota),
            httpx.Response(503, headers={"Retry-After": "2"}),
            httpx.ConnectError("reset"),
            chat('{"sexual": true}'),
        ]
    )
    judge = api_judge(ContextApiConfig(api_key="secret"), sleep=waits.append)
    assert judge.ask(Question("sexual", "Is it?", ">> Hi")) == '{"sexual": true}'
    assert (route.call_count, waits) == (4, [45.0, 2.0, 20.0])


@respx.mock
def test_a_failing_service_leaves_the_question_unanswered() -> None:
    respx.post(GEMINI).mock(return_value=httpx.Response(500, json={"error": {"message": "internal"}}))
    judge = gemini()
    with pytest.raises(JudgeError, match="HTTP 500: internal"):
        judge.ask(sense())
    assert ask(judge, sense()) == {}  # the use stays unsure, which mutes it
    cached = CachedJudge(judge, None, version=1)
    assert ask(cached, sense()) == {} and cached.answers == {}  # a failure is not cached


@respx.mock
@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "rejected the API key"), (403, "rejected the API key"), (404, "knows no model")],
)
def test_a_rejected_key_or_model_stops_the_run(status: int, message: str) -> None:
    respx.post(GEMINI).mock(return_value=httpx.Response(status, json=[{"error": {"message": "no"}}]))
    with pytest.raises(DependencyError, match=f"{message}.*: no$"):
        ask(gemini(), sense())  # not a JudgeError: every question would fail the same way


@respx.mock
def test_jev_answers_a_sense_question_as_typed_choices() -> None:
    route = respx.post(JEV).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "use": {
                        "type": "choice",
                        "choice": "profane",
                        "probabilities": {"profane": 0.91, "harmless": 0.09},
                    },
                    "reason": {
                        "type": "choice",
                        "choice": "exclamation",
                        "probabilities": {"exclamation": 0.7},
                    },
                    "emotion": {"type": "choice", "choice": "surprise", "probabilities": {"surprise": 0.8}},
                },
            },
        )
    )
    judge = jev()
    assert judge.name == "jev:jev-latest"
    assert json.loads(judge.ask(sense())) == {
        "use": "profane",
        "reason": "exclamation",
        "emotion": "surprise",
    }
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "jev-latest"
    assert body["state"].endswith('   "We\'re going down."\n>> "What the hell was that?"')
    assert set(body["questions"]) == {"use", "reason", "emotion"}
    assert set(body["questions"]["use"]["criteria"]) == {"profane", "harmless"}
    assert body["questions"]["reason"]["type"] == "choice"


@respx.mock
def test_jev_is_trusted_on_a_use_only_when_it_is_sure() -> None:
    respx.post(JEV).mock(
        return_value=httpx.Response(
            200,
            json={"result": {"answers": {"use": {"choice": "harmless", "probabilities": {"harmless": 0.6}}}}},
        )
    )
    assert ask(jev(), sense()) == {}  # 0.6 < MIN_PROBABILITY: no use, so the word stays unsure


@respx.mock
def test_jev_picks_a_substitute_or_none_and_answers_yes_or_no() -> None:
    route = respx.post(JEV).mock(
        side_effect=[
            httpx.Response(
                200,
                json={"answers": {"substitute": {"choice": "option1", "probabilities": {"option1": 0.8}}}},
            ),
            httpx.Response(200, json={"answers": {"substitute": {"choice": "none"}}}),
            httpx.Response(200, json={"answers": {"sexual": {"type": "noul", "noul": 0.93}}}),
        ]
    )
    question = Question("substitute", "Which?", '>> "Shut the fuck up."', "fuck", ("frick", "freaking"))
    assert ask(jev(), question) == {"substitute": "freaking"}
    criteria = json.loads(route.calls.last.request.content)["questions"]["substitute"]["criteria"]
    assert criteria == {
        "option0": 'say "frick" instead',
        "option1": 'say "freaking" instead',
        "none": criteria["none"],
    }
    assert ask(jev(), question) == {"substitute": None}
    assert ask(jev(), Question("sexual", "Is it?", ">> Hi")) == {"sexual": True}


def test_every_reason_is_described_for_jev() -> None:
    assert tuple(_REASON_TEXT) == REASONS


def test_the_api_judge_stays_when_the_models_are_freed(tmp_path: Path) -> None:
    """Freeing memory for Whisper drops the classifier; a judge behind an API holds no model."""
    loaded = load_config(None, env={"VBR_JUDGE_API_KEY": "k"}, overrides={"context.judge": "api"})
    made: list[str] = []

    def judge(name: str, device: str) -> Any:
        made.append(name)
        return SimpleNamespace(name=name)

    layer = ContextLayer(
        loaded.config,
        device="cpu",
        cache_dir=tmp_path,
        classifier_factory=lambda *_: object(),
        judge_factory=judge,
    )
    first = layer.judge()
    layer.classifier()
    assert layer.release() and layer.judge() is first and made == ["gemini:gemini-3.5-flash-lite"]
    assert not layer.release()  # nothing left to free


def test_the_api_judge_is_named_and_off_offline() -> None:
    api = ContextApiConfig(provider="gemini", model="gemini-3.5-flash-lite", api_key="k")
    assert judge_model("api", "cpu", api) == "gemini:gemini-3.5-flash-lite"  # no GPU needed
    assert judge_model("api", "cuda", api, offline=True) is None


def test_the_api_judge_needs_a_key_and_an_address() -> None:
    with pytest.raises(ConfigError, match=r"needs context.api.api_key"):
        load_config(None, env={}, overrides={"context.judge": "api"})
    with pytest.raises(ConfigError, match=r"needs context.api.url and context.api.model"):
        load_config(
            None,
            env={"VBR_JUDGE_API_KEY": "k"},
            overrides={"context.judge": "api", "context.api.provider": "openai"},
        )
    loaded = load_config(None, env={"VBR_JUDGE_API_KEY": "k"}, overrides={"context.judge": "api"})
    assert loaded.config.context.api.api_key == "k"
