"""Judges behind an API (DESIGN.md §17.10) against mocked HTTP (respx)."""

import json
import logging
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
from video_beep_remover.context.models import LABELS, CachedJudge, JudgeError, Question
from video_beep_remover.errors import ConfigError, DependencyError
from video_beep_remover.models import Detection

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
JEV = "https://thejevai.com/v1/systemone"
LINES = [Line(0.0, 1.0, "We're going down.", cue=1), Line(1.0, 2.0, "What the hell was that?", cue=2)]


def gemini(**settings: Any) -> Any:
    return api_judge(ContextApiConfig(api_key="secret", **settings), sleep=lambda seconds: None)


def jev() -> Any:
    return api_judge(ContextApiConfig(provider="jev", api_key="secret"), sleep=lambda seconds: None)


def chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


class Clock:
    """Time that passes only when the code under test sleeps, or when the test says so."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


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
    clock = Clock()
    judge = api_judge(ContextApiConfig(api_key="secret"), sleep=clock.sleep, clock=clock)
    assert judge.ask(Question("sexual", "Is it?", ">> Hi")) == '{"sexual": true}'
    assert (route.call_count, clock.waits) == (4, [45.0, 2.0, 20.0])


LITE, FLASH = "gemini-3.5-flash-lite", "gemini-3.8-flash"
QUESTION = Question("sexual", "Is it?", ">> Hi")


def busy(model: str, seconds: int, *, per: str = "Minute") -> httpx.Response:
    """Gemini's 429 for a model whose free quota for the minute (or the day) is used up."""
    error = {
        "code": 429,
        "message": f"You exceeded your current quota. Quota exceeded for metric: ..., model: {model}",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": f"GenerateRequestsPer{per}PerProjectPerModel-FreeTier"}],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{seconds}s"},
        ],
    }
    return httpx.Response(429, json=[{"error": error}])


def models(replies: dict[str, list[httpx.Response]]) -> list[str]:
    """Gemini, each model giving its own replies in turn. Returns the models asked, in order."""
    asked: list[str] = []

    def reply(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        asked.append(model)
        return replies[model].pop(0)

    respx.post(GEMINI).mock(side_effect=reply)
    return asked


def rotating(clock: Clock, model: str = LITE, *fallbacks: str) -> Any:
    api = ContextApiConfig(api_key="secret", model=model, fallback_models=list(fallbacks))
    return api_judge(api, sleep=clock.sleep, clock=clock)


@respx.mock
def test_a_rate_limited_model_rests_while_the_next_one_answers() -> None:
    """Gemini's free tier limits each model separately (context.api.fallback_models)."""
    clock = Clock()
    asked = models(
        {
            LITE: [busy(LITE, 30), chat('{"sexual": true}')],
            FLASH: [chat('{"sexual": false}'), chat('{"sexual": false}')],
        }
    )
    judge = rotating(clock, LITE, FLASH)
    assert judge.ask(QUESTION) == '{"sexual": false}'  # the first is busy: the next answers at once
    assert judge.ask(QUESTION) == '{"sexual": false}'  # the first still rests
    clock.now += 30
    assert judge.ask(QUESTION) == '{"sexual": true}'  # its rest is over
    assert asked == [LITE, FLASH, FLASH, LITE]
    assert clock.waits == [] and judge.answered == {FLASH: 2, LITE: 1}


@respx.mock
def test_a_model_out_of_its_daily_quota_is_asked_nothing_more(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    clock = Clock()
    asked = models({LITE: [busy(LITE, 40, per="Day")], FLASH: [chat("{}"), chat("{}"), chat("{}")]})
    judge = rotating(clock, LITE, FLASH)
    judge.ask(QUESTION)
    clock.now += 3600  # its 40 s are long over, but not the day
    judge.ask(QUESTION)
    judge.ask(QUESTION)
    assert asked == [LITE, FLASH, FLASH, FLASH]
    assert [record.getMessage() for record in caplog.records] == [
        f"the context judge asks {LITE} nothing more in this run: it has used up its daily quota"
    ]


@respx.mock
def test_when_every_model_rests_the_judge_waits_for_the_first_back() -> None:
    clock = Clock()
    asked = models({LITE: [busy(LITE, 20)], FLASH: [busy(FLASH, 10), chat('{"sexual": true}')]})
    assert rotating(clock, LITE, FLASH).ask(QUESTION) == '{"sexual": true}'
    assert (asked, clock.waits) == ([LITE, FLASH, FLASH], [10.0])


@respx.mock
def test_a_model_not_offered_or_overloaded_hands_over_to_the_next() -> None:
    clock = Clock()
    gone = httpx.Response(404, json=[{"error": {"message": "models/gemini-2.5-flash is not found"}}])
    overloaded = httpx.Response(503, json=[{"error": {"message": "The model is overloaded."}}])
    asked = models({"gemini-2.5-flash": [gone], LITE: [overloaded, chat("{}")], FLASH: [chat("{}")]})
    judge = rotating(clock, "gemini-2.5-flash", LITE, FLASH)
    judge.ask(QUESTION)  # 2.5 is not offered to this key, and Flash-Lite is overloaded: Flash answers
    clock.now += 10
    judge.ask(QUESTION)  # Flash-Lite has rested; 2.5 is never asked again
    assert asked == ["gemini-2.5-flash", LITE, FLASH, LITE]


@respx.mock
def test_each_model_gets_its_own_reasoning_effort() -> None:
    clock = Clock()
    route = respx.post(GEMINI).mock(side_effect=[busy("gemini-2.5-flash", 9), busy(FLASH, 9), chat("{}")])
    rotating(clock, "gemini-2.5-flash", FLASH, "gemma-3-27b-it").ask(QUESTION)
    sent = [json.loads(call.request.content) for call in route.calls]
    assert [(body["model"], body.get("reasoning_effort")) for body in sent] == [
        ("gemini-2.5-flash", "none"),  # can answer without thinking
        (FLASH, "low"),  # cannot
        ("gemma-3-27b-it", None),  # takes none
    ]


@respx.mock
def test_the_report_counts_the_answers_of_each_model() -> None:
    clock = Clock()
    models({LITE: [busy(LITE, 30)], FLASH: [chat('{"use": "profane", "reason": "exclamation"}')]})
    overrides = {"context.judge": "api", "context.api.fallback_models": [FLASH]}
    config = load_config(None, env={"VBR_JUDGE_API_KEY": "k"}, overrides=overrides).config
    clean = SimpleNamespace(name="clean", score=lambda texts: [dict.fromkeys(LABELS, 0.0) for _ in texts])
    layer = ContextLayer(
        config,
        device="cpu",
        cache_dir=None,
        classifier_factory=lambda *_: clean,
        judge_factory=lambda name, device: api_judge(config.context.api, sleep=clock.sleep, clock=clock),
    )
    hell = Detection(1.2, 1.4, " hell", "hell", "mild", 0.9, "asr", 2)  # ambiguous: the judge is asked
    _, section = layer.run([hell], LINES)
    assert (section["judge"], section["judge_questions"]) == ("gemini:gemini-3.5-flash-lite", 1)
    assert section["judge_models"] == {FLASH: 1}


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
def test_a_service_that_keeps_failing_is_asked_less_then_not_at_all(caplog: pytest.LogCaptureFixture) -> None:
    """A service that is down would otherwise cost minutes of retries for every question."""
    caplog.set_level(logging.DEBUG)
    route = respx.post(GEMINI).mock(side_effect=httpx.ConnectError("refused"))
    judge = CachedJudge(gemini(), None, version=1)
    questions = [sense_question(LINES, 1, word) for word in ("hell", "damn", "ass", "bitch", "piss")]
    assert [ask(judge, question) for question in questions] == [{}] * 5  # each use stays unsure
    assert route.call_count == 4 + 1 + 1  # every retry, then one attempt each, then nothing
    assert judge.unanswered == 5
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    # One warning per question asked, and one that the service is given up on; skipped questions are quiet.
    assert len(warnings) == 4 and sum("asked nothing more" in warning for warning in warnings) == 1


@respx.mock
def test_an_answer_brings_the_retries_back() -> None:
    route = respx.post(GEMINI).mock(
        side_effect=[
            *[httpx.ConnectError("down")] * 4,
            chat('{"sexual": true}'),
            httpx.ConnectError("blip"),
            chat("{}"),
        ]
    )
    judge = gemini()
    question = Question("sexual", "Is it?", ">> Hi")
    with pytest.raises(JudgeError, match="cannot reach"):
        judge.ask(question)  # four attempts
    assert judge.ask(question) == '{"sexual": true}'  # one attempt
    assert judge.ask(question) == "{}"  # retried again
    assert route.call_count == 4 + 1 + 2


@respx.mock
def test_gemini_rejecting_the_key_stops_the_run() -> None:
    """Gemini answers an unknown key with a 400; any other 400 fails only its question."""
    key = [
        {"error": {"code": 400, "message": "API key not valid.", "details": [{"reason": "API_KEY_INVALID"}]}}
    ]
    route = respx.post(GEMINI).mock(return_value=httpx.Response(400, json=key))
    with pytest.raises(DependencyError, match=r"rejected the API key.*API key not valid"):
        ask(gemini(), sense())
    route.mock(
        return_value=httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}})
    )
    with pytest.raises(JudgeError, match="HTTP 400: Unsupported value"):
        gemini().ask(sense())


@respx.mock
def test_an_empty_answer_is_not_kept() -> None:
    """A model that thinks until it runs out of tokens answers nothing; asked again, it may answer."""
    empty = httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    respx.post(GEMINI).mock(side_effect=[empty, chat('{"sexual": true}')])
    judge = CachedJudge(gemini(), None, version=1)
    question = Question("sexual", "Is it?", ">> Hi")
    with pytest.raises(JudgeError, match=r"an empty answer from .* \(length\)"):
        judge.ask(question)
    assert judge.answers == {} and judge.ask(question) == '{"sexual": true}'


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
