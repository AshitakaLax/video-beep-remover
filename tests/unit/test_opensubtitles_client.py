"""The OpenSubtitles client against mocked HTTP (respx)."""

from typing import Any

import httpx
import pytest
import respx

from video_beep_remover.subtitles.opensubtitles import (
    API,
    KeyRejected,
    OpenSubtitlesClient,
    OpenSubtitlesError,
    QuotaExceeded,
)

SRT = b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"


def result(file_id: int, **attributes: Any) -> dict[str, Any]:
    files = attributes.pop("files", [{"file_id": file_id, "file_name": f"{file_id}.srt"}])
    return {
        "id": str(file_id),
        "type": "subtitle",
        "attributes": {"language": "en", "files": files, **attributes},
    }


def client(sleeps: list[float] | None = None, **kwargs: Any) -> OpenSubtitlesClient:
    record = sleeps if sleeps is not None else []
    return OpenSubtitlesClient("my-key", user_agent="video-beep-remover v0.1", sleep=record.append, **kwargs)


@respx.mock
def test_search_sends_the_key_and_sorted_lowercase_parameters() -> None:
    route = respx.get(f"{API}/subtitles").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    result(1, moviehash_match=True, fps=23.976, hearing_impaired=True, download_count=900,
                           release="The.Movie.2019.1080p.BluRay.x264-SPARKS"),
                    result(2, fps=0, ai_translated=True),
                    result(3, files=[{"file_id": 31}, {"file_id": 32}]),  # two CDs: skipped
                ]
            },
        )
    )  # fmt: skip
    found = client().search(
        languages=["EN", "de"], moviehash="3F1C9A0B7D2E4C55", query="The Movie", year=2019
    )
    request = route.calls.last.request
    assert request.headers["Api-Key"] == "my-key"
    assert request.headers["User-Agent"] == "video-beep-remover v0.1"
    assert (
        request.url.query.decode() == "languages=de%2Cen&moviehash=3f1c9a0b7d2e4c55&query=the+movie&year=2019"
    )
    assert [(r.file_id, r.moviehash_match, r.fps, r.machine_translated) for r in found] == [
        (1, True, 23.976, False),
        (2, False, None, True),
    ]
    assert found[0].release == "The.Movie.2019.1080p.BluRay.x264-SPARKS" and found[0].download_count == 900


@respx.mock
def test_rejected_key() -> None:
    respx.get(f"{API}/subtitles").mock(
        return_value=httpx.Response(403, json={"message": "You cannot consume this service"})
    )
    with pytest.raises(KeyRejected, match="You cannot consume this service"):
        client().search(languages=["en"], query="x")
    respx.get(f"{API}/infos/formats").mock(
        return_value=httpx.Response(401, json={"message": "Invalid API key"})
    )
    with pytest.raises(KeyRejected):
        client().check_key()


@respx.mock
def test_rate_limits_are_retried_after_the_requested_wait() -> None:
    route = respx.get(f"{API}/subtitles").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"data": []}),
        ]
    )
    sleeps: list[float] = []
    assert client(sleeps).search(languages=["en"], query="x") == []
    assert route.call_count == 2 and sleeps == [2.0]


@respx.mock
def test_server_and_network_errors_give_up_after_three_tries() -> None:
    respx.get(f"{API}/subtitles").mock(return_value=httpx.Response(503))
    sleeps: list[float] = []
    with pytest.raises(OpenSubtitlesError, match="HTTP 503"):
        client(sleeps).search(languages=["en"], query="x")
    assert sleeps == [1.0, 2.0]  # capped exponential backoff between the three attempts
    respx.get(f"{API}/subtitles").mock(side_effect=httpx.ConnectError("no route to host"))
    with pytest.raises(OpenSubtitlesError, match=r"cannot reach api\.opensubtitles\.com"):
        client().search(languages=["en"], query="x")


@respx.mock
def test_download_logs_in_and_fetches_the_link_without_the_key() -> None:
    login = respx.post(f"{API}/login").mock(
        return_value=httpx.Response(
            200,
            json={
                "token": "tok",
                "base_url": "vip-api.opensubtitles.com",
                "user": {"allowed_downloads": 1000},
            },
        )
    )
    download = respx.post("https://vip-api.opensubtitles.com/api/v1/download").mock(
        return_value=httpx.Response(
            200,
            json={"link": "https://www.opensubtitles.com/download/abc/1.srt", "file_name": "The.Movie.en.srt",
                  "remaining": 999, "reset_time_utc": "2026-09-27T00:00:00.000Z"},
        )
    )  # fmt: skip
    link = respx.get("https://www.opensubtitles.com/download/abc/1.srt").mock(
        return_value=httpx.Response(200, content=SRT)
    )
    api = client(username="me", password="secret")
    fetched = api.download(1)
    assert (fetched.data, fetched.file_name, fetched.remaining) == (SRT, "The.Movie.en.srt", 999)
    assert fetched.reset_time_utc == "2026-09-27T00:00:00.000Z"
    assert login.calls.last.request.read() == b'{"username":"me","password":"secret"}'
    assert download.calls.last.request.headers["Authorization"] == "Bearer tok"
    assert download.calls.last.request.read() == b'{"file_id":1}'
    assert "Api-Key" not in link.calls.last.request.headers
    assert api.user == {"allowed_downloads": 1000}
    api.download(1)
    assert login.call_count == 1  # logged in once per client


@respx.mock
def test_quota_exhausted_reports_when_it_resets() -> None:
    respx.post(f"{API}/download").mock(
        return_value=httpx.Response(
            406,
            json={"remaining": 0, "message": "You have downloaded your allowed 5 subtitles for 24h",
                  "reset_time_utc": "2026-09-27T01:02:03.000Z"},
        )
    )  # fmt: skip
    with pytest.raises(QuotaExceeded, match="allowed 5 subtitles") as raised:
        client().download(1)
    assert raised.value.reset_time_utc == "2026-09-27T01:02:03.000Z"


@respx.mock
def test_oversized_downloads_are_refused() -> None:
    respx.post(f"{API}/download").mock(
        return_value=httpx.Response(200, json={"link": "https://dl.example/big.srt"})
    )
    respx.get("https://dl.example/big.srt").mock(
        return_value=httpx.Response(200, content=b"x" * (5 * 1024 * 1024 + 1))
    )
    with pytest.raises(OpenSubtitlesError, match="larger than 5 MB"):
        client().download(1)


@respx.mock
def test_failed_login_still_downloads() -> None:
    respx.post(f"{API}/login").mock(return_value=httpx.Response(401, json={"message": "Invalid credentials"}))
    respx.post(f"{API}/download").mock(
        return_value=httpx.Response(200, json={"link": "https://dl.example/1.srt"})
    )
    respx.get("https://dl.example/1.srt").mock(return_value=httpx.Response(200, content=SRT))
    api = client(username="me", password="wrong")
    assert api.download(1).data == SRT
    assert api.token is None


@respx.mock
def test_an_error_page_instead_of_json_is_an_opensubtitles_error() -> None:
    respx.get(f"{API}/subtitles").mock(return_value=httpx.Response(200, text="<html>Bad gateway</html>"))
    with pytest.raises(OpenSubtitlesError, match="search: the answer is not JSON"):
        client().search(languages=["en"], query="the movie")
    respx.get(f"{API}/subtitles").mock(return_value=httpx.Response(200, json=["not", "an", "object"]))
    with pytest.raises(OpenSubtitlesError, match="unexpected answer"):
        client().search(languages=["en"], query="the movie")


@respx.mock
def test_malformed_search_results_are_skipped() -> None:
    respx.get(f"{API}/subtitles").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": ["junk", {"attributes": {"files": "x"}}, result(2, download_count="many"), result(3)]
            },
        )
    )
    assert [r.file_id for r in client().search(languages=["en"], query="the movie")] == [3]


@respx.mock
def test_a_network_error_while_fetching_the_file_is_an_opensubtitles_error() -> None:
    respx.post(f"{API}/download").mock(
        return_value=httpx.Response(200, json={"link": "https://www.opensubtitles.com/download/abc/1.srt"})
    )
    respx.get("https://www.opensubtitles.com/download/abc/1.srt").mock(
        side_effect=httpx.ConnectError("connection reset")
    )
    with pytest.raises(OpenSubtitlesError, match="download failed: connection reset"):
        client().download(1)
