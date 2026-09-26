"""A small client for the OpenSubtitles.com REST API v1 (DESIGN.md §6.4).

Every API request sends the user's own Api-Key and a User-Agent naming the app. Logging in is
optional and raises the daily download quota. Searches are free; each download counts against
the quota, which is why downloaded files are cached (see cache.py)."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from video_beep_remover.errors import VbrError

log = logging.getLogger(__name__)

API = "https://api.opensubtitles.com/api/v1"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
ATTEMPTS = 3  # for rate limits (429), server errors and network errors
MAX_WAIT_S = 30.0


class OpenSubtitlesError(VbrError):
    """The service could not be used for this file; other subtitle sources are still tried."""


class KeyRejected(OpenSubtitlesError):
    pass


class QuotaExceeded(OpenSubtitlesError):
    def __init__(self, message: str, reset_time_utc: str | None) -> None:
        super().__init__(message)
        self.reset_time_utc = reset_time_utc


@dataclass(frozen=True)
class OnlineSubtitle:
    """One search result that has a single subtitle file (multi-CD results are skipped)."""

    file_id: int
    file_name: str | None
    language: str | None
    hearing_impaired: bool
    foreign_parts_only: bool
    machine_translated: bool  # machine or AI translated
    fps: float | None
    release: str | None
    download_count: int
    moviehash_match: bool


@dataclass(frozen=True)
class Download:
    data: bytes
    file_name: str | None
    remaining: int | None  # downloads left today
    reset_time_utc: str | None  # when the quota renews


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:200] or response.reason_phrase
    if isinstance(data, dict):
        for key in ("message", "error", "errors"):
            if data.get(key):
                value = data[key]
                return "; ".join(map(str, value)) if isinstance(value, list) else str(value)
    return response.reason_phrase


def parse_results(data: Any) -> list[OnlineSubtitle]:
    results = []
    for item in (data or {}).get("data") or []:
        attributes = item.get("attributes") or {}
        files = attributes.get("files") or []
        if len(files) != 1 or not isinstance(files[0].get("file_id"), int):
            continue  # split over several CDs, or malformed
        fps = attributes.get("fps")
        results.append(
            OnlineSubtitle(
                file_id=files[0]["file_id"],
                file_name=files[0].get("file_name"),
                language=attributes.get("language"),
                hearing_impaired=bool(attributes.get("hearing_impaired")),
                foreign_parts_only=bool(attributes.get("foreign_parts_only")),
                machine_translated=bool(
                    attributes.get("machine_translated") or attributes.get("ai_translated")
                ),
                fps=float(fps) if isinstance(fps, int | float) and fps > 0 else None,
                release=attributes.get("release") or None,
                download_count=int(attributes.get("download_count") or 0),
                moviehash_match=bool(attributes.get("moviehash_match")),
            )
        )
    return results


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    try:
        return min(MAX_WAIT_S, max(0.0, float(value)))
    except ValueError:
        return None


class OpenSubtitlesClient:
    def __init__(
        self,
        api_key: str,
        *,
        user_agent: str,
        username: str = "",
        password: str = "",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.username = username
        self.password = password
        self.api = API
        self.token: str | None = None
        self.user: dict[str, Any] | None = None  # from /login: allowed_downloads, level, vip...
        self._login_tried = False
        self._sleep = sleep
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _api_headers(self) -> dict[str, str]:
        headers = {"Api-Key": self.api_key}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request, retrying rate limits, server errors and network errors a few times."""
        for attempt in range(ATTEMPTS):
            last = attempt + 1 == ATTEMPTS
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if last:
                    raise OpenSubtitlesError(f"cannot reach {httpx.URL(url).host}: {exc}") from exc
                self._sleep(min(MAX_WAIT_S, 2.0**attempt))
                continue
            if (response.status_code == 429 or response.status_code >= 500) and not last:
                wait = _retry_after(response)
                self._sleep(wait if wait is not None else min(MAX_WAIT_S, 2.0**attempt))
                continue
            return response
        raise AssertionError("unreachable")

    def _api(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self._request(method, f"{self.api}{path}", headers=self._api_headers(), **kwargs)
        if response.status_code in (401, 403) and path != "/login":
            raise KeyRejected(f"the API key was rejected: {_message(response)}")
        if response.status_code == 429:
            raise OpenSubtitlesError("too many requests; try again later")
        return response

    def check_key(self) -> None:
        """Raise KeyRejected unless the API key is accepted. A search is the cheapest request the service
        checks the key for (the /infos endpoints answer any key); searches cost no download quota."""
        self.search(languages=["en"], query="key check")

    def login(self) -> bool:
        """Log in once, if credentials are configured, for the larger download quota."""
        if self._login_tried or not (self.username and self.password):
            return self.token is not None
        self._login_tried = True
        response = self._api("POST", "/login", json={"username": self.username, "password": self.password})
        if response.status_code != 200:
            raise OpenSubtitlesError(f"login failed: {_message(response)}")
        data = response.json()
        self.token = data.get("token") or None
        self.user = data.get("user") if isinstance(data.get("user"), dict) else None
        base_url = data.get("base_url")
        if isinstance(base_url, str) and base_url:  # e.g. the VIP server
            self.api = f"https://{base_url.removeprefix('https://').rstrip('/')}/api/v1"
        return self.token is not None

    def search(
        self,
        *,
        languages: list[str],
        moviehash: str | None = None,
        query: str | None = None,
        year: int | None = None,
        season: int | None = None,
        episode: int | None = None,
        imdb_id: int | None = None,
    ) -> list[OnlineSubtitle]:
        params: dict[str, Any] = {
            "languages": ",".join(sorted({code.lower() for code in languages})),
            "moviehash": moviehash,
            "query": query,
            "year": year,
            "season_number": season,
            "episode_number": episode,
            "imdb_id": imdb_id,
        }
        # The API redirects requests whose parameters are not sorted and lower-case.
        ordered = [
            (key, str(value).lower()) for key, value in sorted(params.items()) if value not in (None, "")
        ]
        response = self._api("GET", "/subtitles", params=ordered)
        if response.status_code != 200:
            raise OpenSubtitlesError(f"search failed: HTTP {response.status_code}: {_message(response)}")
        return parse_results(response.json())

    def download(self, file_id: int) -> Download:
        """Download one subtitle file. Counts against the daily quota."""
        try:
            self.login()
        except OpenSubtitlesError as exc:
            log.warning("OpenSubtitles %s; downloading without logging in", exc)
        response = self._api("POST", "/download", json={"file_id": file_id})
        data: dict[str, Any] = {}
        try:
            parsed = response.json()
            data = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            pass
        if response.status_code == 406 or (response.status_code != 200 and data.get("remaining") == 0):
            raise QuotaExceeded(
                f"the download quota is used up: {_message(response)}", data.get("reset_time_utc")
            )
        if response.status_code != 200 or not data.get("link"):
            raise OpenSubtitlesError(f"download failed: HTTP {response.status_code}: {_message(response)}")
        content = bytearray()
        # The link points at the file itself; the API key is not sent there.
        with self._http.stream("GET", str(data["link"]), headers={"Accept": "*/*"}) as stream:
            if stream.status_code != 200:
                raise OpenSubtitlesError(f"download failed: HTTP {stream.status_code} for the file")
            for chunk in stream.iter_bytes():
                content += chunk
                if len(content) > MAX_DOWNLOAD_BYTES:
                    raise OpenSubtitlesError("the subtitle file is larger than 5 MB; refusing it")
        remaining = data.get("remaining")
        return Download(
            data=bytes(content),
            file_name=data.get("file_name") or None,
            remaining=remaining if isinstance(remaining, int) else None,
            reset_time_utc=data.get("reset_time_utc") or None,
        )
