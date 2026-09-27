#!/bin/bash
# Claude Code on the web: install what the tests and linters need, so a session can run them at once
# (see CLAUDE.md). Local sessions are left alone.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# FFmpeg 5.1+: without ffmpeg and ffprobe on the PATH, the integration tests skip themselves.
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  sudo=""
  if [ "$(id -u)" -ne 0 ]; then sudo="sudo"; fi
  if ! { $sudo apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
      $sudo apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null; }; then
    echo "session-start: could not install FFmpeg; the tests that need it will be skipped" >&2
  fi
fi

# The package, editable, with pytest, ruff and mypy. The container is cached once this has run.
PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet --disable-pip-version-check --root-user-action=ignore \
  -e ".[dev]"
