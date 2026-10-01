# CLAUDE.md

A guide to this repository for coding agents and people. Read it first, then the parts of
[docs/DESIGN.md](docs/DESIGN.md) that your change touches: the design explains why things are as they
are, and its appendices hold the measurements behind the defaults.

## What this is

`vbr` (video-beep-remover) mutes a configurable list of words in video files. For each file it finds
the words with speech recognition, guided by subtitles; mutes each one with a short fade; re-encodes
only the audio; and verifies every muted span before it keeps the output. Two optional model layers
read the dialogue in context (DESIGN.md §17) and say a milder word in the speaker's voice instead of
muting (§16).

**Status.** Milestones M0 to M8 of DESIGN.md §13 are implemented. The next step is to evaluate the
models on a GPU (Appendices C to E hold the CPU measurements to repeat). The open questions in §15 wait
for that evaluation. Nothing is published or versioned yet: don't push tags, and keep changes under
`[0.1.0]` in CHANGELOG.md. The tool is for personal use, which is what F5-TTS's non-commercial weights
allow.

## Commands

```console
pip install -e ".[dev]"            # and FFmpeg 5.1+ (ffmpeg and ffprobe) on the PATH
pytest -q                          # about 430 tests in 1–1.5 min; FFmpeg tests skip themselves without it
pytest -q tests/unit               # about 340 tests in seconds, no FFmpeg needed
VBR_RUN_ASR_TESTS=1 pytest -m asr  # real Whisper on synthesized speech (downloads a model, needs espeak-ng)
VBR_RUN_MODEL_TESTS=1 pytest -m models  # the real context and voice models (the extras; a GPU helps)
ruff check . && ruff format --check . && python -m mypy
```

Use `python -m mypy`: a mypy installed elsewhere lacks the pydantic plugin. It checks for the Python it
runs on, and CI's lint job runs the oldest supported one, 3.11. CI
(`.github/workflows/ci.yml`) runs the same checks on Linux (Python 3.11 to 3.13), macOS and Windows,
and builds the wheel. In Claude Code on the web, `.claude/hooks/session-start.sh` installs all of this.

On a GPU machine: `pip install -e ".[dev,gpu,align,context,voice,sync]"`. On Linux, PyTorch from PyPI
includes CUDA; on Windows, install PyTorch's CUDA build from pytorch.org first. `vbr doctor` checks
every part.

## Where things are

```
src/video_beep_remover/
  cli/          the `vbr` commands (Typer): videos.py (clean, scan, render), subs.py, doctor.py,
                configure.py (vbr config), cache.py; options.py holds the shared options and the
                configuration keys that flags set, console.py the output
  batch.py      files and folders: skips vbr's own outputs and backups, renders in the background
  pipeline.py   one file, stage by stage (its docstring lists the stages); strategy fallbacks; the report
  model_pool.py the large models (Whisper by role, context, voice): loaded on first use, one at a time
  outputs.py    where files go: output.path, --backup and --in-place, report, EDL, review subtitles
  guided.py     the subtitle-guided strategies, targeted and hybrid
  config/       schema.py (pydantic), loader.py, defaults.toml (the same bytes as docs/vbr.example.toml)
  media/        FFmpeg runner, probe, audio decoding, the renderer and its verification (render.py)
  subtitles/    finding, parsing, syncing, censoring and saving subtitles; OpenSubtitles
  asr/          transcribers (faster-whisper, WhisperX), speech detection, the transcript cache
  detect/       word list, matching, window planning, confirmation, muted spans
  context/      context analysis (§17): lines, rules, classifier, judge, verdicts
  voice/        voice replacement (§16): the sentence, separation, the voice model, the check
  report/       JSON report, EDL, review subtitles; reading a report back
  models.py     dataclasses shared by the stages: Word, Detection, Cue, CensorInterval…
tests/unit/          no FFmpeg; tests/integration/: real FFmpeg (marker `ffmpeg`); tests/helpers.py
scripts/             evaluate.py, evaluate_context.py, make_synthetic_set.py; data/ has labelled lines
docs/DESIGN.md       §3 CLI, §4 config, §5 architecture, §6 stages, §13 plan, §15 decisions, appendices
```

## How to

- **Add a flag that sets configuration.** Add the option to `cli/options.py` and its key to
  `overrides()`, then the parameter to each command that takes it. Add the key to `config/schema.py`,
  `config/defaults.toml` and `docs/vbr.example.toml`; the last two must stay identical, and a test
  checks it. Document the flag in README.md ("Use") and DESIGN.md §3.3.
- **Test with media.** `helpers.make_clip` builds a clip in which a 440 Hz tone stands in for speech;
  `Track(surround=True)` makes a 5.1 track. `FakeTranscriber` returns scripted words. `decode` and
  `tone_gain` measure what was muted. `Pipeline` takes stand-ins for every model:
  `transcriber_factory`, `speech_detector`, `context_models` and `voice_models`. `RecordingUI` keeps
  the lines a run prints. The context and voice steps (`context.analyse_file`, `voice.replace_words`)
  can also be tested on their own, without FFmpeg.
- **Change the report.** `pipeline.py` puts it together. Its main sections are typed and built in
  `report/__init__.py`, so mypy checks them; the strategies and the context layer add their own.
  `report/__init__.py` also reads a report back for `vbr render`. Adding a field is compatible. Bump
  `SCHEMA_VERSION` only if a field changes meaning or goes away, since old reports are still read.
- **Change where files go.** Go through `outputs.py`: `place()` decides for every command, and
  `batch.py` uses it to skip outputs and backups.

## Conventions

- Python 3.11+, ruff with 110-character lines, mypy strict on `src/`.
- A docstring says what a function is for and why; comments are rare and explain what the code
  can't. Match the wording and density of the code around your change.
- Times are seconds on the media timeline (§6.2).
- Errors are `VbrError` subclasses (`errors.py`), and each maps to an exit code: `UsageError` for the
  user's mistakes, `ConfigError` for configuration, `DependencyError` for missing tools or models.
- The optional extras (torch, transformers, whisperx, f5_tts, demucs, speechbrain) are imported inside
  the functions that use them, never at module level. CI has none of them.
- Nothing half-written survives: files are written under a temporary name and renamed.
- Docs use short, plain sentences, and British spelling with -ize: analyse, behaviour, licence (the
  noun), recognize. README.md is for users and DESIGN.md for the reasons and the measurements. Update
  them, and CHANGELOG.md, with any change a user would notice.

## Traps

- CLI help is Rich markup: write a literal `[` as `\[`, or Rich swallows `[voice]`.
- The root logger's level leaks between tests, because a `-v` CLI test sets DEBUG. A test that depends
  on the level sets it itself (`caplog.set_level`).
- An MP4 keeps a custom tag such as `VBR_CENSORED` only with `-movflags use_metadata_tags`.
- Voice replacement subtracts the old voice sample-exactly. `read_pcm_windows` and the renderer both
  count samples from the stream's first decoded sample, relative to its `start_time`. Never seek there:
  Matroska timestamps are in milliseconds.
- AC3 tracks in Matroska start 256 samples before zero (AAC at 22.05 kHz, 46 ms) with older FFmpeg
  releases; FFmpeg 9 starts them at zero. Test with `audio_codec="ac3"` when timing matters, and accept
  both.
- Text read from a file keeps its `\r\n`. Write it back with `newline=""`, or Windows doubles each one.
- Tests must not depend on what is installed, since CI has none of the extras. `conftest.py` turns
  ffsubsync off, clears the API keys and gives each test its own temp dir; the extras' modules are
  replaced in `sys.modules`, as in `test_voice.py`. Only the opt-in model tests load the real models.
- On Windows, `asr/cuda.py`'s `load_pip_libraries()` must run before anything imports ctranslate2,
  faster_whisper's VAD included: the Pipeline calls it first. Otherwise ctranslate2 loads its own cuDNN,
  and a PyTorch model that then uses cuDNN aborts the process.
- The first run after a restart reads the models from a cold disk, so don't take its timings.
