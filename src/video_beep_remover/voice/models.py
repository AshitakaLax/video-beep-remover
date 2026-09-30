"""The local models of voice replacement (DESIGN.md §16), all optional: pip install
"video-beep-remover[voice]" brings PyTorch, F5-TTS, Demucs and SpeechBrain, imported only when a word is
replaced.

- A separation model (Demucs) splits the dialogue from music and effects.
- A voice model (F5-TTS) regenerates a span of speech inside the dialogue with new words: the rest of
  the sentence conditions it, so the new word keeps the speaker's voice, pace and pitch.
- A speaker encoder (ECAPA-TDNN, from SpeechBrain) tells whether the new word sounds like the speaker.

F5-TTS's pretrained weights are licensed for non-commercial use (CC BY-NC 4.0): the feature is meant for
personal viewing copies."""

import contextlib
import io
import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from video_beep_remover.errors import DependencyError
from video_beep_remover.voice.splice import FloatArray

INSTALL = 'pip install "video-beep-remover[voice]"'
_MODULES = ("torch", "torchaudio", "f5_tts", "demucs", "speechbrain")
ENCODER = "speechbrain/spkrec-ecapa-voxceleb"


def check_installed() -> None:
    """Fail before a file is analysed, rather than at its first replaced word, when the extra is missing."""
    missing = [name for name in _MODULES if find_spec(name) is None]
    if missing:
        raise DependencyError(f"voice replacement needs {', '.join(missing)}: {INSTALL}")


class Separator(Protocol):
    name: str

    def vocals(self, audio: FloatArray, rate: int) -> FloatArray:
        """The dialogue in each channel of `audio` (channels × samples), at the same rate and length."""
        ...


class Editor(Protocol):
    name: str

    def edit(self, voice: FloatArray, rate: int, text: str, span: tuple[float, float]) -> FloatArray:
        """`voice` (mono) with `span` (seconds into it) spoken again so that the whole says `text`; the
        same rate and length."""
        ...


class SpeakerEncoder(Protocol):
    def embed(self, voice: FloatArray, rate: int) -> FloatArray:
        """A speaker embedding of `voice` (mono)."""
        ...


def _libraries() -> tuple[Any, Any]:
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise DependencyError(f"voice replacement needs PyTorch: {INSTALL}") from exc
    return torch, torchaudio


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Keep the models' progress prints, INFO logs and deprecation warnings off the console, unless the
    run is verbose (-v)."""
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        logging.disable(logging.INFO)  # SpeechBrain sets its own loggers to INFO
        try:
            yield
        finally:
            logging.disable(logging.NOTSET)


def _offline(offline: bool) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # models then load from the Hugging Face cache only


def _failure(what: str, name: str, offline: bool, exc: Exception) -> DependencyError:
    hint = " (offline mode: download it once without --offline)" if offline else ""
    return DependencyError(f"could not load the {what} {name!r}{hint}: {exc}")


def _resample(torchaudio: Any, audio: Any, source: int, target: int) -> Any:
    return audio if source == target else torchaudio.functional.resample(audio, source, target)


def _fit(samples: Any, length: int) -> Any:
    """`samples` (…, n) cut or zero-padded to `length`."""
    import torch

    if samples.shape[-1] >= length:
        return samples[..., :length]
    return torch.nn.functional.pad(samples, (0, length - samples.shape[-1]))


class DemucsSeparator:
    def __init__(self, name: str, *, device: str, offline: bool) -> None:
        self.torch, self.torchaudio = _libraries()
        _offline(offline)
        try:
            with _quiet():
                from demucs.apply import apply_model
                from demucs.pretrained import get_model

                self.model = get_model(name).to(device).eval()
        except Exception as exc:
            raise _failure("separation model", name, offline, exc) from exc
        self.apply: Callable[..., Any] = apply_model
        self.name, self.device = name, device
        self.stem = list(self.model.sources).index("vocals")

    def vocals(self, audio: FloatArray, rate: int) -> FloatArray:
        torch = self.torch
        mixture = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        stereo = mixture if mixture.shape[0] == 2 else mixture.mean(0, keepdim=True).repeat(2, 1)
        model_rate = self.model.samplerate
        with torch.inference_mode(), _quiet():
            separated = self.apply(
                self.model,
                _resample(self.torchaudio, stereo, rate, model_rate)[None],
                device=self.device,
                progress=False,
            )[0]
        vocals = _fit(_resample(self.torchaudio, separated[self.stem], model_rate, rate), audio.shape[1])
        if mixture.shape[0] != 2:
            vocals = vocals.mean(0, keepdim=True).repeat(mixture.shape[0], 1)
        result: FloatArray = vocals.float().cpu().numpy()
        return result


class F5Editor:
    """F5-TTS infilling: the span's mel frames are masked and generated again, the rest of the sentence
    conditioning them (as in F5-TTS's own speech_edit)."""

    RMS = 0.1  # F5-TTS works at this loudness

    def __init__(self, name: str, *, device: str, offline: bool, steps: int, seed: int = 7) -> None:
        self.torch, self.torchaudio = _libraries()
        _offline(offline)
        try:
            with _quiet():
                from f5_tts.api import F5TTS

                self.tts = F5TTS(model=name, device=device)
        except Exception as exc:
            raise _failure("voice model", name, offline, exc) from exc
        self.name, self.steps, self.seed = name, steps, seed
        self.model, self.vocoder = self.tts.ema_model, self.tts.vocoder
        self.rate = int(self.tts.target_sample_rate)
        self.hop = int(self.model.mel_spec.hop_length)

    def edit(self, voice: FloatArray, rate: int, text: str, span: tuple[float, float]) -> FloatArray:
        torch = self.torch
        # The model's sample() puts the text on the device of `cond`, and its mel spectrogram moves
        # itself to its input's device, so the audio must start on the model's device.
        device = next(self.model.parameters()).device
        audio = _resample(
            self.torchaudio, torch.from_numpy(np.ascontiguousarray(voice))[None], rate, self.rate
        ).to(device)
        level = float(audio.square().mean().sqrt())
        gain = self.RMS / level if 0 < level < self.RMS else 1.0
        with torch.inference_mode(), _quiet():
            mel = self.model.mel_spec(audio * gain).permute(0, 2, 1)  # (1, frames, mels)
            first, last = (round(t * self.rate / self.hop) for t in span)
            first, last = max(0, first), min(mel.shape[1], max(last, first + 1))
            cond = mel.clone()
            cond[:, first:last] = 0
            mask = torch.ones(1, mel.shape[1], dtype=torch.bool, device=device)
            mask[:, first:last] = False
            generated, _ = self.model.sample(
                cond=cond,
                text=[text],
                duration=mel.shape[1],
                steps=self.steps,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                seed=self.seed,
                edit_mask=mask,
            )
            wave = self.vocoder.decode(generated.to(torch.float32).permute(0, 2, 1)).cpu() / gain
        edited = _fit(_resample(self.torchaudio, wave, self.rate, rate)[0], voice.size)
        result: FloatArray = edited.float().numpy()
        return result


class EcapaEncoder:
    RATE = 16_000

    def __init__(self, cache_dir: Path, *, device: str, offline: bool) -> None:
        self.torch, self.torchaudio = _libraries()
        _offline(offline)
        try:
            with _quiet():
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                # Windows allows symlinks only in Developer Mode or as an administrator.
                strategy = LocalStrategy.COPY if sys.platform == "win32" else LocalStrategy.SYMLINK
                self.model = EncoderClassifier.from_hparams(
                    source=ENCODER,
                    savedir=str(cache_dir / "ecapa"),
                    run_opts={"device": device},
                    local_strategy=strategy,
                )
        except Exception as exc:
            raise _failure("speaker encoder", ENCODER, offline, exc) from exc

    def embed(self, voice: FloatArray, rate: int) -> FloatArray:
        torch = self.torch
        audio = _resample(
            self.torchaudio, torch.from_numpy(np.ascontiguousarray(voice))[None], rate, self.RATE
        )
        with torch.inference_mode(), _quiet():
            embedding = self.model.encode_batch(audio).reshape(-1)
        result: FloatArray = embedding.float().cpu().numpy()
        return result
