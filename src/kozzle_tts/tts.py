"""Qwen3-TTS wrapper using mlx-audio."""

import gc
import logging
import shutil
import tempfile
import warnings
from pathlib import Path

from kozzle_tts.config import (
    DEFAULT_MODELS_DIR,
    LONG_REF_AUDIO_THRESHOLD_SECONDS,
    MODEL_CONFIGS,
    TTSConfig,
)

logger = logging.getLogger(__name__)

_MIN_DURATION_RATE = 0.065
_MIN_DURATION_FLOOR = 0.5
_SAMPLE_RATE = 24000

# Adaptive max_tokens heuristic. Qwen3-TTS speech tokens are roughly an
# order of magnitude denser per character than text tokens, so a generous
# multiplier per character is fine. The CLI value is the upper bound so the
# user can still cap a runaway sentence.
_MAX_TOKENS_PER_CHAR = 12
_MAX_TOKENS_FLOOR = 256


class TTSError(Exception):
    """TTS-related error."""

    pass


def _resolve_model_path(config: TTSConfig) -> str:
    """Resolve model path from config or default.

    If config.model_path is set, use it directly.
    Otherwise, look up the model folder in ./models/ based on mode.
    Handles both direct folder paths and HuggingFace cache snapshots.
    """
    if config.model_path is not None:
        return str(config.model_path)

    folder_name = MODEL_CONFIGS.get(config.mode, {}).get(f"{config.variant}_{config.quant}")
    if folder_name is None:
        raise TTSError(
            f"Unknown TTS mode/variant/quant: {config.mode!r}/{config.variant!r}/{config.quant!r}"
        )

    base_path = DEFAULT_MODELS_DIR / folder_name
    if not base_path.exists():
        raise TTSError(
            f"Model not found at {base_path}. "
            "Download the model and place it in the ./models/ directory."
        )

    snapshots_dir = base_path / "snapshots"
    if snapshots_dir.exists():
        subfolders = [f for f in snapshots_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]
        if subfolders:
            return str(subfolders[0])

    return str(base_path)


def _compute_max_tokens(text: str, ceiling: int) -> int:
    """Adaptive max_tokens for a given text.

    Avoids letting the GPU run a 4096-step decode for a 2-character word
    (which is one of the triggers of the Metal "Impacting Interactivity"
    abort). The CLI ``--max-tokens`` value is the upper bound.
    """
    return min(ceiling, max(_MAX_TOKENS_FLOOR, len(text) * _MAX_TOKENS_PER_CHAR))


def _silence_known_warnings() -> None:
    """Silence noisy startup warnings that are confirmed harmless.

    - The ``qwen3_tts`` AutoConfig "instantiate a model of type ''"
      warning fires every time mlx-audio loads the model. mlx-audio uses
      its own model class, not transformers'.
    - The ``fix_mistral_regex`` tokenizer warning is about a Mistral-style
      word-boundary regex inherited by the Qwen tokenizer JSON. Verified
      via round-trip on Korean inputs that tokenization is unaffected for
      our use case (see investigation notes in PR description).

    Both are suppressed by lowering transformers' log level and adding a
    warnings filter for the regex message.
    """
    try:
        import transformers  # type: ignore[import-not-found]

        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    warnings.filterwarnings(
        "ignore",
        message=r".*fix_mistral_regex.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*model of type `qwen3_tts`.*",
    )


class TTSModel:
    """Wrapper for Qwen3-TTS model via mlx-audio."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None
        # Path to a trimmed ref-audio temp file owned by this instance, if
        # the worker had to shorten an oversize ref clip at load time.
        self._trimmed_ref_path: Path | None = None

    def load(self, config: TTSConfig | None = None) -> None:
        """Load the Qwen3-TTS model.

        For clone mode, also probes the reference audio: warns if it's
        longer than ``LONG_REF_AUDIO_THRESHOLD_SECONDS`` and, if
        ``config.ref_audio_max_seconds`` is set, trims the clip to a
        worker-owned temp file. The trimmed path replaces ``config.ref_audio``
        in-place so subsequent generate calls use the shorter clip.
        """
        _silence_known_warnings()

        from mlx_audio.tts.utils import load_model as mlx_load_model

        if config is None:
            config = TTSConfig()

        self._cleanup_trimmed_ref()
        if config.mode == "clone" and config.ref_audio is not None:
            self._maybe_trim_ref_audio(config)

        self._model_path = _resolve_model_path(config)
        logger.info("Loading Qwen3-TTS model from %s", self._model_path)
        self._model = mlx_load_model(self._model_path)
        logger.info("Model loaded successfully")

    def close(self) -> None:
        """Release any worker-owned temp files."""
        self._cleanup_trimmed_ref()

    def _cleanup_trimmed_ref(self) -> None:
        if self._trimmed_ref_path is not None:
            try:
                self._trimmed_ref_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._trimmed_ref_path = None

    def _maybe_trim_ref_audio(self, config: TTSConfig) -> None:
        """Probe ref-audio duration; warn if long, trim if requested.

        Mutates ``config.ref_audio`` in-place to point at the trimmed file
        when a trim is performed. Bails quietly if soundfile can't read the
        file (let mlx-audio raise its own error later).
        """
        ref_path = config.ref_audio
        if ref_path is None:
            return
        try:
            from soundfile import info as sf_info
        except Exception:
            return

        try:
            probe = sf_info(str(ref_path))
        except Exception as exc:
            logger.warning("Could not probe reference audio %s: %s", ref_path, exc)
            return

        duration = float(probe.duration)
        sr = int(probe.samplerate)
        logger.info(
            "Reference audio: %s (%.2fs, %d Hz, %d ch)",
            ref_path,
            duration,
            sr,
            probe.channels,
        )

        cap = config.ref_audio_max_seconds
        if cap is not None and duration > cap:
            self._trim_ref_audio(config, ref_path, cap, sr)
            return

        if duration > LONG_REF_AUDIO_THRESHOLD_SECONDS:
            logger.warning(
                "Reference audio is %.1fs long. Long ref clips can trigger "
                "Metal command-buffer aborts (kIOGPUCommandBufferCallback"
                "ErrorImpactingInteractivity). Recommended: 5-10s. Pass "
                "--ref-audio-max-seconds to auto-trim.",
                duration,
            )

    def _trim_ref_audio(
        self, config: TTSConfig, src: Path, cap_seconds: float, sample_rate: int
    ) -> None:
        """Read ``src``, write the first ``cap_seconds`` to a temp WAV."""
        import soundfile as sf

        # Read only the leading window; soundfile streams from disk so this
        # is cheap even for big files.
        frames = int(cap_seconds * sample_rate)
        data, sr = sf.read(str(src), frames=frames, dtype="float32")
        # Write to a temp file we own; cleaned up on next load() / close().
        fd, tmp_str = tempfile.mkstemp(prefix="kozzle_tts_ref_", suffix=".wav")
        # mkstemp gave us an open fd; close it so soundfile can rewrite.
        import os

        os.close(fd)
        sf.write(tmp_str, data, sr)
        tmp = Path(tmp_str)
        self._trimmed_ref_path = tmp
        logger.info(
            "Trimmed reference audio to %.1fs -> %s", cap_seconds, tmp
        )
        config.ref_audio = tmp

    @property
    def model(self):
        """Get the loaded model, loading lazily if needed."""
        if self._model is None:
            self.load()
        return self._model

    def generate_audio(
        self,
        text: str,
        output_path: Path,
        config: TTSConfig,
    ) -> Path:
        """Generate audio for text and save to file.

        Args:
            text: Text to synthesize.
            output_path: Path to save the audio file.
            config: TTS configuration.

        Returns:
            Path to the saved audio file.
        """
        if self._model is None:
            self.load(config)

        with tempfile.TemporaryDirectory(prefix="kozzle_tts_") as temp_dir:
            self._generate_with_retry(text, config, temp_dir)

            source = Path(temp_dir) / "audio_000.wav"
            if not source.exists():
                raise TTSError(
                    f"Model did not produce audio output. Expected: {source}"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(output_path))

        logger.info("Saved audio: %s", output_path)
        gc.collect()
        try:
            import mlx.core as mx

            mx.synchronize()
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()
        return output_path

    def _generate_with_retry(self, text: str, config: TTSConfig, temp_dir: str) -> None:
        """Generate audio, retrying with lower speed if output seems truncated."""
        self._generate(text, config, temp_dir)

        source = Path(temp_dir) / "audio_000.wav"
        if not source.exists():
            return

        from soundfile import info

        info_result = info(str(source))
        duration = info_result.duration
        min_duration = max(len(text) * _MIN_DURATION_RATE, _MIN_DURATION_FLOOR)

        logger.info(
            "Generated audio: text=%r (%d chars), duration=%.2fs, min_expected=%.2fs",
            text[:50],
            len(text),
            duration,
            min_duration,
        )

        if duration < min_duration:
            logger.warning(
                "Audio may be truncated: got %.2fs for %d chars (expected >= %.2fs). "
                "Retrying with lower speed",
                duration,
                len(text),
                min_duration,
            )
            retry_config = config.model_copy(update={"speed": 0.8})
            self._generate(text, retry_config, temp_dir)

            if source.exists():
                retry_info = info(str(source))
                logger.info(
                    "Retry result: duration=%.2fs (previous=%.2fs, expected>=%.2fs)",
                    retry_info.duration,
                    duration,
                    min_duration,
                )

    def _generate(self, text: str, config: TTSConfig, temp_dir: str) -> None:
        """Call mlx-audio generate_audio with the given config."""
        from mlx_audio.tts.generate import generate_audio

        effective_max_tokens = _compute_max_tokens(text, config.max_tokens)
        if effective_max_tokens != config.max_tokens:
            logger.debug(
                "Adaptive max_tokens: text=%d chars -> max_tokens=%d (ceiling=%d)",
                len(text),
                effective_max_tokens,
                config.max_tokens,
            )

        kwargs = {
            "model": self.model,
            "text": text,
            "output_path": temp_dir,
            "lang_code": "ko",
            "verbose": False,
            "max_tokens": effective_max_tokens,
        }

        if config.mode == "custom":
            kwargs["voice"] = config.speaker
            kwargs["instruct"] = config.instruct
            kwargs["speed"] = config.speed
        elif config.mode == "clone":
            if config.ref_audio is None:
                raise TTSError("Clone mode requires --ref-audio")
            kwargs["ref_audio"] = str(config.ref_audio)
            kwargs["ref_text"] = config.ref_text or ""
        elif config.mode == "design":
            kwargs["instruct"] = config.instruct

        generate_audio(**kwargs)

    def generate_word_audio(
        self,
        lemma: str,
        public_id: str,
        output_dir: Path,
        config: TTSConfig,
    ) -> Path:
        """Generate audio for a Korean word.

        Args:
            lemma: The word to synthesize.
            public_id: UUID string for filename.
            output_dir: Directory to save the file.
            config: TTS configuration.

        Returns:
            Path to the saved audio file.
        """
        output_path = output_dir / f"{public_id}_word.wav"
        return self.generate_audio(lemma, output_path, config)

    def generate_example_audio(
        self,
        text: str,
        public_id: str,
        output_dir: Path,
        config: TTSConfig,
    ) -> Path:
        """Generate audio for an example sentence.

        Args:
            text: The sentence to synthesize.
            public_id: UUID string for filename.
            output_dir: Directory to save the file.
            config: TTS configuration.

        Returns:
            Path to the saved audio file.
        """
        output_path = output_dir / f"{public_id}_example.wav"
        return self.generate_audio(text, output_path, config)
