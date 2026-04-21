"""Qwen3-TTS wrapper using mlx-audio."""

import gc
import logging
import shutil
import tempfile
from pathlib import Path

from kozzle_tts.config import DEFAULT_MODELS_DIR, MODEL_CONFIGS, TTSConfig

logger = logging.getLogger(__name__)

_MIN_DURATION_RATE = 0.065
_MIN_DURATION_FLOOR = 0.5
_SAMPLE_RATE = 24000


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

    folder_name = MODEL_CONFIGS.get(config.mode, {}).get("pro")
    if folder_name is None:
        raise TTSError(f"Unknown TTS mode: {config.mode!r}")

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


class TTSModel:
    """Wrapper for Qwen3-TTS model via mlx-audio."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None

    def load(self, config: TTSConfig | None = None) -> None:
        """Load the Qwen3-TTS model.

        Args:
            config: TTS config providing model path. If None, uses default custom mode.
        """
        from mlx_audio.tts.utils import load_model as mlx_load_model

        if config is None:
            config = TTSConfig()

        self._model_path = _resolve_model_path(config)
        logger.info("Loading Qwen3-TTS model from %s", self._model_path)
        self._model = mlx_load_model(self._model_path)
        logger.info("Model loaded successfully")

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

        kwargs = {
            "model": self.model,
            "text": text,
            "output_path": temp_dir,
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