"""Configuration management for kozzle-tts."""

import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings

SPEAKERS = {
    "English": ["Ryan", "Aiden", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}

DEFAULT_MODELS_DIR = Path("./models")
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "kozzle-tts"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"

# In clone mode, ref-audio longer than this triggers a warning. Long ref
# clips inflate the prompt and can cause Metal command-buffer aborts
# (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). 5-10s is the
# documented sweet spot; we warn beyond 15s.
LONG_REF_AUDIO_THRESHOLD_SECONDS = 15.0

MODEL_CONFIGS = {
    "custom": {
        "pro_8bit": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "pro_4bit": "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        "lite_8bit": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "lite_4bit": "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
    },
    "clone": {
        "pro_8bit": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "pro_4bit": "Qwen3-TTS-12Hz-1.7B-Base-4bit",
        "lite_8bit": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "lite_4bit": "Qwen3-TTS-12Hz-0.6B-Base-4bit",
    },
    "design": {
        "pro_8bit": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "pro_4bit": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        "lite_8bit": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
        "lite_4bit": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-4bit",
    },
}

_CONFIG_TEMPLATE = {
    "supabase_url": "https://your-project.supabase.co",
    "supabase_service_role_key": "your-service-role-key-here",
    "output_dir": "./output",
}


class SupabaseConfig(BaseModel):
    """Supabase connection configuration."""

    url: str
    service_role_key: str


class Settings(BaseSettings):
    """Application settings."""

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    output_dir: Path = Path("./output")

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "Settings":
        """Load settings from application configuration file.

        Args:
            config_path: Explicit path to config file. Defaults to
                ~/.config/kozzle-tts/config.json.
        """
        path = config_path or DEFAULT_CONFIG_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Config not found at {path}. "
                "Run 'kozzle-tts init-config' to create one."
            )

        with open(path) as f:
            config = json.load(f)

        supabase_url = config.get("supabase_url")
        supabase_key = config.get("supabase_service_role_key")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase credentials not found in config. "
                "Ensure supabase_url and supabase_service_role_key are set."
            )

        output_dir = config.get("output_dir", "./output")

        return cls(
            supabase_url=supabase_url,
            supabase_service_role_key=supabase_key,
            output_dir=Path(output_dir),
        )

    def get_supabase_config(self) -> SupabaseConfig:
        """Get Supabase configuration."""
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("Supabase credentials not configured")
        return SupabaseConfig(
            url=self.supabase_url,
            service_role_key=self.supabase_service_role_key,
        )


def create_default_config(config_path: Path | None = None) -> Path:
    """Create a template configuration file.

    Args:
        config_path: Explicit path for the config file. Defaults to
            ~/.config/kozzle-tts/config.json.

    Returns:
        Path to the created config file.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(_CONFIG_TEMPLATE, f, indent=2)
        f.write("\n")

    return path


class TTSConfig(BaseModel):
    """TTS generation configuration for Qwen3-TTS."""

    mode: str = "custom"
    variant: str = "pro"
    quant: str = "8bit"
    max_tokens: int = 4096
    speaker: str = "Sohee"
    speed: float = 1.0
    instruct: str = "Normal tone"
    model_path: Path | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None
    # Optional cap for clone-mode reference audio. If set and the ref clip is
    # longer than this, the worker trims to this many seconds at load time
    # and uses the trimmed copy for generation.
    ref_audio_max_seconds: float | None = None

    @classmethod
    def merge_overrides(
        cls,
        stored: dict,
        overrides: dict,
    ) -> tuple["TTSConfig", list[str]]:
        """Merge a stored run-config dict with caller overrides.

        Used by the ``retry-failed`` command. Only override keys whose value
        is not ``None`` actually replace the stored value. Returns the merged
        ``TTSConfig`` plus a list of human-readable notices describing every
        field that was overridden, so the CLI can print them.
        """
        notices: list[str] = []
        merged: dict = dict(stored)
        # Restrict to fields the model actually understands so a stored
        # dict from a future schema doesn't crash older code.
        allowed = set(cls.model_fields.keys())
        for key, new_value in overrides.items():
            if new_value is None or key not in allowed:
                continue
            old_value = stored.get(key)
            if old_value != new_value:
                notices.append(f"{key} = {new_value!r} (was {old_value!r})")
            merged[key] = new_value
        # Only pass keys the model recognises.
        merged = {k: v for k, v in merged.items() if k in allowed}
        return cls(**merged), notices