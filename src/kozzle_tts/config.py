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

MODEL_CONFIGS = {
    "custom": {
        "pro": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "lite": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    },
    "clone": {
        "pro": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "lite": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
    },
    "design": {
        "pro": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "lite": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
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
    speaker: str = "Sohee"
    speed: float = 1.0
    instruct: str = "Normal tone"
    model_path: Path | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None