# AGENTS.md

## Project

- **Python 3.12** — pinned via `.python-version`; do not use a lower version.
- Managed with **uv** — run `uv sync` after adding dependencies.
- Entry point: `kozzle-tts generate` CLI (Typer app at `src/kozzle_tts/cli.py`).
- No tests or CI configured yet. Lint with `uv run ruff check src/`.

## Architecture

```
src/kozzle_tts/
├── cli.py        Typer CLI — —mode, —speaker, —ref-audio, etc.
├── config.py     Settings (Supabase from opencode.json), TTSConfig, MODEL_CONFIGS
├── database.py   Supabase client — KorWord, Example dataclasses
├── main.py       Processor — Rich progress loop tying DB + TTS
└── tts.py        TTSModel wrapping mlx-audio → Qwen3-TTS
```

## TTS Models

- Models live in `./models/` (git-ignored). Default: `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.
- Three modes: `custom` (preset speakers), `clone` (voice from ref audio), `design` (describe voice).
- Each mode maps to a different model folder via `MODEL_CONFIGS` in config.py.
- `_resolve_model_path()` handles both direct paths and HuggingFace `snapshots/` cache layout.

## Key Conventions

- Supabase credentials come from `~/.config/kozzle-tts/config.json` (not env vars). Run `kozzle-tts init-config` to create a template.
- mlx-audio `generate_audio()` writes to a temp dir as `audio_000.wav`; `tts.py` moves it to the final path.
- Sample rate is always **24000 Hz**.
- Truncation detection: if generated duration < `len(text) * 0.065` seconds, retries with `speed=0.8`.
- Korean default speaker: **Sohee**.