# kozzle-tts-qwen

Korean TTS audio generator using [Qwen3-TTS](https://github.com/QwenLM/Qwen2.5-Omni) on Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio). Batch-generates pronunciation audio for Korean words stored in Supabase.

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4) — mlx-audio runs on Metal GPU
- **Python 3.12** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **ffmpeg** — required for audio conversion in clone mode
- Qwen3-TTS model files (see Model Setup below)

## Install

```bash
uv sync
```

## Model Setup

Download Qwen3-TTS model files and place them in `./models/`. You need at least one model variant:

| Mode | Pro (1.7B) | Lite (0.6B) |
|------|-----------|-------------|
| Custom Voice | `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` |
| Voice Cloning | `Qwen3-TTS-12Hz-1.7B-Base-8bit` | `Qwen3-TTS-12Hz-0.6B-Base-8bit` |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` |

The default mode (`custom`) uses the Pro CustomVoice model. Place the model directory under `./models/`:

```
models/
  Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
    config.json
    model.safetensors
    ...
```

You can also point to a custom model path with `--model-path`.

## Configuration

### Initial setup

Create a config file with your Supabase credentials:

```bash
kozzle-tts init-config
```

This creates `~/.config/kozzle-tts/config.json` with template values. Edit it to add your credentials:

```bash
vi ~/.config/kozzle-tts/config.json
```

The file should look like:

```json
{
  "supabase_url": "https://your-project.supabase.co",
  "supabase_service_role_key": "eyJ...",
  "output_dir": "./output"
}
```

You can also specify a custom config path with `--config`:

```bash
kozzle-tts generate --config ./my-config.json
```

## Usage

### Custom mode (preset speakers)

The default mode. Uses built-in speaker voices:

```bash
# Korean with Sohee (default)
kozzle-tts generate

# Specify a different speaker
kozzle-tts generate --speaker Chelsie

# With emotion instruction and speed
kozzle-tts generate --speaker Sohee --instruct "Excited and happy" --speed 1.3
```

### Clone mode (voice cloning)

Clone a voice from a reference audio file (5-10 seconds recommended):

```bash
# Basic cloning
kozzle-tts generate --mode clone --ref-audio ./voice.wav

# With transcript for better quality
kozzle-tts generate --mode clone --ref-audio ./voice.wav --ref-text "안녕하세요"
```

### Design mode (describe a voice)

Describe the desired voice characteristics:

```bash
kozzle-tts generate --mode design --instruct "deep male narrator with warm tone"
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | auto | Path to config file (default: `~/.config/kozzle-tts/config.json`) |
| `--mode` | `-m` | `custom` | TTS mode: `custom`, `clone`, or `design` |
| `--speaker` | `-s` | `Sohee` | Speaker voice name (custom mode) |
| `--speed` | | `1.0` | Speech speed multiplier (0.5-2.0, custom mode) |
| `--instruct` | `-i` | `Normal tone` | Emotion/tone instruction (custom & design modes) |
| `--model-path` | | auto | Path to local model directory |
| `--ref-audio` | `-a` | | Reference audio file (clone mode, required) |
| `--ref-text` | | | Transcript of reference audio (clone mode, improves quality) |
| `--output-dir` | `-o` | `./output` | Output directory |
| `--subset` | | | Max number of words to process |
| `--resume-from` | `-r` | | Resume from word with this id |

### Available Speakers

| Language | Speakers |
|----------|----------|
| Korean | **Sohee** (default) |
| English | Ryan, Aiden, Chelsie, Serena, Vivian |
| Chinese | Vivian, Serena, Uncle_Fu, Dylan, Eric |
| Japanese | Ono_Anna |

## Output

Audio files are saved as WAV (24kHz) in the output directory:

- `{public_id}_word.wav` — pronunciation of the Korean lemma
- `{public_id}_example.wav` — pronunciation of the example sentence

## Development

```bash
uv run kozzzle-tts --version
uv run ruff check src/
uv run mypy src/
uv run pytest
```