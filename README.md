# kozzle-tts-qwen

Korean TTS audio generator using [Qwen3-TTS](https://github.com/QwenLM/Qwen2.5-Omni) on Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio). Batch-generates pronunciation audio for Korean words stored in Supabase.

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4) — mlx-audio runs on Metal GPU
- **Python 3.12** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **ffmpeg** — required for audio conversion in clone mode
- **PyTorch** — required for clone mode (installed separately, see Install below)
- Qwen3-TTS model files (see Model Setup below)

## Install

### 1. Install ffmpeg

Required for audio conversion (clone mode in particular):

```bash
brew install ffmpeg
```

### 2. Sync Python dependencies

```bash
uv sync
```

### 3. Install PyTorch (required for clone mode)

`torch` is **not** declared in `pyproject.toml` but is required by mlx-audio's
clone mode — specifically the Whisper transcriber that runs when you don't
pass `--ref-text`. Without it, the worker crashes on first generation with
`PyTorch was not found`:

```bash
uv pip install torch torchaudio
```

Custom and design modes work without torch, but installing it is the safe
default.

### 4. Verify

```bash
uv run kozzle-tts --help
```

## Model Setup

Download Qwen3-TTS model files and place them in `./models/`. You need at least one model variant:

| Mode | Pro (1.7B) | Lite (0.6B) |
|------|-----------|-------------|
| Custom Voice | `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` |
| Voice Cloning | `Qwen3-TTS-12Hz-1.7B-Base-8bit` | `Qwen3-TTS-12Hz-0.6B-Base-8bit` |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` |

4-bit quantizations (`-4bit` suffix) are also supported via `--quant 4bit` and use less memory at the cost of some quality.

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

Clone a voice from a reference audio file. **Reference clips of 5–10 seconds are
strongly recommended.** Long clips can trigger a Metal command-buffer abort
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) on the GPU.

```bash
# Basic cloning
kozzle-tts generate --mode clone --ref-audio ./voice.wav

# With transcript for better quality
kozzle-tts generate --mode clone --ref-audio ./voice.wav --ref-text "안녕하세요"

# If your reference clip is long, auto-trim it at load time:
kozzle-tts generate --mode clone --ref-audio ./long-voice.wav --ref-audio-max-seconds 8
```

If the reference audio is longer than 15 seconds and you don't pass
`--ref-audio-max-seconds`, the worker prints a warning before loading the model.

### Design mode (describe a voice)

Describe the desired voice characteristics:

```bash
kozzle-tts generate --mode design --instruct "deep male narrator with warm tone"
```

## Incremental runs and recovery

### Skip already-generated outputs (default)

`generate` is incremental by default: items whose output WAV already exists in
`--output-dir` and is non-empty are skipped, the model is **not** loaded if
nothing needs generating, and you'll see a `⊘ Skipped` line per item plus a
final `Generated / Skipped / Failed` summary.

```bash
# Default behavior (skip-existing): cheap to re-run
kozzle-tts generate --subset 100

# Force regeneration of everything
kozzle-tts generate --subset 100 --no-skip-existing
```

Output filenames are content-addressed by the row's `public_id`, so moving or
renaming files in `output/` will cause the next run to regenerate them.

### Failure log and `retry-failed`

Any `generate` run that produces at least one failure writes:

- `output/failed_{ISO_ts}.json` — timestamped per-run log
- `output/failed_latest.json` — stable convenience copy

Each entry records `kind` (`word` or `example`), DB `id`, `public_id`,
the lemma/text (advisory), the number of attempts, and the error.

To re-run only the failed items from a prior run:

```bash
# Use the stable latest copy
kozzle-tts retry-failed output/failed_latest.json

# Or a specific timestamped log
kozzle-tts retry-failed output/failed_2026-05-05T14-23-01Z.json
```

`retry-failed`:

- restores the original run's config (mode, ref-audio, speaker, etc.) from the log;
- accepts the same flags as `generate` to override individual fields, with a printed notice (e.g. `ref_audio = new.wav (was old.wav)`);
- re-queries Supabase by `id` so the retry sees fresh content (cached lemma/text fields in the log are advisory only); items deleted from the DB since the original run are skipped with a warning;
- examples whose parent word succeeded retry independently — the word audio is not regenerated;
- defaults `--skip-existing` to **off** (the whole point is to redo the listed items); pass `--skip-existing` if you want it to behave like `generate`.

### Reliability behavior

- **Per-item retry**: each item gets one retry across a worker restart. Transient Metal GPU crashes recover automatically.
- **Circuit breaker**: a run aborts after 3 consecutive items fail in a row, writing the failure log first so you can resume with `retry-failed`.
- **Adaptive `max_tokens`**: the `--max-tokens` value is treated as a ceiling. The effective value is scaled down for short text to shrink the GPU command buffer (helps avoid Metal aborts on single-word generations).
- **Adaptive timeout**: each generate call is bounded by a per-text-length timeout in `[30s, 300s]`.

### Options

#### `generate`

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | auto | Path to config file (default: `~/.config/kozzle-tts/config.json`) |
| `--mode` | `-m` | `custom` | TTS mode: `custom`, `clone`, or `design` |
| `--variant` | | `pro` | `pro` (1.7B) or `lite` (0.6B) |
| `--quant` | | `8bit` | `8bit` or `4bit` |
| `--max-tokens` | | `4096` | Generation token ceiling (effective value is adaptive) |
| `--speaker` | `-s` | `Sohee` | Speaker voice name (custom mode) |
| `--speed` | | `1.0` | Speech speed multiplier (0.5–2.0, custom mode) |
| `--instruct` | `-i` | `Normal tone` | Emotion/tone instruction (custom & design modes) |
| `--model-path` | | auto | Path to local model directory |
| `--ref-audio` | `-a` | | Reference audio file (clone mode, required) |
| `--ref-text` | | | Transcript of reference audio (clone mode, improves quality) |
| `--ref-audio-max-seconds` | | | Trim reference audio to N seconds at load time (clone mode) |
| `--output-dir` | `-o` | `./output` | Output directory |
| `--subset` | | | Max number of words to process |
| `--resume-from` | `-r` | | Resume from word with this id |
| `--skip-existing` / `--no-skip-existing` | | `--skip-existing` | Skip items whose output already exists |

#### `retry-failed`

```
kozzle-tts retry-failed FAILED_LOG [OPTIONS]
```

`FAILED_LOG` is a positional path to a `failed_*.json` file. All `generate`
flags are accepted as overrides; defaults come from the failed log's stored
config. The default for `--skip-existing` is **off**.

### Available Speakers

| Language | Speakers |
|----------|----------|
| Korean | **Sohee** (default) |
| English | Ryan, Aiden, Chelsie, Serena, Vivian |
| Chinese | Vivian, Serena, Uncle_Fu, Dylan, Eric |
| Japanese | Ono_Anna |

## Output

Audio files are saved as WAV (24 kHz) in the output directory:

- `{public_id}_word.wav` — pronunciation of the Korean lemma
- `{public_id}_example.wav` — pronunciation of the example sentence

If a run produces failures, you'll also see:

- `{output_dir}/failed_{ISO_ts}.json`
- `{output_dir}/failed_latest.json`

## Troubleshooting

### `[METAL] Command buffer execution failed: Impacting Interactivity`

This is the macOS GPU killing a Metal command buffer that ran too long. Common causes:

- **Long reference audio in clone mode.** Trim with `--ref-audio-max-seconds 8` (or pre-trim with `ffmpeg`).
- **Memory pressure.** Try `--quant 4bit` or `--variant lite`.
- The worker is isolated in a subprocess, so a crash kills only that item; the parent restarts the worker and continues.

### `Expecting value: line 1 column 1 (char 0)` from the worker

If you see this error, something is writing non-JSON to the worker subprocess's
stdout (the protocol channel). The worker already redirects mlx-audio /
transformers / tqdm output to stderr; if you see this after modifying
`worker.py` or `main.py`, see the "Worker subprocess protocol" section in
[`AGENTS.md`](AGENTS.md).

### Repeated transformers warnings on every load

The `qwen3_tts` AutoConfig warning and the `fix_mistral_regex` tokenizer
warning are both confirmed harmless for this project (Korean tokenization
round-trips correctly) and are silenced inside the worker. If you see them
again, check `tts._silence_known_warnings()`.

## Development

```bash
uv run kozzle-tts --version
uv run ruff check src/
uv run mypy src/        # database.py + tts.py errors are pre-existing noise
uv run pytest           # no tests yet
```

See [`AGENTS.md`](AGENTS.md) for repo-specific conventions, the worker
subprocess protocol, and operational gotchas.
