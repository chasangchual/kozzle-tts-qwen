# AGENTS.md

## Project

- **Python 3.12 only** (`>=3.12,<3.13`) — pinned via `.python-version` and `pyproject.toml`.
- Managed with **uv** — run `uv sync` after adding dependencies.
- **macOS Apple Silicon required** — mlx-audio runs on Metal GPU.
- **ffmpeg** must be installed (needed for audio conversion in clone mode).
- Entry points: `kozzle-tts generate`, `kozzle-tts retry-failed`, `kozzle-tts organize-by-level`, `kozzle-tts init-config` (Typer app at `src/kozzle_tts/cli.py`).
- No tests, no CI. Pre-existing mypy errors in `database.py` (Supabase response types) and `tts.py` (mlx_audio / soundfile missing stubs) are noise — ignore them and don't try to "fix" without instruction.

```
uv run ruff check src/
uv run mypy src/        # database.py + tts.py errors are pre-existing
uv run pytest           # no tests yet
```

## Architecture

```
src/kozzle_tts/
├── cli.py          Typer CLI — generate, retry-failed, init-config commands
├── config.py       Settings, TTSConfig, MODEL_CONFIGS, SPEAKERS, merge_overrides
├── database.py     Supabase client — KorWord, Example, by-id fetch helpers
├── failure_log.py  Persistent failure log (output/failed_*.json)
├── main.py         Processor + IsolatedWorker; run() and run_retry()
├── tts.py          TTSModel wrapping mlx-audio → Qwen3-TTS
└── worker.py       Subprocess entry — runs the model in isolation
```

`Processor.process()` spawns `worker.py` via `python -m kozzle_tts.worker` and
talks to it with a JSON-per-line protocol over stdin/stdout. Generation runs
in the child so Metal GPU crashes don't kill the parent loop. The worker is
**lazy-started** — `Processor` only spawns it on the first non-skipped item,
so all-skipped runs never load the model.

## Worker subprocess protocol — read this before touching `worker.py` or `main.py`

- Parent ↔ child use **stdout for JSON only**. Anything else on the child's
  stdout breaks the protocol with `Expecting value: line 1 column 1 (char 0)`.
- mlx-audio, transformers, tqdm, and HuggingFace caches print to stdout.
  `worker.py` defends against this at module load time by `os.dup`'ing the
  original stdout to a private fd (`_PROTOCOL_OUT`) and pointing fd 1 at
  fd 2. Heavy libraries are only imported inside `main()`, **after** that
  redirect runs. Do not import mlx-audio / transformers / kozzle_tts.tts at
  the top of `worker.py`.
- The parent inherits the child's stderr (`stderr=None` in `Popen`) so the
  user sees model-loading progress live. Do not switch this to
  `subprocess.PIPE` without adding a reader thread — it deadlocks once the
  pipe buffer fills.
- `IsolatedWorker._recv` skips non-JSON lines defensively (printed with a
  `worker:` prefix) and calls `_stop()` on EOF/timeout to reap the child.
  Don't make it strict.
- `IsolatedWorker._stop` always `wait()`s after `kill()` to silence the
  `resource_tracker: leaked semaphore` warning.

## TTS Models

- Models live in `./models/` (git-ignored). Default: `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.
- Three modes: `custom` (preset speakers), `clone` (voice from ref audio), `design` (describe voice).
- Each mode maps to a different model folder via `MODEL_CONFIGS` in `config.py` (pro/lite × 8bit/4bit).
- `_resolve_model_path()` in `tts.py` accepts both a direct folder and a HuggingFace `snapshots/` cache layout.

## Failure recovery — `retry-failed` workflow

- Every `generate` run that produces at least one failure writes
  `output/failed_{ISO_ts}.json` plus a stable copy `output/failed_latest.json`.
  Schema and reader/writer live in `failure_log.py`. Bump `SCHEMA_VERSION`
  if you change the shape; `FailureLog.load` warns on mismatch but tries to
  read anyway.
- `kozzle-tts retry-failed <path>` re-queries Supabase by id (cached
  lemma/text fields in the file are advisory — fresh DB content wins).
  Stored run config is restored as defaults; CLI flags override per-field
  with a printed notice. Items already-deleted in DB are silently dropped
  with a warning.
- Default `--skip-existing` differs by command:
  - `generate` defaults to **on** (incremental).
  - `retry-failed` defaults to **off** (the whole point is to redo listed items).
- Examples whose parent word succeeded retry independently — `process_retry`
  does not regenerate the word. `_process_word(process_examples=False)` is
  used on the retry path.

## Level filtering and `output/{level}/` layout

- `generate --level N` filters `kor_word.level == N` (Supabase `eq`) and
  redirects all generated files to `{output_dir}/{level}/` instead of the
  flat `{output_dir}/`. Examples inherit the parent word's level
  implicitly because the run only processes that level's words.
- The `level` value is recorded in the failure log's stored run config so
  `retry-failed` lands files in the same subdir without re-passing the
  flag. `Processor` accepts `level` so the stored value is restored.
- `organize-by-level` is a one-shot migration for files generated before
  level support existed. It scans only the *top level* of `output_dir`
  (files already inside a subdir are left alone), parses
  `{public_id}_word.wav` / `{public_id}_example.wav`, looks up each
  file's level in Supabase (examples resolve via `kor_word_id`), and
  moves the file. Files whose record is missing or whose `level` column
  is NULL go to `output/unknown/` (override with `--unknown-dir-name`).
  Always run with `--dry-run` first.
- Database lookups for migration use `kor_word.public_id` /
  `example.public_id` (UUIDs as strings); see
  `Database.get_kor_words_by_public_ids` /
  `Database.get_examples_by_public_ids` /
  `Database.get_kor_word_levels_by_ids`. Requests are chunked
  (200/500) to keep the URL length bounded.

## Reliability tuning (tunable constants)

- `main.py`: `_MAX_RETRIES_PER_ITEM = 1`, `_MAX_CONSECUTIVE_FAILURES = 3`.
  After 3 distinct items fail in a row the run aborts (writes failure log
  first) to avoid burning the queue on a genuinely broken GPU. The
  consecutive-failure counter is shared across GPU failures and Supabase
  fetch failures — if either subsystem is genuinely broken we stop.
- `main.py`: `_compute_timeout(text)` clamps each generate call to
  `[30, 300]` seconds, scaled by text length.
- `tts.py`: `_compute_max_tokens(text, ceiling)` makes the CLI
  `--max-tokens` value the upper bound, scaling down for short text. This
  shrinks the GPU command buffer for single words and helps avoid the
  Metal `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` abort.
- `database.py`: `_DB_READ_TIMEOUT_S = 60.0`, `_DB_MAX_RETRIES = 3`,
  backoff `1s / 3s / 9s` (`_DB_BACKOFF_BASE_S = 1.0`,
  `_DB_BACKOFF_FACTOR = 3.0`). The Supabase client is built with an
  explicit `httpx.Client` whose `HTTPTransport(retries=3)` handles
  connection-level retries; `_with_retry()` adds read-timeout retries
  on top (httpx does NOT retry once the request has been sent). On
  exhaustion the wrapper raises `DatabaseError`. Postgrest's stock 120 s
  read timeout was the original cause of `Unexpected error: The read
  operation timed out` — a single stalled round trip would stall the
  whole queue for two minutes before failing.

## Known noisy warnings (silenced in `tts._silence_known_warnings()`)

- `"You are using a model of type qwen3_tts to instantiate a model of type ''"` —
  transformers AutoConfig warning. mlx-audio uses its own model class; harmless.
- `"incorrect regex pattern ... fix_mistral_regex=True"` — the Qwen tokenizer
  JSON inherits a Mistral-style word-boundary regex. Verified via
  round-trip on Korean inputs (`체개`, `안녕하세요`, sentences with
  punctuation) that tokenization is unaffected for Korean.

Both are silenced via `transformers.logging.set_verbosity_error()` plus
`warnings.filterwarnings(...)`. The call lives in `TTSModel.load()` so it
runs inside the worker subprocess before `mlx_load_model()`.

## Operational gotchas

- **Long `--ref-audio` (clone mode) crashes Metal.** Reference clips longer
  than ~15s inflate the prompt and trigger
  `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. The worker
  warns above `LONG_REF_AUDIO_THRESHOLD_SECONDS = 15.0` (in `config.py`).
  Pass `--ref-audio-max-seconds N` to auto-trim at load time, or pre-trim
  with `ffmpeg`. Recommended: 5–10s.
- **Runs are incremental by default.** `--skip-existing` is on for
  `generate`. Output filenames are content-addressed by `public_id`; moving
  or renaming output files invalidates the skip check. To force
  regeneration without touching files, pass `--no-skip-existing`.
- **`max_tokens` is a ceiling, not a target.** Effective value is adaptive
  per text length (see `tts._compute_max_tokens`).
- **Mid-run Supabase timeouts no longer abort the run.** A `DatabaseError`
  on `get_examples_for_word` is recorded against the parent word in the
  failure log and the loop continues. Re-run `generate` (incremental by
  default) or `retry-failed failed_latest.json` to pick up the missed
  examples — the retry path re-fetches examples from the DB fresh. Three
  consecutive DB failures still trip the shared circuit breaker and abort
  the run cleanly with a failure log written.
- **`process()` / `process_retry()` always write the failure log.** Both
  wrap their body in `try/finally: self._finalize()`, and `_finalize()`
  is idempotent (guarded by `self._finalized`). Even if an unexpected
  exception escapes the iteration loop, the in-flight failure log is
  flushed to disk before the exception propagates to the CLI.

## Key conventions

- Supabase credentials come from `~/.config/kozzle-tts/config.json` (not env vars). Run `kozzle-tts init-config` to create a template.
- `speed` kwarg is **only** passed in `custom` mode — `clone` and `design` modes ignore it (mlx-audio rejects it).
- `mlx_audio.tts.generate.generate_audio()` writes to a temp dir as `audio_000.wav`; `tts.py` moves it to the final path.
- Sample rate is always **24000 Hz**.
- Truncation retry: if generated duration < `max(len(text) * 0.065, 0.5)` seconds, retry once with `speed=0.8`. (This is the in-`tts.py` retry, distinct from the per-item Worker-crash retry in `main.py`.)
- Korean default speaker: **Sohee**.
- Output filenames: `{public_id}_word.wav` and `{public_id}_example.wav`.
