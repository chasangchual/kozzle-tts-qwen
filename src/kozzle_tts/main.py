"""Core processing logic for kozzle-tts."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from kozzle_tts import __version__
from kozzle_tts.config import Settings, TTSConfig
from kozzle_tts.database import Database, Example, KorWord
from kozzle_tts.failure_log import FailureLog, _now_iso
from kozzle_tts.tts import TTSError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

console = Console()

# Per-item retry policy. ``_MAX_RETRIES_PER_ITEM`` is the number of retries
# AFTER the first attempt (so 1 = up to 2 total attempts). After
# ``_MAX_CONSECUTIVE_FAILURES`` distinct items fail in a row we abort the run
# to avoid burning through the queue when the GPU is genuinely broken.
_MAX_RETRIES_PER_ITEM = 1
_MAX_CONSECUTIVE_FAILURES = 3

# Generation timeout bounds in seconds. Computed adaptively from text length.
_TIMEOUT_FLOOR = 30
_TIMEOUT_CEILING = 300


def _compute_timeout(text: str) -> int:
    """Adaptive per-item generation timeout, clamped to [floor, ceiling]."""
    return max(_TIMEOUT_FLOOR, min(_TIMEOUT_CEILING, int(30 + len(text) * 0.5)))


def _output_already_present(path: Path) -> bool:
    """True iff ``path`` exists and is non-empty.

    Non-empty check guards against half-written files from a crashed run.
    """
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


class IsolatedWorker:
    """Manages a persistent worker subprocess for isolated TTS generation.

    Mirrors QwenVoice's XPC pattern: generation runs in a separate process
    so Metal GPU crashes don't kill the main loop. The worker auto-restarts
    on failure.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def start(self, config: TTSConfig) -> None:
        self._stop()
        # NOTE: stderr is intentionally inherited (not piped). The worker
        # redirects mlx-audio/tqdm/transformers stdout to its stderr, and we
        # want the user to see that progress live. Piping stderr without a
        # reader thread would also deadlock once the OS pipe buffer fills.
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "kozzle_tts.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._send({"cmd": "load", "config": config.model_dump(mode="json")})
        resp = self._recv()
        if resp is None or resp.get("status") != "ready":
            msg = resp.get("message", "unknown") if resp else "worker exited"
            raise TTSError(f"Worker failed to load model: {msg}")

    def generate(self, text: str, output_path: str, config: TTSConfig) -> Path:
        if self._proc is None or self._proc.poll() is not None:
            raise TTSError("Worker not running")

        self._send({
            "cmd": "generate",
            "text": text,
            "output_path": output_path,
            "config": config.model_dump(mode="json"),
        })

        resp = self._recv(timeout=_compute_timeout(text))
        if resp is None:
            raise TTSError("Worker crashed during generation")
        if resp.get("status") == "error":
            raise TTSError(resp.get("message", "generation failed"))
        return Path(resp["path"])

    def stop(self) -> None:
        self._stop()

    def _stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._send({"cmd": "shutdown"})
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
            # Always reap so multiprocessing semaphores held by the child
            # are cleaned up; otherwise we get
            #   resource_tracker: There appear to be 1 leaked semaphore
            #   objects to clean up at shutdown
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def _send(self, msg: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout: int | None = None) -> dict | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            # Read lines until we get a JSON object or the worker closes
            # stdout. Skips any stray non-JSON line that may have leaked
            # onto the protocol channel (defensive — the worker also
            # redirects library prints away from fd 1).
            while True:
                ready = sel.select(timeout=timeout)
                if not ready:
                    self._proc.kill()
                    self._stop()
                    return None
                line = self._proc.stdout.readline()
                if not line:
                    # EOF: worker exited. Reap it so semaphores get cleaned.
                    self._stop()
                    return None
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    # Not protocol output; surface it for debugging and
                    # keep waiting for a real response.
                    console.print(f"[dim]worker: {stripped}[/]")
                    continue
        finally:
            sel.close()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def _build_run_config(
    settings: Settings,
    tts_config: TTSConfig,
    skip_existing: bool,
) -> dict[str, Any]:
    """Serialize the effective run config for the failure log."""
    payload = tts_config.model_dump(mode="json")
    payload["output_dir"] = str(settings.output_dir)
    payload["skip_existing"] = skip_existing
    return payload


class Processor:
    """Main processor for generating TTS audio."""

    def __init__(
        self,
        settings: Settings,
        tts_config: TTSConfig,
        subset: int | None = None,
        resume_from: int | None = None,
        skip_existing: bool = True,
    ):
        self.settings = settings
        self.tts_config = tts_config
        self.subset = subset
        self.resume_from = resume_from
        self.skip_existing = skip_existing
        self.db = Database(settings.get_supabase_config())
        self.worker = IsolatedWorker()
        self.failure_log = FailureLog()
        self._n_generated = 0
        self._n_skipped = 0
        self._n_failed = 0
        self._consecutive_failures = 0
        self._aborted = False
        self._worker_started = False

    # ----- worker lifecycle -----

    def _ensure_worker_started(self) -> None:
        """Start the worker subprocess on first need.

        Lazy-start lets all-skipped runs finish without ever loading the
        model.
        """
        if self._worker_started and self.worker.is_alive():
            return
        if not self._worker_started:
            console.print("[bold blue]Loading TTS model (isolated worker)...[/]")
        self.worker.start(self.tts_config)
        self._worker_started = True

    def _restart_worker(self) -> None:
        console.print("[yellow]Restarting worker subprocess...[/]")
        self.worker.start(self.tts_config)
        self._worker_started = True

    # ----- public entry points -----

    def process(self) -> None:
        """Standard run: fetch words via subset/resume_from and process."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._print_effective_config()

        console.print("[bold blue]Fetching words from database...[/]")
        words = self.db.get_kor_words(subset=self.subset, resume_from=self.resume_from)
        console.print(f"[green]Found {len(words)} words to process[/]")

        if not words:
            console.print("[yellow]No words found to process[/]")
            return

        self._run_with_progress(
            description_total=len(words),
            iterate=lambda progress, task: self._iterate_words(words, progress, task),
        )
        self._finalize()

    def process_retry(
        self,
        words: list[KorWord],
        examples: list[Example],
    ) -> None:
        """Retry-only run: process the given words and examples in order."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._print_effective_config()

        total = len(words) + len(examples)
        console.print(
            f"[green]Retrying {len(words)} word(s) and {len(examples)} example(s)[/]"
        )
        if total == 0:
            return

        self._run_with_progress(
            description_total=total,
            iterate=lambda progress, task: self._iterate_retry(
                words, examples, progress, task
            ),
        )
        self._finalize()

    # ----- iteration -----

    def _run_with_progress(self, description_total: int, iterate) -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Processing...",
                total=description_total,
            )
            iterate(progress, main_task)

    def _iterate_words(
        self,
        words: list[KorWord],
        progress: Progress,
        main_task: TaskID,
    ) -> None:
        n = len(words)
        for i, word in enumerate(words, start=1):
            if self._aborted:
                break
            progress.update(
                main_task,
                description=f"[cyan]({i}/{n}) {word.lemma}",
            )
            self._process_word(word, progress)
            progress.update(main_task, advance=1)

    def _iterate_retry(
        self,
        words: list[KorWord],
        examples: list[Example],
        progress: Progress,
        main_task: TaskID,
    ) -> None:
        total = len(words) + len(examples)
        i = 0
        for word in words:
            if self._aborted:
                break
            i += 1
            progress.update(
                main_task,
                description=f"[cyan]({i}/{total}) word: {word.lemma}",
            )
            # Retry path processes the word in isolation: do NOT re-fetch
            # examples from DB. Examples are scheduled separately below.
            self._process_word(word, progress, process_examples=False)
            progress.update(main_task, advance=1)

        for example in examples:
            if self._aborted:
                break
            i += 1
            progress.update(
                main_task,
                description=f"[cyan]({i}/{total}) example: {example.text[:30]}",
            )
            self._process_example(example, progress)
            progress.update(main_task, advance=1)

    # ----- per-item processing -----

    def _process_word(
        self,
        word: KorWord,
        progress: Progress,
        process_examples: bool = True,
    ) -> None:
        progress.console.print(
            f"[yellow]Word:[/] {word.lemma} (id: {word.id})"
        )
        output_path = self.settings.output_dir / f"{word.public_id}_word.wav"
        ok = self._generate_one(
            kind="word",
            text=word.lemma,
            output_path=output_path,
            on_failure=lambda attempts, err: self.failure_log.add_word_failure(
                word, attempts, err
            ),
            on_failure_label=f"word {word.lemma!r} (id={word.id})",
            progress=progress,
        )

        if not ok or self._aborted or not process_examples:
            return

        examples = self.db.get_examples_for_word(word.id)
        if examples:
            progress.console.print(f"  [blue]Found {len(examples)} example(s)[/]")
            for example in examples:
                if self._aborted:
                    break
                self._process_example(example, progress)

    def _process_example(self, example: Example, progress: Progress) -> None:
        output_path = self.settings.output_dir / f"{example.public_id}_example.wav"
        self._generate_one(
            kind="example",
            text=example.text,
            output_path=output_path,
            on_failure=lambda attempts, err: self.failure_log.add_example_failure(
                example, attempts, err
            ),
            on_failure_label=f"example id={example.id}",
            progress=progress,
            indent="    ",
        )

    def _generate_one(
        self,
        kind: str,
        text: str,
        output_path: Path,
        on_failure,
        on_failure_label: str,
        progress: Progress,
        indent: str = "  ",
    ) -> bool:
        """Unified generate-with-skip-and-retry helper.

        Returns ``True`` on success or skip, ``False`` on failure.
        Updates counters and the failure log. Triggers run abort when the
        consecutive-failure circuit breaker trips.
        """
        if self.skip_existing and _output_already_present(output_path):
            progress.console.print(
                f"{indent}[dim]\u2298 Skipped {kind} (already exists): "
                f"{output_path.name}[/]"
            )
            self._n_skipped += 1
            return True

        last_error = ""
        attempts = 0
        for attempt in range(_MAX_RETRIES_PER_ITEM + 1):
            attempts = attempt + 1
            try:
                self._ensure_worker_started()
                self.worker.generate(text, str(output_path), self.tts_config)
                progress.console.print(f"{indent}[green]\u2713[/] Generated {kind}")
                self._n_generated += 1
                self._consecutive_failures = 0
                return True
            except TTSError as e:
                last_error = str(e)
                progress.console.print(
                    f"{indent}[red]\u2717[/] Failed {on_failure_label} "
                    f"(attempt {attempts}/{_MAX_RETRIES_PER_ITEM + 1}): {e}"
                )
                # On any TTSError, the worker is likely dead. Restart for
                # the next attempt OR for the next item.
                try:
                    self._restart_worker()
                except TTSError as restart_err:
                    progress.console.print(
                        f"{indent}[red]\u2717[/] Worker restart failed: "
                        f"{restart_err}"
                    )
                    last_error = f"{last_error}; restart failed: {restart_err}"
                    break

        on_failure(attempts, last_error)
        self._n_failed += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            progress.console.print(
                f"[bold red]Aborting run after "
                f"{_MAX_CONSECUTIVE_FAILURES} consecutive failures[/]"
            )
            self._aborted = True
        return False

    # ----- finalization & reporting -----

    def _print_effective_config(self) -> None:
        cfg = self.tts_config
        console.print("[bold]Effective config[/]")
        console.print(f"  mode: {cfg.mode}")
        console.print(f"  variant/quant: {cfg.variant}/{cfg.quant}")
        if cfg.mode == "custom":
            console.print(f"  speaker: {cfg.speaker}  speed: {cfg.speed}")
        if cfg.mode == "clone":
            console.print(f"  ref_audio: {cfg.ref_audio}")
            if cfg.ref_audio_max_seconds is not None:
                console.print(
                    f"  ref_audio_max_seconds: {cfg.ref_audio_max_seconds}"
                )
        if cfg.mode in ("custom", "design"):
            console.print(f"  instruct: {cfg.instruct!r}")
        console.print(f"  max_tokens (ceiling): {cfg.max_tokens}")
        console.print(f"  output_dir: {self.settings.output_dir}")
        console.print(f"  skip_existing: {self.skip_existing}")

    def _finalize(self) -> None:
        try:
            self.worker.stop()
        except Exception:
            pass

        console.print(
            f"\n[bold]Done.[/] "
            f"Generated: [green]{self._n_generated}[/]  "
            f"Skipped: [dim]{self._n_skipped}[/]  "
            f"Failed: [red]{self._n_failed}[/]"
        )

        if not self.failure_log.is_empty:
            run_config = _build_run_config(
                self.settings, self.tts_config, self.skip_existing
            )
            path = self.failure_log.write(
                self.settings.output_dir,
                run_config=run_config,
                kozzle_tts_version=__version__,
                run_id=_now_iso(),
            )
            console.print(
                f"[yellow]Failure log written:[/] {path}\n"
                f"  Retry with: kozzle-tts retry-failed "
                f"{self.settings.output_dir / 'failed_latest.json'}"
            )

        if self._aborted:
            raise TTSError(
                f"Run aborted after {_MAX_CONSECUTIVE_FAILURES} consecutive failures"
            )


def run(
    config_path: Path | None = None,
    mode: str = "custom",
    variant: str = "pro",
    quant: str = "8bit",
    max_tokens: int = 4096,
    speaker: str = "Sohee",
    speed: float = 1.0,
    instruct: str = "Normal tone",
    model_path: Path | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    ref_audio_max_seconds: float | None = None,
    output_dir: Path | None = None,
    subset: int | None = None,
    resume_from: int | None = None,
    skip_existing: bool = True,
) -> None:
    settings = Settings.from_config(config_path)
    if output_dir:
        settings.output_dir = output_dir

    tts_config = TTSConfig(
        mode=mode,
        variant=variant,
        quant=quant,
        max_tokens=max_tokens,
        speaker=speaker,
        speed=speed,
        instruct=instruct,
        model_path=model_path,
        ref_audio=ref_audio,
        ref_text=ref_text,
        ref_audio_max_seconds=ref_audio_max_seconds,
    )

    processor = Processor(
        settings=settings,
        tts_config=tts_config,
        subset=subset,
        resume_from=resume_from,
        skip_existing=skip_existing,
    )

    processor.process()


def run_retry(
    failed_log_path: Path,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    skip_existing: bool = False,
) -> None:
    """Re-process only the failures recorded in a prior run.

    Re-queries Supabase by id so the retry sees fresh content. Cached
    text/lemma fields in the failure log are advisory only.
    """
    overrides = overrides or {}

    log, stored_config = FailureLog.load(failed_log_path)
    if log.is_empty:
        console.print(
            f"[yellow]No failures recorded in {failed_log_path}; nothing to do.[/]"
        )
        return

    # Build TTSConfig by merging stored config with caller overrides.
    tts_config, override_notices = TTSConfig.merge_overrides(
        stored_config, overrides
    )
    if override_notices:
        console.print("[bold]Overrides applied:[/]")
        for n in override_notices:
            console.print(f"  {n}")

    settings = Settings.from_config(config_path)
    # output_dir resolution: explicit CLI override > stored config > Settings default.
    if output_dir is not None:
        if str(settings.output_dir) != str(output_dir):
            console.print(
                f"  output_dir = {output_dir} (was {settings.output_dir})"
            )
        settings.output_dir = output_dir
    else:
        stored_output_dir = stored_config.get("output_dir")
        if stored_output_dir:
            settings.output_dir = Path(stored_output_dir)

    # Validate ref_audio still exists for clone mode (common gotcha after
    # moving files between machines).
    if tts_config.mode == "clone" and tts_config.ref_audio is not None:
        if not tts_config.ref_audio.exists():
            raise FileNotFoundError(
                f"Reference audio missing: {tts_config.ref_audio}. "
                "Pass --ref-audio to override."
            )

    processor = Processor(
        settings=settings,
        tts_config=tts_config,
        skip_existing=skip_existing,
    )

    word_ids = log.word_ids()
    example_ids = log.example_ids()

    console.print(
        f"[bold blue]Fetching {len(word_ids)} word(s) and "
        f"{len(example_ids)} example(s) from database...[/]"
    )
    words = processor.db.get_kor_words_by_ids(word_ids)
    examples = processor.db.get_examples_by_ids(example_ids)

    n_missing_words = len(word_ids) - len(words)
    n_missing_examples = len(example_ids) - len(examples)
    if n_missing_words or n_missing_examples:
        console.print(
            f"[yellow]Skipped (missing in DB): "
            f"{n_missing_words} word(s), {n_missing_examples} example(s)[/]"
        )

    processor.process_retry(words=words, examples=examples)
