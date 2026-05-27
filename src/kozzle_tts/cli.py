"""CLI entry point for kozzle-tts."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from kozzle_tts import __version__
from kozzle_tts.config import SPEAKERS, create_default_config
from kozzle_tts.database import DatabaseError
from kozzle_tts.main import organize_by_level, run, run_retry

app = typer.Typer(
    name="kozzle-tts",
    help="Korean TTS generator using Qwen3-TTS on Apple Silicon",
    add_completion=False,
)
console = Console()

ALL_SPEAKERS = [name for names in SPEAKERS.values() for name in names]


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"kozzle-tts version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Korean TTS generator using Qwen3-TTS."""
    pass


@app.command()
def init_config(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config-path",
        help="Custom path for the config file (default: ~/.config/kozzle-tts/config.json)",
    ),
) -> None:
    """Create a template configuration file with default values."""
    try:
        path = create_default_config(config_path)
        console.print(f"[green]Created config template at:[/] {path}")
        console.print("\nEdit the file and add your Supabase credentials:")
        console.print(f"  [bold]vi {path}[/]")
        console.print("\nRequired fields:")
        console.print("  [cyan]supabase_url[/]              - Your Supabase project URL")
        console.print("  [cyan]supabase_service_role_key[/] - Your Supabase service role key")
    except PermissionError:
        console.print("[red]Error: Permission denied. Check the config path.[/]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error creating config: {e}[/]")
        raise typer.Exit(1) from None


@app.command()
def generate(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: ~/.config/kozzle-tts/config.json)",
    ),
    mode: str = typer.Option(
        "custom",
        "--mode",
        "-m",
        help="TTS mode: custom (preset speakers), clone (voice from reference audio), or design (describe voice)",
    ),
    variant: str = typer.Option(
        "pro",
        "--variant",
        help="Model variant: pro (1.7B) or lite (0.6B). Use lite if Metal GPU times out.",
    ),
    quant: str = typer.Option(
        "8bit",
        "--quant",
        help="Quantization: 8bit (quality) or 4bit (speed, less memory). Use 4bit on 8GB Macs.",
    ),
    max_tokens: int = typer.Option(
        4096,
        "--max-tokens",
        help="Max generation tokens (ceiling). Effective value is adaptive per text length.",
    ),
    speaker: str = typer.Option(
        "Sohee",
        "--speaker",
        "-s",
        help=f"Speaker voice name. Available: {', '.join(ALL_SPEAKERS)}",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        help="Speech speed multiplier (for custom mode). E.g. 0.8=slow, 1.0=normal, 1.3=fast",
        min=0.5,
        max=2.0,
    ),
    instruct: str = typer.Option(
        "Normal tone",
        "--instruct",
        "-i",
        help="Emotion/tone instruction (for custom and design modes). E.g. 'Sad and crying', 'Excited and happy'",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        help="Path to local model directory (overrides default lookup in ./models/)",
    ),
    ref_audio: Optional[Path] = typer.Option(
        None,
        "--ref-audio",
        "-a",
        help="Path to reference audio for voice cloning (clone mode only)",
        exists=True,
        readable=True,
    ),
    ref_text: Optional[str] = typer.Option(
        None,
        "--ref-text",
        help="Transcript of reference audio (clone mode only). Improves cloning quality.",
    ),
    ref_audio_max_seconds: Optional[float] = typer.Option(
        None,
        "--ref-audio-max-seconds",
        help=(
            "If set, trim reference audio to this many seconds at load time "
            "(clone mode). Long ref clips can trigger Metal command-buffer aborts; "
            "5-10s is recommended."
        ),
        min=1.0,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for output files (default: ./output)",
    ),
    subset: Optional[int] = typer.Option(
        None,
        "--subset",
        help="Maximum number of words to process",
    ),
    resume_from: Optional[int] = typer.Option(
        None,
        "--resume-from",
        "-r",
        help="Resume from word with this id",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help=(
            "Skip items whose output WAV already exists (default: enabled). "
            "Use --no-skip-existing to force regeneration."
        ),
    ),
    level: Optional[int] = typer.Option(
        None,
        "--level",
        "-l",
        help=(
            "Only process kor_word rows with this exact level. When set, "
            "output files are written to {output_dir}/{level}/ instead of "
            "{output_dir}/ directly."
        ),
    ),
) -> None:
    """Generate Korean TTS audio files."""
    if mode not in ("custom", "clone", "design"):
        console.print(f"[red]Error: Invalid mode '{mode}'. Must be 'custom', 'clone', or 'design'[/]")
        raise typer.Exit(1)

    if variant not in ("pro", "lite"):
        console.print(f"[red]Error: Invalid variant '{variant}'. Must be 'pro' or 'lite'[/]")
        raise typer.Exit(1)

    if quant not in ("8bit", "4bit"):
        console.print(f"[red]Error: Invalid quant '{quant}'. Must be '8bit' or '4bit'[/]")
        raise typer.Exit(1)

    if mode == "clone" and ref_audio is None:
        console.print("[red]Error: Clone mode requires --ref-audio[/]")
        raise typer.Exit(1)

    if ref_audio_max_seconds is not None and mode != "clone":
        console.print(
            "[yellow]Warning: --ref-audio-max-seconds is only meaningful in clone mode; ignoring[/]"
        )
        ref_audio_max_seconds = None

    try:
        run(
            config_path=config_path,
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
            output_dir=output_dir,
            subset=subset,
            resume_from=resume_from,
            skip_existing=skip_existing,
            level=level,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except DatabaseError as e:
        # Distinct branch so the user sees a network-flavored message
        # instead of the raw httpx string ("The read operation timed
        # out") that confused us in the past. The Database layer already
        # retried; reaching here means the connection is genuinely
        # broken.
        console.print(f"[red]Database error:[/] {e}")
        console.print(
            "[yellow]Check network connectivity and Supabase status, then "
            "re-run. Generate is incremental by default so finished items "
            "will be skipped.[/]"
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        raise typer.Exit(1) from None


@app.command("retry-failed")
def retry_failed(
    failed_log: Path = typer.Argument(
        ...,
        help="Path to a failed_*.json log written by a previous run.",
        exists=True,
        readable=True,
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to Supabase config file (default: ~/.config/kozzle-tts/config.json)",
    ),
    # Override flags. All default to None so we can distinguish 'user did not
    # pass this' from 'user passed the default'. Anything left as None is
    # taken from the failed log's stored config.
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Override TTS mode"),
    variant: Optional[str] = typer.Option(None, "--variant", help="Override model variant"),
    quant: Optional[str] = typer.Option(None, "--quant", help="Override quantization"),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Override max_tokens ceiling"
    ),
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s", help="Override speaker"),
    speed: Optional[float] = typer.Option(
        None, "--speed", help="Override speech speed (custom mode)", min=0.5, max=2.0
    ),
    instruct: Optional[str] = typer.Option(
        None, "--instruct", "-i", help="Override emotion/tone instruction"
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", help="Override local model path"
    ),
    ref_audio: Optional[Path] = typer.Option(
        None,
        "--ref-audio",
        "-a",
        help="Override clone-mode reference audio",
        exists=True,
        readable=True,
    ),
    ref_text: Optional[str] = typer.Option(
        None, "--ref-text", help="Override clone-mode reference transcript"
    ),
    ref_audio_max_seconds: Optional[float] = typer.Option(
        None,
        "--ref-audio-max-seconds",
        help="Override ref-audio trim limit",
        min=1.0,
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Override output directory"
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing/--no-skip-existing",
        help=(
            "Skip items whose output already exists (default: disabled for retry-failed; "
            "the whole point of retry is to regenerate listed items)."
        ),
    ),
) -> None:
    """Retry only the items recorded in a previous failure log.

    The log's stored run config is restored as the default; any CLI flag
    you pass overrides that field with a printed notice. Items are
    re-queried from Supabase by id to pick up content edits since the
    original run (cached lemma/text fields in the log are advisory only).
    """
    overrides: dict = {
        "mode": mode,
        "variant": variant,
        "quant": quant,
        "max_tokens": max_tokens,
        "speaker": speaker,
        "speed": speed,
        "instruct": instruct,
        "model_path": model_path,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "ref_audio_max_seconds": ref_audio_max_seconds,
    }

    try:
        run_retry(
            failed_log_path=failed_log,
            config_path=config_path,
            overrides=overrides,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except DatabaseError as e:
        console.print(f"[red]Database error:[/] {e}")
        console.print(
            "[yellow]Check network connectivity and Supabase status, then "
            "re-run.[/]"
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        raise typer.Exit(1) from None


@app.command("organize-by-level")
def organize_by_level_cmd(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to Supabase config file (default: ~/.config/kozzle-tts/config.json)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory to scan (default: from config, usually ./output)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print intended moves without changing the filesystem.",
    ),
    unknown_dir_name: str = typer.Option(
        "unknown",
        "--unknown-dir-name",
        help=(
            "Subdir name for files whose level cannot be resolved "
            "(record missing in DB, or level column is NULL)."
        ),
    ),
) -> None:
    """Move existing top-level WAVs in the output dir into per-level subdirs.

    Use this once after pulling level support: it scans
    ``{output_dir}/*.wav``, looks up each file's kor_word level in
    Supabase, and moves the file to ``{output_dir}/{level}/``. Files
    already inside a subdirectory are not touched.
    """
    try:
        organize_by_level(
            config_path=config_path,
            output_dir=output_dir,
            dry_run=dry_run,
            unknown_dir_name=unknown_dir_name,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except DatabaseError as e:
        console.print(f"[red]Database error:[/] {e}")
        console.print(
            "[yellow]Check network connectivity and Supabase status, then "
            "re-run.[/]"
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
