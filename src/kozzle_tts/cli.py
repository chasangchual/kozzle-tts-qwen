"""CLI entry point for kozzle-tts."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from kozzle_tts import __version__
from kozzle_tts.config import SPEAKERS, create_default_config
from kozzle_tts.main import run

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
) -> None:
    """Generate Korean TTS audio files."""
    if mode not in ("custom", "clone", "design"):
        console.print(f"[red]Error: Invalid mode '{mode}'. Must be 'custom', 'clone', or 'design'[/]")
        raise typer.Exit(1)

    if mode == "clone" and ref_audio is None:
        console.print("[red]Error: Clone mode requires --ref-audio[/]")
        raise typer.Exit(1)

    try:
        run(
            config_path=config_path,
            mode=mode,
            speaker=speaker,
            speed=speed,
            instruct=instruct,
            model_path=model_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_dir=output_dir,
            subset=subset,
            resume_from=resume_from,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()