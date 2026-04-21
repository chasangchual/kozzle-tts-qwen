"""Core processing logic for kozzle-tts."""

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from kozzle_tts.config import Settings, TTSConfig
from kozzle_tts.database import Database, Example, KorWord
from kozzle_tts.tts import TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

console = Console()


class Processor:
    """Main processor for generating TTS audio."""

    def __init__(
        self,
        settings: Settings,
        tts_config: TTSConfig,
        subset: int | None = None,
        resume_from: int | None = None,
    ):
        self.settings = settings
        self.tts_config = tts_config
        self.subset = subset
        self.resume_from = resume_from
        self.db = Database(settings.get_supabase_config())
        self.tts = TTSModel()

    def process(self) -> None:
        """Process all words and generate TTS audio."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]Loading TTS model...[/]")
        self.tts.load(self.tts_config)
        console.print("[green]Model loaded successfully[/]")

        console.print("[bold blue]Fetching words from database...[/]")
        words = self.db.get_kor_words(subset=self.subset, resume_from=self.resume_from)
        console.print(f"[green]Found {len(words)} words to process[/]")

        if not words:
            console.print("[yellow]No words found to process[/]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Processing words...",
                total=len(words),
            )

            for word in words:
                self._process_word(word, progress)
                progress.update(main_task, advance=1)

        console.print("[bold green]Processing complete![/]")

    def _process_word(self, word: KorWord, progress: Progress) -> None:
        """Process a single word and its examples."""
        word_desc = f"[yellow]Word:[/] {word.lemma} (id: {word.id})"
        progress.console.print(word_desc)

        try:
            self.tts.generate_word_audio(
                lemma=word.lemma,
                public_id=str(word.public_id),
                output_dir=self.settings.output_dir,
                config=self.tts_config,
            )
            progress.console.print("  [green]✓[/] Generated word audio")
        except Exception as e:
            progress.console.print(f"  [red]✗[/] Failed word: {e}")
            return

        examples = self.db.get_examples_for_word(word.id)
        if examples:
            progress.console.print(f"  [blue]Found {len(examples)} example(s)[/]")

            for example in examples:
                self._process_example(example, progress)

    def _process_example(self, example: Example, progress: Progress) -> None:
        """Process a single example sentence."""
        try:
            self.tts.generate_example_audio(
                text=example.text,
                public_id=str(example.public_id),
                output_dir=self.settings.output_dir,
                config=self.tts_config,
            )
            progress.console.print(f"    [green]✓[/] Generated example: {example.text[:30]}...")
        except Exception as e:
            progress.console.print(f"    [red]✗[/] Failed example: {e}")


def run(
    config_path: Path | None = None,
    mode: str = "custom",
    speaker: str = "Sohee",
    speed: float = 1.0,
    instruct: str = "Normal tone",
    model_path: Path | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    output_dir: Path | None = None,
    subset: int | None = None,
    resume_from: int | None = None,
) -> None:
    """Run the TTS generation process.

    Args:
        config_path: Path to config file (default: ~/.config/kozzle-tts/config.json).
        mode: TTS mode - "custom", "clone", or "design".
        speaker: Speaker voice name (for custom mode).
        speed: Speech speed multiplier (for custom mode).
        instruct: Emotion/tone instruction (for custom and design modes).
        model_path: Path to local model directory (overrides default lookup).
        ref_audio: Path to reference audio file (for clone mode).
        ref_text: Transcript of reference audio (for clone mode).
        output_dir: Directory for output files (default: ./output).
        subset: Maximum number of words to process.
        resume_from: Resume from word with this id.
    """
    settings = Settings.from_config(config_path)
    if output_dir:
        settings.output_dir = output_dir

    tts_config = TTSConfig(
        mode=mode,
        speaker=speaker,
        speed=speed,
        instruct=instruct,
        model_path=model_path,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    processor = Processor(
        settings=settings,
        tts_config=tts_config,
        subset=subset,
        resume_from=resume_from,
    )

    processor.process()