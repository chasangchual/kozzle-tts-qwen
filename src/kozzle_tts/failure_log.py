"""Persistent failure log for kozzle-tts runs.

Each generate run that produces at least one failure writes a JSON file
``output/failed_{ISO_ts}.json`` plus a stable copy ``output/failed_latest.json``.
The ``retry-failed`` CLI command consumes one of these files to re-run only
the recorded failures.

The cached ``text``/``lemma`` fields are advisory — they're useful when a
human ``cat``s the file, but ``retry-failed`` re-queries Supabase by id to
pick up any DB edits made since the original run.

Schema (versioned via ``schema_version``):

    {
      "schema_version": 1,
      "run_id": "2026-05-05T14-23-01Z",
      "kozzle_tts_version": "0.1.0",
      "config": { ...effective TTSConfig + Settings...,
                  "skip_existing": bool, "output_dir": str },
      "failures": [
        {
          "kind": "word" | "example",
          "id": int,
          "public_id": str (uuid),
          "text": str,             # lemma for word, sentence for example
          "kor_word_id": int|null, # populated for examples
          "attempts": int,
          "error": str
        }, ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kozzle_tts.database import Example, KorWord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
LATEST_FILENAME = "failed_latest.json"


def _now_iso() -> str:
    """ISO-8601 UTC timestamp safe to use in a filename (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@dataclass
class FailureRecord:
    """A single failed item from a kozzle-tts run."""

    kind: str  # "word" or "example"
    id: int
    public_id: str
    text: str
    attempts: int
    error: str
    kor_word_id: int | None = None  # populated for examples

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureRecord":
        return cls(
            kind=str(data["kind"]),
            id=int(data["id"]),
            public_id=str(data["public_id"]),
            text=str(data.get("text", "")),
            attempts=int(data.get("attempts", 1)),
            error=str(data.get("error", "")),
            kor_word_id=(
                int(data["kor_word_id"])
                if data.get("kor_word_id") is not None
                else None
            ),
        )


@dataclass
class FailureLog:
    """In-memory accumulator for failed items + writer/loader."""

    records: list[FailureRecord] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def add_word_failure(self, word: KorWord, attempts: int, error: str) -> None:
        self.records.append(
            FailureRecord(
                kind="word",
                id=word.id,
                public_id=str(word.public_id),
                text=word.lemma,
                attempts=attempts,
                error=error,
            )
        )

    def add_example_failure(
        self, example: Example, attempts: int, error: str
    ) -> None:
        self.records.append(
            FailureRecord(
                kind="example",
                id=example.id,
                public_id=str(example.public_id),
                text=example.text,
                attempts=attempts,
                error=error,
                kor_word_id=example.kor_word_id,
            )
        )

    def word_ids(self) -> list[int]:
        return [r.id for r in self.records if r.kind == "word"]

    def example_ids(self) -> list[int]:
        return [r.id for r in self.records if r.kind == "example"]

    def write(
        self,
        output_dir: Path,
        run_config: dict[str, Any],
        kozzle_tts_version: str,
        run_id: str | None = None,
    ) -> Path:
        """Write the log to ``failed_{ts}.json`` plus ``failed_latest.json``.

        Returns the path of the timestamped file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_id or _now_iso()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "kozzle_tts_version": kozzle_tts_version,
            "config": run_config,
            "failures": [asdict(r) for r in self.records],
        }
        timestamped = output_dir / f"failed_{run_id}.json"
        with open(timestamped, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        # Stable convenience copy. We write the file directly (not a symlink)
        # so it survives moving / copying the output directory.
        latest = output_dir / LATEST_FILENAME
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info("Wrote failure log: %s (and %s)", timestamped, latest)
        return timestamped

    @classmethod
    def load(cls, path: Path) -> tuple["FailureLog", dict[str, Any]]:
        """Load a failure log file. Returns ``(log, stored_run_config)``."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected JSON object at top level")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            logger.warning(
                "%s: schema_version %r != expected %d; trying to load anyway",
                path,
                version,
                SCHEMA_VERSION,
            )
        records = [
            FailureRecord.from_dict(r)
            for r in payload.get("failures", [])
            if isinstance(r, dict)
        ]
        config = payload.get("config", {})
        if not isinstance(config, dict):
            config = {}
        return cls(records=records), config
