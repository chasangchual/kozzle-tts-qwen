"""Supabase database client and queries."""

import logging
from dataclasses import dataclass
from uuid import UUID

from supabase import Client, create_client

from kozzle_tts.config import SupabaseConfig

logger = logging.getLogger(__name__)


class KozzleTTSError(Exception):
    """Base exception for kozzle-tts."""

    pass


class DatabaseError(KozzleTTSError):
    """Database-related error."""

    pass


@dataclass
class KorWord:
    """Korean word model."""

    id: int
    public_id: UUID
    lemma: str
    created_at: str
    definition: str | None = None
    pos_id: int | None = None
    level: int | None = None
    pronunciation: str | None = None


@dataclass
class Example:
    """Example sentence model."""

    id: int
    public_id: UUID
    kor_word_id: int
    text: str
    created_at: str
    type: str | None = None
    source: str | None = None


class Database:
    """Supabase database client."""

    def __init__(self, config: SupabaseConfig):
        self._client: Client | None = None
        self._config = config

    @property
    def client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            self._client = create_client(self._config.url, self._config.service_role_key)
        return self._client

    def get_kor_words(
        self,
        subset: int | None = None,
        resume_from: int | None = None,
    ) -> list[KorWord]:
        """Fetch Korean words from database.

        Args:
            subset: Maximum number of words to fetch.
            resume_from: Start from words with id >= resume_from.

        Returns:
            List of KorWord objects.
        """
        query = self.client.table("kor_word").select("*").order("id", desc=False)

        if resume_from is not None:
            query = query.gte("id", resume_from)

        if subset is not None:
            query = query.limit(subset)

        response = query.execute()

        if response.data is None:
            return []

        return [
            KorWord(
                id=row["id"],
                public_id=UUID(row["public_id"]),
                lemma=row["lemma"],
                created_at=row["created_at"],
                definition=row.get("definition"),
                pos_id=row.get("pos_id"),
                level=row.get("level"),
                pronunciation=row.get("pronunciation"),
            )
            for row in response.data
        ]

    def get_kor_words_by_ids(self, ids: list[int]) -> list[KorWord]:
        """Fetch Korean words by a list of ids.

        Used by the ``retry-failed`` command. Logs a warning for any
        requested ids that didn't come back (deleted in DB since the
        original run).
        """
        if not ids:
            return []

        response = (
            self.client.table("kor_word")
            .select("*")
            .in_("id", ids)
            .order("id", desc=False)
            .execute()
        )

        rows = response.data or []
        words = [
            KorWord(
                id=row["id"],
                public_id=UUID(row["public_id"]),
                lemma=row["lemma"],
                created_at=row["created_at"],
                definition=row.get("definition"),
                pos_id=row.get("pos_id"),
                level=row.get("level"),
                pronunciation=row.get("pronunciation"),
            )
            for row in rows
        ]

        returned_ids = {w.id for w in words}
        missing = [i for i in ids if i not in returned_ids]
        if missing:
            logger.warning(
                "kor_word ids missing from DB (likely deleted): %s",
                missing,
            )
        return words

    def get_examples_by_ids(self, ids: list[int]) -> list[Example]:
        """Fetch examples by a list of ids.

        Used by the ``retry-failed`` command. Logs a warning for any
        requested ids that didn't come back.
        """
        if not ids:
            return []

        response = (
            self.client.table("example")
            .select("*")
            .in_("id", ids)
            .order("id", desc=False)
            .execute()
        )

        rows = response.data or []
        examples = [
            Example(
                id=row["id"],
                public_id=UUID(row["public_id"]),
                kor_word_id=row["kor_word_id"],
                text=row["text"],
                created_at=row["created_at"],
                type=row.get("type"),
                source=row.get("source"),
            )
            for row in rows
        ]

        returned_ids = {e.id for e in examples}
        missing = [i for i in ids if i not in returned_ids]
        if missing:
            logger.warning(
                "example ids missing from DB (likely deleted): %s",
                missing,
            )
        return examples

    def get_examples_for_word(self, kor_word_id: int) -> list[Example]:
        """Fetch example sentences for a Korean word.

        Args:
            kor_word_id: The kor_word.id to fetch examples for.

        Returns:
            List of Example objects.
        """
        response = self.client.table("example").select("*").eq("kor_word_id", kor_word_id).execute()

        if response.data is None:
            return []

        return [
            Example(
                id=row["id"],
                public_id=UUID(row["public_id"]),
                kor_word_id=row["kor_word_id"],
                text=row["text"],
                created_at=row["created_at"],
                type=row.get("type"),
                source=row.get("source"),
            )
            for row in response.data
        ]