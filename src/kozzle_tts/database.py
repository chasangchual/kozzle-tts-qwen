"""Supabase database client and queries."""

import logging
from dataclasses import dataclass
from uuid import UUID

from supabase import Client, create_client

from kozzle_tts.config import SupabaseConfig

logger = logging.getLogger(__name__)


# PostgREST (and therefore Supabase) caps a single response at 1000 rows by
# default. We page through results with .range(from, to) so callers always
# get the full set regardless of table size. Keep this <= the server's
# max-rows setting; 1000 is the safe default.
_PAGE_SIZE = 1000


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
        level: int | None = None,
    ) -> list[KorWord]:
        """Fetch Korean words from database.

        Pages through results with PostgREST's ``.range(from, to)`` because
        Supabase caps single responses at 1000 rows by default.

        Args:
            subset: Maximum number of words to fetch (across all pages).
            resume_from: Start from words with id >= resume_from.
            level: If set, only return words with this exact level.

        Returns:
            List of KorWord objects.
        """
        all_rows: list[dict] = []
        offset = 0

        while True:
            # Cap this page if a subset cap would be hit before the full
            # page size; saves a partial round trip.
            page_size = _PAGE_SIZE
            if subset is not None:
                remaining = subset - len(all_rows)
                if remaining <= 0:
                    break
                page_size = min(_PAGE_SIZE, remaining)

            query = self.client.table("kor_word").select("*").order("id", desc=False)

            if resume_from is not None:
                query = query.gte("id", resume_from)

            if level is not None:
                query = query.eq("level", level)

            query = query.range(offset, offset + page_size - 1)

            response = query.execute()
            page = response.data or []
            all_rows.extend(page)

            # Last page reached when the server returns fewer rows than asked.
            if len(page) < page_size:
                break

            offset += page_size

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
            for row in all_rows
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

    def get_kor_words_by_public_ids(
        self, public_ids: list[UUID]
    ) -> list[KorWord]:
        """Fetch Korean words by a list of public_ids (UUIDs).

        Used by the ``organize-by-level`` migration to look up the level
        of each already-generated file. Missing public_ids are silently
        dropped (the caller compares input vs output to detect them).
        """
        if not public_ids:
            return []

        # Supabase ``in_`` expects strings for UUID columns.
        str_ids = [str(p) for p in public_ids]
        # Batch in chunks to keep URL length bounded.
        out: list[KorWord] = []
        chunk = 200
        for i in range(0, len(str_ids), chunk):
            response = (
                self.client.table("kor_word")
                .select("*")
                .in_("public_id", str_ids[i : i + chunk])
                .execute()
            )
            for row in response.data or []:
                out.append(
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
                )
        return out

    def get_examples_by_public_ids(
        self, public_ids: list[UUID]
    ) -> list[Example]:
        """Fetch examples by a list of public_ids (UUIDs).

        Used by the ``organize-by-level`` migration. The returned
        ``kor_word_id`` is then looked up to find each example's level.
        """
        if not public_ids:
            return []

        str_ids = [str(p) for p in public_ids]
        out: list[Example] = []
        chunk = 200
        for i in range(0, len(str_ids), chunk):
            response = (
                self.client.table("example")
                .select("*")
                .in_("public_id", str_ids[i : i + chunk])
                .execute()
            )
            for row in response.data or []:
                out.append(
                    Example(
                        id=row["id"],
                        public_id=UUID(row["public_id"]),
                        kor_word_id=row["kor_word_id"],
                        text=row["text"],
                        created_at=row["created_at"],
                        type=row.get("type"),
                        source=row.get("source"),
                    )
                )
        return out

    def get_kor_word_levels_by_ids(
        self, ids: list[int]
    ) -> dict[int, int | None]:
        """Return a {kor_word.id: level} map for the given ids.

        Used to resolve the level of each example via its parent word.
        Missing ids are simply absent from the returned map.
        """
        if not ids:
            return {}

        out: dict[int, int | None] = {}
        chunk = 500
        unique = list({i for i in ids})
        for i in range(0, len(unique), chunk):
            response = (
                self.client.table("kor_word")
                .select("id,level")
                .in_("id", unique[i : i + chunk])
                .execute()
            )
            for row in response.data or []:
                out[row["id"]] = row.get("level")
        return out

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