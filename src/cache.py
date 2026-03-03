import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from src.agents.state import FileSummary, FileTypes

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(os.getenv("CACHE_DB_PATH", "cache/cache.db"))


class CacheManager:
    """Lightweight SQLite cache for scraped pages and file summaries."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS page_cache (
                url         TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                cached_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS file_summary_cache (
                file_url    TEXT PRIMARY KEY,
                file_name   TEXT NOT NULL,
                file_type   TEXT NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                error       TEXT,
                cached_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS cache_stats (
                key         TEXT PRIMARY KEY,
                value       INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # seed the four stat keys if they don't exist yet
        for key in ("page_hits", "page_misses", "file_hits", "file_misses"):
            conn.execute(
                "INSERT OR IGNORE INTO cache_stats (key, value) VALUES (?, 0)",
                (key,),
            )
        conn.commit()

    def get_page(self, url: str) -> Optional[str]:
        """Return cached page markdown for *url*, or ``None``."""
        row = self._get_conn().execute(
            "SELECT content FROM page_cache WHERE url = ?", (url,)
        ).fetchone()
        if row:
            self._record("page_hits")
            logger.info("Page cache HIT for %s", url)
            return row[0]
        self._record("page_misses")
        logger.debug("Page cache MISS for %s", url)
        return None

    def set_page(self, url: str, content: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO page_cache (url, content, cached_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (url, content),
        )
        conn.commit()
        logger.info("Cached page content for %s (%d chars)", url, len(content))

    def get_file_summary(self, file_url: str) -> Optional[FileSummary]:
        """Return a cached ``FileSummary`` or ``None``."""
        row = self._get_conn().execute(
            "SELECT file_name, file_type, summary, error "
            "FROM file_summary_cache WHERE file_url = ?",
            (file_url,),
        ).fetchone()
        if row:
            self._record("file_hits")
            logger.info("File cache HIT for %s", file_url)
            return FileSummary(
                file_name=row[0],
                file_type=FileTypes(row[1]),
                file_url=file_url,
                summary=row[2],
                error=row[3],
            )
        self._record("file_misses")
        return None

    def set_file_summary(self, summary: FileSummary) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO file_summary_cache "
            "(file_url, file_name, file_type, summary, error, cached_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                summary.file_url,
                summary.file_name,
                summary.file_type.value,
                summary.summary,
                summary.error,
            ),
        )
        conn.commit()

    def _record(self, key: str) -> None:
        """Atomically increment a stats counter in SQLite."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE cache_stats SET value = value + 1 WHERE key = ?",
            (key,),
        )
        conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Return hit/miss stats (from DB) + row counts."""
        conn = self._get_conn()
        stats: Dict[str, int] = {}
        for row in conn.execute("SELECT key, value FROM cache_stats").fetchall():
            stats[row[0]] = row[1]
        stats["cached_pages"] = conn.execute(
            "SELECT COUNT(*) FROM page_cache"
        ).fetchone()[0]
        stats["cached_files"] = conn.execute(
            "SELECT COUNT(*) FROM file_summary_cache"
        ).fetchone()[0]
        return stats

    def clear_all(self) -> None:
        """Delete every cached row and reset counters."""
        conn = self._get_conn()
        conn.executescript(
            """
            DELETE FROM page_cache;
            DELETE FROM file_summary_cache;
            UPDATE cache_stats SET value = 0;
            """
        )
        conn.commit()
        logger.info("All caches cleared")


_instance: Optional[CacheManager] = None
_instance_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """Return (or create) the global ``CacheManager`` singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CacheManager()
    return _instance
