from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .migrations import apply_migrations


class Database:
    def __init__(self, path: str | Path = "research_registry/research_pipeline.sqlite3"):
        self.path = Path(path)

    def initialize(self) -> None:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as connection:
            apply_migrations(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def session(self):
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
