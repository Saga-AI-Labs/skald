"""Append-only result store: SQLite index over immutable per-run files.

Implements the OPEN-1 decision (plan §4.2): a SQLite database acts as the
query index and each ``put`` call additionally persists the validated batch
as one immutable JSON file under ``runs/``.  Query results are reconstructed
from those run files, so the JSON is the authoritative record and SQLite is
purely an index — the store stays inspectable without a server.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

from .schema import FILTERABLE, RECORD_FIELDS, normalize

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    record_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS record_index (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 TEXT NOT NULL,
    row_id                 INTEGER NOT NULL,
    model_checkpoint_sha256 TEXT NOT NULL,
    adapter                TEXT NOT NULL,
    suite                  TEXT NOT NULL,
    task                   TEXT NOT NULL,
    metric                 TEXT NOT NULL,
    value                  REAL NOT NULL,
    n                      INTEGER,
    ci_low                 REAL,
    ci_high                REAL,
    protocol               TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    host                   TEXT,
    script_sha256          TEXT,
    runtime_sha256         TEXT,
    seed                   TEXT,
    artifacts              TEXT NOT NULL DEFAULT '[]',
    UNIQUE (run_id, row_id)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_record_model   ON record_index (model_checkpoint_sha256);
CREATE INDEX IF NOT EXISTS idx_record_adapter ON record_index (adapter);
CREATE INDEX IF NOT EXISTS idx_record_suite   ON record_index (suite);
CREATE INDEX IF NOT EXISTS idx_record_task    ON record_index (task);
CREATE INDEX IF NOT EXISTS idx_record_protocol ON record_index (protocol);
CREATE INDEX IF NOT EXISTS idx_record_created ON record_index (created_at);
CREATE INDEX IF NOT EXISTS idx_record_runtime ON record_index (runtime_sha256);
"""

_INDEX_COLUMNS = ", ".join(RECORD_FIELDS)


class Store:
    """Append-only, server-inspectable unified result store."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.db_path = self.root / "index.sqlite3"
        self.runs_dir = self.root / "runs"

    def initialize(self) -> None:
        """Create the store layout and schema if they do not exist yet."""
        if not self.db_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_CREATE_TABLES)
            # Columns before indexes: old stores gain the column here so
            # the index creation below never references a missing column.
            self._migrate(conn)
            conn.executescript(_CREATE_INDEXES)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive index migrations for stores created before a column.

        The JSON run files are the authoritative records and need no
        migration; only the SQLite index gains columns. Each step is guarded
        by a presence check so re-running is a no-op.
        """
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(record_index)")
        }
        if "runtime_sha256" not in columns:
            conn.execute("ALTER TABLE record_index ADD COLUMN runtime_sha256 TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        return conn

    def put(self, records: list[dict]) -> list[dict]:
        """Persist one run of records and return the normalized copies.

        Each call creates one immutable JSON run file and appends one index
        row per record to SQLite.  Storage is append-only: nothing is
        overwritten or deleted.
        """
        normalized = [normalize(r) for r in records]
        self.initialize()

        run_id = f"RUN-{uuid.uuid4().hex}"
        run_file = self.runs_dir / f"{run_id}.json"
        run_payload = {
            "run_id": run_id,
            "created_at": normalized[0]["created_at"] if normalized else None,
            "records": normalized,
        }
        _atomic_write_json(run_file, run_payload)

        columns = _INDEX_COLUMNS
        placeholders = ", ".join(["?"] * len(RECORD_FIELDS))
        insert_sql = (
            f"INSERT INTO record_index (run_id, row_id, {columns}) "
            f"VALUES (?, ?, {placeholders})"
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, file_path, created_at, record_count)"
                    " VALUES (?, ?, ?, ?)",
                    (run_id, str(run_file.relative_to(self.root)), run_payload["created_at"],
                     len(normalized)),
                )
                for row_id, record in enumerate(normalized):
                    values = [record[field] for field in RECORD_FIELDS]
                    # artifacts is a list; the index stores its JSON encoding.
                    values[RECORD_FIELDS.index("artifacts")] = json.dumps(values[RECORD_FIELDS.index("artifacts")])
                    conn.execute(insert_sql, (run_id, row_id, *values))
                conn.commit()
        except Exception:
            run_file.unlink(missing_ok=True)
            raise

        return normalized

    def query(
        self,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return records matching equality *filters*, oldest run first.

        Filters select scalar record fields (at least model/checkpoint, suite,
        adapter, protocol).  Records are reconstructed from the immutable
        per-run files, so returned dicts match exactly what was stored.
        """
        self.initialize()

        where, params = [], []
        for field, value in (filters or {}).items():
            if field not in FILTERABLE:
                raise ValueError(
                    f"cannot filter on {field!r}; filterable fields: "
                    f"{sorted(FILTERABLE)}"
                )
            where.append(f"{field} = ?")
            params.append(value)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT run_id, row_id FROM record_index{where_sql} "
                "ORDER BY id",
                params,
            ).fetchall()

        loaded: dict[str, dict] = {}
        results: list[dict] = []
        for run_id, row_id in rows:
            if run_id not in loaded:
                loaded[run_id] = json.loads(
                    (self.runs_dir / f"{run_id}.json").read_text()
                )
            record = loaded[run_id]["records"][row_id]
            # Run files predate newer optional columns (no back-fill, per
            # the runtime-manifest spec §4): default them on read and
            # restore canonical order so records stay RECORD_FIELDS-exact.
            for field in RECORD_FIELDS:
                record.setdefault(field, None)
            results.append({field: record[field] for field in RECORD_FIELDS})
            if limit is not None and len(results) >= limit:
                break
        return results


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)