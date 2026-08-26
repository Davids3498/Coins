"""SQLite prediction log -- one row per /predict request. The raw material every drift signal
is computed from: with nothing observing production there is no distribution to compare against,
so this is the prerequisite for the whole monitoring block.

SQLite rather than JSONL for three reasons: the drift check's core operation is a QUERY
("predictions since T"), not a scan; sqlite3 is stdlib, so the slim serving image gains no
dependency (it has no pandas -- see coin_clf.image_meta on why that matters); and each row
commits atomically, so a crash mid-write cannot leave a half-row the way an interrupted append
can. The cost is single-writer locking, which WAL plus a bounded busy_timeout makes a non-issue
for one container writing sub-millisecond inserts.

THE CONTRACT THIS MODULE OWES ITS CALLER: log() never raises. Not for a missing directory, not
for a locked database, not for a disk that filled up. Monitoring is an observer of the serving
path, never a participant in it -- a prediction that would have succeeded must still succeed
when the log is broken. log() returns False instead, and app/main.py wraps the call in its own
guard on top of that, because either layer alone is a single point of failure.

read_records() is the other side, consumed by the drift check on the host. It DOES raise on a
missing database: that means the operator pointed at the wrong path, which is worth surfacing
loudly, not answering with a confident empty list.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TIMEOUT_S = 1.0  # bounded: a lock must never block a prediction indefinitely

# Executed on every connection rather than once behind a flag. It is two sqlite_master lookups,
# and it means a deleted database file heals itself on the next write instead of turning every
# subsequent insert into a silent failure.
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    predicted_label TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    width           INTEGER,
    height          INTEGER,
    mode            TEXT,
    latency_ms      REAL,
    source          TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""

_COLUMNS = (
    "ts", "model_version", "predicted_label", "confidence",
    "width", "height", "mode", "latency_ms", "source",
)
_INSERT = f"INSERT INTO predictions ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})"


def utc_now_iso() -> str:
    """Timestamps are ISO-8601 UTC TEXT so they sort lexicographically -- that is what makes
    `WHERE ts >= ?` a valid time window without a date type or a parse step.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PredictionRecord:
    """One logged prediction.

    width/height/mode come from coin_clf.image_meta, read from the image AS UPLOADED -- see
    image_metadata()'s docstring on why reading them post-convert would silently flatten the
    mode signal.

    source is the traffic tag: NULL for real traffic, set by the replay script so a normal
    replay and a deliberately skewed one stay distinguishable when two demo runs land minutes
    apart and a time window can no longer separate them.
    """

    ts: str
    model_version: str
    predicted_label: str
    confidence: float
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    latency_ms: float | None = None
    source: str | None = None


class PredictionLog:
    """A handle on the log file. Constructing one touches no disk.

    That is deliberate: it is built in the serving app's lifespan, and a monitoring object that
    raised on an unwritable directory would take down STARTUP -- monitoring breaking serving,
    the exact failure mode this module exists to avoid. The file and schema are created on the
    first write, inside the guard.

    A connection is opened per write rather than shared. At ~0.1ms that is noise next to
    inference, and it sidesteps sqlite3's cross-thread connection rules entirely (uvicorn
    dispatches across a threadpool).
    """

    def __init__(self, path: str | Path, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.path = Path(path)
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"PredictionLog(path={str(self.path)!r})"

    def _connect(self, create: bool = True) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        conn.executescript(SCHEMA)
        return conn

    def log(self, record: PredictionRecord) -> bool:
        """Write one row. Returns True on success, False on ANY failure. Never raises.

        The bare `except Exception` is the point of this method, not an oversight: the caller is
        a live prediction request, and there is no failure here worth converting into a 500.
        """
        try:
            with self._connect() as conn:
                conn.execute(_INSERT, (
                    record.ts,
                    record.model_version,
                    record.predicted_label,
                    record.confidence,
                    record.width,
                    record.height,
                    record.mode,
                    record.latency_ms,
                    record.source,
                ))
            return True
        except Exception:
            return False

    def read_records(self, since: str | None = None, limit: int | None = None) -> list[dict]:
        """Rows oldest-first, as plain dicts -- the drift check's input.

        since: ISO-8601 UTC string; keeps rows with ts >= since (lexicographic, see utc_now_iso).
        limit: keeps the MOST RECENT n rows, still returned oldest-first, so "the last 500
            predictions" is one call and not a reversal dance at every call site.

        Raises FileNotFoundError for a missing database rather than creating an empty one and
        reporting no traffic -- "you pointed at the wrong path" and "nothing has been served
        yet" are different answers and must not look alike.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"no prediction log at {self.path} -- has the serving app handled any requests, "
                "and is PREDICTION_LOG_PATH pointing at the same file?"
            )

        where, params = "", []
        if since is not None:
            where, params = " WHERE ts >= ?", [since]

        if limit is None:
            sql = f"SELECT * FROM predictions{where} ORDER BY id"
        else:
            # newest n by id, then flipped back to oldest-first for the caller
            sql = (f"SELECT * FROM (SELECT * FROM predictions{where} ORDER BY id DESC LIMIT ?)"
                   " ORDER BY id")
            params.append(limit)

        with self._connect(create=False) as conn:
            return [dict(row) for row in conn.execute(sql, params)]
