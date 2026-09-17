from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
DATA_DIR = RUNTIME / "data"
AUDIT_DIR = RUNTIME / "audit"
LOG_DIR = RUNTIME / "logs"
DEFAULT_CHANNEL_ID = 1549051139675132034


def load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def resolve_discord_config() -> tuple[str, int, str]:
    candidates = (
        ("FORWARD_TEST_DISCORD_TOKEN", os.getenv("FORWARD_TEST_DISCORD_TOKEN", "")),
        ("FIVELV2_DISCORD_TOKEN", os.getenv("FIVELV2_DISCORD_TOKEN", "")),
        ("DISCORD_BOT_TOKEN", os.getenv("DISCORD_BOT_TOKEN", "")),
    )
    source, token = next(((name, value.strip()) for name, value in candidates if value.strip()),
                         ("none", ""))
    channel = int(os.getenv("FORWARD_TEST_CHANNEL_ID", str(DEFAULT_CHANNEL_ID)) or 0)
    return token, channel, source


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


class AuditJSONL:
    def __init__(self, strategy: str, symbol: str = "MGC"):
        self.strategy = strategy
        self.symbol = symbol
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> None:
        now = time.time()
        et = datetime.fromtimestamp(now, timezone.utc).astimezone(ET)
        record = {
            "ts_epoch": round(now, 3),
            "ts_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "ts_et": et.isoformat(),
            "strategy": self.strategy,
            "symbol": self.symbol,
            "mode": "SIGNAL_ONLY",
            "kind": kind,
            **{k: jsonable(v) for k, v in fields.items()},
        }
        path = AUDIT_DIR / f"{self.strategy}_{self.symbol}_{et:%Y-%m-%d}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as exc:
            print(f"[{self.strategy}][AUDIT] {kind} write failed: {exc}")


DDL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_key TEXT NOT NULL UNIQUE,
    signal_ts TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target1 REAL,
    target2 REAL,
    contracts INTEGER NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state_json TEXT,
    exit_ts TEXT,
    exit_px REAL,
    exit_reason TEXT,
    pnl_points REAL,
    pnl_usd REAL,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE TABLE IF NOT EXISTS daily_state (
    day TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_ts REAL NOT NULL
);
"""


class SignalStore:
    def __init__(self, strategy: str):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = DATA_DIR / f"{strategy}.db"
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)

    def close(self) -> None:
        self.conn.close()

    def seen_keys(self) -> set[str]:
        return {str(r[0]) for r in self.conn.execute("SELECT signal_key FROM signals")}

    def next_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM signals").fetchone()
        return int(row[0])

    def insert(self, *, key: str, signal_ts: str, side: str, entry: float,
               stop: float, target1: Optional[float], target2: Optional[float],
               contracts: int, source: str, payload: dict, state: dict) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO signals(signal_key,signal_ts,side,entry,stop,target1,target2,"
            "contracts,source,payload_json,state_json,created_ts,updated_ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, signal_ts, side, entry, stop, target1, target2, contracts, source,
             json.dumps(jsonable(payload)), json.dumps(jsonable(state)), now, now),
        )
        return int(cur.lastrowid)

    def update_state(self, signal_id: int, state: dict) -> None:
        self.conn.execute(
            "UPDATE signals SET state_json=?,updated_ts=? WHERE id=?",
            (json.dumps(jsonable(state)), time.time(), signal_id),
        )

    def finalize(self, signal_id: int, *, exit_ts: str, exit_px: float,
                 reason: str, pnl_points: float, pnl_usd: float) -> None:
        self.conn.execute(
            "UPDATE signals SET status='closed',exit_ts=?,exit_px=?,exit_reason=?,"
            "pnl_points=?,pnl_usd=?,updated_ts=? WHERE id=?",
            (exit_ts, exit_px, reason, pnl_points, pnl_usd, time.time(), signal_id),
        )

    def open_rows(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM signals WHERE status='open' ORDER BY id")]

    def recent(self, limit: int = 10) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (int(limit),))]

    def summary(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) total,"
            "SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) open,"
            "SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) wins,"
            "SUM(CASE WHEN status='closed' AND COALESCE(pnl_usd,0)<=0 THEN 1 ELSE 0 END) losses,"
            "COALESCE(SUM(pnl_usd),0) pnl FROM signals").fetchone()
        return dict(row)

    def trades_on(self, day_iso: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE substr(signal_ts,1,10)=?", (day_iso,)
        ).fetchone()
        return int(row[0])

    def save_day(self, day_iso: str, state: dict) -> None:
        self.conn.execute(
            "INSERT INTO daily_state(day,state_json,updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(day) DO UPDATE SET state_json=excluded.state_json,"
            "updated_ts=excluded.updated_ts",
            (day_iso, json.dumps(jsonable(state)), time.time()),
        )

    def load_day(self, day_iso: str) -> dict:
        row = self.conn.execute(
            "SELECT state_json FROM daily_state WHERE day=?", (day_iso,)
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return {}


class LiveBarBuilder:
    """Aggregate ticks into closed UTC one-minute bars."""

    def __init__(self, on_close: Callable[[dict], None]):
        self.on_close = on_close
        self.minute: Optional[datetime] = None
        self._reset()

    def _reset(self) -> None:
        self.o = self.h = self.l = self.c = None
        self.volume = 0

    def on_tick(self, ts: Any, price: float, size: int) -> None:
        if price <= 0 or size <= 0:
            return
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        stamp = stamp.tz_convert("UTC").floor("min").to_pydatetime()
        if self.minute is None:
            self.minute = stamp
            self.o = self.h = self.l = price
        while stamp != self.minute:
            self._close()
            self.minute += timedelta(minutes=1)
            self._reset()
            self.o = self.h = self.l = price
        self.h = max(float(self.h), price)
        self.l = min(float(self.l), price)
        self.c = price
        self.volume += size

    def _close(self) -> None:
        if self.o is None or not self.volume:
            return
        self.on_close({
            "minute_utc": self.minute,
            "open": float(self.o), "high": float(self.h),
            "low": float(self.l), "close": float(self.c),
            "volume": int(self.volume),
        })

    def force_close(self) -> None:
        if self.minute is not None:
            self._close()
            self.minute = None
            self._reset()


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz=ET, name="timestamp"),
    )


def append_bar(frame: pd.DataFrame, bar: dict, *, keep_days: int = 21) -> pd.DataFrame:
    ts = pd.Timestamp(bar["minute_utc"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(ET)
    frame.loc[ts] = [bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.tail(keep_days * 1440).copy()


def in_maintenance(now_et: datetime) -> bool:
    weekday = now_et.weekday()
    minute = now_et.hour * 60 + now_et.minute
    return (
        17 * 60 <= minute < 18 * 60
        or (weekday == 4 and minute >= 17 * 60)
        or weekday == 5
        or (weekday == 6 and minute < 18 * 60)
    )
