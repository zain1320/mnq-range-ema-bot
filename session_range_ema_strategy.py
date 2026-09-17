from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


MNQ_TICK_SIZE = 0.25


@dataclass(frozen=True)
class SessionSpec:
    name: str
    timezone: str
    window: tuple[int, int]
    range_window: tuple[int, int]
    enabled: bool = True


SESSIONS = (
    SessionSpec("TOK", "Asia/Tokyo", (9 * 60, 15 * 60 + 30), (10 * 60, 10 * 60 + 5)),
    SessionSpec("HK", "Asia/Hong_Kong", (9 * 60 + 30, 16 * 60), (10 * 60 + 30, 10 * 60 + 35)),
    SessionSpec("LON", "Europe/London", (8 * 60, 16 * 60 + 30), (9 * 60, 9 * 60 + 5)),
    SessionSpec("NY", "America/New_York", (9 * 60 + 30, 16 * 60), (10 * 60 + 30, 10 * 60 + 35)),
)


@dataclass(frozen=True)
class StrategyConfig:
    contracts: int = 10
    max_trades: int = 1
    ema_len: int = 20
    fast_ema_len: int = 15
    min_rr: float = 0.0
    flat_at_end: bool = True
    target_deviations: tuple[float, float, float] = (1.28, 2.01, 2.51)
    target_contracts: tuple[int, int, int] = (5, 3, 2)
    exchange_timezone: str = "America/Chicago"


@dataclass
class SessionState:
    in_window: bool = False
    window_date: Optional[str] = None
    last_bar: bool = False
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_ts: Optional[str] = None
    direction: int = 0
    break_streak_direction: int = 0
    break_streak_count: int = 0
    trades: int = 0
    break_ts: Optional[str] = None
    break_open: Optional[float] = None
    break_high: Optional[float] = None
    break_low: Optional[float] = None
    break_close: Optional[float] = None

    def reset(self) -> None:
        self.range_high = None
        self.range_low = None
        self.range_ts = None
        self.direction = 0
        self.break_streak_direction = 0
        self.break_streak_count = 0
        self.trades = 0
        self.break_ts = None
        self.break_open = None
        self.break_high = None
        self.break_low = None
        self.break_close = None

    @classmethod
    def restore(cls, raw: dict) -> "SessionState":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: raw[key] for key in allowed if key in raw})


@dataclass
class OrderCandidate:
    session: str
    side: str
    direction: int
    entry: float
    stop: float
    targets: tuple[float, float, float]
    risk_points: float
    reward_points: float
    rr: float
    armed_at: pd.Timestamp
    range_high: float
    range_low: float
    range_ts: str
    break_ts: str
    break_open: float
    break_high: float
    break_low: float
    break_close: float
    ema20: float
    ema15: float
    vwap: float
    deviation: float

    def state(self) -> dict:
        raw = asdict(self)
        raw["armed_at"] = self.armed_at.isoformat()
        return raw

    @classmethod
    def restore(cls, raw: dict) -> "OrderCandidate":
        values = dict(raw)
        values["armed_at"] = pd.Timestamp(values["armed_at"])
        values["targets"] = tuple(values["targets"])
        return cls(**values)


def _local_minute(ts: pd.Timestamp, timezone: str) -> int:
    local = pd.Timestamp(ts).tz_convert(timezone)
    return local.hour * 60 + local.minute


def _inside(minute: int, window: tuple[int, int]) -> bool:
    start, end = window
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _round_tick(price: float) -> float:
    return round(float(price) / MNQ_TICK_SIZE) * MNQ_TICK_SIZE


def candidate_is_working(candidate: OrderCandidate, ts: pd.Timestamp) -> bool:
    """Reject stale limits outside their originating local session date."""
    spec = next(item for item in SESSIONS if item.name == candidate.session)
    now_local = pd.Timestamp(ts).tz_convert(spec.timezone)
    armed_local = candidate.armed_at.tz_convert(spec.timezone)
    minute = now_local.hour * 60 + now_local.minute
    return (
        _inside(minute, spec.window)
        and now_local.date() == armed_local.date()
    )


def add_indicators(frame_5m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Pine-equivalent closed-bar EMA and exchange-day VWAP statistics."""
    out = frame_5m.copy()
    out["ema20"] = out["close"].ewm(span=cfg.ema_len, adjust=False).mean()
    out["ema15"] = out["close"].ewm(span=cfg.fast_ema_len, adjust=False).mean()
    out["hl2"] = (out["high"] + out["low"]) / 2.0
    volume = out["volume"].fillna(0.0).astype(float)
    day = pd.Index(out.index.tz_convert(cfg.exchange_timezone).date)
    weighted = out["hl2"] * volume
    weighted2 = out["hl2"] * out["hl2"] * volume
    out["vol_sum"] = volume.groupby(day).cumsum()
    out["vwap"] = weighted.groupby(day).cumsum() / out["vol_sum"].replace(0, np.nan)
    second = weighted2.groupby(day).cumsum() / out["vol_sum"].replace(0, np.nan)
    out["deviation"] = np.sqrt((second - out["vwap"] ** 2).clip(lower=0.0))
    return out


class SessionRangeEma:
    """Closed-5m state machine matching the supplied Pine strategy."""

    def __init__(self, cfg: StrategyConfig = StrategyConfig()):
        self.cfg = cfg
        self.states = {spec.name: SessionState() for spec in SESSIONS}

    def state(self) -> dict:
        return {name: asdict(state) for name, state in self.states.items()}

    def restore(self, raw: dict) -> None:
        for name, values in raw.items():
            if name in self.states and isinstance(values, dict):
                self.states[name] = SessionState.restore(values)

    def record_fill(self, session: str) -> None:
        self.states[session].trades += 1

    def process_bar(
        self,
        ts: pd.Timestamp,
        bar: pd.Series,
        *,
        position_open: bool,
    ) -> Optional[OrderCandidate]:
        ts = pd.Timestamp(ts)
        for spec in SESSIONS:
            state = self.states[spec.name]
            local = ts.tz_convert(spec.timezone)
            minute = local.hour * 60 + local.minute
            live = spec.enabled and _inside(minute, spec.window)
            window_date = local.date().isoformat()
            if (
                live != state.in_window
                or (live and state.window_date != window_date)
            ):
                state.reset()
            state.in_window = live
            state.window_date = window_date if live else None
            state.last_bar = live and _local_minute(
                ts + pd.Timedelta(minutes=5), spec.timezone
            ) == spec.window[1]
            if not live:
                continue
            if _inside(minute, spec.range_window):
                state.range_high = float(bar["high"])
                state.range_low = float(bar["low"])
                state.range_ts = ts.isoformat()
                state.direction = 0
                state.break_streak_direction = 0
                state.break_streak_count = 0
                state.break_ts = None
                continue
            if state.range_high is None or state.direction:
                continue
            clean_direction = 0
            if float(bar["low"]) > state.range_high:
                clean_direction = 1
            elif float(bar["high"]) < state.range_low:
                clean_direction = -1
            if clean_direction == 0:
                state.break_streak_direction = 0
                state.break_streak_count = 0
                continue
            if clean_direction == state.break_streak_direction:
                state.break_streak_count += 1
            else:
                state.break_streak_direction = clean_direction
                state.break_streak_count = 1
            if state.break_streak_count >= 2:
                state.direction = clean_direction
                state.break_ts = ts.isoformat()
                state.break_open = float(bar["open"])
                state.break_high = float(bar["high"])
                state.break_low = float(bar["low"])
                state.break_close = float(bar["close"])

        if position_open:
            return None

        for spec in SESSIONS:  # Pine array priority: TOK, HK, LON, NY.
            state = self.states[spec.name]
            if (
                not state.in_window
                or state.last_bar
                or state.direction == 0
                or state.trades >= self.cfg.max_trades
                or state.range_high is None
                or state.range_low is None
                or state.range_ts is None
                or state.break_ts is None
            ):
                continue
            entry = _round_tick(float(bar["ema20"]))
            ema_outside_range = (
                entry > state.range_high
                if state.direction == 1
                else entry < state.range_low
            )
            if not ema_outside_range:
                continue
            close = float(bar["close"])
            pullback_side = close > entry if state.direction == 1 else close < entry
            if not pullback_side:
                continue
            deviation = float(bar["deviation"])
            vwap = float(bar["vwap"])
            if not np.isfinite(deviation) or not np.isfinite(vwap):
                continue
            targets = tuple(
                _round_tick(vwap + state.direction * dev * deviation)
                for dev in self.cfg.target_deviations
            )
            stop = float(state.range_low if state.direction == 1 else state.range_high)
            geometry_ok = (
                stop < entry and all(target > entry for target in targets)
                if state.direction == 1
                else stop > entry and all(target < entry for target in targets)
            )
            risk = abs(entry - stop)
            reward = abs(targets[0] - entry)
            if not geometry_ok or risk <= 0:
                continue
            rr = reward / risk
            if self.cfg.min_rr > 0 and rr < self.cfg.min_rr:
                continue
            return OrderCandidate(
                session=spec.name,
                side="long" if state.direction == 1 else "short",
                direction=state.direction,
                entry=entry,
                stop=stop,
                targets=targets,
                risk_points=risk,
                reward_points=reward,
                rr=rr,
                armed_at=ts + pd.Timedelta(minutes=5),
                range_high=float(state.range_high),
                range_low=float(state.range_low),
                range_ts=state.range_ts,
                break_ts=state.break_ts,
                break_open=float(state.break_open),
                break_high=float(state.break_high),
                break_low=float(state.break_low),
                break_close=float(state.break_close),
                ema20=float(bar["ema20"]),
                ema15=float(bar["ema15"]),
                vwap=vwap,
                deviation=deviation,
            )
        return None
