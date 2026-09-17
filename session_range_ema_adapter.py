from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from session_range_ema_strategy import (
    MNQ_TICK_SIZE,
    OrderCandidate,
    SessionRangeEma,
    StrategyConfig,
    add_indicators,
    candidate_is_working,
)
from shared import AuditJSONL, SignalStore


BOT_NAME = "range-ema-ft-mnq"
MNQ_POINT_VALUE = 2.0
COMMISSION_PER_SIDE = 2.5
RUNTIME_KEY = "__runtime__"


@dataclass
class RangeEmaLifecycle:
    signal_id: int
    session: str
    side: str
    entry: float
    stop: float
    initial_stop: float
    targets: tuple[float, float, float]
    target_contracts: tuple[int, int, int]
    contracts: int
    contracts_remaining: int
    next_target_index: int
    realized_contract_points: float
    entry_ts: pd.Timestamp

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    def state(self) -> dict:
        return {
            "session": self.session,
            "side": self.side,
            "entry": self.entry,
            "stop": self.stop,
            "initial_stop": self.initial_stop,
            "targets": list(self.targets),
            "target_contracts": list(self.target_contracts),
            "contracts": self.contracts,
            "contracts_remaining": self.contracts_remaining,
            "next_target_index": self.next_target_index,
            "realized_contract_points": self.realized_contract_points,
            "entry_ts": self.entry_ts.isoformat(),
        }

    @classmethod
    def restore(cls, signal_id: int, raw: dict) -> "RangeEmaLifecycle":
        return cls(
            signal_id=int(signal_id),
            session=str(raw["session"]),
            side=str(raw["side"]),
            entry=float(raw["entry"]),
            stop=float(raw["stop"]),
            initial_stop=float(raw.get("initial_stop", raw["stop"])),
            targets=tuple(float(value) for value in raw["targets"]),
            target_contracts=tuple(int(value) for value in raw["target_contracts"]),
            contracts=int(raw["contracts"]),
            contracts_remaining=int(raw.get("contracts_remaining", raw["contracts"])),
            next_target_index=int(raw.get("next_target_index", 0)),
            realized_contract_points=float(raw.get("realized_contract_points", 0.0)),
            entry_ts=pd.Timestamp(raw["entry_ts"]),
        )


class SessionRangeEmaAdapter:
    def __init__(self, notify: Callable[[str], None], contracts: int = 10):
        self.name = BOT_NAME
        self.symbol = "MNQ"
        self.cfg = StrategyConfig(contracts=int(contracts))
        if self.cfg.contracts != sum(self.cfg.target_contracts):
            raise ValueError("Range EMA contracts must equal the 5/3/2 target allocation")
        self.engine = SessionRangeEma(self.cfg)
        self.notify = notify
        self.store = SignalStore("session_range_ema_ft_mnq")
        self.audit = AuditJSONL(self.name, self.symbol)
        self.pending: Optional[OrderCandidate] = None
        self.lifecycle: Optional[RangeEmaLifecycle] = None
        self.last_gate = "startup"
        self.last_5m: Optional[pd.Timestamp] = None
        self.bootstrapped = False
        self.execution_revision = 0
        self._restore()

    def _restore(self) -> None:
        runtime = self.store.load_day(RUNTIME_KEY)
        if runtime:
            self.engine.restore(runtime.get("sessions", {}))
            if runtime.get("pending"):
                self.pending = OrderCandidate.restore(runtime["pending"])
            if runtime.get("last_5m"):
                self.last_5m = pd.Timestamp(runtime["last_5m"])
            self.bootstrapped = bool(runtime.get("bootstrapped", False))
        rows = self.store.open_rows()
        if rows:
            row = rows[-1]
            try:
                raw = json.loads(row.get("state_json") or "{}")
                self.lifecycle = RangeEmaLifecycle.restore(int(row["id"]), raw)
                self.pending = None
                self.audit.event("restore", signal_id=row["id"], state=raw)
            except Exception as exc:
                self.audit.event("restore_failed", row_id=row.get("id"), error=str(exc))

    def _persist_runtime(self) -> None:
        self.store.save_day(RUNTIME_KEY, {
            "sessions": self.engine.state(),
            "pending": self.pending.state() if self.pending else None,
            "last_5m": self.last_5m.isoformat() if self.last_5m is not None else None,
            "bootstrapped": self.bootstrapped,
        })

    def close(self) -> None:
        self._persist_runtime()
        self.store.close()

    @staticmethod
    def _resample_5m(frame_et: pd.DataFrame) -> pd.DataFrame:
        return (
            frame_et.resample("5min", label="left", closed="left")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
        )

    def on_tick(self, ts: pd.Timestamp, price: float) -> None:
        now = pd.Timestamp(ts)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        price = float(price)

        if self.lifecycle is not None:
            lc = self.lifecycle
            stop_hit = price <= lc.stop if lc.direction == 1 else price >= lc.stop
            if stop_hit:
                exit_px = lc.stop - lc.direction * MNQ_TICK_SIZE
                self._finish_remaining(now, exit_px, "stop")
                return
            while (
                self.lifecycle is not None
                and lc.next_target_index < len(lc.targets)
            ):
                index = lc.next_target_index
                target = lc.targets[index]
                target_hit = (
                    price >= target if lc.direction == 1 else price <= target
                )
                if not target_hit:
                    break
                self._take_target(now, index)
            return

        candidate = self.pending
        if candidate is not None and not candidate_is_working(candidate, now):
            self.audit.event(
                "order_cancel",
                session=candidate.session,
                side=candidate.side,
                old_entry=candidate.entry,
                reason="outside_originating_session",
            )
            self.pending = None
            self.last_gate = "stale_limit_cancelled"
            self._persist_runtime()
            return
        if candidate is None or now < candidate.armed_at.tz_convert("UTC"):
            return
        touched = (
            price <= candidate.entry
            if candidate.direction == 1
            else price >= candidate.entry
        )
        if not touched:
            return
        self.engine.record_fill(candidate.session)
        key = f"{candidate.session}|{candidate.break_ts}|{candidate.side}"
        if key in self.store.seen_keys():
            self.pending = None
            self.last_gate = "duplicate_fill_blocked"
            self.audit.event("duplicate_blocked", signal_key=key, observed_tick=price)
            self._persist_runtime()
            return
        state = {
            "session": candidate.session,
            "side": candidate.side,
            "entry": candidate.entry,
            "stop": candidate.stop,
            "initial_stop": candidate.stop,
            "targets": list(candidate.targets),
            "target_contracts": list(self.cfg.target_contracts),
            "contracts": self.cfg.contracts,
            "contracts_remaining": self.cfg.contracts,
            "next_target_index": 0,
            "realized_contract_points": 0.0,
            "entry_ts": now.isoformat(),
        }
        signal_id = self.store.insert(
            key=key,
            signal_ts=now.isoformat(),
            side=candidate.side,
            entry=candidate.entry,
            stop=candidate.stop,
            target1=candidate.targets[0],
            target2=candidate.targets[1],
            contracts=self.cfg.contracts,
            source=f"{candidate.session}_EMA_PULLBACK",
            payload=candidate.state(),
            state=state,
        )
        self.lifecycle = RangeEmaLifecycle.restore(signal_id, state)
        self.execution_revision += 1
        self.pending = None
        self.last_gate = "filled"
        self._persist_runtime()
        self.audit.event(
            "entry",
            signal_id=signal_id,
            session=candidate.session,
            side=candidate.side,
            entry=candidate.entry,
            observed_tick=price,
            stop=candidate.stop,
            targets=candidate.targets,
            contracts=self.cfg.contracts,
        )
        risk_dollars = candidate.risk_points * MNQ_POINT_VALUE * self.cfg.contracts
        self.notify(
            f"🟡 **#{signal_id} FILLED | "
            f"{'▲ LONG' if candidate.direction == 1 else '▼ SHORT'} | "
            f"MNQ · {candidate.session}**\n"
            f"EMA limit `{candidate.entry:.2f}` touched by live tick `{price:.2f}` · "
            f"SL `{candidate.stop:.2f}` · "
            f"TPs `{' / '.join(f'{value:.2f}' for value in candidate.targets)}`\n"
            f"Risk `{candidate.risk_points:.2f}pt` / ~`${risk_dollars:,.2f}` · "
            f"reward `{candidate.reward_points:.2f}pt` / `{candidate.rr:.2f}R` · "
            "10 MNQ"
        )

    def on_bar(self, frame_et: pd.DataFrame, bar_ts_et: pd.Timestamp) -> None:
        ts = pd.Timestamp(bar_ts_et)
        if ts.minute % 5 != 4:
            return
        frame_5m = add_indicators(self._resample_5m(frame_et), self.cfg)
        if len(frame_5m) < self.cfg.ema_len:
            self.last_gate = f"warmup_lt_{self.cfg.ema_len}"
            return
        completed_ts = ts.floor("5min")
        if completed_ts not in frame_5m.index:
            return

        if not self.bootstrapped:
            replay = frame_5m.tail(300)
            for replay_ts, replay_bar in replay.iloc[:-1].iterrows():
                self.engine.process_bar(replay_ts, replay_bar, position_open=True)
            self.bootstrapped = True

        bar = frame_5m.loc[completed_ts]
        candidate = self.engine.process_bar(
            completed_ts, bar, position_open=self.lifecycle is not None
        )
        self.last_5m = completed_ts

        if self.lifecycle is not None:
            lc = self.lifecycle
            state = self.engine.states[lc.session]
            if len(frame_5m.loc[:completed_ts]) >= 2:
                previous = frame_5m.loc[:completed_ts].iloc[-2]
                crossed = (
                    float(previous.ema12) >= float(previous.ema20)
                    and float(bar.ema12) < float(bar.ema20)
                    if lc.direction == 1
                    else float(previous.ema12) <= float(previous.ema20)
                    and float(bar.ema12) > float(bar.ema20)
                )
                if crossed:
                    boundary_stop = (
                        float(state.range_high)
                        if lc.direction == 1
                        else float(state.range_low)
                    )
                    new_stop = (
                        max(lc.stop, boundary_stop)
                        if lc.direction == 1
                        else min(lc.stop, boundary_stop)
                    )
                    if new_stop != lc.stop:
                        old_stop = lc.stop
                        lc.stop = new_stop
                        self.store.update_state(lc.signal_id, lc.state())
                        self.execution_revision += 1
                        self.last_gate = "ema12_cross_stop_tightened"
                        self.audit.event(
                            "ema_cross_stop_tightened",
                            signal_id=lc.signal_id,
                            old_stop=old_stop,
                            new_stop=new_stop,
                            range_boundary=boundary_stop,
                            bar=completed_ts,
                        )
                        self.notify(
                            f"🛡️ **#{lc.signal_id} EMA12/EMA20 CROSS | "
                            f"MNQ {lc.side.upper()} · {lc.session}**\n"
                            f"Trade remains open · SL `{old_stop:.2f}` → "
                            f"`{new_stop:.2f}` at the broken range boundary"
                        )
            if (
                self.lifecycle is not None
                and self.cfg.flat_at_end
                and (state.last_bar or not state.in_window)
            ):
                self._finish_remaining(
                    completed_ts + pd.Timedelta(minutes=5),
                    float(bar.close),
                    "session_end",
                )

        if self.lifecycle is not None:
            self.pending = None
            self.last_gate = "position_open"
        elif candidate is None:
            if self.pending is not None:
                self.audit.event(
                    "order_cancel",
                    session=self.pending.session,
                    side=self.pending.side,
                    old_entry=self.pending.entry,
                    bar=completed_ts,
                )
            self.pending = None
            self.last_gate = "no_eligible_setup"
        else:
            previous = self.pending
            self.pending = candidate
            self.last_gate = "limit_armed"
            identity_changed = (
                previous is None
                or previous.session != candidate.session
                or previous.break_ts != candidate.break_ts
                or previous.side != candidate.side
            )
            price_changed = previous is not None and previous.entry != candidate.entry
            self.audit.event(
                "order_arm" if identity_changed else "order_reprice",
                bar=completed_ts,
                candidate=candidate.state(),
            )
            if identity_changed:
                self._notify_armed(candidate)
            elif price_changed:
                self.notify(
                    f"🔄 **MNQ {candidate.session} EMA LIMIT UPDATED** · "
                    f"`{previous.entry:.2f}` → `{candidate.entry:.2f}` · "
                    f"SL `{candidate.stop:.2f}` · "
                    f"TPs `{' / '.join(f'{value:.2f}' for value in candidate.targets)}`"
                )
        self._persist_runtime()

    def _notify_armed(self, candidate: OrderCandidate) -> None:
        break_rule = (
            f"low `{candidate.break_low:.2f}` > range high `{candidate.range_high:.2f}`"
            if candidate.direction == 1
            else f"high `{candidate.break_high:.2f}` < range low `{candidate.range_low:.2f}`"
        )
        risk_dollars = candidate.risk_points * MNQ_POINT_VALUE * self.cfg.contracts
        target_text = " · ".join(
            f"TP{index + 1} `{target:.2f}` ({deviation:.2f}σ/{qty}ct)"
            for index, (target, deviation, qty) in enumerate(zip(
                candidate.targets,
                self.cfg.target_deviations,
                self.cfg.target_contracts,
            ))
        )
        self.notify(
            f"🟢 **LIMIT ARMED | "
            f"{'▲ LONG' if candidate.direction == 1 else '▼ SHORT'} | "
            f"MNQ · {candidate.session}**\n"
            f"**Range bar** · `{pd.Timestamp(candidate.range_ts).tz_convert('America/New_York'):%H:%M ET}` · "
            f"low `{candidate.range_low:.2f}` · high `{candidate.range_high:.2f}`\n"
            f"**Break bar** · O `{candidate.break_open:.2f}` · "
            f"H `{candidate.break_high:.2f}` · L `{candidate.break_low:.2f}` · "
            f"C `{candidate.break_close:.2f}`\n"
            f"**Why {candidate.side.upper()}** · two consecutive full candles cleared "
            f"the range; second candle: {break_rule}; close remains on the breakout "
            f"side of EMA20\n"
            f"**Pullback order** · limit at EMA20 `{candidate.entry:.2f}` · "
            f"outside range `{candidate.range_low:.2f}–{candidate.range_high:.2f}` · "
            f"armed after `{candidate.armed_at.tz_convert('America/New_York'):%H:%M ET}`\n"
            f"**Stop** · opposite range side `{candidate.stop:.2f}` · "
            f"`{candidate.risk_points:.2f}pt` / ~`${risk_dollars:,.2f}`\n"
            f"**Targets** · {target_text} · TP2 moves runner stop to breakeven"
        )

    def _take_target(self, ts: pd.Timestamp, index: int) -> None:
        lc = self.lifecycle
        if lc is None:
            return
        qty = lc.target_contracts[index]
        exit_px = lc.targets[index]
        points = lc.direction * (exit_px - lc.entry)
        lc.realized_contract_points += points * qty
        lc.contracts_remaining -= qty
        lc.next_target_index = index + 1
        if index == 1 and lc.contracts_remaining:
            lc.stop = lc.entry
        self.execution_revision += 1
        self.audit.event(
            "partial_target",
            signal_id=lc.signal_id,
            target=index + 1,
            exit_ts=ts,
            exit_px=exit_px,
            contracts=qty,
            contracts_remaining=lc.contracts_remaining,
            stop=lc.stop,
        )
        self.notify(
            f"🎯 **#{lc.signal_id} TP{index + 1} HIT | MNQ "
            f"{lc.side.upper()} · {lc.session}**\n"
            f"`{qty}` contracts at `{exit_px:.2f}` · "
            f"`{lc.contracts_remaining}` remaining"
            + (
                f" · runner SL moved to breakeven `{lc.entry:.2f}`"
                if index == 1 and lc.contracts_remaining else ""
            )
        )
        if lc.contracts_remaining == 0:
            self._finalize(ts, exit_px, "target3")
        else:
            self.store.update_state(lc.signal_id, lc.state())
            self._persist_runtime()

    def _finish_remaining(self, ts: pd.Timestamp, exit_px: float, reason: str) -> None:
        lc = self.lifecycle
        if lc is None:
            return
        qty = lc.contracts_remaining
        points = lc.direction * (float(exit_px) - lc.entry)
        lc.realized_contract_points += points * qty
        lc.contracts_remaining = 0
        self.execution_revision += 1
        self._finalize(ts, exit_px, reason)

    def _finalize(self, ts: pd.Timestamp, exit_px: float, reason: str) -> None:
        lc = self.lifecycle
        if lc is None:
            return
        average_points = lc.realized_contract_points / lc.contracts
        pnl = (
            lc.realized_contract_points * MNQ_POINT_VALUE
            - 2 * COMMISSION_PER_SIDE * lc.contracts
        )
        self.store.finalize(
            lc.signal_id,
            exit_ts=pd.Timestamp(ts).isoformat(),
            exit_px=float(exit_px),
            reason=reason,
            pnl_points=average_points,
            pnl_usd=pnl,
        )
        self.audit.event(
            "lifecycle_close",
            signal_id=lc.signal_id,
            session=lc.session,
            reason=reason,
            exit_ts=ts,
            exit_px=exit_px,
            pnl_points=average_points,
            pnl_usd=pnl,
        )
        self.notify(
            f"{'✅' if pnl > 0 else '🛑'} **#{lc.signal_id} CLOSED {reason.upper()} | "
            f"MNQ {lc.side.upper()} · {lc.session}**\n"
            f"Entry `{lc.entry:.2f}` · weighted result `{average_points:+.2f}pt` · "
            f"**${pnl:+,.2f}** including commissions"
        )
        self.lifecycle = None
        self._persist_runtime()

    def summary(self) -> dict:
        return self.store.summary()

    def health(self) -> dict:
        return {
            "strategy": self.name,
            "symbol": self.symbol,
            "last_gate": self.last_gate,
            "pending": ({
                "session": self.pending.session,
                "side": self.pending.side,
                "entry": self.pending.entry,
                "stop": self.pending.stop,
                "targets": self.pending.targets,
            } if self.pending else None),
            "open": self.lifecycle.state() if self.lifecycle else None,
            "sessions": {
                name: {
                    "active": state.in_window,
                    "direction": state.direction,
                    "trades": state.trades,
                    "range": (
                        [state.range_low, state.range_high]
                        if state.range_low is not None else None
                    ),
                }
                for name, state in self.engine.states.items()
            },
        }
