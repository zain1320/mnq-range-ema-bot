from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import shared
from session_range_ema_adapter import SessionRangeEmaAdapter
from session_range_ema_strategy import (
    OrderCandidate,
    SessionRangeEma,
    StrategyConfig,
    add_indicators,
    candidate_is_working,
)


def bar(
    *,
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 98.0,
    close: float = 101.0,
    ema: float = 100.0,
    ema15: float = 101.0,
    vwap: float = 101.0,
    deviation: float = 2.0,
) -> pd.Series:
    return pd.Series({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
        "ema20": ema,
        "ema15": ema15,
        "vwap": vwap,
        "deviation": deviation,
    })


def candidate(**overrides) -> OrderCandidate:
    values = {
        "session": "NY",
        "side": "long",
        "direction": 1,
        "entry": 100.0,
        "stop": 95.0,
        "targets": (110.0, 112.0, 114.0),
        "risk_points": 5.0,
        "reward_points": 10.0,
        "rr": 2.0,
        "armed_at": pd.Timestamp("2026-01-06 15:00", tz="UTC"),
        "range_high": 99.0,
        "range_low": 95.0,
        "range_ts": pd.Timestamp("2026-01-06 10:30", tz="America/New_York").isoformat(),
        "break_ts": pd.Timestamp("2026-01-06 10:40", tz="America/New_York").isoformat(),
        "break_open": 100.0,
        "break_high": 104.0,
        "break_low": 101.0,
        "break_close": 103.0,
        "ema20": 100.0,
        "ema15": 101.0,
        "vwap": 102.0,
        "deviation": 2.0,
    }
    values.update(overrides)
    return OrderCandidate(**values)


class TempDataMixin:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_data = shared.DATA_DIR
        self.old_audit = shared.AUDIT_DIR
        shared.DATA_DIR = Path(self.temp.name) / "data"
        shared.AUDIT_DIR = Path(self.temp.name) / "audit"

    def tearDown(self):
        shared.DATA_DIR = self.old_data
        shared.AUDIT_DIR = self.old_audit
        self.temp.cleanup()


class StrategyTests(unittest.TestCase):
    def test_working_limit_expires_outside_originating_session(self):
        order = candidate(
            armed_at=pd.Timestamp("2026-01-06 15:00", tz="UTC"),
        )
        self.assertTrue(candidate_is_working(
            order, pd.Timestamp("2026-01-06 15:01", tz="UTC")
        ))
        self.assertFalse(candidate_is_working(
            order, pd.Timestamp("2026-01-07 15:01", tz="UTC")
        ))

    def test_new_york_range_is_dst_safe(self):
        for ts in (
            pd.Timestamp("2026-07-01 14:30", tz="UTC"),
            pd.Timestamp("2026-01-06 15:30", tz="UTC"),
        ):
            engine = SessionRangeEma()
            engine.process_bar(ts, bar(high=105, low=95), position_open=False)
            self.assertEqual(engine.states["NY"].range_high, 105.0)
            self.assertEqual(engine.states["NY"].range_low, 95.0)

    def test_break_requires_entire_candle_outside_range(self):
        engine = SessionRangeEma()
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:30", tz="America/New_York"),
            bar(high=100, low=95),
            position_open=False,
        )
        overlap = engine.process_bar(
            pd.Timestamp("2026-01-06 10:35", tz="America/New_York"),
            bar(high=104, low=99, close=103, ema=101),
            position_open=False,
        )
        self.assertIsNone(overlap)
        self.assertEqual(engine.states["NY"].direction, 0)
        first_clean = engine.process_bar(
            pd.Timestamp("2026-01-06 10:40", tz="America/New_York"),
            bar(high=105, low=101, close=104, ema=101, vwap=103, deviation=2),
            position_open=False,
        )
        self.assertIsNone(first_clean)
        self.assertEqual(engine.states["NY"].direction, 0)
        clean = engine.process_bar(
            pd.Timestamp("2026-01-06 10:45", tz="America/New_York"),
            bar(high=106, low=102, close=105, ema=101, vwap=103, deviation=2),
            position_open=False,
        )
        self.assertEqual(clean.side, "long")
        self.assertGreater(clean.break_low, clean.range_high)

    def test_limit_reprices_then_cancels_when_pullback_side_fails(self):
        engine = SessionRangeEma()
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:30", tz="America/New_York"),
            bar(high=100, low=95),
            position_open=False,
        )
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:35", tz="America/New_York"),
            bar(high=105, low=101, close=104, ema=101, vwap=104, deviation=3),
            position_open=False,
        )
        first = engine.process_bar(
            pd.Timestamp("2026-01-06 10:40", tz="America/New_York"),
            bar(high=106, low=102, close=105, ema=102, vwap=104, deviation=3),
            position_open=False,
        )
        second = engine.process_bar(
            pd.Timestamp("2026-01-06 10:45", tz="America/New_York"),
            bar(high=106, low=102, close=105, ema=103, vwap=104, deviation=3),
            position_open=False,
        )
        cancelled = engine.process_bar(
            pd.Timestamp("2026-01-06 10:50", tz="America/New_York"),
            bar(high=103, low=99, close=100, ema=102, vwap=104, deviation=3),
            position_open=False,
        )
        self.assertEqual((first.entry, second.entry), (102.0, 103.0))
        self.assertIsNone(cancelled)

    def test_ema_inside_range_rejects_setup_until_it_moves_outside(self):
        engine = SessionRangeEma()
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:30", tz="America/New_York"),
            bar(high=100, low=95),
            position_open=False,
        )
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:35", tz="America/New_York"),
            bar(high=105, low=101, close=104, ema=99),
            position_open=False,
        )
        rejected = engine.process_bar(
            pd.Timestamp("2026-01-06 10:40", tz="America/New_York"),
            bar(high=106, low=102, close=105, ema=99, vwap=104, deviation=3),
            position_open=False,
        )
        accepted = engine.process_bar(
            pd.Timestamp("2026-01-06 10:45", tz="America/New_York"),
            bar(high=107, low=103, close=106, ema=101, vwap=104, deviation=3),
            position_open=False,
        )
        self.assertIsNone(rejected)
        self.assertEqual(accepted.entry, 101.0)
        self.assertTrue(all(target > accepted.entry for target in accepted.targets))

    def test_tokyo_has_priority_during_overlap(self):
        engine = SessionRangeEma()
        for name in ("TOK", "HK"):
            state = engine.states[name]
            state.in_window = True
            state.window_date = "2026-01-06"
            state.range_high = 100.0
            state.range_low = 95.0
            state.range_ts = pd.Timestamp("2026-01-06", tz="UTC").isoformat()
            state.direction = 1
            state.break_ts = pd.Timestamp("2026-01-06 02:45", tz="UTC").isoformat()
            state.break_open = 101.0
            state.break_high = 105.0
            state.break_low = 101.0
            state.break_close = 104.0
        result = engine.process_bar(
            pd.Timestamp("2026-01-06 03:00", tz="UTC"),
            bar(high=105, low=101, close=104, ema=101, vwap=104, deviation=3),
            position_open=False,
        )
        self.assertEqual(result.session, "TOK")

    def test_max_trades_is_per_session(self):
        engine = SessionRangeEma(StrategyConfig(max_trades=1))
        state = engine.states["NY"]
        state.in_window = True
        state.window_date = "2026-01-06"
        state.range_high = 100.0
        state.range_low = 95.0
        state.range_ts = pd.Timestamp("2026-01-06", tz="UTC").isoformat()
        state.direction = 1
        state.break_ts = pd.Timestamp("2026-01-06", tz="UTC").isoformat()
        state.break_open = state.break_low = 101.0
        state.break_high = state.break_close = 104.0
        state.trades = 1
        result = engine.process_bar(
            pd.Timestamp("2026-01-06 11:00", tz="America/New_York"),
            bar(high=105, low=101, close=104, ema=101, vwap=104, deviation=3),
            position_open=False,
        )
        self.assertIsNone(result)

    def test_new_local_session_date_resets_stale_range_without_gap_bars(self):
        engine = SessionRangeEma()
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:30", tz="America/New_York"),
            bar(high=100, low=95),
            position_open=False,
        )
        engine.process_bar(
            pd.Timestamp("2026-01-06 10:35", tz="America/New_York"),
            bar(high=105, low=101, close=104, ema=101, vwap=104, deviation=3),
            position_open=False,
        )
        result = engine.process_bar(
            pd.Timestamp("2026-01-07 10:35", tz="America/New_York"),
            bar(high=106, low=102, close=105, ema=102, vwap=104, deviation=3),
            position_open=False,
        )
        self.assertIsNone(result)
        self.assertIsNone(engine.states["NY"].range_high)
        self.assertEqual(engine.states["NY"].direction, 0)

    def test_vwap_resets_on_chicago_calendar_day(self):
        idx = pd.DatetimeIndex([
            pd.Timestamp("2026-01-06 23:55", tz="America/Chicago"),
            pd.Timestamp("2026-01-07 00:00", tz="America/Chicago"),
        ])
        frame = pd.DataFrame({
            "open": [100, 200],
            "high": [102, 202],
            "low": [100, 200],
            "close": [101, 201],
            "volume": [10, 10],
        }, index=idx)
        result = add_indicators(frame, StrategyConfig())
        self.assertEqual(result.iloc[0].vwap, 101.0)
        self.assertEqual(result.iloc[1].vwap, 201.0)


class AdapterTests(TempDataMixin, unittest.TestCase):
    def test_tick_before_closed_bar_arm_time_cannot_fill(self):
        adapter = SessionRangeEmaAdapter(lambda _: None)
        adapter.pending = candidate()
        adapter.on_tick(pd.Timestamp("2026-01-06 14:59:59", tz="UTC"), 99.0)
        self.assertIsNone(adapter.lifecycle)
        self.assertIsNotNone(adapter.pending)
        adapter.close()

    def test_live_tick_fill_and_adverse_stop_slippage(self):
        messages = []
        adapter = SessionRangeEmaAdapter(messages.append)
        adapter.pending = candidate()
        adapter.on_tick(pd.Timestamp("2026-01-06 15:00:01", tz="UTC"), 100.0)
        self.assertIsNotNone(adapter.lifecycle)
        self.assertIn("FILLED", messages[-1])
        adapter.on_tick(pd.Timestamp("2026-01-06 15:01", tz="UTC"), 94.0)
        row = adapter.store.recent(1)[0]
        self.assertEqual(row["exit_px"], 94.75)
        self.assertEqual(row["pnl_points"], -5.25)
        self.assertEqual(row["pnl_usd"], -155.0)
        adapter.close()

    def test_open_position_survives_restart(self):
        adapter = SessionRangeEmaAdapter(lambda _: None)
        adapter.pending = candidate()
        adapter.on_tick(pd.Timestamp("2026-01-06 15:00:01", tz="UTC"), 100.0)
        signal_id = adapter.lifecycle.signal_id
        adapter.close()
        restored = SessionRangeEmaAdapter(lambda _: None)
        self.assertIsNotNone(restored.lifecycle)
        self.assertEqual(restored.lifecycle.signal_id, signal_id)
        restored.close()

    def test_targets_scale_five_three_two_and_move_stop_at_tp2(self):
        messages = []
        adapter = SessionRangeEmaAdapter(messages.append)
        adapter.pending = candidate()
        adapter.on_tick(pd.Timestamp("2026-01-06 15:00:01", tz="UTC"), 100.0)
        adapter.on_tick(pd.Timestamp("2026-01-06 15:01", tz="UTC"), 110.0)
        self.assertEqual(adapter.lifecycle.contracts_remaining, 5)
        self.assertEqual(adapter.lifecycle.stop, 95.0)
        adapter.on_tick(pd.Timestamp("2026-01-06 15:02", tz="UTC"), 112.0)
        self.assertEqual(adapter.lifecycle.contracts_remaining, 2)
        self.assertEqual(adapter.lifecycle.stop, 100.0)
        adapter.on_tick(pd.Timestamp("2026-01-06 15:03", tz="UTC"), 114.0)
        self.assertIsNone(adapter.lifecycle)
        row = adapter.store.recent(1)[0]
        self.assertEqual(row["exit_reason"], "target3")
        self.assertEqual(row["pnl_usd"], 178.0)
        adapter.close()

    def test_pending_order_survives_restart(self):
        adapter = SessionRangeEmaAdapter(lambda _: None)
        adapter.pending = candidate(entry=101.25)
        adapter._persist_runtime()
        adapter.close()
        restored = SessionRangeEmaAdapter(lambda _: None)
        self.assertIsNotNone(restored.pending)
        self.assertEqual(restored.pending.entry, 101.25)
        restored.close()

    def test_reason_message_contains_geometry(self):
        messages = []
        adapter = SessionRangeEmaAdapter(messages.append)
        adapter._notify_armed(candidate())
        message = messages[-1]
        self.assertIn("**Why LONG**", message)
        self.assertIn("two consecutive full candles", message)
        self.assertIn("TP1", message)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
