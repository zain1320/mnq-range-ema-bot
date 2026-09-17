#!/usr/bin/env python3
"""Standalone MNQ Session Range EMA forward-test and practice executor."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from discord_reporter import SharedDiscordReporter
from session_range_ema_adapter import SessionRangeEmaAdapter
from session_range_ema_executor import SessionRangeEmaExecutor
from shared import (
    DEFAULT_CHANNEL_ID,
    ET,
    LiveBarBuilder,
    append_bar,
    empty_frame,
    in_maintenance,
    load_env_file,
    resolve_discord_config,
)

SYMBOL = "MNQ"
ROLL_DAYS = 21


class RangeEmaHost:
    STALE_LIMIT_SEC = 180.0

    def __init__(
        self,
        *,
        channel_id: int,
        execution_mode: str,
        account_id: int,
        account_name: str,
    ):
        token, env_channel, token_source = resolve_discord_config()
        self.channel_id = int(channel_id or env_channel or DEFAULT_CHANNEL_ID)
        self.token_source = token_source
        self.discord = SharedDiscordReporter(token, self.channel_id)
        self.execution_mode = execution_mode.lower().strip()
        self.account_id = int(account_id)
        self.account_name = account_name.strip()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.suite = None
        self.executor = None
        self.frame = empty_frame()
        self.bar_queue: asyncio.Queue = asyncio.Queue()
        self.builder = LiveBarBuilder(self.bar_queue.put_nowait)
        self.adapter = SessionRangeEmaAdapter(
            lambda message: self.notify(message),
            contracts=10,
        )
        self.ticks = 0
        self.bars = 0
        self.last_tick_wall = 0.0
        self.last_price = 0.0
        self._register_commands()

    def notify(self, message: str) -> None:
        if self.loop and self.loop.is_running():
            self.loop.create_task(self.discord.post(self.adapter.name, message))
        else:
            print(f"[{self.adapter.name}] {message}")

    def _register_commands(self) -> None:
        async def health(_):
            payload = self.adapter.health()
            payload["topstep"] = (
                self.executor.status() if self.executor else {"enabled": False}
            )
            return f"**[{self.adapter.name}]** `{json.dumps(payload, default=str)}`"

        async def stats(_):
            summary = self.adapter.summary()
            closed = int(summary.get("wins") or 0) + int(summary.get("losses") or 0)
            win_rate = int(summary.get("wins") or 0) / closed * 100 if closed else 0
            return {
                "title": "📊 MNQ Range EMA — Stats",
                "description": (
                    f"**Trades** `{int(summary.get('total') or 0)}` · "
                    f"**W/L** `{int(summary.get('wins') or 0)}/"
                    f"{int(summary.get('losses') or 0)}` · "
                    f"**Open** `{int(summary.get('open') or 0)}`\n"
                    f"**Win rate** `{win_rate:.0f}%` · "
                    f"**Simulated P&L** `${float(summary.get('pnl') or 0):+,.2f}`"
                ),
                "color": 0x3498DB,
            }

        self.discord.register("range-ema-health", health)
        self.discord.register("range-ema-stats", stats)
        self.discord.register("stats", stats)

    async def connect(self) -> None:
        api_key = os.getenv("TOPSTEP_API_KEY") or os.getenv("PROJECTX_API_KEY", "")
        username = os.getenv("TOPSTEP_USERNAME") or os.getenv("PROJECTX_USERNAME", "")
        if not api_key or not username:
            raise RuntimeError("Topstep credentials are missing")
        os.environ["PROJECT_X_API_KEY"] = api_key
        os.environ["PROJECT_X_USERNAME"] = username
        if self.execution_mode == "topstep":
            if not self.account_id or not self.account_name:
                raise RuntimeError("Practice account ID and name are required")
            os.environ["PROJECT_X_ACCOUNT_ID"] = str(self.account_id)
            os.environ["PROJECT_X_ACCOUNT_NAME"] = self.account_name

        from project_x_py import TradingSuite

        self.suite = await TradingSuite.create([SYMBOL], initial_days=1)
        context = self.suite[SYMBOL]
        context.data.timeframes.clear()
        context.data.data.clear()

        async def on_data(payload):
            data = getattr(payload, "data", None) or payload or {}
            try:
                price = float(data.get("price"))
                size = int(data.get("volume", 0))
                ts = pd.Timestamp(
                    data.get("timestamp") or pd.Timestamp.now(tz="UTC")
                )
            except (TypeError, ValueError):
                return
            if price > 10000 and size > 0:
                self.on_tick(ts, price, size)

        await context.data.add_callback("data_update", on_data)
        if self.execution_mode == "topstep":
            self.executor = SessionRangeEmaExecutor(
                self.suite,
                self.adapter,
                self.notify,
                account_id=self.account_id,
                account_name=self.account_name,
            )
            await self.executor.initialize()
        print(f"[TOPSTEP] connected {SYMBOL} range_ema={self.execution_mode}")

    async def warmup(self) -> None:
        from project_x_py import ProjectX

        account = os.getenv("PROJECT_X_ACCOUNT_NAME") or None
        history = None
        for attempt in range(1, 4):
            try:
                async with ProjectX.from_env(account_name=account) as client:
                    if hasattr(client, "authenticate"):
                        await client.authenticate()
                    history = await client.get_bars(
                        SYMBOL, days=ROLL_DAYS, interval=1, unit=2
                    )
                    if hasattr(history, "to_pandas"):
                        history = history.to_pandas()
                if history is not None and len(history):
                    break
            except Exception as exc:
                print(f"[WARMUP] attempt {attempt}/3 failed: {exc}")
                await asyncio.sleep(5 * attempt)
        if history is None or len(history) == 0:
            raise RuntimeError("MNQ historical warmup unavailable")

        frame = history.copy()
        columns = {str(column).lower(): column for column in frame.columns}
        timestamp = columns.get("timestamp") or columns.get("t") or frame.columns[0]
        frame = frame.rename(columns={
            timestamp: "timestamp",
            columns.get("open", "open"): "open",
            columns.get("high", "high"): "high",
            columns.get("low", "low"): "low",
            columns.get("close", "close"): "close",
            columns.get("volume", "volume"): "volume",
        })
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
        frame.index = frame.index.tz_convert(ET)
        self.frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
        self.frame = self.frame[~self.frame.index.duplicated(keep="last")]
        self.frame = self.frame.tail(ROLL_DAYS * 1440).copy()
        print(f"[WARMUP] successful bars={len(self.frame):,}")

    def on_tick(self, ts: pd.Timestamp, price: float, size: int) -> None:
        before = self.adapter.execution_revision
        self.ticks += 1
        self.last_tick_wall = time.time()
        self.last_price = float(price)
        self.adapter.on_tick(ts, price)
        self.builder.on_tick(ts, price, size)
        if self.executor is not None and before != self.adapter.execution_revision:
            asyncio.create_task(self.executor.sync())

    async def bar_worker(self) -> None:
        while True:
            bar = await self.bar_queue.get()
            try:
                self.frame = append_bar(self.frame, bar, keep_days=ROLL_DAYS)
                ts = self.frame.index[-1]
                self.bars += 1
                before = self.adapter.execution_revision
                self.adapter.on_bar(self.frame, ts)
                if self.executor is not None and before != self.adapter.execution_revision:
                    asyncio.create_task(self.executor.sync())
            finally:
                self.bar_queue.task_done()

    async def watchdog(self) -> None:
        await asyncio.sleep(90)
        while True:
            await asyncio.sleep(30)
            age = time.time() - self.last_tick_wall if self.last_tick_wall else 999999
            if not in_maintenance(datetime.now(ET)) and age > self.STALE_LIMIT_SEC:
                await self.discord.post(self.adapter.name, f"MNQ feed stale {age:.0f}s")
                os._exit(1)

    async def status_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            age = time.time() - self.last_tick_wall if self.last_tick_wall else 999999
            print(
                f"[STATUS:MNQ] mode={self.execution_mode} ticks={self.ticks} "
                f"bars={self.bars} frame={len(self.frame)} tick_age={age:.1f}s "
                f"pending={bool(self.adapter.pending)} open={bool(self.adapter.lifecycle)}"
            )

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        print(
            f"MNQ RANGE EMA | 10 contracts | {self.execution_mode.upper()} | "
            f"channel={self.channel_id}"
        )
        await self.connect()
        await self.warmup()
        await self.discord.start()
        await self.discord.wait_ready(30)
        await self.discord.post(
            self.adapter.name,
            f"Host online · MNQ ×10 Session Range EMA `{self.execution_mode}`",
        )
        await asyncio.gather(self.bar_worker(), self.watchdog(), self.status_loop())

    async def shutdown(self) -> None:
        self.builder.force_close()
        try:
            await asyncio.wait_for(self.bar_queue.join(), 5)
        except asyncio.TimeoutError:
            pass
        if self.executor is not None:
            await self.executor.flatten("host_shutdown")
        self.adapter.close()
        await self.discord.stop()
        if self.suite is not None:
            await self.suite.disconnect()


async def amain(args) -> None:
    host = RangeEmaHost(
        channel_id=args.channel,
        execution_mode=args.execution,
        account_id=args.account_id,
        account_name=args.account_name,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    task = asyncio.create_task(host.run())
    waiter = asyncio.create_task(stop.wait())
    await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        waiter.cancel()
        await host.shutdown()


def main() -> None:
    load_env_file(HERE / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL_ID)
    parser.add_argument(
        "--execution",
        choices=("signal_only", "topstep"),
        default=os.getenv("RANGE_EMA_EXECUTION_MODE", "signal_only"),
    )
    parser.add_argument(
        "--account-id",
        type=int,
        default=int(os.getenv("RANGE_EMA_TOPSTEP_ACCOUNT_ID", "0") or 0),
    )
    parser.add_argument(
        "--account-name",
        default=os.getenv("RANGE_EMA_TOPSTEP_ACCOUNT_NAME", ""),
    )
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
