from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd


HERE = Path(__file__).resolve()
BOT_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for path in (BOT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from session_range_ema_strategy import (
    SessionRangeEma,
    StrategyConfig,
    add_indicators,
    candidate_is_working,
)
from shared import DEFAULT_CHANNEL_ID, load_env_file, resolve_discord_config


POINT_VALUE = 2.0
TICK_SIZE = 0.25
CONTRACTS = 10
COMMISSION_ROUND_TURN = 5.0


def _normalize(raw) -> pd.DataFrame:
    frame = raw.to_pandas() if hasattr(raw, "to_pandas") else raw.copy()
    if frame.empty:
        return pd.DataFrame()
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
    frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
    return frame[~frame.index.duplicated(keep="last")]


async def _fetch_all_available(
    client, days: int, limit: int, interval: int
) -> tuple[pd.DataFrame, int]:
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=days)
    cursor = end
    chunks: list[pd.DataFrame] = []
    for _ in range(30):
        raw = await client.get_bars(
            "MNQ",
            interval=interval,
            unit=2,
            limit=limit,
            partial=False,
            start_time=start.to_pydatetime(),
            end_time=cursor.to_pydatetime(),
        )
        chunk = _normalize(raw)
        if chunk.empty:
            break
        chunks.append(chunk)
        earliest = chunk.index.min()
        print(
            f"topstep_{interval}m_chunk={len(chunk)} "
            f"{earliest.isoformat()}..{chunk.index.max().isoformat()}"
        )
        if earliest <= start:
            break
        next_cursor = earliest - pd.Timedelta(minutes=1)
        if next_cursor >= cursor:
            break
        cursor = next_cursor
    if not chunks:
        return pd.DataFrame(), 0
    frame = pd.concat(chunks).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.loc[frame.index >= start], len(chunks)


def replay(frame: pd.DataFrame, native_5m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = StrategyConfig()
    engine = SessionRangeEma(cfg)
    bars_5m = add_indicators(native_5m, cfg)
    if len(bars_5m) < 100:
        raise RuntimeError("Topstep returned fewer than 100 complete five-minute bars")

    chicago_days = pd.Index(bars_5m.index.tz_convert("America/Chicago").date)
    first_day = chicago_days[0]
    eligible_at = bars_5m.index[
        max(100, next(
            (index for index, day in enumerate(chicago_days) if day != first_day),
            100,
        ))
    ]
    pending = None
    position = None
    trades: list[dict] = []
    five_index = 0

    for minute_ts, minute in frame.iterrows():
        if minute_ts < bars_5m.index[0]:
            continue

        if position is not None:
            direction = position["direction"]
            stop_hit = (
                float(minute.low) <= position["stop"]
                if direction == 1 else float(minute.high) >= position["stop"]
            )
            if stop_hit:
                exit_price = position["stop"] - direction * TICK_SIZE
                reason = "stop"
                points = direction * (exit_price - position["entry"])
                position["realized_contract_points"] += (
                    points * position["contracts_remaining"]
                )
                position["contracts_remaining"] = 0
            else:
                exit_price = None
                reason = None
                while position["next_target_index"] < 3:
                    index = position["next_target_index"]
                    target = position["targets"][index]
                    target_hit = (
                        float(minute.high) >= target
                        if direction == 1 else float(minute.low) <= target
                    )
                    if not target_hit:
                        break
                    qty = position["target_contracts"][index]
                    points = direction * (target - position["entry"])
                    position["realized_contract_points"] += points * qty
                    position["contracts_remaining"] -= qty
                    position["next_target_index"] += 1
                    if index == 1 and position["contracts_remaining"]:
                        position["stop"] = position["entry"]
                    if position["contracts_remaining"] == 0:
                        exit_price = target
                        reason = "target3"
                        break
            if exit_price is not None:
                points = position["realized_contract_points"] / CONTRACTS
                pnl = (
                    position["realized_contract_points"] * POINT_VALUE
                    - COMMISSION_ROUND_TURN * CONTRACTS
                )
                risk_dollars = position["risk_points"] * POINT_VALUE
                trades.append({
                    **position,
                    "exit_time": minute_ts.isoformat(),
                    "exit": exit_price,
                    "exit_reason": reason,
                    "pnl_points": points,
                    "pnl_usd": pnl,
                    "r_multiple": (
                        points * POINT_VALUE / risk_dollars
                        if risk_dollars else 0.0
                    ),
                    "status": "closed",
                })
                position = None

        if pending is not None and not candidate_is_working(pending, minute_ts):
            pending = None

        if position is None and pending is not None and minute_ts >= pending.armed_at:
            touched = (
                float(minute.low) <= pending.entry
                if pending.direction == 1
                else float(minute.high) >= pending.entry
            )
            if touched:
                engine.record_fill(pending.session)
                position = {
                    "session": pending.session,
                    "side": pending.side,
                    "direction": pending.direction,
                    "entry_time": minute_ts.isoformat(),
                    "entry": pending.entry,
                    "stop": pending.stop,
                    "targets": pending.targets,
                    "target_contracts": cfg.target_contracts,
                    "next_target_index": 0,
                    "contracts_remaining": CONTRACTS,
                    "realized_contract_points": 0.0,
                    "risk_points": pending.risk_points,
                    "reward_points": pending.reward_points,
                    "planned_rr": pending.rr,
                    "armed_at": pending.armed_at.isoformat(),
                    "range_ts": pending.range_ts,
                    "break_ts": pending.break_ts,
                }
                pending = None

                # Intraminute ordering is unknowable from OHLC; resolve the
                # adverse stop first, then target, on the entry minute too.
                stop_hit = (
                    float(minute.low) <= position["stop"]
                    if position["direction"] == 1
                    else float(minute.high) >= position["stop"]
                )
                if stop_hit:
                    direction = position["direction"]
                    reason = "stop"
                    exit_price = position["stop"] - direction * TICK_SIZE
                    points = direction * (exit_price - position["entry"])
                    position["realized_contract_points"] = points * CONTRACTS
                    position["contracts_remaining"] = 0
                    pnl = (
                        position["realized_contract_points"] * POINT_VALUE
                        - COMMISSION_ROUND_TURN * CONTRACTS
                    )
                    trades.append({
                        **position,
                        "exit_time": minute_ts.isoformat(),
                        "exit": exit_price,
                        "exit_reason": reason,
                        "pnl_points": points,
                        "pnl_usd": pnl,
                        "r_multiple": points / position["risk_points"],
                        "status": "closed",
                    })
                    position = None

        minute_close = minute_ts + pd.Timedelta(minutes=1)
        while (
            five_index < len(bars_5m)
            and bars_5m.index[five_index] + pd.Timedelta(minutes=5) <= minute_close
        ):
            five_ts = bars_5m.index[five_index]
            five_index += 1
            candidate = engine.process_bar(
                five_ts,
                bars_5m.loc[five_ts],
                position_open=position is not None,
            )
            if position is not None:
                current = bars_5m.loc[five_ts]
                previous = (
                    bars_5m.iloc[five_index - 2]
                    if five_index >= 2 else None
                )
                crossed = previous is not None and (
                    (
                        float(previous.ema12) >= float(previous.ema20)
                        and float(current.ema12) < float(current.ema20)
                    )
                    if position["direction"] == 1
                    else (
                        float(previous.ema12) <= float(previous.ema20)
                        and float(current.ema12) > float(current.ema20)
                    )
                )
                session_state = engine.states[position["session"]]
                if crossed:
                    boundary_stop = (
                        float(session_state.range_high)
                        if position["direction"] == 1
                        else float(session_state.range_low)
                    )
                    position["stop"] = (
                        max(position["stop"], boundary_stop)
                        if position["direction"] == 1
                        else min(position["stop"], boundary_stop)
                    )
                exit_reason = (
                    "session_end"
                    if session_state.last_bar or not session_state.in_window
                    else None
                )
                if exit_reason:
                    exit_price = float(current.close)
                    qty = position["contracts_remaining"]
                    points = position["direction"] * (
                        exit_price - position["entry"]
                    )
                    position["realized_contract_points"] += points * qty
                    position["contracts_remaining"] = 0
                    average_points = (
                        position["realized_contract_points"] / CONTRACTS
                    )
                    pnl = (
                        position["realized_contract_points"] * POINT_VALUE
                        - COMMISSION_ROUND_TURN * CONTRACTS
                    )
                    trades.append({
                        **position,
                        "exit_time": (
                            five_ts + pd.Timedelta(minutes=5)
                        ).isoformat(),
                        "exit": exit_price,
                        "exit_reason": exit_reason,
                        "pnl_points": average_points,
                        "pnl_usd": pnl,
                        "r_multiple": (
                            average_points / position["risk_points"]
                            if position["risk_points"] else 0.0
                        ),
                        "status": "closed",
                    })
                    position = None
            if five_ts < eligible_at:
                pending = None
            elif position is None:
                pending = candidate

    if position is not None:
        trades.append({
            **position,
            "exit_time": None,
            "exit": None,
            "exit_reason": None,
            "pnl_points": None,
            "pnl_usd": None,
            "r_multiple": None,
            "status": "open",
        })

    result = pd.DataFrame(trades)
    closed = result[result.status == "closed"].copy() if len(result) else result
    pnl = closed.pnl_usd.astype(float) if len(closed) else pd.Series(dtype=float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    gross_win = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gross_loss = abs(float(pnl[pnl < 0].sum())) if len(pnl) else 0.0
    metrics = {
        "bars_1m": len(frame),
        "bars_5m_complete": len(bars_5m),
        "history_start_utc": frame.index.min().isoformat(),
        "history_end_utc": frame.index.max().isoformat(),
        "eligible_start_utc": eligible_at.isoformat(),
        "closed_trades": len(closed),
        "open_trades": int((result.status == "open").sum()) if len(result) else 0,
        "wins": int((pnl > 0).sum()) if len(pnl) else 0,
        "losses": int((pnl <= 0).sum()) if len(pnl) else 0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "gross_profit": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "max_drawdown": abs(float(drawdown.min())) if len(drawdown) else 0.0,
    }
    return result, metrics


def _chart(trades: pd.DataFrame, path: Path) -> bool:
    closed = trades[trades.status == "closed"].copy() if len(trades) else trades
    if not len(closed):
        return False
    import matplotlib.pyplot as plt

    closed["exit_time"] = pd.to_datetime(closed["exit_time"], utc=True)
    closed["equity"] = closed["pnl_usd"].astype(float).cumsum()
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(closed["exit_time"], closed["equity"], color="#2563eb", linewidth=2)
    axis.axhline(0, color="#6b7280", linewidth=0.8)
    axis.set_title("MNQ Session Range EMA — Topstep historical bar replay")
    axis.set_xlabel("Exit time (UTC)")
    axis.set_ylabel("Cumulative simulated P&L (USD)")
    axis.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


async def _post_discord(
    channel_id: int,
    token: str,
    embed: dict,
    files: list[Path],
) -> None:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    form = aiohttp.FormData()
    form.add_field("payload_json", json.dumps({"embeds": [embed]}))
    handles = []
    try:
        for index, path in enumerate(files):
            handle = path.open("rb")
            handles.append(handle)
            form.add_field(
                f"files[{index}]",
                handle,
                filename=path.name,
                content_type=(
                    "image/png"
                    if path.suffix.lower() == ".png"
                    else (
                        "application/json"
                        if path.suffix.lower() == ".json" else "text/csv"
                    )
                ),
            )
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                url,
                headers={"Authorization": f"Bot {token}"},
                data=form,
            ) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"Discord HTTP {response.status}: "
                        f"{(await response.text())[:300]}"
                    )
    finally:
        for handle in handles:
            handle.close()


async def main(args) -> int:
    load_env_file(REPO_ROOT / ".env")
    load_env_file(BOT_ROOT / ".env.forward-test", override=True)
    os.environ["PROJECT_X_API_KEY"] = (
        os.getenv("TOPSTEP_API_KEY") or os.getenv("PROJECTX_API_KEY", "")
    )
    os.environ["PROJECT_X_USERNAME"] = (
        os.getenv("TOPSTEP_USERNAME") or os.getenv("PROJECTX_USERNAME", "")
    )
    account_name = (
        os.getenv("RANGE_EMA_TOPSTEP_ACCOUNT_NAME")
        or os.getenv("TOPSTEP_ACCOUNT_NAME", "")
    )
    os.environ["PROJECT_X_ACCOUNT_NAME"] = account_name

    from project_x_py import ProjectX

    async with ProjectX.from_env(account_name=account_name) as client:
        await client.authenticate()
        instrument = await client.get_instrument("MNQ")
        frame, one_minute_chunks = await _fetch_all_available(
            client, args.days, args.limit, 1
        )
        native_5m, five_minute_chunks = await _fetch_all_available(
            client, args.days, args.limit, 5
        )
    if frame.empty or native_5m.empty:
        raise RuntimeError("Topstep returned incomplete MNQ historical datasets")

    output_dir = BOT_ROOT / "runtime" / "replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_dir / f"range_ema_topstep_MNQ_1m_{stamp}.csv"
    raw_5m_path = output_dir / f"range_ema_topstep_MNQ_5m_{stamp}.csv"
    trade_path = output_dir / f"range_ema_topstep_trades_{stamp}.csv"
    metrics_path = output_dir / f"range_ema_topstep_metrics_{stamp}.json"
    chart_path = output_dir / f"range_ema_topstep_equity_{stamp}.png"
    frame.to_csv(raw_path, index_label="timestamp")
    native_5m.to_csv(raw_5m_path, index_label="timestamp")
    source_1m_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    source_5m_hash = hashlib.sha256(raw_5m_path.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(
        f"{source_1m_hash}:{source_5m_hash}".encode()
    ).hexdigest()

    trades, metrics = replay(frame, native_5m)
    metrics.update({
        "source": "TopstepX /History/retrieveBars",
        "symbol": "MNQ",
        "contract_id": instrument.id,
        "requested_days": args.days,
        "requested_limit": args.limit,
        "one_minute_fetch_chunks": one_minute_chunks,
        "five_minute_fetch_chunks": five_minute_chunks,
        "source_1m_csv_sha256": source_1m_hash,
        "source_5m_csv_sha256": source_5m_hash,
        "source_csv_sha256": source_hash,
        "model": (
            "actual Topstep closed 5m decisions; subsequent actual Topstep 1m OHLC touches; "
            "stop before target; one-tick adverse stop slippage; $5 round-turn commission"
        ),
    })
    trades.to_csv(trade_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    has_chart = _chart(trades, chart_path)

    print(json.dumps(metrics, indent=2))
    print(f"raw_source={raw_path}")
    print(f"raw_5m_source={raw_5m_path}")
    print(f"trades={trade_path}")
    print(f"metrics={metrics_path}")

    if args.post:
        token, configured_channel, _ = resolve_discord_config()
        channel = int(args.channel or configured_channel or DEFAULT_CHANNEL_ID)
        pf = (
            f"{metrics['profit_factor']:.2f}"
            if metrics["profit_factor"] is not None else "n/a"
        )
        session_lines = []
        if len(trades):
            closed = trades[trades.status == "closed"]
            for session_name, group in closed.groupby("session"):
                session_lines.append(
                    f"`{session_name}` · {len(group)} trades · "
                    f"{int((group.pnl_usd > 0).sum())}W/"
                    f"{int((group.pnl_usd <= 0).sum())}L · "
                    f"`${float(group.pnl_usd.sum()):+,.2f}`"
                )
        embed = {
            "title": "MNQ Session Range EMA — Topstep History Replay",
            "description": (
                "**Source:** direct TopstepX native MNQ 5m decision bars and "
                "1m fill-resolution bars\n"
                f"**Contract:** `{instrument.id}` · **SHA-256:** `{source_hash[:16]}…`\n"
                "**Important:** these are real Topstep bars, but historical ticks "
                "are unavailable; fills are conservative one-minute simulations."
            ),
            "color": 0x3498DB,
            "fields": [
                {
                    "name": "Coverage (UTC)",
                    "value": (
                        f"`{frame.index.min():%Y-%m-%d %H:%M}` → "
                        f"`{frame.index.max():%Y-%m-%d %H:%M}`\n"
                        f"{len(frame):,} 1m bars · "
                        f"{metrics['bars_5m_complete']:,} complete 5m bars"
                    ),
                    "inline": False,
                },
                {
                    "name": "Results",
                    "value": (
                        f"**{metrics['closed_trades']}** closed · "
                        f"**{metrics['open_trades']}** open · "
                        f"**{metrics['wins']}W/{metrics['losses']}L** · "
                        f"**{metrics['win_rate'] * 100:.1f}% WR**\n"
                        f"Net **${metrics['net_pnl']:+,.2f}** · PF **{pf}** · "
                        f"max drawdown **${metrics['max_drawdown']:,.2f}**"
                    ),
                    "inline": False,
                },
                {
                    "name": "By session",
                    "value": "\n".join(session_lines) or "No closed trades",
                    "inline": False,
                },
                {
                    "name": "Execution assumptions",
                    "value": (
                        "EMA order may fill only after its closed 5m arm bar; "
                        "1m high/low determines touch; SL resolves before TP when "
                        "both occur in one minute; stop receives 1 adverse MNQ tick; "
                        "$5 round-turn commission."
                    ),
                    "inline": False,
                },
            ],
            "footer": {
                "text": "Not synthetic data · modeled fills are not broker fills"
            },
        }
        upload = [trade_path, metrics_path]
        if has_chart:
            upload.append(chart_path)
        await _post_discord(channel, token, embed, upload)
        print(f"discord_posted_channel={channel}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL_ID)
    args = parser.parse_args()
    return asyncio.run(main(args))


if __name__ == "__main__":
    raise SystemExit(cli())
