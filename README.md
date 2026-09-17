# MNQ Session Range EMA Bot

A standalone live forward-test bot for the Session Range Break + EMA20
Pullback strategy. It consumes TopstepX MNQ market data, records simulated
performance, posts Discord alerts, and can optionally execute on a validated
Topstep practice account.

## Current rules

- Instrument: MNQ, fixed size of 10 contracts.
- Sessions: Tokyo, Hong Kong, London, and New York using DST-aware local
  timezones.
- Break confirmation: two consecutive closed 5-minute candles entirely beyond
  the same side of the session range.
- Entry: a later live tick touches EMA20 after EMA20 is outside the range.
- Stop: opposite side of the original session range.
- TP1: VWAP ± 1.28σ, exit 5 contracts; stop unchanged.
- TP2: VWAP ± 2.01σ, exit 3 contracts; move the remaining stop to breakeven.
- TP3: VWAP ± 2.51σ, exit the final 2 contracts.
- On an adverse closed-bar EMA12/EMA20 crossover, keep the trade open and
  tighten the stop to the broken range boundary (range high for longs, range
  low for shorts). Never loosen an existing breakeven stop.
- Exit any remaining contracts at the originating session's end.
- Maximum one trade per session.

Long targets use VWAP plus the listed deviation; short targets use VWAP minus
the deviation. VWAP statistics reset on the Chicago calendar day.

## Safety

Broker execution is practice-only. The executor requires:

- `PAPER_MODE=true`
- `LIVE_MODE` not set to `true`
- An account name beginning with `PRAC`
- Exactly 10 MNQ contracts

It rejects entries when the account already has an MNQ position or MNQ order.
Do not treat this software as financial advice or deploy it to a funded account
without independent review and testing.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item env.example .env
```

Fill in `.env` with your own Topstep and Discord credentials. Keep
`RANGE_EMA_EXECUTION_MODE=signal_only` until you have verified alerts and
simulated behavior. To enable practice execution:

```env
PAPER_MODE=true
LIVE_MODE=false
RANGE_EMA_EXECUTION_MODE=topstep
RANGE_EMA_TOPSTEP_ACCOUNT_NAME=PRAC-...
RANGE_EMA_TOPSTEP_ACCOUNT_ID=...
```

Run directly:

```powershell
python -u range_ema_live.py
```

Run with automatic restart:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_range_ema.ps1
```

Discord commands:

- `!stats`
- `!range-ema-stats`
- `!range-ema-health`

## Verification

```powershell
python -m unittest discover -s tests -v
```

Runtime SQLite ledgers, JSONL audits, logs, `.env`, credentials, and account
details are ignored by Git.
