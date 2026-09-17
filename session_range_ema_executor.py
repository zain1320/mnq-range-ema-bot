from __future__ import annotations

import os
import time
from typing import Callable, Optional

import httpx
import pandas as pd

from safety import check_safety_gates, response_order_id


STATE_KEY = "__range_ema_mnq_topstep__"


class SessionRangeEmaExecutor:
    """Practice-only MNQ execution synchronized to the tick simulator."""

    def __init__(
        self,
        suite,
        adapter,
        notify: Callable[[str], None],
        *,
        account_id: int,
        account_name: str,
    ):
        self.suite = suite
        self.context = suite["MNQ"]
        self.adapter = adapter
        self.notify = notify
        self.account_id = int(account_id)
        self.account_name = str(account_name)
        self.contract_id = str(self.context.instrument_info.id)
        self.orders = self.context.orders
        self.lock = __import__("asyncio").Lock()
        self.started_at = pd.Timestamp.now(tz="UTC")
        raw = adapter.store.load_day(STATE_KEY)
        self.active_signal_id: Optional[int] = (
            int(raw["active_signal_id"]) if raw.get("active_signal_id") else None
        )
        self.completed_signal_ids = {
            int(value) for value in raw.get("completed_signal_ids", [])
        }
        self.entry_order_id = raw.get("entry_order_id")
        self.stop_order_id = raw.get("stop_order_id")
        self.broker_remaining = int(raw.get("broker_remaining", 0))
        self.applied_target_index = int(raw.get("applied_target_index", 0))
        self.placed = int(raw.get("placed", 0))
        self.failed = int(raw.get("failed", 0))
        self.last_error = raw.get("last_error")
        self.enabled = False

    def _persist(self) -> None:
        self.adapter.store.save_day(STATE_KEY, {
            "active_signal_id": self.active_signal_id,
            "completed_signal_ids": sorted(self.completed_signal_ids)[-100:],
            "entry_order_id": self.entry_order_id,
            "stop_order_id": self.stop_order_id,
            "broker_remaining": self.broker_remaining,
            "applied_target_index": self.applied_target_index,
            "placed": self.placed,
            "failed": self.failed,
            "last_error": self.last_error,
            "updated_at": time.time(),
        })

    def _rest(self) -> httpx.AsyncClient:
        client = self.suite.client
        base = (
            getattr(client, "base_url", None)
            or getattr(client, "_base_url", None)
            or "https://api.topstepx.com/api"
        )
        token = (
            getattr(client, "session_token", None)
            or getattr(client, "_session_token", None)
            or getattr(client, "token", None)
            or getattr(client, "_token", None)
            or getattr(client, "jwt", None)
            or getattr(client, "_jwt", None)
        )
        return httpx.AsyncClient(
            base_url=base,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )

    async def initialize(self) -> None:
        gate_ok, gate_reason = check_safety_gates()
        if not gate_ok:
            raise RuntimeError(gate_reason)
        if not self.account_name.upper().startswith("PRAC"):
            raise RuntimeError(
                f"practice-only guard rejected account name {self.account_name!r}"
            )
        if int(self.adapter.cfg.contracts) != 10:
            raise RuntimeError("practice executor is hard-limited to exactly 10 MNQ")
        env_id = (
            os.getenv("TOPSTEP_ACCOUNT_ID")
            or os.getenv("PROJECTX_ACCOUNT_ID")
            or os.getenv("PROJECT_X_ACCOUNT_ID")
        )
        if not env_id or int(env_id) != self.account_id:
            raise RuntimeError(
                f"bound account {env_id or 'missing'} does not match allowed "
                f"practice account {self.account_id}"
            )
        async with self._rest() as client:
            response = await client.post("/Account/search", json={})
            response.raise_for_status()
            payload = response.json()
        accounts = payload if isinstance(payload, list) else payload.get("accounts", [])
        account = next(
            (item for item in accounts if int(item.get("id", 0)) == self.account_id),
            None,
        )
        if account is None:
            raise RuntimeError("allowed practice account was not returned by Topstep")
        api_name = str(account.get("name") or "")
        if api_name != self.account_name or not bool(account.get("canTrade", True)):
            raise RuntimeError(
                f"practice account validation failed for {self.account_id}"
            )
        self.enabled = True
        await self._reconcile()
        self.notify(
            f"🔫 **TOPSTEP PRACTICE EXECUTION ARMED**\n"
            f"**Account** · `{self.account_name}` (`{self.account_id}`)\n"
            "**Instrument** · `MNQ` · **Size** · `10 contracts`\n"
            "**Execution** · market entry only after live EMA-touch fill; "
            "broker stop placed immediately; 5/3/2 scale-outs follow live TP touches"
        )
        self.adapter.audit.event(
            "broker_armed",
            mode="TOPSTEP_PRAC",
            account_id=self.account_id,
            account_name=self.account_name,
            contract_id=self.contract_id,
        )
        print(
            f"[RANGE-EMA-EXEC] armed account={self.account_name} "
            f"id={self.account_id} contract={self.contract_id}"
        )

    async def _reconcile(self) -> None:
        lifecycle = self.adapter.lifecycle
        if self.active_signal_id is None:
            return
        if lifecycle is None or lifecycle.signal_id != self.active_signal_id:
            await self._flatten_locked("startup_reconcile")
            return
        position = await self.position_size()
        if position == 0:
            signal_id = self.active_signal_id
            self.completed_signal_ids.add(signal_id)
            self.active_signal_id = None
            self._clear_order_ids()
            self._persist()
            self.notify(
                f"⚠️ **TOPSTEP RECONCILE #{signal_id}** · broker is flat; "
                "the persisted simulated trade will not be re-entered"
            )

    async def sync(self) -> None:
        if not self.enabled:
            return
        async with self.lock:
            lifecycle = self.adapter.lifecycle
            if lifecycle is None:
                if self.active_signal_id is not None:
                    await self._flatten_locked("simulated_exit")
                return
            signal_id = int(lifecycle.signal_id)
            if signal_id == self.active_signal_id:
                await self._sync_partial_targets(lifecycle)
                return
            if signal_id in self.completed_signal_ids:
                return
            if pd.Timestamp(lifecycle.entry_ts).tz_convert("UTC") < self.started_at:
                self.completed_signal_ids.add(signal_id)
                self._persist()
                self.notify(
                    f"ℹ️ **TOPSTEP SKIP #{signal_id}** · simulated position "
                    "predates executor startup; no retroactive broker entry"
                )
                return
            await self._place(lifecycle)

    async def _place(self, lifecycle) -> None:
        signal_id = int(lifecycle.signal_id)
        direction = 1 if lifecycle.side == "long" else -1
        side_entry = 0 if direction == 1 else 1
        side_exit = 1 if direction == 1 else 0
        try:
            if await self.position_size() != 0:
                raise RuntimeError("MNQ broker position is not flat")
            if await self.open_order_count() != 0:
                raise RuntimeError("MNQ broker has existing open orders")
            entry = await self.orders.place_market_order(
                contract_id=self.contract_id,
                side=side_entry,
                size=10,
                account_id=self.account_id,
            )
            self.entry_order_id = response_order_id(entry)
            stop = await self.orders.place_stop_order(
                contract_id=self.contract_id,
                side=side_exit,
                size=10,
                stop_price=float(lifecycle.stop),
                account_id=self.account_id,
            )
            self.stop_order_id = response_order_id(stop)
            self.active_signal_id = signal_id
            self.broker_remaining = 10
            self.applied_target_index = 0
            self.placed += 1
            self.last_error = None
            self._persist()
            self.notify(
                f"📤 **TOPSTEP PRAC ORDER #{signal_id} | "
                f"{lifecycle.side.upper()} MNQ ×10**\n"
                f"**Account** · `{self.account_name}`\n"
                f"**Entry** · market after live EMA touch · "
                f"**Stop** · `{lifecycle.stop:.2f}`\n"
                f"**Targets** · `{' / '.join(f'{value:.2f}' for value in lifecycle.targets)}` "
                "(5/3/2)\n"
                f"**Broker IDs** · entry `{self.entry_order_id}` · "
                f"stop `{self.stop_order_id}`"
            )
            self.adapter.audit.event(
                "broker_entry",
                mode="TOPSTEP_PRAC",
                signal_id=signal_id,
                side=lifecycle.side,
                size=10,
                account_id=self.account_id,
                contract_id=self.contract_id,
                entry_order_id=self.entry_order_id,
                stop_order_id=self.stop_order_id,
                stop=lifecycle.stop,
                targets=lifecycle.targets,
            )
        except Exception as exc:
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:240]
            self._persist()
            await self._flatten_locked(f"placement_failure#{signal_id}", force=True)
            self.completed_signal_ids.add(signal_id)
            self._persist()
            self.notify(
                f"❌ **TOPSTEP ORDER FAILED #{signal_id}** · `{self.last_error}` · "
                "cleanup/flatten attempted; no retry for this signal"
            )
            self.adapter.audit.event(
                "broker_rejected",
                mode="TOPSTEP_PRAC",
                signal_id=signal_id,
                error=self.last_error,
            )

    async def _sync_partial_targets(self, lifecycle) -> None:
        desired = int(lifecycle.contracts_remaining)
        if desired >= self.broker_remaining:
            return
        qty = self.broker_remaining - desired
        side_exit = 1 if lifecycle.side == "long" else 0
        try:
            scale = await self.orders.place_market_order(
                contract_id=self.contract_id,
                side=side_exit,
                size=qty,
                account_id=self.account_id,
            )
            scale_order_id = response_order_id(scale)
            await self._cancel_stop()
            self.stop_order_id = None
            if desired:
                stop = await self.orders.place_stop_order(
                    contract_id=self.contract_id,
                    side=side_exit,
                    size=desired,
                    stop_price=float(lifecycle.stop),
                    account_id=self.account_id,
                )
                self.stop_order_id = response_order_id(stop)
            self.broker_remaining = desired
            self.applied_target_index = int(lifecycle.next_target_index)
            self._persist()
            self.notify(
                f"📉 **TOPSTEP PRAC SCALE-OUT #{lifecycle.signal_id}** · "
                f"`{qty}` MNQ market exit · `{desired}` remain · "
                f"SL `{lifecycle.stop:.2f}` · order `{scale_order_id}`"
            )
            self.adapter.audit.event(
                "broker_scale_out",
                mode="TOPSTEP_PRAC",
                signal_id=lifecycle.signal_id,
                size=qty,
                remaining=desired,
                stop=lifecycle.stop,
                scale_order_id=scale_order_id,
                stop_order_id=self.stop_order_id,
            )
        except Exception as exc:
            self.failed += 1
            self.last_error = (
                f"scale-out #{lifecycle.signal_id}: {type(exc).__name__}: {exc}"
            )[:240]
            self._persist()
            await self._flatten_locked(
                f"scaleout_failure#{lifecycle.signal_id}", force=True
            )
            self.notify(
                f"❌ **TOPSTEP SCALE-OUT FAILED #{lifecycle.signal_id}** · "
                f"`{self.last_error}` · safety flatten attempted"
            )

    async def _cancel_stop(self) -> None:
        if self.stop_order_id is None:
            return
        async with self._rest() as client:
            response = await client.post(
                "/Order/cancel",
                json={
                    "accountId": self.account_id,
                    "orderId": self.stop_order_id,
                },
            )
            response.raise_for_status()

    async def position_size(self) -> int:
        async with self._rest() as client:
            response = await client.post(
                "/Position/searchOpen", json={"accountId": self.account_id}
            )
            response.raise_for_status()
            payload = response.json()
        size = 0
        for position in payload.get("positions", []):
            if position.get("contractId") != self.contract_id:
                continue
            amount = int(position.get("size") or 0)
            size += amount if int(position.get("type") or 0) == 1 else -amount
        return size

    async def open_order_count(self) -> int:
        orders = await self._open_orders()
        return len(orders)

    async def _open_orders(self) -> list[dict]:
        async with self._rest() as client:
            response = await client.post(
                "/Order/searchOpen", json={"accountId": self.account_id}
            )
            response.raise_for_status()
            payload = response.json()
        return [
            order for order in payload.get("orders", [])
            if order.get("contractId") == self.contract_id
        ]

    async def flatten(self, reason: str) -> None:
        async with self.lock:
            await self._flatten_locked(reason)

    async def _flatten_locked(self, reason: str, *, force: bool = False) -> None:
        signal_id = self.active_signal_id
        if signal_id is None and not force:
            return
        try:
            async with self._rest() as client:
                for order in await self._open_orders():
                    response = await client.post(
                        "/Order/cancel",
                        json={"accountId": self.account_id, "orderId": order["id"]},
                    )
                    response.raise_for_status()
                if await self.position_size() != 0:
                    response = await client.post(
                        "/Position/closeContract",
                        json={
                            "accountId": self.account_id,
                            "contractId": self.contract_id,
                        },
                    )
                    response.raise_for_status()
        except Exception as exc:
            self.last_error = f"flatten {reason}: {type(exc).__name__}: {exc}"[:240]
            print(f"[RANGE-EMA-EXEC] {self.last_error}")
        if signal_id is not None:
            self.completed_signal_ids.add(signal_id)
        self.active_signal_id = None
        self._clear_order_ids()
        self._persist()
        if signal_id is not None:
            self.notify(
                f"🧹 **TOPSTEP PRAC FLATTEN #{signal_id}** · `{reason}` · "
                "MNQ orders cancelled and broker position close requested"
            )
            self.adapter.audit.event(
                "broker_flatten",
                mode="TOPSTEP_PRAC",
                signal_id=signal_id,
                reason=reason,
                error=self.last_error,
            )

    def _clear_order_ids(self) -> None:
        self.entry_order_id = None
        self.stop_order_id = None
        self.broker_remaining = 0
        self.applied_target_index = 0

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "account": self.account_name,
            "account_id": self.account_id,
            "contract_id": self.contract_id,
            "active_signal_id": self.active_signal_id,
            "placed": self.placed,
            "failed": self.failed,
            "last_error": self.last_error,
        }
