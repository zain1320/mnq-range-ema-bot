from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from session_range_ema_executor import SessionRangeEmaExecutor


class StopUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_ema_cross_replaces_stop_without_closing_position(self):
        saved = {}
        orders = SimpleNamespace(
            place_market_order=AsyncMock(),
            place_stop_order=AsyncMock(return_value={"orderId": 99}),
        )
        executor = SessionRangeEmaExecutor.__new__(SessionRangeEmaExecutor)
        executor.orders = orders
        executor.contract_id = "CON.F.US.MNQ.Z26"
        executor.account_id = 123
        executor.stop_order_id = "88"
        executor.broker_remaining = 10
        executor.applied_target_index = 0
        executor.applied_stop = 95.0
        executor.failed = 0
        executor.last_error = None
        executor.active_signal_id = 7
        executor.completed_signal_ids = set()
        executor.notify = lambda _: None
        executor._cancel_stop = AsyncMock()
        executor._persist = lambda: saved.update({
            "remaining": executor.broker_remaining,
            "stop": executor.applied_stop,
        })
        executor._flatten_locked = AsyncMock()
        executor.adapter = SimpleNamespace(
            audit=SimpleNamespace(event=lambda *args, **kwargs: None)
        )
        lifecycle = SimpleNamespace(
            signal_id=7,
            side="long",
            stop=99.0,
            contracts_remaining=10,
            next_target_index=0,
        )

        await executor._sync_partial_targets(lifecycle)

        orders.place_market_order.assert_not_awaited()
        executor._cancel_stop.assert_awaited_once()
        orders.place_stop_order.assert_awaited_once_with(
            contract_id="CON.F.US.MNQ.Z26",
            side=1,
            size=10,
            stop_price=99.0,
            account_id=123,
        )
        self.assertEqual(saved, {"remaining": 10, "stop": 99.0})


if __name__ == "__main__":
    unittest.main()
