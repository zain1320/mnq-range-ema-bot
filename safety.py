from __future__ import annotations

import os
from typing import Optional


def check_safety_gates() -> tuple[bool, str]:
    """Allow order routing only in explicitly configured paper mode."""
    paper = os.getenv("PAPER_MODE", "").strip().lower()
    live = os.getenv("LIVE_MODE", "").strip().lower()
    if live == "true":
        return False, "LIVE_MODE=true detected; funded execution is blocked"
    if paper != "true":
        return False, "PAPER_MODE=true is required before sending orders"
    return True, "PAPER_MODE=true; LIVE_MODE off"


def response_order_id(response) -> Optional[str]:
    """Best-effort extraction of an order ID from project-x-py responses."""
    if response is None:
        return None
    for attribute in ("order_id", "id", "orderId"):
        value = getattr(response, attribute, None)
        if value is not None:
            return str(value)
    if isinstance(response, dict):
        for key in ("order_id", "id", "orderId"):
            if key in response:
                return str(response[key])
    return None
