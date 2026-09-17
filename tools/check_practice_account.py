from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx


HERE = Path(__file__).resolve()
BOT_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for path in (BOT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared import load_env_file


async def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    load_env_file(BOT_ROOT / ".env.forward-test", override=True)
    account_id = int(os.getenv("RANGE_EMA_TOPSTEP_ACCOUNT_ID", "0"))
    account_name = os.getenv("RANGE_EMA_TOPSTEP_ACCOUNT_NAME", "")
    if not account_id or not account_name.upper().startswith("PRAC"):
        print("practice_account_config=invalid")
        return 1
    os.environ["PROJECT_X_API_KEY"] = (
        os.getenv("TOPSTEP_API_KEY") or os.getenv("PROJECTX_API_KEY", "")
    )
    os.environ["PROJECT_X_USERNAME"] = (
        os.getenv("TOPSTEP_USERNAME") or os.getenv("PROJECTX_USERNAME", "")
    )
    os.environ["PROJECT_X_ACCOUNT_NAME"] = account_name
    os.environ["PROJECT_X_ACCOUNT_ID"] = str(account_id)

    from project_x_py import ProjectX

    async with ProjectX.from_env(account_name=account_name) as client:
        await client.authenticate()
        token = (
            getattr(client, "session_token", None)
            or getattr(client, "_session_token", None)
            or getattr(client, "token", None)
            or getattr(client, "_token", None)
        )
        async with httpx.AsyncClient(
            base_url="https://api.topstepx.com/api",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        ) as http:
            account_response = await http.post("/Account/search", json={})
            positions_response = await http.post(
                "/Position/searchOpen", json={"accountId": account_id}
            )
            orders_response = await http.post(
                "/Order/searchOpen", json={"accountId": account_id}
            )
            account_response.raise_for_status()
            positions_response.raise_for_status()
            orders_response.raise_for_status()

    raw_accounts = account_response.json()
    accounts = (
        raw_accounts
        if isinstance(raw_accounts, list)
        else raw_accounts.get("accounts", [])
    )
    account = next(
        (item for item in accounts if int(item.get("id", 0)) == account_id),
        None,
    )
    positions = positions_response.json().get("positions", [])
    orders = orders_response.json().get("orders", [])
    mnq_positions = [
        item for item in positions
        if "MNQ" in str(item.get("contractId", ""))
    ]
    mnq_orders = [
        item for item in orders
        if "MNQ" in str(item.get("contractId", ""))
    ]
    valid = bool(
        account
        and account.get("name") == account_name
        and account_name.upper().startswith("PRAC")
        and bool(account.get("canTrade", True))
    )
    print(f"practice_account_valid={str(valid).lower()}")
    print(f"account_name={account_name}")
    print(f"account_id={account_id}")
    print(f"all_open_positions={len(positions)}")
    print(f"all_open_orders={len(orders)}")
    print(f"mnq_open_positions={len(mnq_positions)}")
    print(f"mnq_open_orders={len(mnq_orders)}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
