from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Awaitable, Callable

import aiohttp

from shared import DATA_DIR

try:
    import discord
except ImportError:  # pragma: no cover - exercised on minimal test installs
    discord = None

Command = Callable[[list[str]], Awaitable[object]]


def _discord_connector():
    """Avoid aiodns failures seen on the Windows forward-test host."""
    import aiohttp
    return aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())


class Outbox:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS outbox("
            "id INTEGER PRIMARY KEY,ts REAL NOT NULL,content TEXT NOT NULL)"
        )

    def add(self, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO outbox(ts,content) VALUES(?,?)",
            (time.time(), json.dumps(payload)),
        )

    def batch(self, limit: int = 30) -> list[tuple[int, dict]]:
        rows = []
        for row_id, content in self.conn.execute(
            "SELECT id,content FROM outbox ORDER BY id LIMIT ?", (limit,)
        ):
            try:
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise TypeError
            except (json.JSONDecodeError, TypeError):
                payload = {"content": str(content)}
            rows.append((int(row_id), payload))
        return rows

    def delete(self, row_id: int) -> None:
        self.conn.execute("DELETE FROM outbox WHERE id=?", (row_id,))

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    def close(self) -> None:
        self.conn.close()


class SharedDiscordReporter:
    """REST delivery plus a separate gateway connection for commands."""

    def __init__(self, token: str, channel_id: int):
        self.token = token.strip()
        self.channel_id = int(channel_id)
        self.enabled = bool(self.token and self.channel_id)
        self.outbox = Outbox(DATA_DIR / "discord_outbox.db")
        self.commands: dict[str, Command] = {}
        self.client = None
        self.channel = None
        self.http = None
        self.ready = asyncio.Event()
        self.task = None
        self.delivery_task = None
        self.stopping = False

    def register(self, name: str, handler: Command) -> None:
        self.commands[name.lower().lstrip("!")] = handler

    async def start(self) -> None:
        if not self.enabled:
            print("[DISCORD] disabled (token/channel/dependency unavailable)")
            return
        self.http = aiohttp.ClientSession(connector=_discord_connector())
        self.ready.set()
        self.delivery_task = asyncio.create_task(self._delivery_loop())
        if discord is not None:
            self.task = asyncio.create_task(self._connection_loop())

    async def wait_ready(self, timeout: float = 30.0) -> bool:
        if not self.enabled:
            return False
        try:
            await asyncio.wait_for(self._rest_check(), timeout)
            return True
        except (asyncio.TimeoutError, RuntimeError) as exc:
            print(f"[DISCORD] REST access check failed: {exc}")
            return False

    async def stop(self) -> None:
        self.stopping = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.delivery_task:
            self.delivery_task.cancel()
            try:
                await self.delivery_task
            except asyncio.CancelledError:
                pass
        if self.client and not self.client.is_closed():
            await self.client.close()
        if self.http and not self.http.closed:
            await self.http.close()
        self.outbox.close()

    async def post(self, strategy: str, content: str) -> None:
        payload = {"embeds": [self.format_embed(strategy, content)]}
        try:
            await self._rest_send(payload)
        except Exception as exc:
            print(f"[DISCORD] send failed; queued: {exc}")
            self.outbox.add(payload)

    @staticmethod
    def format_embed(strategy: str, content: str) -> dict:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        raw_title = lines[0] if lines else strategy
        title = re.sub(r"\*\*", "", raw_title)
        is_short = "SHORT" in title.upper()
        is_long = "LONG" in title.upper()
        is_loss = "🛑" in title or ("CLOSED" in title and "-$" in content)
        if is_short or is_loss:
            colour = 0xE74C3C
        elif is_long or "✅" in title or "🟢" in title:
            colour = 0x2ECC71
        else:
            colour = 0x3498DB

        fields = []
        description = []
        for line in lines[1:]:
            match = re.match(r"^\*\*(.+?)\*\*\s*·\s*(.*)$", line)
            if not match:
                description.append(line)
                continue
            name, value = match.groups()
            inline = not (
                name.startswith("Why")
                or name in {"Trigger bar", "Break bar", "Range", "Range bar"}
            )
            fields.append({
                "name": name[:256],
                "value": value[:1024] or "—",
                "inline": inline,
            })
        embed = {
            "title": f"{strategy} · {title}"[:256],
            "color": colour,
            "footer": {"text": f"{strategy} · signal-only forward test"},
        }
        if description:
            embed["description"] = "\n".join(description)[:4096]
        if fields:
            embed["fields"] = fields[:25]
        return embed

    async def post_embed(self, embed: dict) -> None:
        payload = {"embeds": [embed]}
        try:
            await self._rest_send(payload)
        except Exception as exc:
            print(f"[DISCORD] embed send failed; queued: {exc}")
            self.outbox.add(payload)

    async def _rest_send(self, payload: dict) -> None:
        if self.http is None or self.http.closed:
            raise RuntimeError("REST session unavailable")
        url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
        headers = {"Authorization": f"Bot {self.token}"}
        for attempt in range(2):
            async with self.http.post(url, headers=headers, json=payload) as response:
                if 200 <= response.status < 300:
                    return
                if response.status == 429 and attempt == 0:
                    payload = await response.json(content_type=None)
                    await asyncio.sleep(float(payload.get("retry_after", 1.0)))
                    continue
                detail = (await response.text())[:300]
                raise RuntimeError(f"HTTP {response.status}: {detail}")

    async def _rest_check(self) -> None:
        if self.http is None or self.http.closed:
            raise RuntimeError("REST session unavailable")
        url = f"https://discord.com/api/v10/channels/{self.channel_id}"
        headers = {"Authorization": f"Bot {self.token}"}
        async with self.http.get(url, headers=headers) as response:
            if response.status == 200:
                return
            detail = (await response.text())[:300]
            raise RuntimeError(f"HTTP {response.status}: {detail}")

    async def _flush(self) -> None:
        for row_id, payload in self.outbox.batch():
            if payload.get("content"):
                payload["content"] = f"**[LATE]** {payload['content']}"
            elif payload.get("embeds"):
                footer = payload["embeds"][0].setdefault("footer", {})
                footer["text"] = f"LATE · {footer.get('text', '')}"[:2048]
            await self._rest_send(payload)
            self.outbox.delete(row_id)

    async def _delivery_loop(self) -> None:
        while not self.stopping:
            try:
                if self.outbox.count():
                    await self._flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[DISCORD] REST backlog retry: {exc}")
            await asyncio.sleep(2.0)

    async def _connect_once(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents, connector=_discord_connector())

        @self.client.event
        async def on_ready():
            channel = self.client.get_channel(self.channel_id)
            if channel is None:
                channel = await self.client.fetch_channel(self.channel_id)
            self.channel = channel
            print(f"[DISCORD] command gateway connected channel={self.channel_id}")

        @self.client.event
        async def on_disconnect():
            self.channel = None

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user or message.channel.id != self.channel_id:
                return
            content = (message.content or "").strip()
            if not content.startswith("!"):
                return
            parts = content[1:].split()
            if not parts:
                return
            handler = self.commands.get(parts[0].lower())
            if handler:
                try:
                    response = await handler(parts[1:])
                    if response:
                        if isinstance(response, dict):
                            await message.channel.send(
                                embed=discord.Embed.from_dict(response)
                            )
                        else:
                            await message.channel.send(str(response))
                except Exception as exc:
                    await message.channel.send(f"Command failed: `{type(exc).__name__}: {exc}`")

        # discord.py 2.6 can dereference a missing websocket while handling an
        # initial Windows connection failure. Let our outer loop reconnect
        # with a fresh client instead of using that broken internal path.
        await self.client.start(self.token, reconnect=False)

    async def _connection_loop(self) -> None:
        backoff = 5.0
        while not self.stopping:
            try:
                await self._connect_once()
                backoff = 5.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.channel = None
                self.ready.clear()
                if self.client and not self.client.is_closed():
                    await self.client.close()
                print(f"[DISCORD] connection failed; retry {backoff:.0f}s: {exc}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 120.0)
