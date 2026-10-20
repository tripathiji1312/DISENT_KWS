"""Action rendering, webhook safety, shell allowlist, and cooldown."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app import actions
from app.actions import ActionRunner, execute, render_message
from app.registry import ActionConfig


def test_render_message_and_announce():
    assert render_message("{name} said {keyword}", name="Alice", keyword="stop") == "Alice said stop"
    assert render_message("{bad}", name="Alice", keyword="stop") == "Alice said stop"
    result = asyncio.run(
        execute(ActionConfig(type="announce"), name="Alice", keyword="stop")
    )
    assert result == {"type": "announce", "status": "ok", "text": "Alice said stop"}


def test_webhook_mock_transport():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/hook"
        assert b"Alice" in request.content
        return httpx.Response(200, text="ok")

    async def run() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute(
                ActionConfig(type="webhook", webhook_url="https://example.test/hook"),
                name="Alice",
                keyword="stop",
                decision={"type": "keyword"},
                client=client,
            )

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert result["status_code"] == 200


def test_webhook_timeout_does_not_raise():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    async def run() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute(
                ActionConfig(type="webhook", webhook_url="https://example.test/hook"),
                name="Bob",
                keyword="yes",
                client=client,
            )

    result = asyncio.run(run())
    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_shell_allowlist_rejects_dangerous_command():
    result = asyncio.run(
        execute(ActionConfig(type="shell", shell_command="rm -rf /"), name="Alice", keyword="stop")
    )
    assert result["status"] == "error"
    assert "not allowed" in result["error"]


def test_shell_allowed_command(monkeypatch):
    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    calls = []

    async def fake_exec(*argv, **kwargs):
        calls.append(argv)
        return FakeProcess()

    monkeypatch.setattr(actions.asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(
        execute(
            ActionConfig(type="shell", shell_command="notify-send 'Alice said stop'"),
            name="Alice",
            keyword="stop",
        )
    )
    assert result["status"] == "ok"
    assert calls == [("notify-send", "Alice said stop")]


def test_action_runner_cooldown():
    now = [100.0]
    runner = ActionRunner(clock=lambda: now[0])

    async def run() -> tuple[dict, dict, dict]:
        action = ActionConfig(type="announce")
        first = await runner.run(
            action, person_id="p1", name="Alice", keyword="stop", cooldown_sec=2.0
        )
        second = await runner.run(
            action, person_id="p1", name="Alice", keyword="stop", cooldown_sec=2.0
        )
        now[0] += 2.1
        third = await runner.run(
            action, person_id="p1", name="Alice", keyword="stop", cooldown_sec=2.0
        )
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first["status"] == "ok"
    assert second["status"] == "cooldown"
    assert third["status"] == "ok"


def test_disabled_action():
    result = asyncio.run(
        execute(
            ActionConfig(type="announce", enabled=False), name="Alice", keyword="stop"
        )
    )
    assert result["status"] == "disabled"


def test_smart_home_action():
    result = asyncio.run(
        execute(
            ActionConfig(type="smart_home", smart_device="lights"),
            name="Alice",
            keyword="lights on",
        )
    )
    assert result == {
        "type": "smart_home",
        "status": "ok",
        "device": "lights",
        "text": "Alice said lights on",
    }

