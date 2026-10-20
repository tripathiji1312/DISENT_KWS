"""Per-person actions. Every action is best-effort and never raises into the WS loop."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import Any, Callable, Mapping

import httpx

from app.registry import ActionConfig

log = logging.getLogger(__name__)

ALLOWED_SHELL = {"notify-send", "paplay", "aplay"}


def render_message(template: str, *, name: str, keyword: str) -> str:
    try:
        return str(template).format(name=name, keyword=keyword)
    except (KeyError, IndexError, ValueError, AttributeError):
        return f"{name} said {keyword}"


def _coerce(action: ActionConfig | Mapping[str, Any]) -> ActionConfig:
    if isinstance(action, ActionConfig):
        return action
    return ActionConfig.model_validate(dict(action))


async def execute(
    action: ActionConfig | Mapping[str, Any],
    *,
    name: str,
    keyword: str,
    decision: Mapping[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    try:
        config = _coerce(action)
    except Exception as exc:
        return {"type": "invalid", "status": "error", "error": str(exc)}

    if not config.enabled:
        return {"type": config.type, "status": "disabled"}

    text = render_message(config.message_template, name=name, keyword=keyword)

    if config.type == "announce":
        return {"type": "announce", "status": "ok", "text": text}

    if config.type == "smart_home":
        device = getattr(config, "smart_device", None) or "auto"
        return {"type": "smart_home", "status": "ok", "device": device, "text": text}

    if config.type == "log":
        log.info("ACTION log person=%s keyword=%s text=%s", name, keyword, text)
        return {"type": "log", "status": "ok", "text": text}

    if config.type == "webhook":
        payload = {
            "name": name,
            "keyword": keyword,
            "text": text,
            "event": dict(decision or {}),
        }
        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=3.0)
        try:
            response = await http.post(str(config.webhook_url), json=payload)
            return {
                "type": "webhook",
                "status": "ok" if response.is_success else "error",
                "status_code": response.status_code,
                "body": response.text[:300],
                "text": text,
            }
        except Exception as exc:
            log.warning("webhook action failed: %s", exc)
            return {"type": "webhook", "status": "error", "error": str(exc), "text": text}
        finally:
            if owns_client:
                await http.aclose()

    if config.type == "shell":
        try:
            argv = shlex.split(str(config.shell_command or ""))
        except ValueError as exc:
            return {"type": "shell", "status": "error", "error": f"invalid command: {exc}"}
        if not argv:
            return {"type": "shell", "status": "error", "error": "empty command"}
        executable = argv[0].rsplit("/", 1)[-1]
        if executable not in ALLOWED_SHELL:
            return {
                "type": "shell",
                "status": "error",
                "error": f"executable '{executable}' is not allowed (allowlist: {sorted(ALLOWED_SHELL)})",
            }
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return {"type": "shell", "status": "timeout", "error": "command timed out after 3s"}
            return {
                "type": "shell",
                "status": "ok" if process.returncode == 0 else "error",
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", "replace")[:300],
                "stderr": stderr.decode("utf-8", "replace")[:300],
            }
        except Exception as exc:
            log.warning("shell action failed: %s", exc)
            return {"type": "shell", "status": "error", "error": str(exc), "text": text}

    return {"type": config.type, "status": "error", "error": "unknown action type"}


class ActionRunner:
    """Cooldown-aware wrapper so one utterance cannot fire twice in a row."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._last: dict[str, float] = {}
        self._clock = clock or time.monotonic

    def reset(self) -> None:
        self._last.clear()

    async def run(
        self,
        action: ActionConfig | Mapping[str, Any],
        *,
        person_id: str | None,
        name: str,
        keyword: str,
        decision: Mapping[str, Any] | None = None,
        cooldown_sec: float = 0.0,
        enabled: bool = True,
    ) -> dict:
        config = _coerce(action)
        if not enabled:
            return {"type": config.type, "status": "disabled", "reason": "actions disabled"}
        if person_id and cooldown_sec > 0:
            now = self._clock()
            last = self._last.get(person_id)
            if last is not None and now - last < cooldown_sec:
                return {
                    "type": config.type,
                    "status": "cooldown",
                    "cooldown": True,
                    "remaining_sec": round(cooldown_sec - (now - last), 2),
                }
            self._last[person_id] = now
        return await execute(action, name=name, keyword=keyword, decision=decision)
