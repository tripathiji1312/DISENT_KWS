"""Uvicorn entry point: `uvicorn app.main:app --workers 1`."""

from __future__ import annotations

from app.server import create_app

app = create_app()
