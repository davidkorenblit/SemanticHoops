"""
main.py — top-level entry point.

Uvicorn target: `uvicorn main:app --reload`
(Delegates directly to the API package so the ASGI app is importable from both
the root and from `src.api.main`.)
"""
from src.api.main import app  # noqa: F401  (re-export)

__all__ = ["app"]
