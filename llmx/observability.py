from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from langfuse import observe, get_client as _get_client
    _enabled = True
except ImportError:
    _enabled = False
    _get_client = None

    def observe(func=None, *, name=None, as_type=None, **_kwargs):
        def decorator(fn):
            return fn
        return decorator if func is None else func


def lf():
    """Return the Langfuse client, or None if not available."""
    if not _enabled:
        return None
    try:
        return _get_client()
    except Exception:
        return None
