"""Utilities for recording fact-flow diagnostics across the workflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRACE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "fact_flow_trace.json"
_MAX_TRACE_ENTRIES = 100


def _coerce_jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(item) for item in value]
    return str(value)


def record_fact_flow_event(
    stage: str,
    stats: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a structured entry describing fact availability at a workflow stage."""
    entry = {
        "stage": stage,
        "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "stats": _coerce_jsonable(stats or {}),
    }
    if extra:
        entry["extra"] = _coerce_jsonable(extra)

    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive filesystem guard
        logger.debug("Unable to create fact-flow trace directory: %s", exc)
        return

    entries: List[Dict[str, Any]] = []
    if _TRACE_PATH.exists():
        try:
            with _TRACE_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, list):
                    entries = loaded[-_MAX_TRACE_ENTRIES:]
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.debug("Unable to read fact-flow trace file: %s", exc)

    entries.append(entry)
    if len(entries) > _MAX_TRACE_ENTRIES:
        entries = entries[-_MAX_TRACE_ENTRIES:]

    try:
        with _TRACE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.debug("Unable to write fact-flow trace file: %s", exc)


__all__ = ["record_fact_flow_event"]

