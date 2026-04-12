"""
parser.py
---------
Extract the observation dict from an incoming A2A message.

The MAizeBargAIn green agent sends observation payloads like:
  {
    "role": "row",
    "round": 2,
    "valuations": [45, 72, 33],
    "batna": 85,
    "quantities": [7, 4, 1],
    "last_offer": [3, 2, 0],
    "history": [...]
  }

These can arrive as:
  1. Raw JSON as the full message body.
  2. JSON embedded within text prose / markdown fences.
  3. Some other framing the green agent uses for its "circle" prompts.

We also format our outgoing action into the wire format the green agent expects:
  {"action": "COUNTEROFFER", "offer": [4, 2, 1]}
  {"action": "ACCEPT"}
  {"action": "WALK"}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_incoming(raw_text: str) -> dict[str, Any]:
    """
    Best-effort extraction of an observation dict from incoming text.
    Returns {} if nothing parseable is found — the caller should handle that
    by still producing a valid action (likely a safe default COUNTEROFFER).
    """
    if not raw_text or not raw_text.strip():
        return {}

    text = raw_text.strip()

    # Fast path: whole message is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _normalize(obj)
    except json.JSONDecodeError:
        pass

    # Strip common markdown fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    if fenced != text:
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return _normalize(obj)
        except json.JSONDecodeError:
            pass

    # Scan for embedded JSON object(s) and take the largest one that parses
    best: dict | None = None
    best_len = 0
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and len(match.group(0)) > best_len:
            best = obj
            best_len = len(match.group(0))
    if best is not None:
        return _normalize(best)

    logger.warning("Could not parse observation from message; got: %s", text[:200])
    return {}


def _normalize(obj: dict) -> dict:
    """
    If the dict nests the observation under keys like 'observation' or 'state',
    unwrap it. Also accept alias keys.
    """
    # Unwrap known container keys
    for wrapper in ("observation", "state", "obs", "payload"):
        inner = obj.get(wrapper)
        if isinstance(inner, dict):
            merged = {**obj, **inner}
            merged.pop(wrapper, None)
            obj = merged

    # Normalize alias keys
    if "my_valuations" in obj and "valuations" not in obj:
        obj["valuations"] = obj["my_valuations"]
    if "outside_option" in obj and "batna" not in obj:
        obj["batna"] = obj["outside_option"]
    if "last_offer" not in obj and "latest_offer" in obj:
        obj["last_offer"] = obj["latest_offer"]

    return obj


def format_action(action: dict) -> str:
    """
    Serialize our action for the wire. We emit a single-line JSON object.
    """
    act = str(action.get("action", "")).upper()
    if act == "COUNTEROFFER":
        offer = action.get("offer", [])
        return json.dumps({"action": "COUNTEROFFER", "offer": list(offer)})
    if act == "ACCEPT":
        return json.dumps({"action": "ACCEPT"})
    if act == "WALK":
        return json.dumps({"action": "WALK"})
    # Fallback: serialize whatever we were given
    return json.dumps(action)


def parse_llm_action(text: str) -> dict | None:
    """Extract an action dict from the LLM's raw response."""
    if not text:
        return None

    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate).strip()

    # First try to parse the whole thing
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Scan for JSON blocks
    for match in _JSON_BLOCK_RE.finditer(candidate):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj

    return None
