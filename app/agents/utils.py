from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional


def _try_load_json(candidate: str) -> Optional[Any]:
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _extract_fenced_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _iter_json_candidates(text: str) -> Iterable[str]:
    for start_idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        stack = [ch]
        for idx in range(start_idx + 1, len(text)):
            current = text[idx]
            if current in "{[":
                stack.append(current)
            elif current in "}]":
                if not stack:
                    break
                opening = stack.pop()
                if opening == "{" and current != "}":
                    break
                if opening == "[" and current != "]":
                    break
                if not stack:
                    yield text[start_idx : idx + 1]
                    break


def extract_json(text: str) -> Any:
    if not text:
        raise ValueError("empty response")
    stripped = text.strip()
    direct = _try_load_json(stripped)
    if direct is not None:
        return direct
    fenced = _extract_fenced_block(stripped)
    if fenced:
        fenced_obj = _try_load_json(fenced)
        if fenced_obj is not None:
            return fenced_obj
    for candidate in _iter_json_candidates(stripped):
        obj = _try_load_json(candidate)
        if obj is not None:
            return obj
    raise ValueError("no valid json found")
