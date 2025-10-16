"""Compatibility helpers for normalising study names."""

from __future__ import annotations

try:  # pragma: no cover - passthrough when renamer is available
    from .schema_renamer import normalize_study_name
except Exception:  # pragma: no cover - optional dependency missing
    import re

    _WORD_RE = re.compile(r"[0-9A-Za-z]+")

    def normalize_study_name(raw: str) -> str:
        text = "" if raw is None else str(raw)
        text = text.strip()
        if not text:
            return ""

        pieces: list[str] = []
        last_word: str | None = None
        cursor = 0

        for match in _WORD_RE.finditer(text):
            start, end = match.span()

            if start > cursor:
                separator = text[cursor:start]
                if separator:
                    pieces.append(separator)

            word = match.group(0)
            lowered = word.lower()

            if lowered != last_word:
                pieces.append(word)
                last_word = lowered
            else:
                if start > cursor and pieces:
                    separator = text[cursor:start]
                    if pieces[-1] == separator:
                        pieces.pop()

            cursor = end

        if cursor < len(text):
            pieces.append(text[cursor:])

        return "".join(pieces).strip()


__all__ = ["normalize_study_name"]

