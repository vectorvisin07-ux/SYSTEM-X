"""Bounded defensive redaction for authentication-adjacent diagnostics."""

from __future__ import annotations

import re
from typing import Any


MAX_DIAGNOSTIC_CHARACTERS = 4_096
RAW_KEY_PATTERN = re.compile(
    r"sxk_v1_[0-9a-f]{32}_[A-Za-z0-9_-]{43}"
)
AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*)(?:bearer\s+)?[^\s,;]+"
)
API_KEY_HEADER_PATTERN = re.compile(
    r"(?i)(x-api-key\s*:\s*)[^\s,;]+"
)
BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
REDACTED_KEY = "[REDACTED_SYSTEM_X_API_KEY]"
REDACTED_HEADER = "[REDACTED]"


def redact_text(value: Any) -> str:
    """Render bounded text with raw keys and credential values removed."""

    rendered = str(value)
    rendered = RAW_KEY_PATTERN.sub(REDACTED_KEY, rendered)
    rendered = AUTHORIZATION_PATTERN.sub(
        lambda match: match.group(1) + REDACTED_HEADER,
        rendered,
    )
    rendered = API_KEY_HEADER_PATTERN.sub(
        lambda match: match.group(1) + REDACTED_HEADER,
        rendered,
    )
    rendered = BEARER_PATTERN.sub(
        lambda match: match.group(1) + REDACTED_HEADER,
        rendered,
    )
    if len(rendered) > MAX_DIAGNOSTIC_CHARACTERS:
        rendered = rendered[:MAX_DIAGNOSTIC_CHARACTERS] + "[TRUNCATED]"
    return rendered
