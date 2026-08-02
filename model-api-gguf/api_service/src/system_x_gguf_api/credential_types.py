"""Canonical private API-key identities, parsing, and verifier primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
import secrets
from typing import Literal


AUTHENTICATION_CONTRACT = "system-x.private-authentication.v1"
CREDENTIAL_SCHEMA_IDENTITY = "system-x.api-credentials.v1"
CREDENTIAL_SCHEMA_VERSION = 1
VERIFIER_ALGORITHM = "system-x-hmac-sha256-v1"
VERIFIER_DOMAIN = b"system-x-api-key-v1\0"
KEY_PREFIX = "sxk_v1_"
KEY_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
RAW_KEY_PATTERN = re.compile(
    r"^sxk_v1_(?P<key_id>[0-9a-f]{32})_(?P<secret>[A-Za-z0-9_-]{43})$"
)
DUMMY_VERIFIER = b"\x00" * 32
CredentialStatus = Literal["ACTIVE", "REVOKED"]
CredentialScheme = Literal["bearer", "x-api-key", "dual"]


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    """A syntactically valid key split into non-secret ID and secret material."""

    key_id: str
    raw_key: str


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    """A newly generated key that must remain transient until handoff."""

    key_id: str
    raw_key: str


@dataclass(frozen=True, slots=True)
class CredentialVerification:
    """Internal verification outcome; never carries the supplied raw key."""

    accepted: bool
    reason: str
    key_id: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class AuthenticationContext:
    """Immutable, non-secret identity attached to one authenticated request."""

    request_id: str
    key_id: str
    label: str
    credential_scheme: CredentialScheme
    authenticated: Literal[True] = True


def parse_api_key(raw_key: str) -> ParsedApiKey | None:
    """Strictly parse one complete System X API key."""

    if not isinstance(raw_key, str):
        return None
    match = RAW_KEY_PATTERN.fullmatch(raw_key)
    if match is None:
        return None
    return ParsedApiKey(key_id=match.group("key_id"), raw_key=raw_key)


def generate_api_key() -> GeneratedApiKey:
    """Generate a 128-bit public ID and 256-bit bearer secret."""

    key_id = secrets.token_hex(16)
    secret = secrets.token_urlsafe(32)
    if KEY_ID_PATTERN.fullmatch(key_id) is None:
        raise RuntimeError("generated API-key identity is invalid")
    if SECRET_PATTERN.fullmatch(secret) is None:
        raise RuntimeError("generated API-key secret encoding is invalid")
    raw_key = f"{KEY_PREFIX}{key_id}_{secret}"
    if RAW_KEY_PATTERN.fullmatch(raw_key) is None:
        raise RuntimeError("generated System X API key is invalid")
    return GeneratedApiKey(key_id=key_id, raw_key=raw_key)


def compute_verifier(pepper: bytes, raw_key: str) -> bytes:
    """Compute the versioned HMAC verifier for a complete raw key."""

    if not isinstance(pepper, bytes) or len(pepper) != 32:
        raise ValueError("credential pepper must contain exactly 32 bytes")
    if not isinstance(raw_key, str):
        raise TypeError("raw API key must be text")
    return hmac.new(
        pepper,
        VERIFIER_DOMAIN + raw_key.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def verifier_matches(stored: bytes, candidate: bytes) -> bool:
    """Compare fixed-size verifier bytes using the constant-time primitive."""

    if not isinstance(stored, bytes) or len(stored) != 32:
        raise ValueError("stored credential verifier is invalid")
    if not isinstance(candidate, bytes) or len(candidate) != 32:
        raise ValueError("candidate credential verifier is invalid")
    return hmac.compare_digest(stored, candidate)


def execute_dummy_verification(pepper: bytes, supplied: str) -> None:
    """Execute one HMAC and constant-time compare for an unresolvable key."""

    candidate = compute_verifier(pepper, supplied)
    verifier_matches(DUMMY_VERIFIER, candidate)
