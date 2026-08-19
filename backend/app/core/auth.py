"""Supabase JWT verification: the auth gate on the query endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# How long a fetched key set is trusted before being re-fetched.
_JWKS_TTL_SECONDS = 600

# Floor on time between re-fetches triggered by an unknown `kid`, to prevent amplification.
_JWKS_MIN_REFETCH_INTERVAL_SECONDS = 30

_JWKS_TIMEOUT_SECONDS = 5.0

# Asymmetric only, see the module docstring on algorithm confusion. HMAC (`HS*`) and
# `none` are absent by design, not by omission.
_ALLOWED_ALGORITHMS = frozenset({"ES256", "ES384", "ES512", "RS256", "RS384", "RS512", "EdDSA"})

# Every Supabase user access token carries this audience; the anon key is rejected here.
_EXPECTED_AUDIENCE = "authenticated"

# `auto_error=False` so this class only parses the header; the handler below decides the response.
_bearer_scheme = HTTPBearer(auto_error=False, description="Supabase access token (JWT).")

# Builds the HTTP client used to fetch the key set. Injectable for tests.
ClientFactory = Callable[[], httpx.AsyncClient]


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_JWKS_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class AuthenticatedUser:
    """The verified identity behind a request."""

    id: str
    email: str | None
    role: str | None
    claims: dict[str, Any]


class JWKSCache:
    """Fetches and caches the Supabase project's public signing keys."""

    def __init__(self, jwks_url: str, client_factory: ClientFactory | None = None) -> None:
        self._jwks_url = jwks_url
        # Injected so tests can drive the fetch/parse/cache path with a mock transport.
        self._client_factory = client_factory or _default_client
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str | None) -> jwt.PyJWK:
        """Return the signing key for `kid`, fetching or re-fetching as needed."""
        key = self._lookup(kid)
        if key is not None and not self._is_stale():
            return key

        async with self._lock:
            # Re-check inside the lock: another coroutine may have already refreshed.
            key = self._lookup(kid)
            if key is not None and not self._is_stale():
                return key
            if self._can_refetch():
                await self._fetch()
            key = self._lookup(kid)

        if key is None:
            raise _unauthorized("Token signing key is not recognised by this project.")
        return key

    def _lookup(self, kid: str | None) -> jwt.PyJWK | None:
        if not self._keys:
            return None
        if kid is None:
            # A single-key set makes `kid` optional; with several keys an absent `kid` fails.
            return next(iter(self._keys.values())) if len(self._keys) == 1 else None
        return self._keys.get(kid)

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > _JWKS_TTL_SECONDS

    def _can_refetch(self) -> bool:
        elapsed = time.monotonic() - self._fetched_at
        return not self._keys or elapsed >= _JWKS_MIN_REFETCH_INTERVAL_SECONDS

    async def _fetch(self) -> None:
        try:
            async with self._client_factory() as client:
                response = await client.get(self._jwks_url)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("JWKS fetch failed (%s): %s", self._jwks_url, exc)
            # Keep any previously cached keys; only a cold cache 503s.
            if not self._keys:
                raise _auth_unavailable("Could not fetch the token signing keys.") from exc
            return

        keys = {}
        for key in jwt.PyJWKSet.from_dict(payload).keys:
            if key.key_id and key.algorithm_name in _ALLOWED_ALGORITHMS:
                keys[key.key_id] = key
        if not keys:
            logger.warning("JWKS at %s contained no usable asymmetric keys", self._jwks_url)
            if not self._keys:
                raise _auth_unavailable("Token signing keys are unusable.")
            return

        self._keys = keys
        self._fetched_at = time.monotonic()
        logger.info("fetched %d Supabase signing key(s)", len(keys))


def _unauthorized(detail: str) -> HTTPException:
    """401 with the RFC 6750 challenge header, so a client knows how to retry."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": f'Bearer error="invalid_token", error_description="{detail}"'},
    )


def _auth_unavailable(detail: str) -> HTTPException:
    """503: our side is misconfigured or the auth provider is unreachable."""
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def jwks_url_for(settings: Settings) -> str:
    """Build the project's JWKS URL, or 503 if Supabase is not configured."""
    base = settings.supabase_url.strip().rstrip("/")
    if not base:
        raise _auth_unavailable(
            "Authentication is not configured on this server (SUPABASE_URL is unset)."
        )
    return f"{base}/auth/v1/.well-known/jwks.json"


@lru_cache
def get_jwks_cache() -> JWKSCache:
    """Process-wide cache instance (built once, on first authenticated request)."""
    return JWKSCache(jwks_url_for(get_settings()))


async def require_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> AuthenticatedUser:
    """FastAPI dependency: verify the bearer token, or reject the request."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Send 'Authorization: Bearer <supabase-access-token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()
    issuer = f"{settings.supabase_url.strip().rstrip('/')}/auth/v1"
    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise _unauthorized("Malformed token.") from None

    signing_key = await get_jwks_cache().get_key(header.get("kid"))

    try:
        claims = jwt.decode(
            token,
            key=signing_key.key,
            # Pinned to the fetched key's algorithm, never to the token's `alg` header.
            algorithms=[signing_key.algorithm_name],
            audience=_EXPECTED_AUDIENCE,
            issuer=issuer,
            options={"require": ["exp", "sub", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired.") from None
    except jwt.InvalidAudienceError:
        raise _unauthorized("Token audience is not 'authenticated'.") from None
    except jwt.InvalidIssuerError:
        raise _unauthorized("Token was not issued by this project's Supabase instance.") from None
    except jwt.PyJWTError as exc:
        # Signature failures land here; the message stays generic.
        logger.info("rejected token: %s", exc)
        raise _unauthorized("Token is invalid.") from None

    subject = claims.get("sub")
    if not subject:
        raise _unauthorized("Token has no subject claim.")

    return AuthenticatedUser(
        id=str(subject),
        email=claims.get("email"),
        role=claims.get("role"),
        claims=claims,
    )


# Handlers that need the caller's identity annotate a parameter with this.
CurrentUser = Annotated[AuthenticatedUser, Depends(require_user)]
