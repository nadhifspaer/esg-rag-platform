"""Tests for Supabase JWT verification against a mocked JWKS, including algorithm confusion."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.core import auth as auth_module
from app.core.auth import JWKSCache, require_user
from app.main import app

client = TestClient(app)

PROJECT_URL = "https://test-project.supabase.co"
ISSUER = f"{PROJECT_URL}/auth/v1"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
KID = "test-key-0001"
USER_ID = "11111111-2222-4333-8444-555555555555"

# Routes that are public on purpose; everything else must be behind the gate.
PUBLIC_PATHS = {"/health", "/documents"}


# --- key material and token minting -----------------------------------------


def _keypair() -> tuple[ec.EllipticCurvePrivateKey, dict]:
    """Generate an ES256 keypair and the JWKS dict advertising its public half."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": KID, "use": "sig", "alg": "ES256"})
    return private_key, {"keys": [jwk]}


PRIVATE_KEY, JWKS = _keypair()
PRIVATE_PEM = PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
PUBLIC_PEM = PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)


def _mint(
    *,
    issuer: str = ISSUER,
    audience: str = "authenticated",
    expires_in: int = 3600,
    kid: str | None = KID,
    key: bytes | None = None,
    subject: str | None = USER_ID,
) -> str:
    """Sign an access token shaped like Supabase's, with any field made wrong on request."""
    now = int(time.time())
    claims: dict = {"iss": issuer, "aud": audience, "iat": now, "exp": now + expires_in}
    if subject is not None:
        claims["sub"] = subject
    claims.update({"email": "real-user@example.com", "role": "authenticated"})
    return jwt.encode(claims, key or PRIVATE_PEM, algorithm="ES256", headers={"kid": kid})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- wiring -----------------------------------------------------------------


def _jwks_transport(payload: dict | None = None, status_code: int = 200) -> tuple:
    """A mock transport serving the JWKS, plus a mutable call counter."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status_code != 200:
            return httpx.Response(status_code, text="nope")
        return httpx.Response(200, json=payload if payload is not None else JWKS)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory, calls


@pytest.fixture(autouse=True)
def _real_auth(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo conftest's stub user: this module tests the gate, not what is behind it."""
    app.dependency_overrides.pop(require_user, None)
    factory, _ = _jwks_transport()
    monkeypatch.setattr(auth_module, "get_jwks_cache", lambda: JWKSCache(JWKS_URL, factory))
    monkeypatch.setattr(
        auth_module, "get_settings", lambda: SimpleNamespace(supabase_url=PROJECT_URL)
    )
    yield


def _verify(token: str) -> auth_module.AuthenticatedUser:
    """Run the dependency directly, so identity can be asserted without a live engine."""
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return asyncio.run(require_user(creds))


# --- the accepted case ------------------------------------------------------


def test_valid_token_is_accepted_and_carries_the_user_identity() -> None:
    user = _verify(_mint())
    assert user.id == USER_ID
    assert user.email == "real-user@example.com"
    assert user.role == "authenticated"
    assert user.claims["iss"] == ISSUER


# --- rejections -------------------------------------------------------------


def test_missing_token_is_401_with_a_bearer_challenge() -> None:
    response = client.post("/chat", json={"query": "scope 1 emissions"})
    assert response.status_code == 401
    assert "Missing bearer token" in response.json()["detail"]
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_malformed_token_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        _verify("not-a-jwt")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Malformed token."


def test_expired_token_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(expires_in=-60))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Token has expired."


def test_token_signed_by_a_foreign_key_is_401() -> None:
    """A well-formed token from someone else's keypair: the plain forgery case."""
    other_key = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(key=other_key))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Token is invalid."


def test_hs256_forgery_with_the_public_key_is_rejected() -> None:
    """Algorithm confusion: sign with the *public* key as an HMAC secret, claim HS256."""

    def b64(raw: bytes) -> bytes:
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    now = int(time.time())
    header = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": KID}).encode())
    claims = {
        "iss": ISSUER,
        "aud": "authenticated",
        "sub": "attacker",
        "role": "service_role",
        "iat": now,
        "exp": now + 3600,
    }
    signing_input = header + b"." + b64(json.dumps(claims).encode())
    signature = b64(hmac.new(PUBLIC_PEM, signing_input, hashlib.sha256).digest())

    with pytest.raises(HTTPException) as exc:
        _verify((signing_input + b"." + signature).decode())
    assert exc.value.status_code == 401


def test_wrong_audience_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(audience="anon"))
    assert exc.value.status_code == 401
    assert "audience" in exc.value.detail


def test_wrong_issuer_is_401() -> None:
    """A genuine token from a *different* Supabase project must not open this one."""
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(issuer="https://someone-elses.supabase.co/auth/v1"))
    assert exc.value.status_code == 401
    assert "not issued by this project" in exc.value.detail


def test_token_without_a_subject_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(subject=None))
    assert exc.value.status_code == 401


def test_unknown_kid_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        _verify(_mint(kid="some-other-key"))
    assert exc.value.status_code == 401
    assert "not recognised" in exc.value.detail


# --- the key cache ----------------------------------------------------------


def test_keys_are_fetched_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hot path must not call Supabase per request."""
    factory, calls = _jwks_transport()
    cache = JWKSCache(JWKS_URL, factory)
    monkeypatch.setattr(auth_module, "get_jwks_cache", lambda: cache)

    for _ in range(5):
        _verify(_mint())
    assert len(calls) == 1


def test_concurrent_cold_start_fetches_once() -> None:
    """A burst against an empty cache collapses into a single fetch, not one per request."""
    factory, calls = _jwks_transport()
    cache = JWKSCache(JWKS_URL, factory)

    async def race() -> None:
        await asyncio.gather(*(cache.get_key(KID) for _ in range(10)))

    asyncio.run(race())
    assert len(calls) == 1


def test_cold_cache_with_unreachable_jwks_is_503_not_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage on our side must not tell users their login is bad."""
    factory, _ = _jwks_transport(status_code=500)
    monkeypatch.setattr(auth_module, "get_jwks_cache", lambda: JWKSCache(JWKS_URL, factory))
    with pytest.raises(HTTPException) as exc:
        _verify(_mint())
    assert exc.value.status_code == 503


def test_cached_keys_survive_a_transient_fetch_failure() -> None:
    """Once warm, a Supabase blip must not take authentication down."""
    failing = {"now": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if failing["now"]:
            return httpx.Response(503, text="down")
        return httpx.Response(200, json=JWKS)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    cache = JWKSCache(JWKS_URL, factory)
    assert asyncio.run(cache.get_key(KID)) is not None

    failing["now"] = True
    cache._fetched_at -= 10_000  # force staleness, so the next call attempts a re-fetch
    assert asyncio.run(cache.get_key(KID)) is not None


def test_unconfigured_supabase_url_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "get_settings", lambda: SimpleNamespace(supabase_url=""))
    with pytest.raises(HTTPException) as exc:
        auth_module.jwks_url_for(SimpleNamespace(supabase_url=""))  # type: ignore[arg-type]
    assert exc.value.status_code == 503
    assert "not configured" in exc.value.detail


# --- route policy -----------------------------------------------------------


def _security_by_path() -> dict[str, object]:
    """Which paths the OpenAPI schema declares as requiring a bearer token."""
    spec = app.openapi()
    return {
        path: operation.get("security")
        for path, operations in spec["paths"].items()
        for operation in operations.values()
    }


def test_every_non_public_route_requires_authentication() -> None:
    """Structural guard: a new endpoint added without auth fails here, not silently."""
    unprotected = [
        path
        for path, security in _security_by_path().items()
        if path not in PUBLIC_PATHS and not security
    ]
    assert unprotected == []


def test_the_three_query_endpoints_are_protected() -> None:
    security = _security_by_path()
    for path in ("/chat", "/compliance-check", "/rank"):
        assert security[path] == [{"HTTPBearer": []}], path


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat", {"query": "scope 1 emissions"}),
        ("/compliance-check", {"company": "BCA", "category": "environmental"}),
        ("/rank", {"query": "top 5 banks by earliest net-zero target year"}),
    ],
)
def test_protected_endpoints_reject_an_unauthenticated_request(path: str, body: dict) -> None:
    """Behavioural counterpart to the schema check: a real request with no token is a 401."""
    response = client.post(path, json=body)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_health_stays_public_so_deploy_probes_keep_working() -> None:
    """No token, and no 401. (The status itself depends on whether Qdrant is reachable,
    which is `test_health.py`'s subject, not this file's.)"""
    assert _security_by_path()["/health"] is None
    assert client.get("/health").status_code != 401
