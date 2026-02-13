"""Agent SDK — governed HTTP client for Nomotic-certified agents.

The SDK makes carrying a certificate as easy as using ``urllib.request``.
Load the certificate and key once, and every outbound HTTP request
automatically carries the ``X-Nomotic-*`` headers with a fresh signature.

Usage::

    from nomotic.sdk import GovernedAgent

    agent = GovernedAgent.from_cert_id("nmc-abc123")
    response = agent.get("https://api.example.com/data")
    response = agent.post("https://api.example.com/actions", json={"do": "thing"})

Zero runtime dependencies beyond the Python standard library.
"""

from __future__ import annotations

import http.client
import json as _json
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nomotic.certificate import AgentCertificate
from nomotic.headers import generate_headers
from nomotic.keys import SigningKey

__all__ = [
    "GovernedAgent",
    "GovernedResponse",
    "NomoticSDKError",
    "CertificateLoadError",
    "GovernedRequestError",
]


# ── Exceptions ─────────────────────────────────────────────────────────


class NomoticSDKError(Exception):
    """Base exception for SDK errors."""


class CertificateLoadError(NomoticSDKError):
    """Failed to load certificate or key from file."""


class GovernedRequestError(NomoticSDKError):
    """HTTP request failed at the network level."""


# ── GovernedResponse ───────────────────────────────────────────────────


@dataclass
class GovernedResponse:
    """Response from a governed HTTP request.

    Wraps the stdlib HTTPResponse with convenient access to status,
    headers, and body.
    """

    status: int
    headers: dict[str, str]
    body: bytes
    url: str

    @property
    def ok(self) -> bool:
        """True if status is 2xx."""
        return 200 <= self.status < 300

    def json(self) -> Any:
        """Parse body as JSON."""
        return _json.loads(self.body)

    @property
    def text(self) -> str:
        """Decode body as UTF-8 text."""
        return self.body.decode("utf-8")


# ── GovernedAgent ──────────────────────────────────────────────────────


class GovernedAgent:
    """HTTP client that carries a Nomotic certificate on every request.

    Every outbound request includes ``X-Nomotic-*`` headers with the
    agent's certificate metadata and a signature over the request body.
    Receiving services can validate the certificate and make trust-based
    access decisions.

    Usage::

        agent = GovernedAgent.from_cert_id("nmc-abc123")
        response = agent.get("https://api.example.com/data")
        response = agent.post("https://api.example.com/actions", json={"do": "thing"})
    """

    def __init__(
        self,
        certificate: AgentCertificate,
        signing_key: SigningKey,
        *,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        verify_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            certificate: The agent's birth certificate.
            signing_key: The agent's private signing key.
            base_url: Optional base URL prepended to all request paths.
            extra_headers: Additional headers added to every request.
            verify_url: If set, the agent verifies its own certificate
                        on initialization by calling this endpoint.
            timeout: Default request timeout in seconds.
        """
        self._certificate = certificate
        self._signing_key = signing_key
        self._base_url = base_url.rstrip("/") if base_url else None
        self._extra_headers = extra_headers or {}
        self._timeout = timeout
        self._lock = threading.Lock()

        if verify_url:
            self.refresh_certificate(verify_url)

    @classmethod
    def from_files(
        cls,
        cert_path: str | Path,
        key_path: str | Path,
        **kwargs: Any,
    ) -> GovernedAgent:
        """Load certificate and key from file paths."""
        cert_path = Path(cert_path).expanduser()
        key_path = Path(key_path).expanduser()

        try:
            cert_data = cert_path.read_text(encoding="utf-8")
            cert = AgentCertificate.from_json(cert_data)
        except (OSError, ValueError, KeyError) as exc:
            raise CertificateLoadError(
                f"Failed to load certificate from {cert_path}: {exc}"
            ) from exc

        try:
            key_bytes = key_path.read_bytes()
            signing_key = SigningKey.from_bytes(key_bytes)
        except (OSError, ValueError) as exc:
            raise CertificateLoadError(
                f"Failed to load signing key from {key_path}: {exc}"
            ) from exc

        return cls(certificate=cert, signing_key=signing_key, **kwargs)

    @classmethod
    def from_cert_id(
        cls,
        cert_id: str,
        base_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> GovernedAgent:
        """Auto-discover certificate and key from ``~/.nomotic/certs/``.

        Loads ``<cert_id>.json`` and ``<cert_id>.key`` from the certs
        directory.
        """
        if base_dir is None:
            base_dir = Path.home() / ".nomotic"
        base_dir = Path(base_dir)
        cert_path = base_dir / "certs" / f"{cert_id}.json"
        key_path = base_dir / "certs" / f"{cert_id}.key"
        return cls.from_files(cert_path, key_path, **kwargs)

    # ── HTTP methods ──────────────────────────────────────────────────

    def get(self, url: str, **kwargs: Any) -> GovernedResponse:
        """Send a GET request with certificate headers."""
        return self.request("GET", url, **kwargs)

    def post(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> GovernedResponse:
        """Send a POST request with certificate headers and body signature."""
        return self.request("POST", url, json=json, data=data, **kwargs)

    def put(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> GovernedResponse:
        """Send a PUT request with certificate headers and body signature."""
        return self.request("PUT", url, json=json, data=data, **kwargs)

    def patch(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> GovernedResponse:
        """Send a PATCH request with certificate headers and body signature."""
        return self.request("PATCH", url, json=json, data=data, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> GovernedResponse:
        """Send a DELETE request with certificate headers."""
        return self.request("DELETE", url, **kwargs)

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> GovernedResponse:
        """Send an arbitrary HTTP request with certificate headers.

        This is the core method. All convenience methods delegate here.

        Steps:
            1. Resolve URL (prepend base_url if set)
            2. Serialize body (json.dumps if json provided, else use data,
               else ``b""``)
            3. Generate ``X-Nomotic-*`` headers (signs the body)
            4. Merge with extra_headers and request-specific headers
            5. Send via ``urllib.request.urlopen``
            6. Return :class:`GovernedResponse`
        """
        # 1. Resolve URL
        resolved_url = self._resolve_url(url)

        # 2. Serialize body
        body: bytes
        content_type: str | None = None
        if json is not None:
            body = _json.dumps(
                json, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            content_type = "application/json"
        elif data is not None:
            body = data
        else:
            body = b""

        # 3. Generate X-Nomotic-* headers
        nomotic_headers = generate_headers(self._certificate, self._signing_key, body)

        # 4. Merge headers
        all_headers: dict[str, str] = {}
        all_headers.update(self._extra_headers)
        all_headers.update(nomotic_headers)
        if content_type:
            all_headers["Content-Type"] = content_type
        if headers:
            all_headers.update(headers)

        # 5. Send request using http.client directly.
        # urllib.request normalises header names with capitalize(),
        # which mangles X-Nomotic-Cert-ID to X-nomotic-cert-id.
        # http.client preserves exact header casing.
        effective_timeout = timeout if timeout is not None else self._timeout

        try:
            parsed = urlparse(resolved_url)
            host = parsed.hostname or ""
            port = parsed.port
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"

            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    host, port=port, timeout=effective_timeout, context=ctx,
                )
            else:
                conn = http.client.HTTPConnection(
                    host, port=port, timeout=effective_timeout,
                )

            conn.request(method, path, body=body or None, headers=all_headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_headers = {k: v for k, v in resp.getheaders()}
            conn.close()

            return GovernedResponse(
                status=resp.status,
                headers=resp_headers,
                body=resp_body,
                url=resolved_url,
            )
        except (http.client.HTTPException, OSError) as exc:
            raise GovernedRequestError(
                f"{method} {resolved_url} failed: {exc}"
            ) from exc

    # ── Properties ────────────────────────────────────────────────────

    @property
    def certificate(self) -> AgentCertificate:
        """The agent's current certificate."""
        return self._certificate

    @property
    def cert_id(self) -> str:
        """Shortcut to certificate_id."""
        return self._certificate.certificate_id

    @property
    def trust_score(self) -> float:
        """Current trust score from the certificate."""
        return self._certificate.trust_score

    @property
    def behavioral_age(self) -> int:
        """Current behavioral age from the certificate."""
        return self._certificate.behavioral_age

    # ── Certificate refresh ───────────────────────────────────────────

    def refresh_certificate(
        self, verify_url: str | None = None,
    ) -> AgentCertificate:
        """Fetch the latest certificate state from the governance API.

        Calls ``GET /v1/certificates/{id}`` and updates the local
        certificate with current trust_score and behavioral_age.
        """
        if verify_url is None:
            raise NomoticSDKError("No verify_url provided for certificate refresh")

        url = f"{verify_url.rstrip('/')}/v1/certificates/{self._certificate.certificate_id}"

        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port
            path = parsed.path or "/"

            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    host, port=port, timeout=self._timeout, context=ctx,
                )
            else:
                conn = http.client.HTTPConnection(
                    host, port=port, timeout=self._timeout,
                )

            conn.request("GET", path)
            resp = conn.getresponse()
            data = _json.loads(resp.read())
            conn.close()
        except (http.client.HTTPException, OSError, ValueError) as exc:
            raise GovernedRequestError(
                f"Certificate refresh failed: {exc}"
            ) from exc

        with self._lock:
            self._certificate.trust_score = data.get(
                "trust_score", self._certificate.trust_score
            )
            self._certificate.behavioral_age = data.get(
                "behavioral_age", self._certificate.behavioral_age
            )

        return self._certificate

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_url(self, url: str) -> str:
        """Resolve a URL against the base_url."""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if self._base_url:
            return f"{self._base_url}/{url.lstrip('/')}"
        return url
