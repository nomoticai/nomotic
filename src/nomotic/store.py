"""Certificate storage — pluggable persistence for agent certificates.

v1 is an in-memory store backed by a simple dict. The interface is defined
as a Protocol so it can be swapped for file-based JSON, SQLite, Postgres,
or any other backend without changing calling code.

The file-based store (FileCertificateStore) persists certificates and keys
to ``~/.nomotic/`` for the CLI and standalone usage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from nomotic.certificate import AgentCertificate, CertStatus

__all__ = [
    "CertificateStore",
    "MemoryCertificateStore",
    "FileCertificateStore",
]


@runtime_checkable
class CertificateStore(Protocol):
    """Abstract certificate storage interface."""

    def save(self, certificate: AgentCertificate) -> None: ...

    def get(self, certificate_id: str) -> AgentCertificate | None: ...

    def list(
        self,
        org: str | None = None,
        status: CertStatus | None = None,
        archetype: str | None = None,
    ) -> list[AgentCertificate]: ...

    def update(self, certificate: AgentCertificate) -> None: ...

    def move_to_revoked(self, certificate_id: str) -> None: ...


class MemoryCertificateStore:
    """In-memory certificate store.

    Good for testing and for embedding in the GovernanceRuntime where
    persistence isn't needed.
    """

    def __init__(self) -> None:
        self._certs: dict[str, AgentCertificate] = {}
        self._revoked: dict[str, AgentCertificate] = {}

    def save(self, certificate: AgentCertificate) -> None:
        self._certs[certificate.certificate_id] = certificate

    def get(self, certificate_id: str) -> AgentCertificate | None:
        cert = self._certs.get(certificate_id)
        if cert is None:
            cert = self._revoked.get(certificate_id)
        return cert

    def list(
        self,
        org: str | None = None,
        status: CertStatus | None = None,
        archetype: str | None = None,
    ) -> list[AgentCertificate]:
        results: list[AgentCertificate] = []
        for cert in self._certs.values():
            if org is not None and cert.organization != org:
                continue
            if status is not None and cert.status != status:
                continue
            if archetype is not None and cert.archetype != archetype:
                continue
            results.append(cert)
        return results

    def update(self, certificate: AgentCertificate) -> None:
        if certificate.certificate_id in self._revoked:
            self._revoked[certificate.certificate_id] = certificate
        else:
            self._certs[certificate.certificate_id] = certificate

    def move_to_revoked(self, certificate_id: str) -> None:
        cert = self._certs.pop(certificate_id, None)
        if cert is not None:
            self._revoked[certificate_id] = cert


class FileCertificateStore:
    """File-based certificate store using JSON.

    Stores certificates in ``<base_dir>/certs/`` and revoked certificates
    in ``<base_dir>/revoked/``. Keys are stored alongside certificates.

    Directory layout::

        <base_dir>/
            certs/
                <cert-id>.json
                <cert-id>.key   (agent private key, only on issuing machine)
                <cert-id>.pub   (agent public key)
            revoked/
                <cert-id>.json
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".nomotic"
        self._base = Path(base_dir)
        self._certs_dir = self._base / "certs"
        self._revoked_dir = self._base / "revoked"
        self._certs_dir.mkdir(parents=True, exist_ok=True)
        self._revoked_dir.mkdir(parents=True, exist_ok=True)

    def save(self, certificate: AgentCertificate) -> None:
        path = self._certs_dir / f"{certificate.certificate_id}.json"
        path.write_text(
            json.dumps(certificate.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def get(self, certificate_id: str) -> AgentCertificate | None:
        path = self._certs_dir / f"{certificate_id}.json"
        if not path.exists():
            path = self._revoked_dir / f"{certificate_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentCertificate.from_dict(data)

    def list(
        self,
        org: str | None = None,
        status: CertStatus | None = None,
        archetype: str | None = None,
    ) -> list[AgentCertificate]:
        results: list[AgentCertificate] = []
        for path in sorted(self._certs_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            cert = AgentCertificate.from_dict(data)
            if org is not None and cert.organization != org:
                continue
            if status is not None and cert.status != status:
                continue
            if archetype is not None and cert.archetype != archetype:
                continue
            results.append(cert)
        return results

    def update(self, certificate: AgentCertificate) -> None:
        # Check revoked dir first
        revoked_path = self._revoked_dir / f"{certificate.certificate_id}.json"
        if revoked_path.exists():
            revoked_path.write_text(
                json.dumps(certificate.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        else:
            self.save(certificate)

    def move_to_revoked(self, certificate_id: str) -> None:
        src = self._certs_dir / f"{certificate_id}.json"
        if src.exists():
            dst = self._revoked_dir / f"{certificate_id}.json"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            src.unlink()

    # ── Key file helpers ─────────────────────────────────────────────

    def save_agent_key(self, certificate_id: str, key_bytes: bytes) -> None:
        """Save the agent's private key alongside the certificate."""
        path = self._certs_dir / f"{certificate_id}.key"
        path.write_bytes(key_bytes)
        os.chmod(path, 0o600)

    def save_agent_pub(self, certificate_id: str, pub_bytes: bytes) -> None:
        """Save the agent's public key alongside the certificate."""
        path = self._certs_dir / f"{certificate_id}.pub"
        path.write_bytes(pub_bytes)

    def get_agent_key(self, certificate_id: str) -> bytes | None:
        """Load the agent's private key."""
        path = self._certs_dir / f"{certificate_id}.key"
        if not path.exists():
            return None
        return path.read_bytes()

    def get_agent_pub(self, certificate_id: str) -> bytes | None:
        """Load the agent's public key."""
        path = self._certs_dir / f"{certificate_id}.pub"
        if not path.exists():
            return None
        return path.read_bytes()
