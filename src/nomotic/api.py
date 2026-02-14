"""HTTP API server for the Nomotic governance framework.

A thin HTTP layer over :class:`CertificateAuthority`, the validation
registries, and optionally :class:`GovernanceRuntime`.  Uses only
:mod:`http.server` from the standard library — zero extra dependencies.

Start the server::

    from nomotic.api import NomoticAPIServer
    server = NomoticAPIServer(ca)
    server.serve_forever()

Or via the CLI::

    nomotic serve --port 8420
"""

from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from nomotic.authority import CertificateAuthority
from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.evaluator import EvaluatorConfig, ProtocolEvaluator
from nomotic.protocol import ReasoningArtifact
from nomotic.registry import (
    ArchetypeRegistry,
    OrganizationRegistry,
    OrgStatus,
    ZoneValidator,
)

__all__ = ["NomoticAPIServer"]

_VERSION = "0.1.0"


# ── JSON helpers ────────────────────────────────────────────────────────


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=2).encode("utf-8")


def _error(status_code: int, error: str, message: str, **extra: Any) -> tuple[int, bytes]:
    body: dict[str, Any] = {
        "error": error,
        "message": message,
        "status_code": status_code,
    }
    body.update(extra)
    return status_code, _json_bytes(body)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Route matching ──────────────────────────────────────────────────────

_CERT_ID_RE = re.compile(r"^/v1/certificates/(nmc-[^/]+)$")
_CERT_ACTION_RE = re.compile(r"^/v1/certificates/(nmc-[^/]+)/(verify|verify/live|suspend|reactivate|revoke|renew|zone|reputation)$")
_VERIFY_RE = re.compile(r"^/v1/verify/(nmc-[^/]+)$")
_ARCHETYPE_NAME_RE = re.compile(r"^/v1/archetypes/([a-z0-9][a-z0-9-]*[a-z0-9])$")
_ORG_NAME_RE = re.compile(r"^/v1/organizations/([a-z0-9][a-z0-9-]*[a-z0-9])$")
_ORG_ACTION_RE = re.compile(r"^/v1/organizations/([a-z0-9][a-z0-9-]*[a-z0-9])/(suspend|revoke)$")
_FINGERPRINT_RE = re.compile(r"^/v1/fingerprint/(.+)$")
_DRIFT_RE = re.compile(r"^/v1/drift/(.+)$")
_ALERT_ACK_RE = re.compile(r"^/v1/alerts/([^/]+)/(\d+)/acknowledge$")
_TRUST_RE = re.compile(r"^/v1/trust/([^/]+)$")
_TRUST_TRAJECTORY_RE = re.compile(r"^/v1/trust/([^/]+)/trajectory$")
_PROVENANCE_HISTORY_RE = re.compile(r"^/v1/provenance/history/([^/]+)/(.+)$")
_OWNER_ACTIVITY_RE = re.compile(r"^/v1/owner/([^/]+)/activity$")
_OWNER_ENGAGEMENT_RE = re.compile(r"^/v1/owner/([^/]+)/engagement$")
_USER_STATS_RE = re.compile(r"^/v1/user/([^/]+)/stats$")


# ── Request handler ─────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler that delegates to the shared server state."""

    server: NomoticHTTPServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""
        pass

    # ── Dispatch ────────────────────────────────────────────────────

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def _dispatch(self, method: str) -> None:
        try:
            status, body = self._route(method)
        except Exception:
            status, body = _error(500, "internal_error", traceback.format_exc())
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _query_params(self) -> dict[str, str]:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        return {k: v[0] for k, v in qs.items()}

    def _clean_path(self) -> str:
        from urllib.parse import urlparse
        return urlparse(self.path).path

    # ── Router ──────────────────────────────────────────────────────

    def _route(self, method: str) -> tuple[int, bytes]:
        path = self._clean_path()
        ctx = self.server.ctx

        # Health
        if path == "/v1/health" and method == "GET":
            return self._handle_health(ctx)
        if path == "/v1/info" and method == "GET":
            return self._handle_info(ctx)

        # Revocations
        if path == "/v1/revocations" and method == "GET":
            return self._handle_revocations(ctx)

        # Certificates collection
        if path == "/v1/certificates":
            if method == "GET":
                return self._handle_list_certs(ctx)
            if method == "POST":
                return self._handle_issue_cert(ctx)

        # Single certificate
        m = _CERT_ID_RE.match(path)
        if m and method == "GET":
            return self._handle_get_cert(ctx, m.group(1))

        # Certificate actions
        m = _CERT_ACTION_RE.match(path)
        if m:
            cert_id, action = m.group(1), m.group(2)
            if action == "verify" and method == "POST":
                return self._handle_verify_cert(ctx, cert_id)
            if action == "verify/live" and method == "POST":
                return self._handle_verify_live(ctx, cert_id)
            if action == "suspend" and method == "PATCH":
                return self._handle_suspend_cert(ctx, cert_id)
            if action == "reactivate" and method == "PATCH":
                return self._handle_reactivate_cert(ctx, cert_id)
            if action == "revoke" and method == "PATCH":
                return self._handle_revoke_cert(ctx, cert_id)
            if action == "renew" and method == "POST":
                return self._handle_renew_cert(ctx, cert_id)
            if action == "zone" and method == "PATCH":
                return self._handle_transfer_zone(ctx, cert_id)
            if action == "reputation" and method == "GET":
                return self._handle_reputation(ctx, cert_id)

        # Quick verify
        m = _VERIFY_RE.match(path)
        if m and method == "GET":
            return self._handle_quick_verify(ctx, m.group(1))

        # Archetypes
        if path == "/v1/archetypes":
            if method == "GET":
                return self._handle_list_archetypes(ctx)
            if method == "POST":
                return self._handle_register_archetype(ctx)
        if path == "/v1/archetypes/validate" and method == "POST":
            return self._handle_validate_archetype(ctx)
        m = _ARCHETYPE_NAME_RE.match(path)
        if m and method == "GET":
            return self._handle_get_archetype(ctx, m.group(1))

        # Organizations
        if path == "/v1/organizations":
            if method == "GET":
                return self._handle_list_orgs(ctx)
            if method == "POST":
                return self._handle_register_org(ctx)
        if path == "/v1/organizations/validate" and method == "POST":
            return self._handle_validate_org(ctx)
        m = _ORG_ACTION_RE.match(path)
        if m:
            org_name, action = m.group(1), m.group(2)
            if action == "suspend" and method == "PATCH":
                return self._handle_suspend_org(ctx, org_name)
            if action == "revoke" and method == "PATCH":
                return self._handle_revoke_org(ctx, org_name)
        m = _ORG_NAME_RE.match(path)
        if m and method == "GET":
            return self._handle_get_org(ctx, m.group(1))

        # Zones
        if path == "/v1/zones/validate" and method == "POST":
            return self._handle_validate_zone(ctx)

        # Fingerprints
        m = _FINGERPRINT_RE.match(path)
        if m and method == "GET":
            return self._handle_get_fingerprint(ctx, m.group(1))

        # Drift
        m = _DRIFT_RE.match(path)
        if m and method == "GET":
            return self._handle_get_drift(ctx, m.group(1))

        # Alerts
        if path == "/v1/alerts" and method == "GET":
            return self._handle_get_alerts(ctx)

        m = _ALERT_ACK_RE.match(path)
        if m and method == "POST":
            return self._handle_acknowledge_alert(ctx, m.group(1), int(m.group(2)))

        # Trust
        m = _TRUST_TRAJECTORY_RE.match(path)
        if m and method == "GET":
            return self._handle_get_trust_trajectory(ctx, m.group(1))

        m = _TRUST_RE.match(path)
        if m and method == "GET":
            return self._handle_get_trust(ctx, m.group(1))

        # Audit (Phase 5)
        if path == "/v1/audit" and method == "GET":
            return self._handle_audit_query(ctx)
        if path == "/v1/audit/summary" and method == "GET":
            return self._handle_audit_summary(ctx)

        # Provenance (Phase 5)
        if path == "/v1/provenance" and method == "GET":
            return self._handle_provenance_query(ctx)
        m = _PROVENANCE_HISTORY_RE.match(path)
        if m and method == "GET":
            return self._handle_provenance_history(ctx, m.group(1), m.group(2))

        # Owner (Phase 5)
        m = _OWNER_ACTIVITY_RE.match(path)
        if m and method == "GET":
            return self._handle_owner_activity(ctx, m.group(1))
        m = _OWNER_ENGAGEMENT_RE.match(path)
        if m and method == "GET":
            return self._handle_owner_engagement(ctx, m.group(1))

        # User stats (Phase 5)
        m = _USER_STATS_RE.match(path)
        if m and method == "GET":
            return self._handle_user_stats(ctx, m.group(1))

        # ── Nomotic Protocol endpoints ─────────────────────────────
        if path == "/v1/reason" and method == "POST":
            return self._handle_reason(ctx)
        if path == "/v1/reason/summary" and method == "POST":
            return self._handle_reason_summary(ctx)
        if path == "/v1/reason/posthoc" and method == "POST":
            return self._handle_reason_posthoc(ctx)
        if path == "/v1/token/validate" and method == "POST":
            return self._handle_token_validate(ctx)
        if path == "/v1/token/introspect" and method == "POST":
            return self._handle_token_introspect(ctx)
        if path == "/v1/token/revoke" and method == "POST":
            return self._handle_token_revoke(ctx)
        if path == "/v1/schema" and method == "GET":
            return self._handle_schema(ctx)
        if path == "/v1/schema/version" and method == "GET":
            return self._handle_schema_version(ctx)

        return _error(404, "not_found", f"No route for {method} {path}")

    # ── Health / Info ───────────────────────────────────────────────

    def _handle_health(self, ctx: _ServerContext) -> tuple[int, bytes]:
        return 200, _json_bytes({"status": "ok"})

    def _handle_info(self, ctx: _ServerContext) -> tuple[int, bytes]:
        certs = ctx.ca.list()
        return 200, _json_bytes({
            "version": _VERSION,
            "issuer_id": ctx.ca.issuer_id,
            "issuer_fingerprint": ctx.ca.issuer_fingerprint,
            "uptime_seconds": round(time.time() - ctx.started_at, 1),
            "certificate_count": len(certs),
        })

    # ── Revocations ─────────────────────────────────────────────────

    def _handle_revocations(self, ctx: _ServerContext) -> tuple[int, bytes]:
        revoked = ctx.ca.get_revocation_list()
        return 200, _json_bytes({
            "revoked": revoked,
            "generated_at": _utcnow_iso(),
        })

    # ── Certificates ────────────────────────────────────────────────

    def _handle_issue_cert(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        required = ["agent_id", "archetype", "organization", "zone_path"]
        missing = [k for k in required if k not in data]
        if missing:
            return _error(400, "validation_error", f"Missing fields: {', '.join(missing)}")
        try:
            cert, _sk = ctx.ca.issue(
                agent_id=data["agent_id"],
                archetype=data["archetype"],
                organization=data["organization"],
                zone_path=data["zone_path"],
            )
        except ValueError as exc:
            suggestion = None
            msg = str(exc)
            if "did you mean" in msg:
                import re as _re
                m = _re.search(r"did you mean '([^']+)'", msg)
                if m:
                    suggestion = m.group(1)
            kwargs: dict[str, Any] = {}
            if suggestion:
                kwargs["suggestion"] = suggestion
            return _error(400, "validation_error", str(exc), **kwargs)
        return 201, _json_bytes(cert.to_dict())

    def _handle_list_certs(self, ctx: _ServerContext) -> tuple[int, bytes]:
        params = self._query_params()
        org = params.get("org")
        status_str = params.get("status")
        archetype = params.get("archetype")
        status = CertStatus[status_str.upper()] if status_str else None
        certs = ctx.ca.list(org=org, status=status, archetype=archetype)
        return 200, _json_bytes([c.to_dict() for c in certs])

    def _handle_get_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        cert = ctx.ca.get(cert_id)
        if cert is None:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        return 200, _json_bytes(cert.to_dict())

    def _handle_verify_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        cert = ctx.ca.get(cert_id)
        if cert is None:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        result = ctx.ca.verify_certificate(cert)
        return 200, _json_bytes({
            "valid": result.valid,
            "certificate_id": result.certificate_id,
            "issues": result.issues,
            "status": result.status.name if result.status else None,
        })

    def _handle_verify_live(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        cert = ctx.ca.get(cert_id)
        if cert is None:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        result = ctx.ca.verify_live(cert)
        return 200, _json_bytes({
            "valid": result.valid,
            "certificate_id": result.certificate_id,
            "issues": result.issues,
            "status": result.status.name if result.status else None,
            "trust_score": result.trust_score,
            "behavioral_age": result.behavioral_age,
            "governance_hash": result.governance_hash,
            "healthy": result.healthy,
        })

    def _handle_suspend_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        data = self._read_json()
        reason = data.get("reason", "no reason given")
        try:
            cert = ctx.ca.suspend(cert_id, reason)
        except KeyError:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 200, _json_bytes(cert.to_dict())

    def _handle_reactivate_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        try:
            cert = ctx.ca.reactivate(cert_id)
        except KeyError:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 200, _json_bytes(cert.to_dict())

    def _handle_revoke_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        data = self._read_json()
        reason = data.get("reason", "no reason given")
        try:
            cert = ctx.ca.revoke(cert_id, reason)
        except KeyError:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 200, _json_bytes(cert.to_dict())

    def _handle_renew_cert(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        try:
            cert, _sk = ctx.ca.renew(cert_id)
        except KeyError:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        return 200, _json_bytes(cert.to_dict())

    def _handle_transfer_zone(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        data = self._read_json()
        zone_path = data.get("zone_path")
        if not zone_path:
            return _error(400, "validation_error", "Missing 'zone_path' field")
        try:
            cert = ctx.ca.transfer_zone(cert_id, zone_path)
        except KeyError:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        return 200, _json_bytes(cert.to_dict())

    def _handle_reputation(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        cert = ctx.ca.get(cert_id)
        if cert is None:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        return 200, _json_bytes({
            "certificate_id": cert.certificate_id,
            "agent_id": cert.agent_id,
            "trust_score": cert.trust_score,
            "behavioral_age": cert.behavioral_age,
            "status": cert.status.name,
            "archetype": cert.archetype,
            "issued_at": cert.issued_at.isoformat(),
            "lineage": cert.lineage,
        })

    # ── Quick verify ────────────────────────────────────────────────

    def _handle_quick_verify(self, ctx: _ServerContext, cert_id: str) -> tuple[int, bytes]:
        cert = ctx.ca.get(cert_id)
        if cert is None:
            return _error(404, "not_found", f"Certificate not found: {cert_id}")
        result = ctx.ca.verify_live(cert)
        return 200, _json_bytes({
            "valid": result.valid,
            "certificate_id": cert.certificate_id,
            "status": cert.status.name,
            "trust_score": result.trust_score,
            "behavioral_age": result.behavioral_age,
            "archetype": cert.archetype,
            "organization": cert.organization,
            "zone_path": cert.zone_path,
            "governance_hash": result.governance_hash,
            "healthy": result.healthy,
        })

    # ── Archetypes ──────────────────────────────────────────────────

    def _handle_list_archetypes(self, ctx: _ServerContext) -> tuple[int, bytes]:
        params = self._query_params()
        category = params.get("category")
        archetypes = ctx.archetype_registry.list(category=category)
        return 200, _json_bytes([
            {"name": a.name, "description": a.description, "category": a.category, "builtin": a.builtin}
            for a in archetypes
        ])

    def _handle_register_archetype(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        required = ["name", "description", "category"]
        missing = [k for k in required if k not in data]
        if missing:
            return _error(400, "validation_error", f"Missing fields: {', '.join(missing)}")
        try:
            defn = ctx.archetype_registry.register(
                data["name"], data["description"], data["category"],
            )
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 201, _json_bytes({
            "name": defn.name, "description": defn.description,
            "category": defn.category, "builtin": defn.builtin,
        })

    def _handle_get_archetype(self, ctx: _ServerContext, name: str) -> tuple[int, bytes]:
        defn = ctx.archetype_registry.get(name)
        if defn is None:
            return _error(404, "not_found", f"Archetype not found: {name}")
        return 200, _json_bytes({
            "name": defn.name, "description": defn.description,
            "category": defn.category, "builtin": defn.builtin,
        })

    def _handle_validate_archetype(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        name = data.get("name", "")
        result = ctx.archetype_registry.validate(name)
        body: dict[str, Any] = {
            "valid": result.valid,
            "name": result.name,
            "warnings": result.warnings,
            "errors": result.errors,
        }
        if result.suggestion:
            body["suggestion"] = result.suggestion
        return 200, _json_bytes(body)

    # ── Organizations ───────────────────────────────────────────────

    def _handle_list_orgs(self, ctx: _ServerContext) -> tuple[int, bytes]:
        params = self._query_params()
        status_str = params.get("status")
        status = OrgStatus[status_str.upper()] if status_str else None
        orgs = ctx.org_registry.list(status=status)
        return 200, _json_bytes([o.to_dict() for o in orgs])

    def _handle_register_org(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        name = data.get("name")
        if not name:
            return _error(400, "validation_error", "Missing 'name' field")
        issuer_fp = ctx.ca.issuer_fingerprint
        try:
            org = ctx.org_registry.register(
                name,
                issuer_fp,
                contact_email=data.get("contact_email"),
            )
        except ValueError as exc:
            msg = str(exc)
            if "already registered" in msg:
                return _error(409, "conflict", msg)
            return _error(400, "validation_error", msg)
        return 201, _json_bytes(org.to_dict())

    def _handle_get_org(self, ctx: _ServerContext, name: str) -> tuple[int, bytes]:
        org = ctx.org_registry.get(name)
        if org is None:
            return _error(404, "not_found", f"Organization not found: {name}")
        return 200, _json_bytes(org.to_dict())

    def _handle_validate_org(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        name = data.get("name", "")
        result = ctx.org_registry.validate(name)
        return 200, _json_bytes({
            "valid": result.valid,
            "name": result.name,
            "warnings": result.warnings,
            "errors": result.errors,
        })

    def _handle_suspend_org(self, ctx: _ServerContext, name: str) -> tuple[int, bytes]:
        data = self._read_json()
        reason = data.get("reason", "no reason given")
        try:
            org = ctx.org_registry.suspend(name, reason)
        except KeyError:
            return _error(404, "not_found", f"Organization not found: {name}")
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 200, _json_bytes(org.to_dict())

    def _handle_revoke_org(self, ctx: _ServerContext, name: str) -> tuple[int, bytes]:
        data = self._read_json()
        reason = data.get("reason", "no reason given")
        try:
            org = ctx.org_registry.revoke(name, reason)
        except KeyError:
            return _error(404, "not_found", f"Organization not found: {name}")
        except ValueError as exc:
            return _error(400, "validation_error", str(exc))
        return 200, _json_bytes(org.to_dict())

    # ── Zones ───────────────────────────────────────────────────────

    def _handle_validate_zone(self, ctx: _ServerContext) -> tuple[int, bytes]:
        data = self._read_json()
        zone_path = data.get("zone_path", "")
        result = ctx.zone_validator.validate(zone_path)
        return 200, _json_bytes({
            "valid": result.valid,
            "name": result.name,
            "warnings": result.warnings,
            "errors": result.errors,
        })

    # ── Fingerprints ───────────────────────────────────────────────

    def _handle_get_fingerprint(self, ctx: _ServerContext, agent_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Fingerprints require a GovernanceRuntime")
        fp = ctx.runtime.get_fingerprint(agent_id)
        if fp is None:
            return _error(404, "not_found", f"No fingerprint for agent: {agent_id}")
        return 200, _json_bytes({
            "agent_id": fp.agent_id,
            "total_observations": fp.total_observations,
            "confidence": fp.confidence,
            "action_distribution": fp.action_distribution,
            "target_distribution": fp.target_distribution,
            "temporal_pattern": fp.temporal_pattern.to_dict(),
            "outcome_distribution": fp.outcome_distribution,
        })

    # ── Drift ──────────────────────────────────────────────────────

    def _handle_get_drift(self, ctx: _ServerContext, agent_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Drift detection requires a GovernanceRuntime")
        drift = ctx.runtime.get_drift(agent_id)
        if drift is None:
            return _error(404, "not_found", f"No drift data for agent: {agent_id}")
        alerts = ctx.runtime.get_drift_alerts(agent_id)
        return 200, _json_bytes({
            "agent_id": agent_id,
            "drift": drift.to_dict(),
            "alerts": [a.to_dict() for a in alerts],
        })

    def _handle_get_alerts(self, ctx: _ServerContext) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Drift detection requires a GovernanceRuntime")
        params = self._query_params()
        agent_id = params.get("agent_id")
        alerts = ctx.runtime.get_drift_alerts(agent_id)
        unacked = sum(1 for a in alerts if not a.acknowledged)
        return 200, _json_bytes({
            "alerts": [a.to_dict() for a in alerts],
            "total": len(alerts),
            "unacknowledged": unacked,
        })

    def _handle_acknowledge_alert(self, ctx: _ServerContext, agent_id: str, index: int) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Drift detection requires a GovernanceRuntime")
        if ctx.runtime._fingerprint_observer is None:
            return _error(404, "not_found", "Drift detection is not enabled")
        ok = ctx.runtime._fingerprint_observer.drift_monitor.acknowledge_alert(agent_id, index)
        if not ok:
            return _error(404, "not_found", f"Alert not found: agent={agent_id}, index={index}")
        return 200, _json_bytes({"acknowledged": True})

    # ── Trust ─────────────────────────────────────────────────────────

    def _handle_get_trust(self, ctx: _ServerContext, agent_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Trust reports require a GovernanceRuntime")
        profile = ctx.runtime.get_trust_profile(agent_id)
        # Return 404 if the agent has never been evaluated (baseline profile
        # with no actions indicates no evaluation has occurred).
        if profile.successful_actions == 0 and profile.violation_count == 0:
            trajectory = ctx.runtime.trust_calibrator.get_trajectory(agent_id)
            if len(trajectory) == 0:
                return _error(404, "not_found", f"No trust data for agent: {agent_id}")
        report = ctx.runtime.get_trust_report(agent_id)
        return 200, _json_bytes(report)

    def _handle_get_trust_trajectory(self, ctx: _ServerContext, agent_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None:
            return _error(404, "not_found", "Trust reports require a GovernanceRuntime")
        trajectory = ctx.runtime.trust_calibrator.get_trajectory(agent_id)
        params = self._query_params()

        events = trajectory.events

        # Filter by since
        since_str = params.get("since")
        if since_str is not None:
            try:
                since = float(since_str)
                events = [e for e in events if e.timestamp > since]
            except ValueError:
                pass

        # Filter by source prefix
        source_prefix = params.get("source")
        if source_prefix is not None:
            events = [e for e in events if e.source.startswith(source_prefix)]

        # Limit
        limit_str = params.get("limit")
        if limit_str is not None:
            try:
                limit = min(int(limit_str), 500)
                events = events[-limit:]
            except ValueError:
                pass

        return 200, _json_bytes({
            "agent_id": agent_id,
            "events": [e.to_dict() for e in events],
            "total": len(events),
        })

    # ── Audit (Phase 5) ─────────────────────────────────────────────

    def _handle_audit_query(self, ctx: _ServerContext) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.audit_trail is None:
            return _error(404, "not_found", "Audit trail requires a GovernanceRuntime with audit enabled")
        params = self._query_params()
        kwargs: dict[str, Any] = {}
        for key in ("agent_id", "severity", "category", "verdict", "context_code", "owner_id", "user_id"):
            if key in params:
                kwargs[key] = params[key]
        if "since" in params:
            try:
                kwargs["since"] = float(params["since"])
            except ValueError:
                pass
        if "until" in params:
            try:
                kwargs["until"] = float(params["until"])
            except ValueError:
                pass
        if "limit" in params:
            try:
                kwargs["limit"] = min(int(params["limit"]), 500)
            except ValueError:
                pass
        records = ctx.runtime.audit_trail.query(**kwargs)
        return 200, _json_bytes({
            "records": [r.to_dict() for r in records],
            "total": len(records),
        })

    def _handle_audit_summary(self, ctx: _ServerContext) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.audit_trail is None:
            return _error(404, "not_found", "Audit trail requires a GovernanceRuntime with audit enabled")
        params = self._query_params()
        kwargs: dict[str, Any] = {}
        if "agent_id" in params:
            kwargs["agent_id"] = params["agent_id"]
        if "since" in params:
            try:
                kwargs["since"] = float(params["since"])
            except ValueError:
                pass
        summary = ctx.runtime.audit_trail.summary(**kwargs)
        return 200, _json_bytes(summary)

    # ── Provenance (Phase 5) ─────────────────────────────────────────

    def _handle_provenance_query(self, ctx: _ServerContext) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.provenance_log is None:
            return _error(404, "not_found", "Provenance log requires a GovernanceRuntime with audit enabled")
        params = self._query_params()
        kwargs: dict[str, Any] = {}
        for key in ("actor", "target_type", "target_id", "change_type"):
            if key in params:
                kwargs[key] = params[key]
        if "since" in params:
            try:
                kwargs["since"] = float(params["since"])
            except ValueError:
                pass
        if "limit" in params:
            try:
                kwargs["limit"] = min(int(params["limit"]), 500)
            except ValueError:
                pass
        records = ctx.runtime.provenance_log.query(**kwargs)
        return 200, _json_bytes({
            "records": [r.to_dict() for r in records],
            "total": len(records),
        })

    def _handle_provenance_history(self, ctx: _ServerContext, target_type: str, target_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.provenance_log is None:
            return _error(404, "not_found", "Provenance log requires a GovernanceRuntime with audit enabled")
        records = ctx.runtime.provenance_log.history(target_type, target_id)
        return 200, _json_bytes({
            "target_type": target_type,
            "target_id": target_id,
            "records": [r.to_dict() for r in records],
            "total": len(records),
        })

    # ── Owner (Phase 5) ──────────────────────────────────────────────

    def _handle_owner_activity(self, ctx: _ServerContext, owner_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.owner_activity is None:
            return _error(404, "not_found", "Owner tracking requires a GovernanceRuntime with audit enabled")
        params = self._query_params()
        kwargs: dict[str, Any] = {}
        if "activity_type" in params:
            kwargs["activity_type"] = params["activity_type"]
        if "since" in params:
            try:
                kwargs["since"] = float(params["since"])
            except ValueError:
                pass
        if "limit" in params:
            try:
                kwargs["limit"] = min(int(params["limit"]), 500)
            except ValueError:
                pass
        activities = ctx.runtime.owner_activity.get_activities(owner_id, **kwargs)
        return 200, _json_bytes({
            "owner_id": owner_id,
            "activities": [a.to_dict() for a in activities],
            "total": len(activities),
        })

    def _handle_owner_engagement(self, ctx: _ServerContext, owner_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.owner_activity is None:
            return _error(404, "not_found", "Owner tracking requires a GovernanceRuntime with audit enabled")
        params = self._query_params()
        window_days = 30
        if "window_days" in params:
            try:
                window_days = int(params["window_days"])
            except ValueError:
                pass
        score = ctx.runtime.owner_activity.engagement_score(owner_id, window_days=window_days)
        return 200, _json_bytes(score)

    # ── User stats (Phase 5) ─────────────────────────────────────────

    def _handle_user_stats(self, ctx: _ServerContext, user_id: str) -> tuple[int, bytes]:
        if ctx.runtime is None or ctx.runtime.user_tracker is None:
            return _error(404, "not_found", "User tracking requires a GovernanceRuntime with audit enabled")
        stats = ctx.runtime.user_tracker.get_stats(user_id)
        if stats is None:
            return _error(404, "not_found", f"No interaction data for user: {user_id}")
        return 200, _json_bytes(stats.to_dict())

    # ── Nomotic Protocol ──────────────────────────────────────────────

    def _handle_reason(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/reason — Full Deliberation Flow."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        try:
            artifact = ReasoningArtifact.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, "validation_error", f"Invalid reasoning artifact: {exc}")
        response = ctx.evaluator.evaluate(artifact)
        return 200, _json_bytes(response.to_dict())

    def _handle_reason_summary(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/reason/summary — Summary Flow."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        try:
            artifact = ReasoningArtifact.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, "validation_error", f"Invalid reasoning artifact: {exc}")
        response = ctx.evaluator.evaluate_summary(artifact)
        return 200, _json_bytes(response.to_dict())

    def _handle_reason_posthoc(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/reason/posthoc — Post-Hoc Flow."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        artifact_data = data.get("artifact", data)
        action_result = data.get("action_result")
        try:
            artifact = ReasoningArtifact.from_dict(artifact_data)
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, "validation_error", f"Invalid reasoning artifact: {exc}")
        assessment = ctx.evaluator.evaluate_posthoc(artifact, action_result)
        return 200, _json_bytes(assessment.to_dict())

    def _handle_token_validate(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/token/validate — Validate a Governance Token."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        token = data.get("token")
        if not token:
            return _error(400, "validation_error", "Missing 'token' field")
        result = ctx.evaluator.validate_token(token)
        return 200, _json_bytes(result.to_dict())

    def _handle_token_introspect(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/token/introspect — Full governance context for a token."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        token = data.get("token")
        if not token:
            return _error(400, "validation_error", "Missing 'token' field")
        result = ctx.evaluator.introspect_token(token)
        if result is None:
            return _error(400, "validation_error", "Invalid or expired token")
        return 200, _json_bytes(result)

    def _handle_token_revoke(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """POST /v1/token/revoke — Revoke a token before expiration."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        data = self._read_json()
        token_id = data.get("token_id")
        if not token_id:
            return _error(400, "validation_error", "Missing 'token_id' field")
        revoked = ctx.evaluator.revoke_token(token_id)
        return 200, _json_bytes({"revoked": revoked, "token_id": token_id})

    def _handle_schema(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """GET /v1/schema — Returns the current Reasoning Artifact JSON Schema."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        schema = ctx.evaluator.get_schema()
        return 200, _json_bytes(schema)

    def _handle_schema_version(self, ctx: _ServerContext) -> tuple[int, bytes]:
        """GET /v1/schema/version — Returns supported schema versions."""
        if ctx.evaluator is None:
            return _error(404, "not_found", "Protocol evaluator not configured")
        versions = ctx.evaluator.get_supported_versions()
        return 200, _json_bytes({"supported_versions": versions})


# ── Server context ──────────────────────────────────────────────────────


class _ServerContext:
    """Shared state accessible to all request handlers."""

    def __init__(
        self,
        ca: CertificateAuthority,
        archetype_registry: ArchetypeRegistry,
        zone_validator: ZoneValidator,
        org_registry: OrganizationRegistry,
        runtime: Any = None,
        evaluator: ProtocolEvaluator | None = None,
    ) -> None:
        self.ca = ca
        self.archetype_registry = archetype_registry
        self.zone_validator = zone_validator
        self.org_registry = org_registry
        self.runtime = runtime
        self.evaluator = evaluator
        self.started_at = time.time()


class NomoticHTTPServer(HTTPServer):
    """HTTPServer subclass that carries the shared context."""

    ctx: _ServerContext


# ── Public API ──────────────────────────────────────────────────────────


class NomoticAPIServer:
    """HTTP API server for the Nomotic governance framework.

    Wraps :class:`CertificateAuthority`, registries, and optionally
    :class:`GovernanceRuntime` to expose governance operations over HTTP.
    """

    def __init__(
        self,
        ca: CertificateAuthority,
        *,
        archetype_registry: ArchetypeRegistry | None = None,
        zone_validator: ZoneValidator | None = None,
        org_registry: OrganizationRegistry | None = None,
        runtime: Any = None,
        evaluator: ProtocolEvaluator | None = None,
        host: str = "0.0.0.0",
        port: int = 8420,
    ) -> None:
        self._ca = ca
        self._archetype_registry = archetype_registry or ArchetypeRegistry.with_defaults()
        self._zone_validator = zone_validator or ZoneValidator()
        self._org_registry = org_registry or OrganizationRegistry()
        self._runtime = runtime
        self._evaluator = evaluator
        self._host = host
        self._port = port
        self._server: NomoticHTTPServer | None = None

    def _build_server(self) -> NomoticHTTPServer:
        server = NomoticHTTPServer((self._host, self._port), _Handler)
        server.ctx = _ServerContext(
            ca=self._ca,
            archetype_registry=self._archetype_registry,
            zone_validator=self._zone_validator,
            org_registry=self._org_registry,
            runtime=self._runtime,
            evaluator=self._evaluator,
        )
        return server

    def serve_forever(self) -> None:
        """Start serving requests (blocks)."""
        self._server = self._build_server()
        self._server.serve_forever()

    def shutdown(self) -> None:
        """Shut down the server."""
        if self._server is not None:
            self._server.shutdown()

    @property
    def server_address(self) -> tuple[str, int]:
        """Return the bound (host, port) tuple."""
        if self._server is not None:
            return self._server.server_address  # type: ignore[return-value]
        return (self._host, self._port)
