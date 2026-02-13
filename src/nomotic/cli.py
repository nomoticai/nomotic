"""Nomotic CLI — manage agent birth certificates from the command line.

Usage::

    nomotic birth --agent-id myagent --archetype customer-experience --org acme
    nomotic verify <cert-id>
    nomotic inspect <cert-id>
    nomotic suspend <cert-id> --reason "policy violation"
    nomotic revoke <cert-id> --reason "decommissioned"
    nomotic reactivate <cert-id>
    nomotic renew <cert-id>
    nomotic list [--status ACTIVE] [--archetype ...] [--org ...]
    nomotic reputation <cert-id>
    nomotic export <cert-id>

    nomotic archetype list [--category ...]
    nomotic archetype register --name <name> --description <desc> --category <cat>
    nomotic archetype validate <name>

    nomotic org register --name <name> [--email <email>]
    nomotic org list [--status ACTIVE]
    nomotic org validate <name>

    nomotic zone validate <path>

    nomotic serve [--host 0.0.0.0] [--port 8420]

Certificates are stored in ``~/.nomotic/certs/``.
The issuer key is stored in ``~/.nomotic/issuer/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from nomotic.authority import CertificateAuthority
from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.keys import SigningKey, VerifyKey
from nomotic.registry import (
    ArchetypeRegistry,
    FileOrgStore,
    OrganizationRegistry,
    OrgStatus,
    ZoneValidator,
)
from nomotic.store import FileCertificateStore

__all__ = ["main"]

_DEFAULT_BASE = Path.home() / ".nomotic"


# ── Issuer key management ────────────────────────────────────────────────


def _issuer_dir(base: Path) -> Path:
    d = base / "issuer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_or_create_issuer(base: Path) -> tuple[SigningKey, VerifyKey, str]:
    """Load the issuer key pair, or generate one on first run."""
    issuer = _issuer_dir(base)
    key_path = issuer / "issuer.key"
    pub_path = issuer / "issuer.pub"
    meta_path = issuer / "issuer.json"

    if key_path.exists():
        sk = SigningKey.from_bytes(key_path.read_bytes())
        vk = sk.verify_key()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return sk, vk, meta["issuer_id"]

    # First run — generate
    sk, vk = SigningKey.generate()
    key_path.write_bytes(sk.to_bytes())
    os.chmod(key_path, 0o600)
    pub_path.write_bytes(vk.to_bytes())
    issuer_id = f"nomotic-cli-{vk.fingerprint()[-12:]}"
    meta = {
        "issuer_id": issuer_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": vk.fingerprint(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sk, vk, issuer_id


def _build_ca(base: Path) -> tuple[CertificateAuthority, FileCertificateStore]:
    """Build a CertificateAuthority backed by the file store."""
    sk, _vk, issuer_id = _load_or_create_issuer(base)
    store = FileCertificateStore(base)
    ca = CertificateAuthority(issuer_id=issuer_id, signing_key=sk, store=store)
    return ca, store


def _build_registries(base: Path) -> tuple[ArchetypeRegistry, ZoneValidator, OrganizationRegistry]:
    """Build the validation registries for CLI use."""
    archetype_reg = ArchetypeRegistry.with_defaults()
    zone_val = ZoneValidator()
    org_store = FileOrgStore(base)
    org_reg = OrganizationRegistry(store=org_store)
    return archetype_reg, zone_val, org_reg


# ── CLI commands ─────────────────────────────────────────────────────────


def _cmd_birth(args: argparse.Namespace) -> None:
    ca, store = _build_ca(args.base_dir)
    zone_path = args.zone or "global"
    arch_reg, zone_val, _org_reg = _build_registries(args.base_dir)

    # Validate archetype
    archetype = args.archetype
    arch_result = arch_reg.validate(archetype)
    if not arch_result.valid:
        msg = f"Invalid archetype '{archetype}': {'; '.join(arch_result.errors)}"
        if arch_result.suggestion:
            msg += f"\n  Did you mean '{arch_result.suggestion}'?"
        print(msg, file=sys.stderr)
        sys.exit(1)
    if arch_result.warnings:
        for w in arch_result.warnings:
            print(f"  Warning: {w}", file=sys.stderr)
        if arch_result.suggestion:
            print(f"  Did you mean '{arch_result.suggestion}'?", file=sys.stderr)

    # Validate zone
    zone_path = args.zone or "global"
    zone_result = zone_val.validate(zone_path)
    if not zone_result.valid:
        print(f"Invalid zone path '{zone_path}': {'; '.join(zone_result.errors)}", file=sys.stderr)
        sys.exit(1)

    cert, agent_sk = ca.issue(
        agent_id=args.agent_id,
        archetype=archetype,
        organization=args.org,
        zone_path=zone_path,
        owner=args.owner or "",
    )
    # Save agent keys alongside certificate
    store.save_agent_key(cert.certificate_id, agent_sk.to_bytes())
    store.save_agent_pub(cert.certificate_id, cert.public_key)

    print(f"Certificate issued: {cert.certificate_id}")
    print(f"  Agent:     {cert.agent_id}")
    print(f"  Owner:     {cert.owner}")
    print(f"  Archetype: {cert.archetype}")
    print(f"  Org:       {cert.organization}")
    print(f"  Zone:      {cert.zone_path}")
    print(f"  Trust:     {cert.trust_score}")
    print(f"  Age:       {cert.behavioral_age}")
    print(f"  Status:    {cert.status.name}")
    print(f"  Issued:    {cert.issued_at.isoformat()}")
    print(f"  Fingerprint: {cert.fingerprint}")


def _cmd_verify(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert = ca.get(args.cert_id)
    if cert is None:
        print(f"Certificate not found: {args.cert_id}", file=sys.stderr)
        sys.exit(1)
    result = ca.verify_certificate(cert)
    if result.valid:
        print(f"VALID — {cert.certificate_id}")
        print(f"  Status: {cert.status.name}")
        print(f"  Trust:  {cert.trust_score}")
        print(f"  Age:    {cert.behavioral_age}")
    else:
        print(f"INVALID — {cert.certificate_id}")
        for issue in result.issues:
            print(f"  - {issue}")
        sys.exit(1)


def _cmd_inspect(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert = ca.get(args.cert_id)
    if cert is None:
        print(f"Certificate not found: {args.cert_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(cert.to_dict(), indent=2, sort_keys=True))


def _cmd_suspend(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    try:
        cert = ca.suspend(args.cert_id, args.reason)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Suspended: {cert.certificate_id}")


def _cmd_reactivate(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    try:
        cert = ca.reactivate(args.cert_id)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Reactivated: {cert.certificate_id}")


def _cmd_revoke(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    try:
        cert = ca.revoke(args.cert_id, args.reason)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Revoked: {cert.certificate_id}")


def _cmd_renew(args: argparse.Namespace) -> None:
    ca, store = _build_ca(args.base_dir)
    try:
        cert, agent_sk = ca.renew(args.cert_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    store.save_agent_key(cert.certificate_id, agent_sk.to_bytes())
    store.save_agent_pub(cert.certificate_id, cert.public_key)
    print(f"Renewed: {cert.certificate_id}")
    print(f"  Lineage: {cert.lineage}")


def _cmd_list(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    status = CertStatus[args.status.upper()] if args.status else None
    certs = ca.list(org=args.org, status=status, archetype=args.archetype)
    if not certs:
        print("No certificates found.")
        return
    for cert in certs:
        print(f"{cert.certificate_id}  {cert.agent_id:20s}  {cert.status.name:10s}  trust={cert.trust_score:.2f}  age={cert.behavioral_age}")


def _cmd_reputation(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert = ca.get(args.cert_id)
    if cert is None:
        print(f"Certificate not found: {args.cert_id}", file=sys.stderr)
        sys.exit(1)
    print(f"Reputation for {cert.certificate_id}")
    print(f"  Agent:          {cert.agent_id}")
    print(f"  Owner:          {cert.owner}")
    print(f"  Trust Score:    {cert.trust_score}")
    print(f"  Behavioral Age: {cert.behavioral_age}")
    print(f"  Status:         {cert.status.name}")
    print(f"  Archetype:      {cert.archetype}")
    print(f"  Issued:         {cert.issued_at.isoformat()}")
    if cert.lineage:
        print(f"  Lineage:        {cert.lineage}")


def _cmd_export(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert = ca.get(args.cert_id)
    if cert is None:
        print(f"Certificate not found: {args.cert_id}", file=sys.stderr)
        sys.exit(1)
    # Export public certificate (no private key)
    filename = f"{cert.certificate_id}.cert.json"
    Path(filename).write_text(
        json.dumps(cert.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Exported to {filename}")


# ── Archetype commands ───────────────────────────────────────────────────


def _cmd_archetype(args: argparse.Namespace) -> None:
    arch_reg = ArchetypeRegistry.with_defaults()
    sub = args.archetype_command

    if sub == "list":
        archetypes = arch_reg.list(category=args.category)
        if not archetypes:
            print("No archetypes found.")
            return
        for a in archetypes:
            print(f"  {a.name:30s}  {a.category:20s}  {'(builtin)' if a.builtin else '(custom)'}")
            print(f"    {a.description}")

    elif sub == "register":
        try:
            defn = arch_reg.register(args.name, args.description, args.category)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Registered archetype: {defn.name}")

    elif sub == "validate":
        result = arch_reg.validate(args.name)
        if result.valid:
            print(f"VALID: {result.name}")
            for w in result.warnings:
                print(f"  Warning: {w}")
            if result.suggestion:
                print(f"  Suggestion: {result.suggestion}")
        else:
            print(f"INVALID: {result.name}")
            for e in result.errors:
                print(f"  Error: {e}")
            if result.suggestion:
                print(f"  Did you mean '{result.suggestion}'?")
            sys.exit(1)

    else:
        print("Unknown archetype subcommand", file=sys.stderr)
        sys.exit(1)


# ── Organization commands ────────────────────────────────────────────────


def _cmd_org(args: argparse.Namespace) -> None:
    _arch_reg, _zone_val, org_reg = _build_registries(args.base_dir)
    sub = args.org_command

    if sub == "register":
        sk, vk, _issuer_id = _load_or_create_issuer(args.base_dir)
        try:
            org = org_reg.register(
                args.name,
                vk.fingerprint(),
                contact_email=getattr(args, "email", None),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Registered organization: {org.name}")
        print(f"  Display name: {org.display_name}")
        print(f"  Issuer:       {org.issuer_fingerprint}")

    elif sub == "list":
        status_str = getattr(args, "status", None)
        status = OrgStatus[status_str.upper()] if status_str else None
        orgs = org_reg.list(status=status)
        if not orgs:
            print("No organizations found.")
            return
        for org in orgs:
            print(f"  {org.name:30s}  {org.status.name:10s}  {org.display_name}")

    elif sub == "validate":
        result = org_reg.validate(args.name)
        if result.valid:
            print(f"VALID: {result.name} (available)")
        else:
            print(f"INVALID: {result.name}")
            for e in result.errors:
                print(f"  Error: {e}")
            sys.exit(1)

    else:
        print("Unknown org subcommand", file=sys.stderr)
        sys.exit(1)


# ── Zone commands ────────────────────────────────────────────────────────


def _cmd_zone(args: argparse.Namespace) -> None:
    zone_val = ZoneValidator()
    sub = args.zone_command

    if sub == "validate":
        result = zone_val.validate(args.path)
        if result.valid:
            print(f"VALID: {result.name}")
        else:
            print(f"INVALID: {result.name}")
            for e in result.errors:
                print(f"  Error: {e}")
            sys.exit(1)

    else:
        print("Unknown zone subcommand", file=sys.stderr)
        sys.exit(1)


# ── Serve command ────────────────────────────────────────────────────────


def _cmd_hello(args: argparse.Namespace) -> None:
    """Run the hello-nomo governance demo (all in-memory)."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from typing import Any

    from nomotic.authority import CertificateAuthority
    from nomotic.headers import generate_headers
    from nomotic.middleware import GatewayConfig, NomoticGateway
    from nomotic.sdk import GovernedAgent
    from nomotic.store import MemoryCertificateStore

    # ANSI helpers
    _no_color = not sys.stdout.isatty()

    def _c(code: str, text: str) -> str:
        return text if _no_color else f"\033[{code}m{text}\033[0m"

    def _bold(t: str) -> str:
        return _c("1", t)

    def _green(t: str) -> str:
        return _c("32", t)

    def _red(t: str) -> str:
        return _c("31", t)

    def _yellow(t: str) -> str:
        return _c("33", t)

    def _cyan(t: str) -> str:
        return _c("36", t)

    # Set up infrastructure
    print()
    print(_bold(_cyan("=" * 60)))
    print(_bold(_cyan("  Nomotic Governance Demo")))
    print(_bold(_cyan("=" * 60)))
    print()

    issuer_sk, issuer_vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(
        issuer_id="hello-nomo-issuer",
        signing_key=issuer_sk,
        store=store,
    )
    print(f"  {_green('OK')} CertificateAuthority created")

    cert, agent_sk = ca.issue(
        agent_id="hello-agent",
        archetype="customer-experience",
        organization="hello-corp",
        zone_path="global/us",
    )
    print(f"  {_green('OK')} Certificate issued: {cert.certificate_id}")
    print(f"      Agent: hello-agent | Archetype: customer-experience")
    print(f"      Trust: {cert.trust_score} | Age: {cert.behavioral_age}")

    gateway = NomoticGateway(config=GatewayConfig(
        require_cert=True,
        min_trust=0.3,
        local_ca=ca,
        verify_signature=True,
    ))

    # Tiny service handler
    request_log: list[dict[str, Any]] = []

    class _DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            pass

        def do_GET(self) -> None:
            hdrs = {k: v for k, v in self.headers.items()}
            result = gateway.check(hdrs)
            request_log.append({"allowed": result.allowed, "reason": result.reason})
            if not result.allowed:
                body = json.dumps(result.to_dict()).encode()
                self.send_response(403)
            else:
                body = json.dumps({"status": "ok", "trust": result.trust_score}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _DemoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  {_green('OK')} Service running on http://127.0.0.1:{port}")

    agent = GovernedAgent(
        certificate=cert,
        signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )

    # Normal requests
    print()
    print(_bold("  Normal operations:"))
    trust_history = [cert.trust_score]
    for i in range(5):
        resp = agent.get("/api/data")
        ca.record_action(cert.certificate_id, min(0.95, cert.trust_score + 0.05))
        trust_history.append(cert.trust_score)
        status = _green(str(resp.status)) if resp.ok else _red(str(resp.status))
        print(f"    [{i+1}] GET /api/data -> {status}  trust={_green(f'{cert.trust_score:.2f}')}")
        # Recreate agent to pick up new trust in headers
        agent = GovernedAgent(
            certificate=cert, signing_key=agent_sk,
            base_url=f"http://127.0.0.1:{port}",
        )

    # Trust drop
    print()
    print(_bold("  Trust degradation:"))
    ca.update_trust(cert.certificate_id, 0.2)
    trust_history.append(cert.trust_score)
    agent = GovernedAgent(
        certificate=cert, signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )
    resp = agent.get("/api/data")
    status = _green(str(resp.status)) if resp.ok else _red(str(resp.status))
    print(f"    Trust dropped to {_red(f'{cert.trust_score:.2f}')} -> {status} (gateway denies below 0.30)")

    # Recovery
    ca.update_trust(cert.certificate_id, 0.6)
    trust_history.append(cert.trust_score)
    agent = GovernedAgent(
        certificate=cert, signing_key=agent_sk,
        base_url=f"http://127.0.0.1:{port}",
    )
    resp = agent.get("/api/data")
    status = _green(str(resp.status)) if resp.ok else _red(str(resp.status))
    print(f"    Trust restored to {_green(f'{cert.trust_score:.2f}')} -> {status} (access restored)")

    # Summary
    print()
    print(_bold("  Trust trajectory:"))
    for i, t in enumerate(trust_history):
        bar_len = int(t * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        color = _green if t >= 0.5 else (_yellow if t >= 0.3 else _red)
        print(f"    [{i:2d}] {color(f'{t:.2f}')} |{bar}|")

    allowed = sum(1 for r in request_log if r["allowed"])
    denied = len(request_log) - allowed
    print()
    print(f"  Requests: {len(request_log)} total, {_green(str(allowed))} allowed, {_red(str(denied))} denied")

    # Behavioral fingerprint demo
    from nomotic.monitor import DriftConfig as _DriftConfig
    from nomotic.runtime import GovernanceRuntime, RuntimeConfig
    from nomotic.types import Action as GovAction, AgentContext as GovCtx, TrustProfile as GovTP

    print()
    print(_bold("  Behavioral Fingerprint:"))
    drift_cfg = _DriftConfig(window_size=15, check_interval=5, min_observations=5)
    runtime = GovernanceRuntime(config=RuntimeConfig(enable_fingerprints=True, drift_config=drift_cfg))
    demo_actions = [
        ("read", "/api/data"),
        ("read", "/api/data"),
        ("read", "/api/users"),
        ("write", "/api/data"),
        ("read", "/api/data"),
    ]
    for action_type, target in demo_actions:
        action = GovAction(agent_id="hello-agent", action_type=action_type, target=target)
        ctx = GovCtx(agent_id="hello-agent", trust_profile=GovTP(agent_id="hello-agent"))
        runtime.evaluate(action, ctx)

    fp = runtime.get_fingerprint("hello-agent")
    if fp is not None:
        obs = fp.total_observations
        print(f"    After {obs} governance evaluations:")
        if fp.action_distribution:
            parts = [f"{k}={v:.0%}" for k, v in sorted(fp.action_distribution.items(), key=lambda x: -x[1])]
            print(f"      Actions:  {', '.join(parts)}  (confidence: {fp.confidence:.2f})")
        if fp.target_distribution:
            parts = [f"{k}={v:.0%}" for k, v in sorted(fp.target_distribution.items(), key=lambda x: -x[1])]
            print(f"      Targets:  {', '.join(parts)}")
        if fp.outcome_distribution:
            parts = [f"{k}={v:.0%}" for k, v in sorted(fp.outcome_distribution.items(), key=lambda x: -x[1])]
            print(f"      Outcomes: {', '.join(parts)}")

    # Drift detection demo
    print()
    print(_bold("  Drift Detection:"))
    print("    Building normal baseline (20 read operations)...")
    for i in range(20):
        action = GovAction(agent_id="hello-agent", action_type="read", target="/api/data")
        ctx = GovCtx(agent_id="hello-agent", trust_profile=GovTP(agent_id="hello-agent"))
        runtime.evaluate(action, ctx)

    drift_score = runtime.get_drift("hello-agent")
    if drift_score is not None:
        print(f"    Baseline drift: {_green(f'{drift_score.overall:.2f}')} ({drift_score.severity})")
    else:
        print(f"    Baseline drift: {_green('not yet computed')}")

    print("    Agent changes behavior to mass deletes on sensitive targets...")
    # Shift behavior to mostly deletes on sensitive targets
    drift_actions = [
        ("delete", "/api/sensitive/data"),
        ("delete", "/api/user/records"),
        ("delete", "/api/config/keys"),
        ("delete", "/api/sensitive/data"),
        ("delete", "/api/user/records"),
        ("delete", "/api/config/keys"),
        ("delete", "/api/sensitive/data"),
        ("delete", "/api/user/records"),
        ("delete", "/api/config/keys"),
        ("delete", "/api/sensitive/data"),
        ("delete", "/api/user/records"),
        ("delete", "/api/config/keys"),
        ("delete", "/api/sensitive/data"),
        ("delete", "/api/user/records"),
        ("delete", "/api/config/keys"),
    ]
    for i, (action_type, target) in enumerate(drift_actions):
        action = GovAction(agent_id="hello-agent", action_type=action_type, target=target)
        ctx = GovCtx(agent_id="hello-agent", trust_profile=GovTP(agent_id="hello-agent"))
        runtime.evaluate(action, ctx)
        status = _red(action_type.upper())
        print(f"    [{i+1:2d}] {status} {target}")

    drift_score = runtime.get_drift("hello-agent")
    if drift_score is not None:
        sev = drift_score.severity
        color = _red if sev in ("high", "critical") else (_yellow if sev == "moderate" else _green)
        print()
        print(f"    Drift detected!")
        print(f"      Overall: {color(f'{drift_score.overall:.2f}')} ({sev})")
        print(f"      Action:  {drift_score.action_drift:.2f}")
        print(f"      Target:  {drift_score.target_drift:.2f}")
        if drift_score.detail:
            print(f"      Detail:  {drift_score.detail}")

    alerts = runtime.get_drift_alerts("hello-agent")
    if alerts:
        print(f"    Alerts: {len(alerts)}")
        for alert in alerts:
            color = _red if alert.severity == "critical" else (_yellow if alert.severity == "high" else _cyan)
            print(f"      {color(alert.severity)}: {alert.drift_score.detail}")
    print()

    server.shutdown()


def _cmd_drift(args: argparse.Namespace) -> None:
    """Show the current behavioral drift for an agent (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/drift/{args.agent_id}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"No drift data for agent: {args.agent_id}", file=sys.stderr)
            sys.exit(1)
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        print("Is the Nomotic API server running? (nomotic serve)", file=sys.stderr)
        sys.exit(1)

    drift = data["drift"]
    print(f"Behavioral Drift: {data['agent_id']}")
    print(f"  Overall:    {drift['overall']:.4f} ({drift['severity']})")
    print(f"  Confidence: {drift['confidence']:.2f}")
    print()
    print("  Per-distribution:")
    for name in ["action_drift", "target_drift", "temporal_drift", "outcome_drift"]:
        val = drift[name]
        bar_full = int(val * 10)
        bar = "#" * bar_full + "." * (10 - bar_full)
        label = name.replace("_drift", "").title()
        print(f"    {label:10s} {val:.2f}  {bar}")
    print()
    if drift.get("detail"):
        print(f"  Detail: {drift['detail']}")
        print()

    alerts = data.get("alerts", [])
    unacked = sum(1 for a in alerts if not a.get("acknowledged"))
    if alerts:
        print(f"  Alerts: {unacked} unacknowledged")
        for i, alert in enumerate(alerts):
            ack = " (ack)" if alert.get("acknowledged") else ""
            print(f"    [{i}] {alert['severity']}{ack} - {alert['drift_score'].get('detail', '')}")
    print()


def _cmd_alerts(args: argparse.Namespace) -> None:
    """List drift alerts (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/alerts"
    if args.agent_id:
        url += f"?agent_id={args.agent_id}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        print("Is the Nomotic API server running? (nomotic serve)", file=sys.stderr)
        sys.exit(1)

    alerts = data.get("alerts", [])
    if args.unacknowledged:
        alerts = [a for a in alerts if not a.get("acknowledged")]

    if not alerts:
        print("No drift alerts.")
        return

    print(f"Drift Alerts ({len(alerts)} total, {data.get('unacknowledged', 0)} unacknowledged):")
    for i, alert in enumerate(alerts):
        ack = " [ack]" if alert.get("acknowledged") else ""
        agent = alert.get("agent_id", "?")
        sev = alert.get("severity", "?")
        detail = alert.get("drift_score", {}).get("detail", "")
        print(f"  [{i}] {agent:20s} {sev:10s}{ack}  {detail}")


def _cmd_fingerprint(args: argparse.Namespace) -> None:
    """Show the behavioral fingerprint for an agent (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/fingerprint/{args.agent_id}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"No fingerprint found for agent: {args.agent_id}", file=sys.stderr)
            sys.exit(1)
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        print("Is the Nomotic API server running? (nomotic serve)", file=sys.stderr)
        sys.exit(1)

    print(f"Behavioral Fingerprint for {data['agent_id']}")
    print(f"  Observations: {data['total_observations']}")
    print(f"  Confidence:   {data['confidence']:.3f}")
    print()

    if data.get("action_distribution"):
        print("  Action Distribution:")
        for action_type, freq in sorted(data["action_distribution"].items(), key=lambda x: -x[1]):
            bar = "#" * int(freq * 40)
            print(f"    {action_type:20s} {freq:6.1%} {bar}")
        print()

    if data.get("target_distribution"):
        print("  Target Distribution:")
        for target, freq in sorted(data["target_distribution"].items(), key=lambda x: -x[1]):
            bar = "#" * int(freq * 40)
            print(f"    {target:20s} {freq:6.1%} {bar}")
        print()

    tp = data.get("temporal_pattern", {})
    if tp.get("active_hours"):
        print(f"  Temporal Pattern:")
        print(f"    Active hours: {tp['active_hours']}")
        print(f"    Rate: {tp.get('actions_per_hour_mean', 0):.1f} +/- {tp.get('actions_per_hour_std', 0):.1f} actions/hr")
        print()

    if data.get("outcome_distribution"):
        print("  Outcome Distribution:")
        for outcome, freq in sorted(data["outcome_distribution"].items(), key=lambda x: -x[1]):
            bar = "#" * int(freq * 40)
            print(f"    {outcome:20s} {freq:6.1%} {bar}")
        print()


def _cmd_serve(args: argparse.Namespace) -> None:
    from nomotic.api import NomoticAPIServer

    ca, _store = _build_ca(args.base_dir)
    arch_reg, zone_val, org_reg = _build_registries(args.base_dir)

    server = NomoticAPIServer(
        ca,
        archetype_registry=arch_reg,
        zone_validator=zone_val,
        org_registry=org_reg,
        host=args.host,
        port=args.port,
    )
    sk, vk, issuer_id = _load_or_create_issuer(args.base_dir)
    print(f"Nomotic API server starting on http://{args.host}:{args.port}")
    print(f"  Issuer: {issuer_id}")
    print(f"  Fingerprint: {vk.fingerprint()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# ── Argument parser ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nomotic",
        description="Nomotic — runtime governance for agentic AI",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_DEFAULT_BASE,
        help="Base directory for nomotic data (default: ~/.nomotic)",
    )

    sub = parser.add_subparsers(dest="command")

    # birth
    birth = sub.add_parser("birth", help="Issue a new agent certificate")
    birth.add_argument("--agent-id", required=True, help="Agent identifier")
    birth.add_argument("--owner", default=None, help="Accountable owner of the agent")
    birth.add_argument("--archetype", required=True, help="Behavioral archetype")
    birth.add_argument("--org", required=True, help="Organization")
    birth.add_argument("--zone", default=None, help="Governance zone path")

    # verify
    verify = sub.add_parser("verify", help="Verify a certificate")
    verify.add_argument("cert_id", help="Certificate ID")

    # inspect
    inspect_ = sub.add_parser("inspect", help="Inspect a certificate")
    inspect_.add_argument("cert_id", help="Certificate ID")

    # suspend
    suspend = sub.add_parser("suspend", help="Suspend a certificate")
    suspend.add_argument("cert_id", help="Certificate ID")
    suspend.add_argument("--reason", required=True, help="Reason for suspension")

    # reactivate
    reactivate = sub.add_parser("reactivate", help="Reactivate a suspended certificate")
    reactivate.add_argument("cert_id", help="Certificate ID")

    # revoke
    revoke = sub.add_parser("revoke", help="Permanently revoke a certificate")
    revoke.add_argument("cert_id", help="Certificate ID")
    revoke.add_argument("--reason", required=True, help="Reason for revocation")

    # renew
    renew = sub.add_parser("renew", help="Renew a certificate with lineage link")
    renew.add_argument("cert_id", help="Certificate ID")

    # list
    list_ = sub.add_parser("list", help="List certificates")
    list_.add_argument("--status", default=None, help="Filter by status")
    list_.add_argument("--archetype", default=None, help="Filter by archetype")
    list_.add_argument("--org", default=None, help="Filter by organization")

    # reputation
    rep = sub.add_parser("reputation", help="Show trust trajectory")
    rep.add_argument("cert_id", help="Certificate ID")

    # export
    export = sub.add_parser("export", help="Export public certificate to file")
    export.add_argument("cert_id", help="Certificate ID")

    # ── archetype subcommands ────────────────────────────────────────
    archetype = sub.add_parser("archetype", help="Manage archetypes")
    arch_sub = archetype.add_subparsers(dest="archetype_command")

    arch_list = arch_sub.add_parser("list", help="List archetypes")
    arch_list.add_argument("--category", default=None, help="Filter by category")

    arch_reg = arch_sub.add_parser("register", help="Register a custom archetype")
    arch_reg.add_argument("--name", required=True, help="Archetype name")
    arch_reg.add_argument("--description", required=True, help="Description")
    arch_reg.add_argument("--category", required=True, help="Category")

    arch_val = arch_sub.add_parser("validate", help="Validate an archetype name")
    arch_val.add_argument("name", help="Archetype name to validate")

    # ── org subcommands ──────────────────────────────────────────────
    org = sub.add_parser("org", help="Manage organizations")
    org_sub = org.add_subparsers(dest="org_command")

    org_register = org_sub.add_parser("register", help="Register an organization")
    org_register.add_argument("--name", required=True, help="Organization name")
    org_register.add_argument("--email", default=None, help="Contact email")

    org_list = org_sub.add_parser("list", help="List organizations")
    org_list.add_argument("--status", default=None, help="Filter by status")

    org_val = org_sub.add_parser("validate", help="Validate an org name")
    org_val.add_argument("name", help="Organization name to validate")

    # ── zone subcommands ─────────────────────────────────────────────
    zone = sub.add_parser("zone", help="Manage governance zones")
    zone_sub = zone.add_subparsers(dest="zone_command")

    zone_val = zone_sub.add_parser("validate", help="Validate a zone path")
    zone_val.add_argument("path", help="Zone path to validate")

    # ── serve ────────────────────────────────────────────────────────
    serve = sub.add_parser("serve", help="Start the Nomotic API server")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")

    # ── hello ────────────────────────────────────────────────────────
    sub.add_parser("hello", help="Run the hello-nomo governance demo (in-memory)")

    # ── fingerprint ─────────────────────────────────────────────────
    fp_parser = sub.add_parser("fingerprint", help="Show behavioral fingerprint for an agent")
    fp_parser.add_argument("agent_id", help="Agent identifier")
    fp_parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    fp_parser.add_argument("--port", type=int, default=8420, help="API server port (default: 8420)")

    # ── drift ────────────────────────────────────────────────────
    drift_parser = sub.add_parser("drift", help="Show behavioral drift for an agent")
    drift_parser.add_argument("agent_id", help="Agent identifier")
    drift_parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    drift_parser.add_argument("--port", type=int, default=8420, help="API server port (default: 8420)")

    # ── alerts ───────────────────────────────────────────────────
    alerts_parser = sub.add_parser("alerts", help="List drift alerts")
    alerts_parser.add_argument("--agent-id", default=None, help="Filter by agent ID")
    alerts_parser.add_argument("--unacknowledged", action="store_true", help="Show only unacknowledged alerts")
    alerts_parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    alerts_parser.add_argument("--port", type=int, default=8420, help="API server port (default: 8420)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "birth": _cmd_birth,
        "verify": _cmd_verify,
        "inspect": _cmd_inspect,
        "suspend": _cmd_suspend,
        "reactivate": _cmd_reactivate,
        "revoke": _cmd_revoke,
        "renew": _cmd_renew,
        "list": _cmd_list,
        "reputation": _cmd_reputation,
        "export": _cmd_export,
        "archetype": _cmd_archetype,
        "org": _cmd_org,
        "zone": _cmd_zone,
        "serve": _cmd_serve,
        "hello": _cmd_hello,
        "fingerprint": _cmd_fingerprint,
        "drift": _cmd_drift,
        "alerts": _cmd_alerts,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
