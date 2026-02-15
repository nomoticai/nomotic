"""Nomotic CLI — manage agent birth certificates from the command line.

Usage::

    nomotic setup
    nomotic birth --name my-agent
    nomotic birth --name my-agent --archetype customer-experience --org acme
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
    nomotic archetype set <agent>
    nomotic archetype register --name <name> --description <desc> --category <cat>
    nomotic archetype validate <name>

    nomotic org register --name <name> [--email <email>]
    nomotic org list [--status ACTIVE]
    nomotic org validate <name>

    nomotic zone validate <path>

    nomotic serve [--host 0.0.0.0] [--port 8420]

Configuration is stored in ``~/.nomotic/config.json``.
Certificates are stored in ``~/.nomotic/certs/``.
The issuer key is stored in ``~/.nomotic/issuer/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from nomotic.authority import CertificateAuthority
from nomotic.certificate import AgentCertificate, CertStatus
from nomotic.id_registry import AgentIdRegistry
from nomotic.keys import SigningKey, VerifyKey
from nomotic.registry import (
    ArchetypeRegistry,
    FileOrgStore,
    OrganizationRegistry,
    OrgStatus,
    ZoneValidator,
    _normalize_org_name,
)
from nomotic.store import FileCertificateStore

__all__ = ["main"]

_DEFAULT_BASE = Path.home() / ".nomotic"


# ── ANSI color support ───────────────────────────────────────────────────


def _init_colors() -> bool:
    """Initialize ANSI color support. Returns True if colors are available."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Enable ANSI escape sequences on Windows 10+
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


_COLORS_ENABLED = _init_colors()


def _c(code: str, text: str) -> str:
    return text if not _COLORS_ENABLED else f"\033[{code}m{text}\033[0m"


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


def _dim(t: str) -> str:
    return _c("2", t)


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


# ── Nomotic config (setup defaults) ─────────────────────────────────────


def _load_nomotic_config(base: Path) -> dict:
    """Load the nomotic config, or return empty dict if not set up."""
    config_path = base / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


# ── Agent ID registry helpers ────────────────────────────────────────────


def _load_registry(base: Path) -> AgentIdRegistry:
    """Load the AgentIdRegistry, running migration on first use."""
    registry = AgentIdRegistry(base)
    if not registry._path.exists():
        # First use — migrate existing certs
        certs_dir = base / "certs"
        registry.migrate_from_certs(certs_dir)
        registry._save()
    return registry


# ── Identifier resolution ────────────────────────────────────────────────


def _resolve_agent(base: Path, identifier: str) -> tuple[int, str, str]:
    """Resolve an identifier to (agent_numeric_id, name, cert_id).

    Uses the AgentIdRegistry for resolution.
    Falls back to scanning cert files for old-format agents without
    registry entries.
    """
    registry = _load_registry(base)

    try:
        match = registry.resolve_single(identifier)
        return match["agent_id"], match["name"], match["certificate_id"]
    except ValueError:
        pass

    # Fall back: try registry.resolve for multiple matches
    matches = registry.resolve(identifier)
    if len(matches) > 1:
        print(f"Ambiguous identifier '{identifier}' matches multiple agents:", file=sys.stderr)
        for m in matches:
            print(f"  ID {m['agent_id']}: {m['name']} ({m['certificate_id']})", file=sys.stderr)
        print("  Use numeric ID or name+ID combo to disambiguate.", file=sys.stderr)
        sys.exit(1)

    if len(matches) == 1:
        m = matches[0]
        return m["agent_id"], m["name"], m["certificate_id"]

    # Fall back to scanning cert files for old-format agents
    if identifier.startswith("nmc-"):
        # Direct cert-id — check file exists
        certs_dir = base / "certs"
        cert_path = certs_dir / f"{identifier}.json"
        if cert_path.exists():
            data = json.loads(cert_path.read_text(encoding="utf-8"))
            return data.get("agent_numeric_id", 0), data.get("agent_id", ""), identifier
        revoked_path = base / "revoked" / f"{identifier}.json"
        if revoked_path.exists():
            data = json.loads(revoked_path.read_text(encoding="utf-8"))
            return data.get("agent_numeric_id", 0), data.get("agent_id", ""), identifier

    # Try scanning certs for agent name match (old-format without registry)
    certs_dir = base / "certs"
    if certs_dir.exists():
        for cert_file in certs_dir.glob("nmc-*.json"):
            try:
                data = json.loads(cert_file.read_text(encoding="utf-8"))
                if data.get("agent_id") == identifier:
                    return (
                        data.get("agent_numeric_id", 0),
                        data.get("agent_id", ""),
                        data["certificate_id"],
                    )
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"No agent found for identifier '{identifier}'.", file=sys.stderr)
    print("  Use 'nomotic list' to see available agents.", file=sys.stderr)
    sys.exit(1)


def _resolve_cert_id(base: Path, identifier: str) -> str:
    """Backward-compatible wrapper: resolve identifier to cert_id only."""
    _numeric_id, _name, cert_id = _resolve_agent(base, identifier)
    return cert_id


# ── CLI commands ─────────────────────────────────────────────────────────


def _cmd_setup(args: argparse.Namespace) -> None:
    """Interactive first-run configuration."""
    config_path = args.base_dir / "config.json"

    # Load existing config if present (for re-running setup)
    existing = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

    print()
    print(f"  {_bold('Nomotic Setup')}")
    print(f"  {'─' * 35}")
    print()

    # Organization
    default_org = existing.get("organization", "")
    prompt = "  Organization name"
    if default_org:
        prompt += f" [{default_org}]"
    prompt += ": "
    org_input = input(prompt).strip() or default_org
    org = _normalize_org_name(org_input)
    if org_input != org:
        print(f"    \u2192 Normalized to: {org}")
    print()

    # Owner
    default_owner = existing.get("owner", "")
    prompt = "  Your name or email (agent owner)"
    if default_owner:
        prompt += f" [{default_owner}]"
    prompt += ": "
    owner = input(prompt).strip() or default_owner
    print()

    # Zone
    default_zone = existing.get("default_zone", "global")
    print("  Zones define where agents operate (e.g., global, us/production, eu/staging).")
    prompt = f"  Default governance zone [{default_zone}]: "
    zone_input = input(prompt).strip() or default_zone
    zone = zone_input.lower()
    if zone_input != zone:
        print(f"    \u2192 Normalized to: {zone}")
    print()

    # Save
    config = {
        "organization": org,
        "owner": owner,
        "default_zone": zone,
        "retention": existing.get("retention", "5y"),
        "enforce_unique_names": existing.get("enforce_unique_names", False),
    }
    args.base_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"  {_green('Setup complete.')} Configuration saved to {config_path}")
    print()
    print("  Your defaults:")
    print(f"    Organization: {org}")
    print(f"    Owner:        {owner}")
    print(f"    Zone:         {zone}")
    print()
    print(f"  You can change these anytime with 'nomotic setup' or edit {config_path}")
    print()
    print("  Next: Create your first agent with 'nomotic birth --name my-agent'")
    print()


def _cmd_birth(args: argparse.Namespace) -> None:
    # Load config defaults
    config = _load_nomotic_config(args.base_dir)

    # Resolve name from --name or --agent-id (backward compat)
    agent_name = args.name or getattr(args, "agent_id_compat", None)
    if not agent_name:
        print("Error: --name is required.", file=sys.stderr)
        sys.exit(1)
    args.name = agent_name

    # Merge config defaults with CLI args
    org_raw = args.org or config.get("organization", "")
    owner = args.owner or config.get("owner", "")
    zone_raw = args.zone or config.get("default_zone", "global")
    archetype_raw = (args.archetype or "general-purpose")

    # First-run guard: need at least an org
    if not org_raw:
        print("Nomotic is not configured yet.", file=sys.stderr)
        print("  Run 'nomotic setup' first, or provide --org.", file=sys.stderr)
        sys.exit(1)

    # Normalize zone, org, archetype
    org = _normalize_org_name(org_raw)
    zone_path = zone_raw.lower()
    archetype = archetype_raw.lower()

    ca, store = _build_ca(args.base_dir)
    arch_reg, zone_val, _org_reg = _build_registries(args.base_dir)

    # Load or create the AgentIdRegistry
    registry = _load_registry(args.base_dir)

    # Check for duplicate active name within the same org via registry
    existing = registry.list_all(status="ACTIVE")
    for entry in existing:
        if entry["name"] == args.name and entry["organization"] == org:
            print(f"Error: An active agent named '{args.name}' already exists in org '{org}'.", file=sys.stderr)
            print(f"  ID: {entry['agent_id']}, Certificate: {entry['certificate_id']}", file=sys.stderr)
            print(f"  Use a different --name, or revoke the existing agent first.", file=sys.stderr)
            sys.exit(1)

    # Validate archetype
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

    # Validate zone (after normalization)
    zone_result = zone_val.validate(zone_path)
    if not zone_result.valid:
        error = zone_result.errors[0] if zone_result.errors else "Invalid zone format"
        print(f"Invalid zone '{zone_path}': {error}", file=sys.stderr)
        print("  Zones use lowercase with slashes: global, us/production, eu/staging", file=sys.stderr)
        sys.exit(1)

    # Allocate sequential numeric ID
    numeric_id = registry.next_id()

    cert, agent_sk = ca.issue(
        agent_id=args.name,
        archetype=archetype,
        organization=org,
        zone_path=zone_path,
        owner=owner,
    )

    # Set numeric ID on certificate and re-save
    cert.agent_numeric_id = numeric_id
    store.update(cert)

    # Register in the ID registry
    registry.register(numeric_id, args.name, cert.certificate_id, org)

    # Save agent keys alongside certificate
    store.save_agent_key(cert.certificate_id, agent_sk.to_bytes())
    store.save_agent_pub(cert.certificate_id, cert.public_key)

    print()
    print(f"Agent created: {_bold(args.name)}")
    if cert.agent_numeric_id:
        print(f"  ID:          {cert.agent_numeric_id}")
    print(f"  Name:        {cert.agent_id}")
    print(f"  Certificate: {cert.certificate_id}")
    print(f"  Archetype:   {archetype}")
    print(f"  Org:         {org}")
    print(f"  Zone:        {zone_path}")
    print(f"  Owner:       {owner}")
    print(f"  Trust:       {cert.trust_score:.2f}")
    print(f"  Status:      {cert.status.name}")

    # Prompt to set archetype if they didn't specify one
    if not args.archetype:
        print()
        print("  Archetype defaulted to 'general-purpose'.")
        print(f"  Set a specific archetype with: nomotic archetype set {args.name}")

    # Next steps
    print()
    print("  Next steps:")
    print(f"    nomotic scope {args.name} set --actions read,write --boundaries my_db")
    print(f"    nomotic eval {args.name} --action read --target my_db")


def _cmd_verify(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    cert = ca.get(cert_id)
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
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    cert = ca.get(cert_id)
    if cert is None:
        print(f"Certificate not found: {args.cert_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(cert.to_dict(), indent=2, sort_keys=True))


def _cmd_suspend(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    numeric_id, _name, cert_id = _resolve_agent(args.base_dir, args.cert_id)
    try:
        cert = ca.suspend(cert_id, args.reason)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    # Update registry status
    if numeric_id > 0:
        registry = _load_registry(args.base_dir)
        try:
            registry.update_status(numeric_id, "SUSPENDED")
        except KeyError:
            pass
    print(f"Suspended: {cert.certificate_id}")


def _cmd_reactivate(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    numeric_id, _name, cert_id = _resolve_agent(args.base_dir, args.cert_id)
    try:
        cert = ca.reactivate(cert_id)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    # Update registry status
    if numeric_id > 0:
        registry = _load_registry(args.base_dir)
        try:
            registry.update_status(numeric_id, "ACTIVE")
        except KeyError:
            pass
    print(f"Reactivated: {cert.certificate_id}")


def _cmd_revoke(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    numeric_id, name, cert_id = _resolve_agent(args.base_dir, args.cert_id)
    try:
        cert = ca.revoke(cert_id, args.reason)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Update registry status
    registry = _load_registry(args.base_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    if numeric_id > 0:
        try:
            registry.update_status(numeric_id, "REVOKED", revoked_at=now_iso)
        except KeyError:
            pass

    # Create revocation seal
    from nomotic.revocation import create_revocation_seal
    create_revocation_seal(
        args.base_dir,
        agent_id=numeric_id,
        agent_name=name,
        certificate_id=cert_id,
        reason=args.reason,
        cert=cert,
    )

    print(f"Revoked: {cert.certificate_id}")


def _cmd_renew(args: argparse.Namespace) -> None:
    ca, store = _build_ca(args.base_dir)
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    try:
        cert, agent_sk = ca.renew(cert_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    store.save_agent_key(cert.certificate_id, agent_sk.to_bytes())
    store.save_agent_pub(cert.certificate_id, cert.public_key)
    print(f"Renewed: {cert.certificate_id}")
    print(f"  Lineage: {cert.lineage}")


def _cmd_list(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    status_filter = CertStatus[args.status.upper()] if args.status else None

    # Special handling for REVOKED status — show from revocation records
    if status_filter == CertStatus.REVOKED:
        registry = _load_registry(args.base_dir)
        revoked_agents = registry.list_all(status="REVOKED", org=args.org)
        if not revoked_agents:
            print("No revoked agents found.")
            return

        print()
        print("Revoked Agents:")
        print()
        print(f"  {'ID':<6}{'Name':<14}{'Org':<17}{'Revoked':<13}{'Reason'}")
        print(f"  {'─' * 4}  {'─' * 12}  {'─' * 15}  {'─' * 11}  {'─' * 23}")
        for agent in revoked_agents:
            revoked_at = agent.get("revoked_at", "")
            if revoked_at:
                revoked_date = revoked_at[:10]
            else:
                revoked_date = "unknown"
            # Load reason from revocation seal if available
            reason = ""
            seal_path = args.base_dir / "revocations" / f"{agent['agent_id']}.json"
            if seal_path.exists():
                try:
                    seal = json.loads(seal_path.read_text(encoding="utf-8"))
                    reason = seal.get("reason", "")
                except (json.JSONDecodeError, KeyError):
                    pass
            print(f"  {agent['agent_id']:<6}{agent['name']:<14}{agent['organization']:<17}{revoked_date:<13}{reason}")
        print()
        return

    certs = ca.list(org=args.org, status=status_filter, archetype=args.archetype)
    if not certs:
        print("No certificates found.")
        return

    # Load registry for numeric IDs
    registry = _load_registry(args.base_dir)

    for cert in certs:
        nid = cert.agent_numeric_id
        if nid == 0:
            # Try to find in registry
            matches = registry.resolve(cert.agent_id)
            if matches:
                nid = matches[0]["agent_id"]
        id_str = str(nid) if nid > 0 else "?"
        print(f"  {id_str:<6}{cert.agent_id:20s}  {cert.status.name:10s}  trust={cert.trust_score:.2f}  age={cert.behavioral_age}  cert={cert.certificate_id}")


def _cmd_reputation(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    cert = ca.get(cert_id)
    if cert is None:
        print(f"Certificate not found: {cert_id}", file=sys.stderr)
        sys.exit(1)

    trust = cert.trust_score
    age = cert.behavioral_age

    # Estimate violation/success counts from behavioral age and trust
    # Trust starts at 0.500 and moves asymmetrically (violations cost 5x).
    # We estimate based on age and current trust trajectory.
    if age > 0:
        # Rough heuristic: each success adds ~0.002, each violation costs ~0.010
        # Solve: successful * 0.002 - violations * 0.010 = trust - 0.500
        #        successful + violations = age
        delta = trust - 0.500
        # violations ≈ (age * 0.002 - delta) / 0.012
        est_violations = max(0, int((age * 0.002 - delta) / 0.012))
        est_successful = max(0, age - est_violations)
    else:
        est_violations = 0
        est_successful = 0

    total = est_successful + est_violations
    violation_rate = (est_violations / total * 100) if total > 0 else 0.0

    # Trust color
    if trust >= 0.5:
        trust_display = _green(f"{trust:.3f}")
    elif trust >= 0.3:
        trust_display = _yellow(f"{trust:.3f}")
    else:
        trust_display = _red(f"{trust:.3f}")

    # Status color
    if cert.status.name == "ACTIVE":
        status_display = _green(cert.status.name)
    elif cert.status.name == "SUSPENDED":
        status_display = _yellow(cert.status.name)
    else:
        status_display = _red(cert.status.name)

    print()
    print(f"  Reputation: {_bold(cert.agent_id)}")
    print()
    print(f"    Trust Score:        {trust_display}")
    print(f"    Behavioral Age:     {age}")
    print(f"    Violation Count:    {est_violations}")
    print(f"    Successful Actions: {est_successful}")
    print(f"    Violation Rate:     {violation_rate:.1f}%")
    print(f"    Status:             {status_display}")
    print()

    # Trust trajectory
    start_trust = 0.500
    trend = "stable"
    if trust > start_trust + 0.02:
        trend = "rising"
    elif trust < start_trust - 0.02:
        trend = "falling"

    print(f"    Trust Trajectory:")
    print(f"      Started at {start_trust:.3f} on {cert.issued_at.strftime('%Y-%m-%d')}")
    print(f"      Current:   {trust:.3f}")
    print(f"      Trend:     {trend}")
    print()


def _cmd_export(args: argparse.Namespace) -> None:
    ca, _store = _build_ca(args.base_dir)
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    cert = ca.get(cert_id)
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


def _cmd_status(args: argparse.Namespace) -> None:
    """Show quick operational status for an agent."""
    ca, _store = _build_ca(args.base_dir)
    cert_id = _resolve_cert_id(args.base_dir, args.cert_id)
    cert = ca.get(cert_id)
    if cert is None:
        print(f"Certificate not found: {cert_id}", file=sys.stderr)
        sys.exit(1)

    # Status color
    if cert.status.name == "ACTIVE":
        status_display = _green(cert.status.name)
    elif cert.status.name == "SUSPENDED":
        status_display = _yellow(cert.status.name)
    else:
        status_display = _red(cert.status.name)

    # Trust color
    trust = cert.trust_score
    if trust >= 0.5:
        trust_display = _green(f"{trust:.3f}")
    elif trust >= 0.3:
        trust_display = _yellow(f"{trust:.3f}")
    else:
        trust_display = _red(f"{trust:.3f}")

    print()
    print(f"  {_bold(cert.agent_id)}")
    print(f"  {'─' * 40}")
    print(f"  Status:    {status_display}")
    print(f"  Trust:     {trust_display}")
    print(f"  Archetype: {cert.archetype}")
    print(f"  Org:       {cert.organization}")
    print(f"  Zone:      {cert.zone_path}")
    print(f"  Owner:     {cert.owner}")
    print(f"  Issued:    {cert.issued_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Cert:      {cert.certificate_id}")
    print()


# ── Archetype commands ───────────────────────────────────────────────────


def _cmd_archetype(args: argparse.Namespace) -> None:
    arch_reg = ArchetypeRegistry.with_defaults()
    sub = args.archetype_command

    if sub == "list":
        archetypes = arch_reg.list(category=args.category)
        if not archetypes:
            print("No archetypes found.")
            return

        # Group by category
        from collections import defaultdict
        by_category: dict[str, list] = defaultdict(list)
        for a in archetypes:
            by_category[a.category].append(a)

        # Find max name length for alignment
        max_name = max(len(a.name) for a in archetypes)

        print()
        print("Archetypes:")
        for cat in sorted(by_category.keys()):
            print()
            print(f"  {_bold(cat)}")
            for a in sorted(by_category[cat], key=lambda x: x.name):
                print(f"    {a.name:<{max_name + 2}}{a.description}")

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

    elif sub == "set":
        _cmd_archetype_set(args)

    else:
        print("Unknown archetype subcommand", file=sys.stderr)
        sys.exit(1)


def _cmd_archetype_set(args: argparse.Namespace) -> None:
    """Interactive archetype selection for an agent."""
    agent_identifier = args.agent_identifier

    # Resolve agent
    _numeric_id, _name, cert_id = _resolve_agent(args.base_dir, agent_identifier)

    arch_reg = ArchetypeRegistry.with_defaults()
    archetypes = arch_reg.list()

    # Group by category
    by_category: dict[str, list] = {}
    for a in archetypes:
        cat = a.category or "uncategorized"
        by_category.setdefault(cat, []).append(a)

    # Display numbered list
    print()
    print(f"  Select an archetype for {_bold(agent_identifier)}:")
    print()

    numbered = []
    idx = 1
    for cat in sorted(by_category.keys()):
        print(f"  {_bold(cat)}")
        for a in sorted(by_category[cat], key=lambda x: x.name):
            print(f"    {idx:2d}. {a.name:28s} {a.description}")
            numbered.append(a)
            idx += 1
        print()

    # Get selection
    while True:
        choice = input("  Enter number (or 'skip' to keep current): ").strip()
        if choice.lower() == "skip":
            print("  Archetype unchanged.")
            return
        try:
            num = int(choice)
            if 1 <= num <= len(numbered):
                selected = numbered[num - 1]
                break
            print(f"  Please enter a number between 1 and {len(numbered)}.")
        except ValueError:
            print("  Please enter a number or 'skip'.")

    # Update the certificate's archetype
    ca, store = _build_ca(args.base_dir)
    cert = ca.get(cert_id)
    if cert is None:
        print("Certificate not found.", file=sys.stderr)
        sys.exit(1)

    cert.archetype = selected.name
    store.update(cert)

    print(f"  Archetype set to '{_green(selected.name)}' for {agent_identifier}.")
    print()


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
    """Redirect to the tutorial."""
    print("Starting the Nomotic tutorial...")
    print("  (Tip: You can also run 'nomotic tutorial' directly.)")
    print()
    _cmd_tutorial(args)


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


def _cmd_trust(args: argparse.Namespace) -> None:
    """Show the trust report for an agent (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/trust/{args.agent_id}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"No trust data for agent: {args.agent_id}", file=sys.stderr)
            sys.exit(1)
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        print("Is the Nomotic API server running? (nomotic serve)", file=sys.stderr)
        sys.exit(1)

    print(f"Trust Report: {data['agent_id']}")
    print(f"  Current Trust: {data['current_trust']:.2f}")

    traj = data.get("trajectory", {})
    trend = traj.get("trend", "unknown")
    print(f"  Trend: {trend}")
    print()

    total = data.get("successful_actions", 0) + data.get("violation_count", 0)
    print(f"  Actions: {total} total ({data.get('successful_actions', 0)} successful, {data.get('violation_count', 0)} denied)")
    vr = data.get("violation_rate", 0)
    print(f"  Violation Rate: {vr:.1%}")
    print()

    fp = data.get("fingerprint")
    if fp:
        print(f"  Fingerprint: {fp['total_observations']} observations (confidence: {fp['confidence']:.3f})")

    drift = data.get("drift")
    if drift:
        print(f"  Drift: {drift['overall']:.2f} ({drift['severity']})")

    active_alerts = data.get("active_alerts")
    if active_alerts is not None:
        print(f"  Active Alerts: {active_alerts}")
    print()

    # Show recent events
    recent = traj.get("recent_events", [])
    if recent:
        total_events = traj.get("total_events", 0)
        print(f"  Trust History (last {len(recent)} of {total_events} changes):")
        for i, event in enumerate(recent):
            idx = total_events - len(recent) + i
            tb = event.get("trust_before", 0)
            ta = event.get("trust_after", 0)
            delta = event.get("delta", 0)
            src = event.get("source", "")
            reason = event.get("reason", "")
            sign = "+" if delta >= 0 else ""
            print(f"    [{idx}] {tb:.2f} -> {ta:.2f}  {sign}{delta:.3f}  {src:25s} {reason}")
        print()

    # Show trust by source
    sources = traj.get("sources", {})
    if sources:
        print("  Trust by Source:")
        for src, info in sorted(sources.items()):
            nd = info.get("net_delta", 0)
            cnt = info.get("count", 0)
            sign = "+" if nd >= 0 else ""
            print(f"    {src:25s} {sign}{nd:.3f} ({cnt} events)")
        print()


def _cmd_audit(args: argparse.Namespace) -> None:
    """Show audit records (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"

    if args.audit_command == "summary":
        url = f"{base_url}/v1/audit/summary"
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
            sys.exit(1)

        print(f"Audit Trail Summary")
        print(f"  Total records: {data.get('total_records', 0)}")
        by_verdict = data.get("by_verdict", {})
        if by_verdict:
            print(f"  By verdict:")
            for v, c in sorted(by_verdict.items()):
                print(f"    {v:12s} {c}")
        by_severity = data.get("by_severity", {})
        if by_severity:
            print(f"  By severity:")
            for s, c in sorted(by_severity.items()):
                print(f"    {s:12s} {c}")
        alerts = data.get("recent_alerts", [])
        if alerts:
            print(f"  Recent alerts ({len(alerts)}):")
            for a in alerts:
                print(f"    [{a.get('severity', '?')}] {a.get('context_code', '?')} - {a.get('agent_id', '?')}")
        return

    # Default: query
    url = f"{base_url}/v1/audit?"
    params = []
    if args.agent_id:
        params.append(f"agent_id={args.agent_id}")
    if args.severity:
        params.append(f"severity={args.severity}")
    if args.limit:
        params.append(f"limit={args.limit}")
    url += "&".join(params)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    records = data.get("records", [])
    if not records:
        print("No audit records found.")
        return

    print(f"Audit Records ({len(records)} shown):")
    for r in records:
        print(f"  [{r.get('severity', '?'):8s}] {r.get('context_code', '?'):25s} agent={r.get('agent_id', '?'):15s} verdict={r.get('verdict', '?')}")


def _cmd_provenance(args: argparse.Namespace) -> None:
    """Show configuration provenance records (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/provenance?"
    params = []
    if args.target_type:
        params.append(f"target_type={args.target_type}")
    if args.target_id:
        params.append(f"target_id={args.target_id}")
    url += "&".join(params)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    records = data.get("records", [])
    if not records:
        print("No provenance records found.")
        return

    print(f"Provenance Records ({len(records)} shown):")
    for r in records:
        print(f"  [{r.get('change_type', '?'):6s}] {r.get('target_type', '?'):10s}/{r.get('target_id', '?'):15s} by {r.get('actor', '?')}")
        if r.get("reason"):
            print(f"          reason: {r['reason']}")


def _cmd_owner(args: argparse.Namespace) -> None:
    """Show owner activity and engagement (via API)."""
    import urllib.request
    import urllib.error

    base_url = f"http://{args.host}:{args.port}"
    owner_id = args.owner_id

    if args.engagement:
        url = f"{base_url}/v1/owner/{owner_id}/engagement"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as exc:
            print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
            sys.exit(1)

        print(f"Owner Engagement: {owner_id}")
        print(f"  Total activities:     {data.get('total_activities', 0)}")
        print(f"  Activities in window: {data.get('activities_in_window', 0)}")
        print(f"  Alert response rate:  {data.get('alert_response_rate', 0):.1%}")
        print(f"  Engagement level:     {data.get('engagement_level', 'unknown')}")
        return

    url = f"{base_url}/v1/owner/{owner_id}/activity"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"API error: {exc.code} {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot connect to {base_url}: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    activities = data.get("activities", [])
    if not activities:
        print(f"No activity for owner: {owner_id}")
        return

    print(f"Owner Activity: {owner_id} ({len(activities)} shown)")
    for a in activities:
        print(f"  {a.get('activity_type', '?'):25s} agent={a.get('target_agent_id', '-')}")
        if a.get("detail"):
            print(f"    {a['detail']}")


def _cmd_scope(args: argparse.Namespace) -> None:
    """Configure or view an agent's authorized scope."""
    from nomotic.sandbox import (
        AgentConfig,
        load_agent_config,
        save_agent_config,
    )

    agent_id = args.agent_id
    sub = args.scope_command

    if sub == "set":
        config = load_agent_config(args.base_dir, agent_id) or AgentConfig(agent_id=agent_id)
        if args.actions:
            config.actions = [a.strip() for a in args.actions.split(",")]
        if args.boundaries:
            config.boundaries = [b.strip() for b in args.boundaries.split(",")]
        save_agent_config(args.base_dir, config)

        print(f"Scope configured for {agent_id}:")
        print(f"  Allowed actions:  {', '.join(config.actions)}")
        print(f"  Boundaries:       {', '.join(config.boundaries)}")
        print()
        _print_authority_envelope(agent_id, config)

    elif sub == "show":
        config = load_agent_config(args.base_dir, agent_id)
        if config is None:
            print(f"No configuration found for agent: {agent_id}", file=sys.stderr)
            sys.exit(1)
        _print_authority_envelope(agent_id, config)

    else:
        print("Unknown scope subcommand. Use 'set' or 'show'.", file=sys.stderr)
        sys.exit(1)


def _print_authority_envelope(agent_id: str, config: "AgentConfig") -> None:
    """Print the formatted authority envelope box."""
    actions_str = "  ".join(config.actions) if config.actions else "(none)"
    boundaries = config.boundaries or []

    width = 45
    print(f"  {'':>{0}}{_bold(chr(9484))}{chr(9472) * width}{_bold(chr(9488))}")
    print(f"  {_bold(chr(9474))}  {agent_id} Authority Envelope{' ' * (width - len(agent_id) - 22)}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9474))}{' ' * width}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9474))}  Actions: {actions_str}{' ' * max(0, width - len(actions_str) - 12)}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9474))}{' ' * width}{_bold(chr(9474))}")
    if boundaries:
        print(f"  {_bold(chr(9474))}  Can touch:{' ' * (width - 13)}{_bold(chr(9474))}")
        for b in boundaries:
            line = f"    {_green(chr(10003))} {b}"
            # Count visible chars (without ANSI)
            visible_len = len(f"    {chr(10003)} {b}")
            pad = width - visible_len - 2
            print(f"  {_bold(chr(9474))}  {line}{' ' * max(0, pad)}{_bold(chr(9474))}")
    else:
        print(f"  {_bold(chr(9474))}  Boundaries: (none){' ' * (width - 22)}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9474))}{' ' * width}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9474))}  Cannot touch: everything else{' ' * (width - 33)}{_bold(chr(9474))}")
    print(f"  {_bold(chr(9492))}{chr(9472) * width}{_bold(chr(9496))}")


def _cmd_rule(args: argparse.Namespace) -> None:
    """Add governance rules to an agent."""
    from nomotic.sandbox import (
        AgentConfig,
        EthicalRuleSpec,
        HumanOverrideSpec,
        load_agent_config,
        save_agent_config,
    )

    agent_id = args.agent_id
    config = load_agent_config(args.base_dir, agent_id) or AgentConfig(agent_id=agent_id)

    if args.rule_command != "add":
        print("Unknown rule subcommand. Use 'add'.", file=sys.stderr)
        sys.exit(1)

    rule_type = args.type

    if rule_type == "ethical":
        if not args.condition:
            print("--condition is required for ethical rules", file=sys.stderr)
            sys.exit(1)
        if not args.message:
            print("--message is required for ethical rules", file=sys.stderr)
            sys.exit(1)
        rule = EthicalRuleSpec(
            condition=args.condition,
            message=args.message,
            name=getattr(args, "name", "") or "",
        )
        config.ethical_rules.append(rule)
        save_agent_config(args.base_dir, config)

        print(f"Ethical rule added:")
        print(f"  Agent: {agent_id}")
        print(f"  Rule:  {rule.condition}")
        print(f"  If violated: \"{rule.message}\"")
        print(f"  Effect: VETO (cannot be overridden by score)")

    elif rule_type == "human-override":
        if not args.action:
            print("--action is required for human-override rules", file=sys.stderr)
            sys.exit(1)
        override = HumanOverrideSpec(
            action=args.action,
            message=args.message or f"Human approval required for '{args.action}'",
        )
        config.human_overrides.append(override)
        save_agent_config(args.base_dir, config)

        print(f"Human override rule added:")
        print(f"  Agent:  {agent_id}")
        print(f"  Action: {override.action}")
        print(f"  Effect: \"{override.message}\"")

    else:
        print(f"Unknown rule type: {rule_type}. Use 'ethical' or 'human-override'.", file=sys.stderr)
        sys.exit(1)


def _cmd_config(args: argparse.Namespace) -> None:
    """Show the complete governance configuration for an agent, or set config values."""
    from nomotic.sandbox import load_agent_config

    # Handle "config set" subcommand
    config_cmd = getattr(args, "config_command", None)
    if config_cmd == "set":
        _cmd_config_set(args)
        return

    agent_id = args.agent_id

    # Resolve agent identifier
    try:
        _numeric_id, resolved_name, cert_id = _resolve_agent(args.base_dir, agent_id)
        agent_id = resolved_name
    except SystemExit:
        cert_id = None

    # Try to load certificate info
    ca, _store = _build_ca(args.base_dir)
    if cert_id is None:
        from nomotic.sandbox import find_agent_cert_id
        cert_id = find_agent_cert_id(args.base_dir, agent_id)
    cert = ca.get(cert_id) if cert_id else None

    # Load governance config
    config = load_agent_config(args.base_dir, agent_id)

    print()
    print(_bold(f"  Governance Configuration: {agent_id}"))
    print(f"  {'=' * 50}")
    print()

    # Certificate info
    if cert:
        print(_bold("  Identity:"))
        print(f"    Certificate: {cert.certificate_id}")
        print(f"    Archetype:   {cert.archetype}")
        print(f"    Org:         {cert.organization}")
        print(f"    Zone:        {cert.zone_path}")
        print(f"    Owner:       {cert.owner}")
        print(f"    Trust:       {cert.trust_score}")
        print(f"    Status:      {cert.status.name}")
        print()
    else:
        print(_bold("  Identity:"))
        print(f"    No certificate found for '{agent_id}'")
        print()

    if config is None:
        print(_bold("  Authority:"))
        print(f"    No governance configuration found.")
        print(f"    Use 'nomotic scope set {agent_id} --actions ...' to configure.")
        print()
        return

    # Scope
    print(_bold("  Authority:"))
    if config.actions:
        print(f"    Allowed actions: {', '.join(config.actions)}")
    else:
        print(f"    Allowed actions: (not configured)")
    print()

    # Boundaries
    print(_bold("  Boundaries:"))
    if config.boundaries:
        for b in config.boundaries:
            print(f"    {_green(chr(10003))} {b}")
    else:
        print(f"    (not configured)")
    print()

    # Ethical rules
    print(_bold("  Ethical Rules:"))
    if config.ethical_rules:
        for i, rule in enumerate(config.ethical_rules, 1):
            print(f"    [{i}] {_yellow('VETO')} if: {rule.condition}")
            print(f"        Message: \"{rule.message}\"")
    else:
        print(f"    (none)")
    print()

    # Human override rules
    print(_bold("  Human Override Rules:"))
    if config.human_overrides:
        for i, ho in enumerate(config.human_overrides, 1):
            print(f"    [{i}] Action '{_red(ho.action)}' requires human approval")
            print(f"        Message: \"{ho.message}\"")
    else:
        print(f"    (none)")
    print()


def _cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a single action against the governance pipeline."""
    from nomotic.sandbox import (
        AgentConfig,
        build_sandbox_runtime,
        load_agent_config,
    )

    from nomotic.sandbox import format_bar
    from nomotic.types import Action as GovAction, AgentContext as GovCtx, TrustProfile as GovTP, Verdict

    agent_id = args.agent_id
    action_type = args.action
    target = args.target or ""
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Invalid JSON for --params: {args.params}", file=sys.stderr)
            sys.exit(1)

    # Load agent config
    config = load_agent_config(args.base_dir, agent_id)
    if config is None:
        config = AgentConfig(agent_id=agent_id)

    # Load certificate for trust
    ca, _store = _build_ca(args.base_dir)
    from nomotic.sandbox import find_agent_cert_id
    cert_id = find_agent_cert_id(args.base_dir, agent_id)
    cert = ca.get(cert_id) if cert_id else None
    initial_trust = cert.trust_score if cert else 0.5

    # Build runtime and evaluate
    runtime = build_sandbox_runtime(agent_config=config, agent_id=agent_id)
    trust_profile = runtime.get_trust_profile(agent_id)
    trust_profile.overall_trust = initial_trust

    action = GovAction(
        agent_id=agent_id,
        action_type=action_type,
        target=target,
        parameters=params,
    )
    ctx = GovCtx(agent_id=agent_id, trust_profile=trust_profile)

    verdict = runtime.evaluate(action, ctx)

    # Display results
    print()
    print(_bold(_cyan("=" * 55)))
    print(_bold(_cyan("  GOVERNANCE EVALUATION")))
    print(_bold(_cyan("=" * 55)))
    print()
    print(f"  Agent:   {agent_id} (trust: {initial_trust:.3f})")
    target_str = f" {chr(8594)} {target}" if target else ""
    print(f"  Action:  {action_type}{target_str}")
    if params:
        print(f"  Params:  {json.dumps(params)}")
    print()

    # Dimension scores
    print(f"  {_bold(chr(9472) * 2 + ' Dimension Scores ' + chr(9472) * 30)}")
    print()

    for ds in verdict.dimension_scores:
        bar = format_bar(ds.score, 20)
        name_padded = f"{ds.dimension_name:26s}"
        score_str = f"{ds.score:.2f}"

        if ds.veto and ds.score < 0.5:
            check = _red(f"{chr(10007)} VETO: {ds.reasoning}")
            color_score = _red(score_str)
        elif ds.score >= 0.7:
            check = _green(ds.reasoning)
            color_score = _green(score_str)
        elif ds.score >= 0.4:
            check = _yellow(ds.reasoning)
            color_score = _yellow(score_str)
        else:
            check = _red(ds.reasoning)
            color_score = _red(score_str)

        print(f"  {name_padded} {color_score}  {bar}  {check}")

    # Decision
    print()
    print(f"  {_bold(chr(9472) * 2 + ' Decision ' + chr(9472) * 38)}")
    print()

    ucs_str = f"{verdict.ucs:.3f}"
    tier_labels = {1: "veto gate", 2: "threshold evaluation", 3: "deliberative review"}
    tier_label = tier_labels.get(verdict.tier, "unknown")

    if verdict.verdict == Verdict.ALLOW:
        verdict_display = _green(f"{chr(9989)} ALLOW")
    elif verdict.verdict == Verdict.DENY:
        verdict_display = _red(f"{chr(10060)} DENY")
    elif verdict.verdict == Verdict.ESCALATE:
        verdict_display = _yellow(f"{chr(9888)} ESCALATE")
    else:
        verdict_display = _yellow(f"{chr(9888)} {verdict.verdict.name}")

    print(f"  UCS:      {ucs_str}")
    print(f"  Tier:     {verdict.tier} ({tier_label})")
    print(f"  Verdict:  {verdict_display}")
    if verdict.vetoed_by:
        print(f"  Vetoed by: {', '.join(verdict.vetoed_by)}")
    if verdict.reasoning:
        print(f"  Reason:   {verdict.reasoning}")
    print(f"  Time:     {verdict.evaluation_time_ms:.1f}ms")

    # Trust update
    new_trust = runtime.get_trust_profile(agent_id).overall_trust
    delta = new_trust - initial_trust
    sign = "+" if delta >= 0 else ""
    delta_color = _green if delta > 0 else (_red if delta < 0 else _yellow)

    trend = "stable"
    if delta > 0.001:
        trend = f"{chr(8599)} rising"
    elif delta < -0.001:
        trend = f"{chr(8600)} falling"

    print()
    print(f"  {_bold(chr(9472) * 2 + ' Trust Update ' + chr(9472) * 34)}")
    print()
    print(f"  Before:   {initial_trust:.3f}")
    print(f"  After:    {new_trust:.3f}  ({delta_color(f'{sign}{delta:.3f}')})")
    print(f"  Trend:    {trend}")

    if delta < -0.01:
        ratio = abs(delta) / 0.01
        print()
        print(f"  {_yellow(chr(9888))} Note: violations cost 5x more trust than successes earn.")
        if ratio > 1:
            print(f"  This one denial erased ~{int(ratio)} successful actions.")
    print()

    # Update certificate trust if we have one
    if cert is not None:
        ca.update_trust(cert.certificate_id, new_trust)


def _cmd_simulate(args: argparse.Namespace) -> None:
    """Run a batch simulation of governance evaluations."""
    from nomotic.sandbox import (
        AgentConfig,
        build_sandbox_runtime,
        format_bar,
        format_pct_bar,
        load_agent_config,
    )
    from nomotic.scenarios import BUILTIN_SCENARIOS, generate_actions
    from nomotic.types import AgentContext as GovCtx, TrustProfile as GovTP, Verdict

    agent_id = args.agent_id
    scenario_name = args.scenario
    count = args.count

    if scenario_name not in BUILTIN_SCENARIOS:
        print(f"Unknown scenario: {scenario_name}", file=sys.stderr)
        print(f"Available: {', '.join(BUILTIN_SCENARIOS.keys())}", file=sys.stderr)
        sys.exit(1)

    scenario = BUILTIN_SCENARIOS[scenario_name]

    # Load agent config
    config = load_agent_config(args.base_dir, agent_id)
    if config is None:
        config = AgentConfig(agent_id=agent_id)

    # Load initial trust from cert
    ca, _store = _build_ca(args.base_dir)
    from nomotic.sandbox import find_agent_cert_id
    cert_id = find_agent_cert_id(args.base_dir, agent_id)
    cert = ca.get(cert_id) if cert_id else None
    initial_trust = cert.trust_score if cert else 0.5

    # Build runtime
    runtime = build_sandbox_runtime(agent_config=config, agent_id=agent_id)
    trust_profile = runtime.get_trust_profile(agent_id)
    trust_profile.overall_trust = initial_trust

    # Generate actions
    actions = generate_actions(agent_id, scenario, total_count=count if count else None)

    # Run simulation
    print()
    print(f"  {_bold(f'Simulating: {scenario.description}')}")
    print()

    trust_history: list[float] = [initial_trust]
    results = {"ALLOW": 0, "DENY": 0, "ESCALATE": 0, "MODIFY": 0}
    current_phase = ""

    for i, (action, phase_desc) in enumerate(actions):
        if phase_desc != current_phase:
            current_phase = phase_desc
            print(f"  {_cyan(f'Phase: {phase_desc}')}")

        ctx = GovCtx(agent_id=agent_id, trust_profile=runtime.get_trust_profile(agent_id))
        verdict = runtime.evaluate(action, ctx)
        results[verdict.verdict.name] = results.get(verdict.verdict.name, 0) + 1
        trust_history.append(runtime.get_trust_profile(agent_id).overall_trust)

        # Progress indicator
        total = len(actions)
        bar_width = 40
        filled = int((i + 1) / total * bar_width)
        bar = chr(9608) * filled + chr(9617) * (bar_width - filled)
        print(f"\r  [{bar}] {i + 1}/{total}", end="", flush=True)

    print()  # newline after progress bar
    print()

    # Behavioral Fingerprint
    fp = runtime.get_fingerprint(agent_id)
    if fp is not None:
        print(f"  {_bold(chr(9472) * 2 + ' Behavioral Fingerprint ' + chr(9472) * 24)}")
        print()
        print(f"  Observations: {fp.total_observations}")
        print(f"  Confidence:   {fp.confidence:.2f}")
        print()

        if fp.action_distribution:
            print(f"  Action Distribution:")
            for act, freq in sorted(fp.action_distribution.items(), key=lambda x: -x[1]):
                bar = format_pct_bar(freq)
                print(f"    {act:20s} {bar}  {freq:.0%}")
            print()

        if fp.target_distribution:
            print(f"  Target Distribution:")
            for tgt, freq in sorted(fp.target_distribution.items(), key=lambda x: -x[1]):
                bar = format_pct_bar(freq)
                print(f"    {tgt:20s} {bar}  {freq:.0%}")
            print()

    # Drift detection
    drift = runtime.get_drift(agent_id)
    if drift is not None and drift.overall > 0.05:
        print(f"  {_bold(chr(9472) * 2 + ' Drift Detection Report ' + chr(9472) * 24)}")
        print()
        sev = drift.severity
        color = _red if sev in ("high", "critical") else (_yellow if sev == "moderate" else _green)
        print(f"  Drift Score:    {color(f'{drift.overall:.2f}')} ({sev.upper()})")
        print(f"  JSD Divergence: {drift.action_drift:.2f}")
        if drift.detail:
            print(f"  Detail:         {drift.detail}")
        print()

        # Show alerts
        alerts = runtime.get_drift_alerts(agent_id)
        if alerts:
            for alert in alerts:
                sev_color = _red if alert.severity in ("critical", "high") else _yellow
                print(f"  {sev_color(chr(9888))} DRIFT ALERT:")
                print(f"    Severity: {alert.severity.upper()}")
                if alert.drift_score.detail:
                    print(f"    Message:  {alert.drift_score.detail}")
            print()

    # Trust trajectory
    print(f"  {_bold(chr(9472) * 2 + ' Trust Trajectory ' + chr(9472) * 30)}")
    print()
    end_trust = trust_history[-1]
    trend = "stable"
    if end_trust > initial_trust + 0.02:
        trend = "rising"
    elif end_trust < initial_trust - 0.02:
        trend = "falling"

    print(f"  Start: {initial_trust:.3f} {chr(8594)} End: {end_trust:.3f}")
    print(f"  Trend: {trend}")
    print()

    # Visual trust trajectory
    n_rows = 6
    min_t = min(trust_history)
    max_t = max(trust_history)
    t_range = max_t - min_t if max_t > min_t else 0.1
    chart_width = min(40, len(trust_history))
    step = max(1, len(trust_history) // chart_width)
    sampled = [trust_history[i] for i in range(0, len(trust_history), step)]

    for row in range(n_rows, -1, -1):
        level = min_t + (row / n_rows) * t_range
        label = f"  {level:.2f} |"
        chars = []
        for val in sampled:
            val_row = int((val - min_t) / t_range * n_rows) if t_range > 0 else n_rows // 2
            if val_row == row:
                chars.append(".")
            elif val_row > row:
                chars.append(" ")
            else:
                chars.append(" ")
        print(f"{label}{''.join(chars)}|")
    total_actions = len(actions)
    print(f"         {chr(9492)}{chr(9472) * len(sampled)}{chr(9496)}")
    label_str = f"         0{' ' * (len(sampled) - len(str(total_actions)) - 1)}{total_actions}"
    print(label_str)
    print()

    # Summary
    allowed = results.get("ALLOW", 0)
    denied = results.get("DENY", 0)
    escalated = results.get("ESCALATE", 0)

    print(f"  {_green(chr(10003))} {total_actions} actions evaluated")
    print(f"  {_green(chr(10003))} {_green(str(allowed))} allowed, {_red(str(denied))} denied", end="")
    if escalated:
        print(f", {_yellow(str(escalated))} escalated", end="")
    print()
    if fp:
        print(f"  {_green(chr(10003))} Fingerprint established")
    print(f"  {_green(chr(10003))} Trust trajectory: {trend}")
    print()

    # Update certificate trust
    if cert is not None:
        ca.update_trust(cert.certificate_id, end_trust)


def _cmd_config_set(args: argparse.Namespace) -> None:
    """Handle 'nomotic config set --retention <period>'."""
    if args.retention is not None:
        valid_periods = {"2y", "3y", "5y", "forever"}
        if args.retention not in valid_periods:
            print(f"Invalid retention period: {args.retention}", file=sys.stderr)
            print(f"  Valid values: {', '.join(sorted(valid_periods))}", file=sys.stderr)
            sys.exit(1)
        config_path = args.base_dir / "config.json"
        config_data: dict = {}
        if config_path.exists():
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["retention"] = args.retention
        args.base_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        print(f"Retention set to: {args.retention}")
        print("  Note: Retention is recorded but not yet enforced (all records are kept).")
    else:
        print("No configuration value specified.", file=sys.stderr)
        print("  Use --retention to set audit retention period.", file=sys.stderr)
        sys.exit(1)


def _cmd_history(args: argparse.Namespace) -> None:
    """Show the complete historical record for any agent (including revoked)."""
    numeric_id, name, cert_id = _resolve_agent(args.base_dir, args.identifier)

    ca, _store = _build_ca(args.base_dir)
    cert = ca.get(cert_id)

    registry = _load_registry(args.base_dir)
    entry = registry.get(numeric_id) if numeric_id > 0 else None

    status = "UNKNOWN"
    revoked_at = ""
    reason = ""
    seal_data: dict = {}

    if entry:
        status = entry.get("status", "UNKNOWN")
    elif cert:
        status = cert.status.name

    # Check for revocation seal
    seal_path = args.base_dir / "revocations" / f"{numeric_id}.json"
    if seal_path.exists():
        try:
            seal_data = json.loads(seal_path.read_text(encoding="utf-8"))
            revoked_at = seal_data.get("revoked_at", "")
            reason = seal_data.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            pass

    print()
    print(f"History: {name} (ID: {numeric_id})")

    if status == "REVOKED" and revoked_at:
        print(f"  Status: REVOKED ({revoked_at[:10]})")
    else:
        print(f"  Status: {status}")
    print(f"  {'─' * 37}")
    print()

    # Lifecycle
    print("  Lifecycle:")
    if cert:
        print(f"    Born:     {cert.issued_at.strftime('%Y-%m-%d %H:%M')}  (trust: 0.500)")
    elif entry:
        created = entry.get("created_at", "unknown")
        print(f"    Born:     {created[:16] if len(created) >= 16 else created}  (trust: 0.500)")

    if seal_data:
        peak_trust = seal_data.get("peak_trust", 0.5)
        final_trust = seal_data.get("final_trust", 0.05)
        total_evals = seal_data.get("total_evaluations", 0)
        total_violations = seal_data.get("total_violations", 0)
        print(f"    Peak:     (trust: {peak_trust:.3f})")
        if revoked_at:
            print(f"    Revoked:  {revoked_at[:16]}  (trust: {final_trust:.3f})")
        if reason:
            print(f"    Reason:   {reason}")

        print()
        print("  Summary:")
        allowed = total_evals - total_violations
        violation_rate = (total_violations / total_evals * 100) if total_evals > 0 else 0.0
        print(f"    Total evaluations:  {total_evals}")
        print(f"    Allowed:            {allowed}")
        print(f"    Denied:             {total_violations}")
        print(f"    Violation rate:     {violation_rate:.1f}%")

        audit_seal = seal_data.get("audit_seal", "")
        if audit_seal:
            print()
            print(f"  Audit seal: {audit_seal}")

        chain_verified = seal_data.get("chain_records", 0)
        if chain_verified > 0:
            print(f"  Chain integrity: verified ({chain_verified} records)")
    elif cert:
        trust = cert.trust_score
        age = cert.behavioral_age
        print(f"    Current:  (trust: {trust:.3f})")
        print()
        print("  Summary:")
        print(f"    Behavioral age:     {age}")
        print(f"    Current trust:      {trust:.3f}")

    print()


def _cmd_tutorial(args: argparse.Namespace) -> None:
    """Run the interactive governance tutorial — the full developer journey."""
    import time as _time

    from nomotic.monitor import DriftConfig as _DriftConfig
    from nomotic.runtime import GovernanceRuntime, RuntimeConfig
    from nomotic.sandbox import (
        AgentConfig,
        EthicalRuleSpec,
        HumanOverrideSpec,
        apply_config_to_runtime,
        format_bar,
        format_pct_bar,
    )
    from nomotic.store import MemoryCertificateStore
    from nomotic.types import Action as GovAction, AgentContext as GovCtx, TrustProfile as GovTP, Verdict

    interactive = sys.stdin.isatty() and _COLORS_ENABLED

    def _pause(msg: str = "Press Enter to continue...") -> None:
        if interactive:
            input(f"\n  {_dim(msg)}")
        print()

    def _header(chapter: str, title: str) -> None:
        print()
        print(_bold(_cyan("=" * 60)))
        print(_bold(_cyan(f"  {chapter}: {title}")))
        print(_bold(_cyan("=" * 60)))
        print()

    def _section(title: str) -> None:
        print(f"  {_bold(chr(9472) * 2 + ' ' + title + ' ' + chr(9472) * max(0, 45 - len(title)))}")
        print()

    # ── Chapter 1: Identity ─────────────────────────────────────────

    _header("Chapter 1", "Identity \u2014 \"Who is this agent?\"")
    print("  Every AI agent starts with a birth certificate \u2014 a cryptographically")
    print("  signed identity. Let's create one.")
    print()

    issuer_sk, issuer_vk = SigningKey.generate()
    store = MemoryCertificateStore()
    ca = CertificateAuthority(
        issuer_id="playground-issuer",
        signing_key=issuer_sk,
        store=store,
    )

    cert, agent_sk = ca.issue(
        agent_id="claims-bot",
        archetype="customer-experience",
        organization="acme-insurance",
        zone_path="us/production",
        owner="chris@acme.com",
    )

    print(f"  {_green('Certificate issued')}: {cert.certificate_id}")
    print(f"    Agent:     {cert.agent_id}")
    print(f"    Owner:     {cert.owner}")
    print(f"    Archetype: {cert.archetype}")
    print(f"    Org:       {cert.organization}")
    print(f"    Zone:      {cert.zone_path}")
    print(f"    Trust:     {cert.trust_score}")
    print(f"    Age:       {cert.behavioral_age}")
    print(f"    Status:    {cert.status.name}")
    print(f"    Issued:    {cert.issued_at.isoformat()}")
    print(f"    Fingerprint: {cert.fingerprint[:16]}...")
    print()
    print(f"  {_dim('What you learned: Every agent has an identity.')}")
    print(f"  {_dim('It is cryptographically signed, has a human owner,')}")
    print(f"  {_dim('lives in a governance zone, and starts with 0.50 trust.')}")

    _pause()

    # ── Chapter 2: Authority ────────────────────────────────────────

    _header("Chapter 2", 'Authority \u2014 "What can this agent do?"')
    print("  Now we define what this agent is allowed to do. Scope controls which")
    print("  actions. Boundaries control which resources. Rules add hard limits.")
    print()

    agent_config = AgentConfig(
        agent_id="claims-bot",
        actions=["read", "write", "query", "assess"],
        boundaries=["claims_db", "customer_records", "policy_db"],
        ethical_rules=[
            EthicalRuleSpec(
                condition="amount <= 10000",
                message="Claims over $10,000 require manual processing",
                name="max-claim-amount",
            ),
        ],
        human_overrides=[
            HumanOverrideSpec(
                action="approve",
                message="All claim approvals require human sign-off",
            ),
        ],
    )

    drift_cfg = _DriftConfig(window_size=50, check_interval=10, min_observations=10)
    runtime = GovernanceRuntime(config=RuntimeConfig(
        enable_fingerprints=True,
        drift_config=drift_cfg,
    ))
    runtime.set_certificate_authority(ca)
    runtime._cert_map["claims-bot"] = cert.certificate_id

    apply_config_to_runtime(runtime, agent_config)

    # Sync initial trust
    tp = runtime.get_trust_profile("claims-bot")
    tp.overall_trust = cert.trust_score

    print(f"  Scope configured:")
    print(f"    Allowed actions:  {', '.join(agent_config.actions)}")
    print(f"    Boundaries:       {', '.join(agent_config.boundaries)}")
    print()
    _print_authority_envelope("claims-bot", agent_config)
    print()

    print(f"  Ethical rule added:")
    print(f"    Rule:  amount <= 10000")
    print(f"    If violated: \"Claims over $10,000 require manual processing\"")
    print(f"    Effect: {_red('VETO')} (cannot be overridden by score)")
    print()

    print(f"  Human override rule added:")
    print(f"    Action '{_yellow('approve')}' requires human sign-off")
    print()

    print(f"  {_dim('What you learned: Authority is defined before the agent acts.')}")
    print(f"  {_dim('Scope controls actions. Boundaries control resources.')}")
    print(f"  {_dim('Rules add ethical constraints and human checkpoints.')}")

    _pause()

    # ── Chapter 3: Evaluation ───────────────────────────────────────

    _header("Chapter 3", 'Evaluation \u2014 "What happens when the agent acts?"')
    print("  When the agent tries to act, every action passes through 13 governance")
    print("  dimensions simultaneously. Watch what happens when we try allowed")
    print("  actions, then forbidden ones.")
    print()

    # Eval 1: authorized action
    _section("Evaluating: read \u2192 claims_db")

    action1 = GovAction(
        agent_id="claims-bot",
        action_type="read",
        target="claims_db",
        parameters={"claim_id": "CLM-5678"},
    )
    ctx1 = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
    trust_before1 = ctx1.trust_profile.overall_trust
    v1 = runtime.evaluate(action1, ctx1)
    trust_after1 = runtime.get_trust_profile("claims-bot").overall_trust

    _print_evaluation(v1, trust_before1, trust_after1)

    _pause()

    # Eval 2: unauthorized action (out of scope)
    _section("Evaluating: delete \u2192 claims_db")

    action2 = GovAction(agent_id="claims-bot", action_type="delete", target="claims_db")
    ctx2 = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
    trust_before2 = ctx2.trust_profile.overall_trust
    v2 = runtime.evaluate(action2, ctx2)
    trust_after2 = runtime.get_trust_profile("claims-bot").overall_trust

    _print_evaluation(v2, trust_before2, trust_after2)
    if v2.verdict == Verdict.DENY:
        print(f"  {_yellow(chr(9888))} Note: violations cost 5x more trust than successes earn.")
    print()

    _pause()

    # Eval 3: ethical violation
    _section("Evaluating: write \u2192 claims_db (amount=25000)")

    action3 = GovAction(
        agent_id="claims-bot",
        action_type="write",
        target="claims_db",
        parameters={"claim_id": "CLM-9999", "amount": 25000},
    )
    ctx3 = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
    trust_before3 = ctx3.trust_profile.overall_trust
    v3 = runtime.evaluate(action3, ctx3)
    trust_after3 = runtime.get_trust_profile("claims-bot").overall_trust

    _print_evaluation(v3, trust_before3, trust_after3)

    _pause()

    # Eval 4: boundary breach
    _section("Evaluating: read \u2192 payroll_db (boundary breach)")

    action4 = GovAction(agent_id="claims-bot", action_type="read", target="payroll_db")
    ctx4 = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
    trust_before4 = ctx4.trust_profile.overall_trust
    v4 = runtime.evaluate(action4, ctx4)
    trust_after4 = runtime.get_trust_profile("claims-bot").overall_trust

    _print_evaluation(v4, trust_before4, trust_after4)

    print(f"  {_dim('What you learned: Every action goes through 13 dimensions.')}")
    print(f"  {_dim('Some dimensions can veto (instant denial).')}")
    print(f"  {_dim('The UCS aggregates all scores into a single confidence number.')}")
    emdash = "\u2014"
    print(f"  {_dim('Trust updates asymmetrically ' + emdash + ' violations hurt 5x more.')}")

    _pause()

    # ── Chapter 4: Behavior ─────────────────────────────────────────

    _header("Chapter 4", 'Behavior \u2014 "What patterns does the agent develop?"')
    print("  After enough actions, the system learns what 'normal' looks like for")
    print("  this agent. Then we'll change the agent's behavior and watch the")
    print("  system detect the shift.")
    print()

    _section("Simulating 100 normal operations")

    import random
    normal_actions = [
        ("read", "claims_db"),
        ("read", "customer_records"),
        ("query", "claims_db"),
        ("query", "policy_db"),
        ("write", "claims_db"),
        ("read", "policy_db"),
        ("assess", "claims_db"),
    ]
    normal_weights = [25, 15, 15, 10, 10, 15, 10]

    trust_traj: list[float] = [runtime.get_trust_profile("claims-bot").overall_trust]

    for i in range(100):
        act_type, act_target = random.choices(normal_actions, weights=normal_weights, k=1)[0]
        params = {}
        if act_type == "write":
            params = {"claim_id": f"CLM-{random.randint(1000, 9999)}", "amount": random.randint(500, 9000)}
        elif act_type == "assess":
            params = {"claim_id": f"CLM-{random.randint(1000, 9999)}", "amount": random.randint(1000, 8000)}
        action = GovAction(agent_id="claims-bot", action_type=act_type, target=act_target, parameters=params)
        ctx = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
        runtime.evaluate(action, ctx)
        trust_traj.append(runtime.get_trust_profile("claims-bot").overall_trust)

        bar_width = 40
        filled = int((i + 1) / 100 * bar_width)
        bar = chr(9608) * filled + chr(9617) * (bar_width - filled)
        print(f"\r  Simulating 100 normal operations... [{bar}] {i + 1}/100", end="", flush=True)

    print()
    print()

    # Show fingerprint
    fp = runtime.get_fingerprint("claims-bot")
    if fp is not None:
        print(f"  {_bold('Behavioral Fingerprint:')}")
        print(f"    Observations: {fp.total_observations}")
        print(f"    Confidence:   {fp.confidence:.2f}")
        print()

        if fp.action_distribution:
            print(f"    Action Distribution:")
            for act, freq in sorted(fp.action_distribution.items(), key=lambda x: -x[1]):
                bar = format_pct_bar(freq, 36)
                print(f"      {act:8s} {bar}  {freq:.0%}")
            print()

        if fp.target_distribution:
            print(f"    Target Distribution:")
            for tgt, freq in sorted(fp.target_distribution.items(), key=lambda x: -x[1]):
                bar = format_pct_bar(freq, 28)
                print(f"      {tgt:20s} {bar}  {freq:.0%}")
            print()

    # Show trust trajectory
    start_trust = trust_traj[0]
    end_trust = trust_traj[-1]
    print(f"  {_bold('Trust Trajectory:')}")
    print(f"    Start: {start_trust:.3f} {chr(8594)} End: {end_trust:.3f}")
    trend = "rising" if end_trust > start_trust + 0.02 else ("falling" if end_trust < start_trust - 0.02 else "stable")
    print(f"    Trend: {trend}")
    print()

    # Mini trajectory chart
    n_rows = 5
    min_t = min(trust_traj)
    max_t = max(trust_traj)
    t_range = max_t - min_t if max_t > min_t else 0.1
    chart_width = 40
    step = max(1, len(trust_traj) // chart_width)
    sampled = [trust_traj[i] for i in range(0, len(trust_traj), step)]

    for row in range(n_rows, -1, -1):
        level = min_t + (row / n_rows) * t_range
        label = f"  {level:.2f} |"
        chars = []
        for val in sampled:
            val_row = int((val - min_t) / t_range * n_rows) if t_range > 0 else n_rows // 2
            chars.append("." if val_row == row else " ")
        print(f"{label}{''.join(chars)}|")
    print(f"         {chr(9492)}{chr(9472) * len(sampled)}{chr(9496)}")
    print()

    allowed_count = sum(1 for t1, t2 in zip(trust_traj, trust_traj[1:]) if t2 >= t1)
    denied_count = 100 - allowed_count
    print(f"  {_green(chr(10003))} 100 actions evaluated")
    print(f"  {_green(chr(10003))} Fingerprint established")
    print(f"  {_green(chr(10003))} Trust trajectory: {trend}")

    _pause()

    # ── Chapter 4b: Drift ───────────────────────────────────────────

    _section("Introducing behavioral drift")

    trust_before_drift = runtime.get_trust_profile("claims-bot").overall_trust
    print(f"  Agent starts doing mostly deletes instead of reads...")
    print(f"  Trust before drift: {_green(f'{trust_before_drift:.3f}')}")
    print()

    drift_traj: list[float] = [trust_before_drift]

    for i in range(50):
        # 80% deletes, 10% reads on sensitive targets, 10% normal reads
        r = random.random()
        if r < 0.80:
            action = GovAction(agent_id="claims-bot", action_type="delete", target=random.choice(["claims_db", "customer_records", "policy_db"]))
        elif r < 0.90:
            action = GovAction(agent_id="claims-bot", action_type="read", target="payroll_db")
        else:
            action = GovAction(agent_id="claims-bot", action_type="read", target="claims_db")
        ctx = GovCtx(agent_id="claims-bot", trust_profile=runtime.get_trust_profile("claims-bot"))
        runtime.evaluate(action, ctx)
        drift_traj.append(runtime.get_trust_profile("claims-bot").overall_trust)

        bar_width = 40
        filled = int((i + 1) / 50 * bar_width)
        bar = chr(9608) * filled + chr(9617) * (bar_width - filled)
        print(f"\r  Simulating drift... [{bar}] {i + 1}/50", end="", flush=True)

    print()
    print()

    # Show drift
    drift = runtime.get_drift("claims-bot")
    if drift is not None:
        sev = drift.severity
        color = _red if sev in ("high", "critical") else (_yellow if sev == "moderate" else _green)
        print(f"  {_bold('Drift Detection Report:')}")
        print(f"    Drift Score:    {color(f'{drift.overall:.2f}')} ({sev.upper()})")
        print(f"    JSD Divergence: {drift.action_drift:.2f}")
        if drift.detail:
            print(f"    Detail:         {drift.detail}")
        print()

    # Trust erosion
    trust_after_drift = runtime.get_trust_profile("claims-bot").overall_trust
    erosion = trust_after_drift - trust_before_drift
    print(f"  {_bold('Trust Impact:')}")
    print(f"    Trust eroded by drift: {trust_before_drift:.3f} {chr(8594)} {trust_after_drift:.3f}")
    print(f"    Erosion amount: {_red(f'{erosion:+.3f}')}")
    print()

    alerts = runtime.get_drift_alerts("claims-bot")
    if alerts:
        for alert in alerts[-1:]:
            sev_color = _red if alert.severity in ("critical", "high") else _yellow
            print(f"  {sev_color(chr(9888))} DRIFT ALERT generated:")
            print(f"    Severity: {alert.severity.upper()}")
            if alert.drift_score.detail:
                print(f"    Message:  \"{alert.drift_score.detail}\"")
        print()

    print(f"  {_dim('What you learned: The system learns normal patterns,')}")
    print(f"  {_dim('then detects when behavior changes. Drift erodes trust')}")
    print(f"  {_dim('and changes governance outcomes automatically.')}")

    _pause()

    # ── Chapter 5: Interruption ─────────────────────────────────────

    _header("Chapter 5", 'Interruption \u2014 "Can we stop an agent mid-action?"')
    print("  Governance doesn't just decide before execution \u2014 it can stop an agent")
    print("  mid-action and roll back. Watch an interrupt happen.")
    print()

    print(f"  Agent claims-bot is processing a batch write...")
    print()

    rollback_log: list[str] = []

    def _rollback() -> None:
        for item in reversed(rollback_log):
            print(f"    Reverting {item}... done")
        rollback_log.clear()

    batch_action = GovAction(agent_id="claims-bot", action_type="write", target="claims_db")
    handle = runtime.begin_execution(batch_action, ctx1, rollback=_rollback)

    for record_num in range(1, 11):
        if handle.check_interrupt():
            print(f"\n  {_red(chr(9940))} EXECUTION HALTED after record {record_num - 1}")
            break

        rollback_log.append(f"record {record_num}")
        print(f"  {chr(9203)} Processing record {record_num} of 10... {_green(chr(10003))}")
        _time.sleep(0.15)

        if record_num == 3:
            print()
            print(f"  {_yellow(chr(9889))} INTERRUPT SIGNAL: Anomaly detected in write pattern")
            print()
            print(f"  Agent checks interrupt flag at next safe point...")
            print()

            runtime.interrupt_action(
                batch_action.id,
                reason="Anomaly detected in write pattern",
                source="drift_monitor",
            )

    if handle.is_interrupted:
        print()
        print(f"  {_bold(chr(8617))} ROLLBACK executing...")

        handle_rollback = handle.rollback
        if handle_rollback:
            handle_rollback()

        print()
        print(f"  {_green(chr(10003))} Rollback complete. System state restored.")
    else:
        runtime.complete_execution(batch_action.id, ctx1)
        print(f"\n  All records processed successfully.")

    print()
    print(f"  {_dim('What you learned: Governance has teeth.')}")
    print(f"  {_dim('The agent cooperated with the interrupt ' + emdash + ' it checked')}")
    print(f"  {_dim('for interrupts at each safe point and rolled back cleanly.')}")

    _pause()

    # ── Chapter 6: Audit Trail ──────────────────────────────────────

    _header("Chapter 6", 'Audit Trail \u2014 "What happened and why?"')
    print("  Every governance decision is recorded with full justification \u2014 not")
    print("  just what happened, but why.")
    print()

    if runtime.audit_trail is not None:
        _section("Recent audit records")

        records = runtime.audit_trail.query(agent_id="claims-bot", limit=5)
        if records:
            for i, r in enumerate(records):
                verdict_color = _green if r.verdict == "ALLOW" else (_red if r.verdict == "DENY" else _yellow)
                ts_str = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
                print(f"  #{r.record_id}  {ts_str}  {verdict_color(f'GOVERNANCE.{r.verdict}')}")
                print(f"       Action: {r.action_type} {chr(8594)} {r.action_target}")
                print(f"       UCS: {r.ucs:.2f}  Tier: {r.tier}")
                print(f"       Trust: {r.trust_score:.3f} ({r.trust_trend})")
                if r.justification:
                    just = r.justification[:100]
                    if len(r.justification) > 100:
                        just += "..."
                    print(f"       Why: {just}")
                print()

        _section("Audit summary")

        summary = runtime.audit_trail.summary()
        total = summary.get("total_records", 0)
        print(f"  Total evaluations:  {total}")
        print()

        by_verdict = summary.get("by_verdict", {})
        if by_verdict:
            print(f"  By verdict:")
            for v, cnt in sorted(by_verdict.items()):
                pct = cnt / total * 100 if total else 0
                bar = format_pct_bar(cnt / total if total else 0, 40)
                vcolor = _green if v == "ALLOW" else (_red if v == "DENY" else _yellow)
                print(f"    {vcolor(f'{v:10s}')} {cnt:5d}  {bar}  {pct:.1f}%")
            print()

        by_severity = summary.get("by_severity", {})
        if by_severity:
            print(f"  By severity:")
            for s, cnt in sorted(by_severity.items()):
                pct = cnt / total * 100 if total else 0
                bar = format_pct_bar(cnt / total if total else 0, 40)
                print(f"    {s:10s} {cnt:5d}  {bar}  {pct:.1f}%")
            print()

        recent_alerts = summary.get("recent_alerts", [])
        if recent_alerts:
            print(f"  Recent alerts:")
            for a in recent_alerts[-3:]:
                sev = a.get("severity", "?")
                code = a.get("context_code", "?")
                sev_color = _red if sev in ("critical", "alert") else _yellow
                print(f"    [{sev_color(sev.upper())}] {code}")
            print()

    print(f"  {_dim('What you learned: Every decision is recorded with full')}")
    print(f"  {_dim('justification. The audit trail explains the why, not just the what.')}")

    _pause()

    # ── Chapter 7: Full picture ─────────────────────────────────────

    _header("Chapter 7", 'The Full Picture')

    final_trust = runtime.get_trust_profile("claims-bot").overall_trust
    report = runtime.get_trust_report("claims-bot")

    print(f"  {_bold('Governance Report for claims-bot')}")
    print()
    print(f"    Current Trust:     {final_trust:.3f}")
    print(f"    Total Actions:     {report.get('successful_actions', 0) + report.get('violation_count', 0)}")
    print(f"    Successful:        {report.get('successful_actions', 0)}")
    print(f"    Violations:        {report.get('violation_count', 0)}")
    print(f"    Violation Rate:    {report.get('violation_rate', 0):.1%}")
    print()

    fp_info = report.get("fingerprint")
    if fp_info:
        print(f"    Fingerprint:       {fp_info.get('total_observations', 0)} observations (confidence: {fp_info.get('confidence', 0):.2f})")

    drift_info = report.get("drift")
    if drift_info:
        sev = drift_info.get("severity", "none")
        color = _red if sev in ("high", "critical") else (_yellow if sev == "moderate" else _green)
        drift_val = drift_info.get("overall", 0)
        print(f"    Drift:             {color(f'{drift_val:.2f}')} ({sev})")

    alert_count = report.get("active_alerts", 0)
    if alert_count:
        print(f"    Active Alerts:     {_red(str(alert_count))}")
    print()

    traj = report.get("trajectory", {})
    sources = traj.get("sources", {})
    if sources:
        print(f"  {_bold('Trust by source:')}")
        for src, info in sorted(sources.items()):
            nd = info.get("net_delta", 0)
            cnt = info.get("count", 0)
            sign = "+" if nd >= 0 else ""
            color = _green if nd > 0 else (_red if nd < 0 else _yellow)
            print(f"    {src:25s} {color(f'{sign}{nd:.3f}')} ({cnt} events)")
        print()

    print(_bold(_cyan("=" * 60)))
    print(_bold(_cyan("  Tutorial complete.")))
    print(_bold(_cyan("=" * 60)))
    print()
    print(f"  You just saw the entire Nomotic governance lifecycle:")
    print(f"    1. Identity     \u2014 Cryptographic birth certificate")
    print(f"    2. Authority    \u2014 Scope, boundaries, and rules")
    print(f"    3. Evaluation   \u2014 13-dimension scoring with UCS")
    print(f"    4. Behavior     \u2014 Fingerprinting and drift detection")
    print(f"    5. Interruption \u2014 Mechanical halt with rollback")
    print(f"    6. Audit Trail  \u2014 Full accountability record")
    print(f"    7. Trust        \u2014 Continuous calibration from all sources")
    print()
    print(f"  Next steps:")
    print(f"    nomotic birth --agent-id my-agent --archetype customer-experience --org my-org")
    print(f"    nomotic scope set my-agent --actions read,write --boundaries my_db")
    print(f"    nomotic eval my-agent --action read --target my_db")
    print(f"    nomotic simulate my-agent --scenario normal")
    print()


def _print_evaluation(
    verdict: "GovernanceVerdict",
    trust_before: float,
    trust_after: float,
) -> None:
    """Print a formatted governance evaluation result."""
    from nomotic.sandbox import format_bar
    from nomotic.types import Verdict

    print(f"  Trust:   {trust_before:.3f}")
    print()

    for ds in verdict.dimension_scores:
        bar = format_bar(ds.score, 20)
        name_padded = f"{ds.dimension_name:26s}"
        score_str = f"{ds.score:.2f}"

        if ds.veto and ds.score < 0.5:
            check = _red(f"{chr(10007)} VETO: {ds.reasoning}")
            color_score = _red(score_str)
        elif ds.score >= 0.7:
            check = _green(ds.reasoning)
            color_score = _green(score_str)
        elif ds.score >= 0.4:
            check = _yellow(ds.reasoning)
            color_score = _yellow(score_str)
        else:
            check = _red(ds.reasoning)
            color_score = _red(score_str)

        print(f"  {name_padded} {color_score}  {bar}  {check}")

    print()

    ucs_str = f"{verdict.ucs:.3f}"
    tier_labels = {1: "veto gate", 2: "threshold evaluation", 3: "deliberative review"}
    tier_label = tier_labels.get(verdict.tier, "unknown")

    if verdict.verdict == Verdict.ALLOW:
        verdict_display = _green(f"{chr(9989)} ALLOW")
    elif verdict.verdict == Verdict.DENY:
        verdict_display = _red(f"{chr(10060)} DENY")
    elif verdict.verdict == Verdict.ESCALATE:
        verdict_display = _yellow(f"{chr(9888)} ESCALATE")
    else:
        verdict_display = _yellow(f"{chr(9888)} {verdict.verdict.name}")

    print(f"  UCS:      {ucs_str}")
    print(f"  Tier:     {verdict.tier} ({tier_label})")
    print(f"  Verdict:  {verdict_display}")
    if verdict.vetoed_by:
        print(f"  Vetoed by: {', '.join(verdict.vetoed_by)}")
    if verdict.reasoning:
        print(f"  Reason:   {verdict.reasoning}")
    print(f"  Time:     {verdict.evaluation_time_ms:.1f}ms")

    # Trust update
    delta = trust_after - trust_before
    sign = "+" if delta >= 0 else ""
    delta_color = _green if delta > 0 else (_red if delta < 0 else _yellow)
    trend = "stable"
    if delta > 0.001:
        trend = f"{chr(8599)} rising"
    elif delta < -0.001:
        trend = f"{chr(8600)} falling"

    print()
    print(f"  Before:   {trust_before:.3f}")
    print(f"  After:    {trust_after:.3f}  ({delta_color(f'{sign}{delta:.3f}')})")
    print(f"  Trend:    {trend}")
    print()


def _cmd_audit_show(args: argparse.Namespace) -> None:
    """Show audit records from an in-memory simulation or local eval sessions."""
    # This is a convenience alias that displays a help message
    # pointing users to the tutorial or eval commands.
    print("The 'audit show' command works in two modes:")
    print()
    print("  1. After 'nomotic tutorial' — audit records are shown as part")
    print("     of the interactive walkthrough (Chapter 6).")
    print()
    print("  2. Against a running API server:")
    print("     nomotic audit query --agent-id <agent> --host 127.0.0.1 --port 8420")
    print("     nomotic audit summary --agent-id <agent> --host 127.0.0.1 --port 8420")
    print()
    print("  Run 'nomotic tutorial' for the full interactive experience.")


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

    # setup
    sub.add_parser("setup", help="Configure Nomotic defaults (organization, owner, zone)")

    # birth
    birth = sub.add_parser("birth", help="Create a new agent")
    birth.add_argument("--name", default=None, help="Agent name")
    birth.add_argument("--agent-id", dest="agent_id_compat", default=None, help=argparse.SUPPRESS)
    birth.add_argument("--archetype", default=None, help="Behavioral archetype (optional, can be set later)")
    birth.add_argument("--org", default=None, help="Organization (default: from setup config)")
    birth.add_argument("--zone", default=None, help="Governance zone (default: from setup config)")
    birth.add_argument("--owner", default=None, help="Agent owner (default: from setup config)")

    # verify
    verify = sub.add_parser("verify", help="Verify a certificate")
    verify.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # inspect
    inspect_ = sub.add_parser("inspect", help="Inspect a certificate")
    inspect_.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # suspend
    suspend = sub.add_parser("suspend", help="Suspend a certificate")
    suspend.add_argument("cert_id", help="Agent ID, name, or certificate ID")
    suspend.add_argument("--reason", required=True, help="Reason for suspension")

    # reactivate
    reactivate = sub.add_parser("reactivate", help="Reactivate a suspended certificate")
    reactivate.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # revoke
    revoke = sub.add_parser("revoke", help="Permanently revoke a certificate")
    revoke.add_argument("cert_id", help="Agent ID, name, or certificate ID")
    revoke.add_argument("--reason", required=True, help="Reason for revocation")

    # renew
    renew = sub.add_parser("renew", help="Renew a certificate with lineage link")
    renew.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # list
    list_ = sub.add_parser("list", help="List certificates")
    list_.add_argument("--status", default=None, help="Filter by status")
    list_.add_argument("--archetype", default=None, help="Filter by archetype")
    list_.add_argument("--org", default=None, help="Filter by organization")

    # reputation
    rep = sub.add_parser("reputation", help="Show trust trajectory")
    rep.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # export
    export = sub.add_parser("export", help="Export public certificate to file")
    export.add_argument("cert_id", help="Agent ID, name, or certificate ID")

    # ── status ──────────────────────────────────────────────────────
    status = sub.add_parser("status", help="Quick operational status for an agent")
    status.add_argument("cert_id", help="Agent ID, name, or certificate ID")

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

    arch_set = arch_sub.add_parser("set", help="Set archetype for an agent (interactive)")
    arch_set.add_argument("agent_identifier", help="Agent name, ID, or certificate ID")

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

    # ── scope ───────────────────────────────────────────────────────
    scope = sub.add_parser("scope", help="Configure agent authority scope")
    scope.add_argument("agent_id", help="Agent ID, name, or certificate ID")
    scope_sub = scope.add_subparsers(dest="scope_command")
    scope_set = scope_sub.add_parser("set", help="Set scope and boundaries")
    scope_set.add_argument("--actions", default=None, help="Comma-separated allowed actions")
    scope_set.add_argument("--boundaries", default=None, help="Comma-separated allowed targets")
    scope_show = scope_sub.add_parser("show", help="Show current scope")

    # ── rule ────────────────────────────────────────────────────────
    rule = sub.add_parser("rule", help="Manage governance rules")
    rule.add_argument("agent_id", help="Agent ID, name, or certificate ID")
    rule_sub = rule.add_subparsers(dest="rule_command")
    rule_add = rule_sub.add_parser("add", help="Add a governance rule")
    rule_add.add_argument("--type", required=True, choices=["ethical", "human-override"], help="Rule type")
    rule_add.add_argument("--condition", default=None, help="Condition expression (ethical rules)")
    rule_add.add_argument("--action", default=None, help="Action type (human-override rules)")
    rule_add.add_argument("--message", default=None, help="Message when rule triggers")
    rule_add.add_argument("--name", default=None, help="Optional rule name")

    # ── config show ─────────────────────────────────────────────────
    config = sub.add_parser("config", help="Show governance configuration")
    config.add_argument("agent_id", help="Agent ID, name, or certificate ID")
    config_sub = config.add_subparsers(dest="config_command")
    config_set = config_sub.add_parser("set", help="Set a configuration value")
    config_set.add_argument("--retention", default=None, help="Audit retention period (2y, 3y, 5y, forever)")

    # configure (alias for config)
    configure = sub.add_parser("configure", help="Show governance configuration (alias for 'config')")
    configure.add_argument("agent_id", help="Agent ID, name, or certificate ID")

    # ── eval ────────────────────────────────────────────────────────
    eval_parser = sub.add_parser("eval", help="Evaluate a single action through the governance pipeline")
    eval_parser.add_argument("agent_id", help="Agent ID, name, or certificate ID")
    eval_parser.add_argument("--action", required=True, help="Action type to evaluate")
    eval_parser.add_argument("--target", default=None, help="Target resource")
    eval_parser.add_argument("--params", default=None, help="JSON parameters")

    # ── simulate ────────────────────────────────────────────────────
    sim_parser = sub.add_parser("simulate", help="Run a batch governance simulation")
    sim_parser.add_argument("agent_id", help="Agent ID, name, or certificate ID")
    sim_parser.add_argument("--scenario", required=True, help="Scenario name (normal, drift, violations, mixed)")
    sim_parser.add_argument("--count", type=int, default=None, help="Override action count")
    sim_parser.add_argument("--description", default=None, help="Custom scenario description")

    # ── tutorial ───────────────────────────────────────────────────
    sub.add_parser("tutorial", help="Interactive walkthrough of the complete governance lifecycle")

    # ── serve ────────────────────────────────────────────────────────
    serve = sub.add_parser("serve", help="Start the Nomotic API server")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")

    # ── hello ────────────────────────────────────────────────────────
    sub.add_parser("hello", help="Start the Nomotic tutorial (same as 'nomotic tutorial')")

    # ── history ─────────────────────────────────────────────────────
    history = sub.add_parser("history", help="Full historical record for an agent (including revoked)")
    history.add_argument("identifier", help="Agent ID, name, or certificate ID")

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

    # ── trust ────────────────────────────────────────────────────
    trust_parser = sub.add_parser("trust", help="Show trust report for an agent")
    trust_parser.add_argument("agent_id", help="Agent identifier")
    trust_parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    trust_parser.add_argument("--port", type=int, default=8420, help="API server port (default: 8420)")

    # ── alerts ───────────────────────────────────────────────────
    alerts_parser = sub.add_parser("alerts", help="List drift alerts")
    alerts_parser.add_argument("--agent-id", default=None, help="Filter by agent ID")
    alerts_parser.add_argument("--unacknowledged", action="store_true", help="Show only unacknowledged alerts")
    alerts_parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    alerts_parser.add_argument("--port", type=int, default=8420, help="API server port (default: 8420)")

    # ── audit ────────────────────────────────────────────────────
    audit_parser = sub.add_parser("audit", help="Show audit trail records")
    audit_sub = audit_parser.add_subparsers(dest="audit_command")
    audit_query = audit_sub.add_parser("query", help="Query audit records")
    audit_query.add_argument("--agent-id", default=None, help="Filter by agent ID")
    audit_query.add_argument("--severity", default=None, help="Filter by severity")
    audit_query.add_argument("--limit", type=int, default=20, help="Max records to show")
    audit_query.add_argument("--host", default="127.0.0.1", help="API server host")
    audit_query.add_argument("--port", type=int, default=8420, help="API server port")
    audit_summary = audit_sub.add_parser("summary", help="Show audit trail summary")
    audit_summary.add_argument("--agent-id", default=None, help="Filter by agent ID")
    audit_summary.add_argument("--host", default="127.0.0.1", help="API server host")
    audit_summary.add_argument("--port", type=int, default=8420, help="API server port")
    # Default: query
    audit_parser.add_argument("--agent-id", default=None, help="Filter by agent ID")
    audit_parser.add_argument("--severity", default=None, help="Filter by severity")
    audit_parser.add_argument("--limit", type=int, default=20, help="Max records to show")
    audit_parser.add_argument("--host", default="127.0.0.1", help="API server host")
    audit_parser.add_argument("--port", type=int, default=8420, help="API server port")

    # ── provenance ──────────────────────────────────────────────
    prov_parser = sub.add_parser("provenance", help="Show configuration provenance")
    prov_parser.add_argument("--target-type", default=None, help="Filter by target type")
    prov_parser.add_argument("--target-id", default=None, help="Filter by target ID")
    prov_parser.add_argument("--host", default="127.0.0.1", help="API server host")
    prov_parser.add_argument("--port", type=int, default=8420, help="API server port")

    # ── owner ───────────────────────────────────────────────────
    owner_parser = sub.add_parser("owner", help="Show owner activity")
    owner_parser.add_argument("owner_id", help="Owner identifier")
    owner_parser.add_argument("--engagement", action="store_true", help="Show engagement score")
    owner_parser.add_argument("--host", default="127.0.0.1", help="API server host")
    owner_parser.add_argument("--port", type=int, default=8420, help="API server port")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "setup": _cmd_setup,
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
        "status": _cmd_status,
        "history": _cmd_history,
        "archetype": _cmd_archetype,
        "org": _cmd_org,
        "zone": _cmd_zone,
        "scope": _cmd_scope,
        "rule": _cmd_rule,
        "config": _cmd_config,
        "configure": _cmd_config,
        "eval": _cmd_eval,
        "simulate": _cmd_simulate,
        "tutorial": _cmd_tutorial,
        "serve": _cmd_serve,
        "hello": _cmd_hello,
        "fingerprint": _cmd_fingerprint,
        "drift": _cmd_drift,
        "alerts": _cmd_alerts,
        "trust": _cmd_trust,
        "audit": _cmd_audit,
        "provenance": _cmd_provenance,
        "owner": _cmd_owner,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
