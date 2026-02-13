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


# ── CLI commands ─────────────────────────────────────────────────────────


def _cmd_birth(args: argparse.Namespace) -> None:
    ca, store = _build_ca(args.base_dir)
    zone_path = args.zone or "global"
    cert, agent_sk = ca.issue(
        agent_id=args.agent_id,
        archetype=args.archetype,
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
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
