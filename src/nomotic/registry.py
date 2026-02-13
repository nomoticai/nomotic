"""Validation registries for archetypes, zones, and organizations.

Archetypes are behavioral templates that drive governance intelligence.
Zones are hierarchical governance paths.  Organizations are unique
identity namespaces that prevent impersonation.

All three share the same name-validation rules and ``ValidationResult``
type.  They live in a single module because the overlap is large and the
coupling is intentional.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ArchetypeDefinition",
    "ArchetypeRegistry",
    "FileOrgStore",
    "MemoryOrgStore",
    "OrganizationRegistry",
    "OrgRegistration",
    "OrgStatus",
    "OrgStore",
    "ValidationResult",
    "ZoneValidator",
]

# ── Shared name-validation pattern ──────────────────────────────────────

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{1,62}[a-z0-9])?$")
_DOUBLE_HYPHEN = re.compile(r"--")


def _validate_name_format(name: str) -> list[str]:
    """Return a list of format errors for *name* (empty list ⇒ OK)."""
    errors: list[str] = []
    if not name:
        errors.append("name must not be empty")
        return errors
    if len(name) < 3:
        errors.append("name must be at least 3 characters")
    if len(name) > 64:
        errors.append("name must be at most 64 characters")
    if name != name.lower():
        errors.append("name must be lowercase")
    if not re.match(r"^[a-z0-9]", name):
        errors.append("name must start with an alphanumeric character")
    if not re.match(r".*[a-z0-9]$", name):
        errors.append("name must end with an alphanumeric character")
    if re.search(r"[^a-z0-9-]", name):
        errors.append("name must contain only lowercase alphanumeric characters and hyphens")
    if _DOUBLE_HYPHEN.search(name):
        errors.append("name must not contain double hyphens")
    return errors


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── ValidationResult ────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Outcome of a name or path validation check.

    ``valid`` is ``True`` when the input is acceptable for use.
    ``warnings`` carry non-blocking notes (e.g. unknown archetype in
    non-strict mode).  ``errors`` carry blocking issues.
    """

    valid: bool
    name: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggestion: str | None = None


# ── ArchetypeDefinition ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchetypeDefinition:
    """A registered behavioural archetype.

    Built-in archetypes ship with Nomotic.  Custom archetypes can be
    registered by organizations.
    """

    name: str
    description: str
    category: str
    builtin: bool
    prior_name: str | None = None  # Links to PriorRegistry for behavioral fingerprint seeding


# ── Built-in archetypes ─────────────────────────────────────────────────

BUILT_IN_ARCHETYPES: dict[str, dict[str, Any]] = {
    # Customer-facing
    "customer-experience": {
        "description": "Customer service, support, and interaction agents",
        "category": "customer-facing",
        "prior_name": "customer-experience",
    },
    "sales-assistant": {
        "description": "Sales support, lead qualification, and outreach agents",
        "category": "customer-facing",
        "prior_name": "sales-agent",
    },
    # Data and analytics
    "data-processing": {
        "description": "Data transformation, ETL, and pipeline agents",
        "category": "data",
        "prior_name": "data-processor",
    },
    "analytics": {
        "description": "Data analysis, reporting, and insight generation agents",
        "category": "data",
        "prior_name": "research-analyst",
    },
    # Financial
    "financial-transactions": {
        "description": "Payment processing, transfers, and financial operation agents",
        "category": "financial",
        "prior_name": "financial-analyst",
    },
    "underwriting": {
        "description": "Risk assessment, policy evaluation, and approval agents",
        "category": "financial",
        "prior_name": "financial-analyst",
    },
    # Operations
    "system-administration": {
        "description": "Infrastructure management, deployment, and monitoring agents",
        "category": "operations",
        "prior_name": "operations-coordinator",
    },
    "workflow-orchestration": {
        "description": "Multi-step process coordination and task routing agents",
        "category": "operations",
        "prior_name": "operations-coordinator",
    },
    "supply-chain": {
        "description": "Logistics, inventory, and procurement agents",
        "category": "operations",
        "prior_name": "operations-coordinator",
    },
    # Content
    "content-generation": {
        "description": "Text, media, and document creation agents",
        "category": "content",
        "prior_name": "content-creator",
    },
    "content-moderation": {
        "description": "Content review, filtering, and compliance agents",
        "category": "content",
        "prior_name": "security-monitor",
    },
    # Security and compliance
    "security-monitoring": {
        "description": "Threat detection, incident response, and security audit agents",
        "category": "security",
        "prior_name": "security-monitor",
    },
    "compliance-audit": {
        "description": "Regulatory compliance checking and reporting agents",
        "category": "security",
        "prior_name": "security-monitor",
    },
    # Healthcare
    "clinical-support": {
        "description": "Clinical decision support and care coordination agents",
        "category": "healthcare",
        "prior_name": "healthcare-agent",
    },
    # Research
    "research-assistant": {
        "description": "Literature review, data gathering, and research support agents",
        "category": "research",
        "prior_name": "research-analyst",
    },
    # General
    "general-purpose": {
        "description": "Unspecialized agents or agents awaiting archetype assignment",
        "category": "general",
        "prior_name": None,
    },
}


# ── ArchetypeRegistry ───────────────────────────────────────────────────


class ArchetypeRegistry:
    """Registry of known agent archetypes.

    Ships with built-in archetypes.  Organizations can register custom
    archetypes.  Validation at certificate issuance prevents typos and
    inconsistency.
    """

    def __init__(self, *, strict: bool = False) -> None:
        """Create an empty registry.

        Args:
            strict: If ``True``, reject unknown archetypes at validation.
                    If ``False``, warn but allow (returns a validation
                    result with a warning).
        """
        self._strict = strict
        self._archetypes: dict[str, ArchetypeDefinition] = {}

    @classmethod
    def with_defaults(cls, *, strict: bool = False) -> ArchetypeRegistry:
        """Create a registry pre-loaded with built-in archetypes."""
        registry = cls(strict=strict)
        for name, info in BUILT_IN_ARCHETYPES.items():
            registry._archetypes[name] = ArchetypeDefinition(
                name=name,
                description=info["description"],
                category=info["category"],
                builtin=True,
                prior_name=info.get("prior_name"),
            )
        return registry

    def register(self, name: str, description: str, category: str) -> ArchetypeDefinition:
        """Register a custom archetype.

        ``name`` must be unique and pass format validation.

        Raises:
            ValueError: If the name is already registered or has an invalid
                format.
        """
        errors = _validate_name_format(name)
        if errors:
            raise ValueError(f"Invalid archetype name '{name}': {'; '.join(errors)}")
        if name in self._archetypes:
            raise ValueError(f"Archetype '{name}' is already registered")
        defn = ArchetypeDefinition(
            name=name, description=description, category=category, builtin=False,
        )
        self._archetypes[name] = defn
        return defn

    def get(self, name: str) -> ArchetypeDefinition | None:
        """Look up an archetype by name."""
        return self._archetypes.get(name)

    def validate(self, name: str) -> ValidationResult:
        """Validate an archetype name.

        Returns a :class:`ValidationResult` indicating whether the name
        is acceptable, with any warnings or errors.
        """
        format_errors = _validate_name_format(name)
        if format_errors:
            return ValidationResult(
                valid=False, name=name, errors=format_errors,
            )

        if name in self._archetypes:
            return ValidationResult(valid=True, name=name)

        # Unknown archetype — suggest closest match
        suggestion = self._suggest(name)

        if self._strict:
            return ValidationResult(
                valid=False,
                name=name,
                errors=[f"unknown archetype '{name}'"],
                suggestion=suggestion,
            )

        warnings = [f"archetype '{name}' is not registered"]
        return ValidationResult(
            valid=True, name=name, warnings=warnings, suggestion=suggestion,
        )

    def list(self, category: str | None = None) -> list[ArchetypeDefinition]:
        """List all registered archetypes, optionally filtered by category."""
        if category is None:
            return sorted(self._archetypes.values(), key=lambda d: d.name)
        return sorted(
            (d for d in self._archetypes.values() if d.category == category),
            key=lambda d: d.name,
        )

    def categories(self) -> list[str]:
        """List all distinct categories."""
        return sorted({d.category for d in self._archetypes.values()})

    # ── Internal helpers ────────────────────────────────────────────

    def _suggest(self, name: str) -> str | None:
        """Return the closest registered name if edit distance ≤ 3."""
        matches = difflib.get_close_matches(
            name, self._archetypes.keys(), n=1, cutoff=0.6,
        )
        return matches[0] if matches else None


# ── ZoneValidator ───────────────────────────────────────────────────────

_SEGMENT_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")
_MAX_ZONE_DEPTH = 10


class ZoneValidator:
    """Validates governance zone paths.

    Zone paths are slash-separated hierarchical identifiers such as
    ``global/us/production``.  Each segment follows the same naming
    rules as archetype names.
    """

    def validate(self, zone_path: str) -> ValidationResult:
        """Validate a zone path format."""
        errors: list[str] = []

        if not zone_path:
            errors.append("zone path must not be empty")
            return ValidationResult(valid=False, name=zone_path, errors=errors)

        if zone_path.startswith("/"):
            errors.append("zone path must not start with '/'")
        if zone_path.endswith("/"):
            errors.append("zone path must not end with '/'")
        if "//" in zone_path:
            errors.append("zone path must not contain '//'")

        if errors:
            return ValidationResult(valid=False, name=zone_path, errors=errors)

        segments = zone_path.split("/")
        if len(segments) > _MAX_ZONE_DEPTH:
            errors.append(f"zone path must have at most {_MAX_ZONE_DEPTH} segments")
            return ValidationResult(valid=False, name=zone_path, errors=errors)

        for i, seg in enumerate(segments):
            if not seg:
                errors.append(f"segment {i + 1} is empty")
            elif not _SEGMENT_RE.match(seg):
                seg_errors = _validate_name_format(seg) if len(seg) >= 3 else []
                if seg_errors:
                    errors.append(f"segment '{seg}' is invalid: {'; '.join(seg_errors)}")
                elif not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", seg):
                    errors.append(f"segment '{seg}' contains invalid characters")

        return ValidationResult(
            valid=len(errors) == 0, name=zone_path, errors=errors,
        )

    def parse_segments(self, zone_path: str) -> list[str]:
        """Split a valid zone path into segments.

        Raises:
            ValueError: If the zone path is invalid.
        """
        result = self.validate(zone_path)
        if not result.valid:
            raise ValueError(
                f"Invalid zone path '{zone_path}': {'; '.join(result.errors)}"
            )
        return zone_path.split("/")

    def is_parent_of(self, parent: str, child: str) -> bool:
        """Check if *parent* zone is an ancestor of *child* zone."""
        return child.startswith(parent + "/") and child != parent

    def common_ancestor(self, zone_a: str, zone_b: str) -> str | None:
        """Find the deepest common ancestor of two zones."""
        segs_a = zone_a.split("/")
        segs_b = zone_b.split("/")
        common: list[str] = []
        for sa, sb in zip(segs_a, segs_b):
            if sa == sb:
                common.append(sa)
            else:
                break
        return "/".join(common) if common else None


# ── Organization Registry ───────────────────────────────────────────────


class OrgStatus(Enum):
    """Lifecycle status of an organization registration."""

    ACTIVE = auto()
    SUSPENDED = auto()
    REVOKED = auto()


@dataclass
class OrgRegistration:
    """A registered organization."""

    name: str
    display_name: str
    registered_at: datetime
    issuer_fingerprint: str
    contact_email: str | None = None
    status: OrgStatus = OrgStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "registered_at": self.registered_at.isoformat(),
            "issuer_fingerprint": self.issuer_fingerprint,
            "contact_email": self.contact_email,
            "status": self.status.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrgRegistration:
        """Deserialize from a dict."""
        return cls(
            name=d["name"],
            display_name=d["display_name"],
            registered_at=datetime.fromisoformat(d["registered_at"]),
            issuer_fingerprint=d["issuer_fingerprint"],
            contact_email=d.get("contact_email"),
            status=OrgStatus[d["status"]],
        )


# ── OrgStore protocol ──────────────────────────────────────────────────


@runtime_checkable
class OrgStore(Protocol):
    """Persistence backend for organization registrations."""

    def save(self, org: OrgRegistration) -> None: ...
    def get(self, normalized_name: str) -> OrgRegistration | None: ...
    def list(self, status: OrgStatus | None = None) -> list[OrgRegistration]: ...
    def update(self, org: OrgRegistration) -> None: ...


class MemoryOrgStore:
    """In-memory organization store."""

    def __init__(self) -> None:
        self._orgs: dict[str, OrgRegistration] = {}

    def save(self, org: OrgRegistration) -> None:
        self._orgs[org.name] = org

    def get(self, normalized_name: str) -> OrgRegistration | None:
        return self._orgs.get(normalized_name)

    def list(self, status: OrgStatus | None = None) -> list[OrgRegistration]:
        results: list[OrgRegistration] = []
        for org in self._orgs.values():
            if status is not None and org.status != status:
                continue
            results.append(org)
        return sorted(results, key=lambda o: o.name)

    def update(self, org: OrgRegistration) -> None:
        self._orgs[org.name] = org


class FileOrgStore:
    """File-based organization store using JSON.

    Stores registrations in ``<base_dir>/orgs/``.

    Directory layout::

        <base_dir>/
            orgs/
                <normalized-name>.json
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".nomotic"
        self._base = Path(base_dir)
        self._orgs_dir = self._base / "orgs"
        self._orgs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, org: OrgRegistration) -> None:
        path = self._orgs_dir / f"{org.name}.json"
        path.write_text(
            json.dumps(org.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def get(self, normalized_name: str) -> OrgRegistration | None:
        path = self._orgs_dir / f"{normalized_name}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return OrgRegistration.from_dict(data)

    def list(self, status: OrgStatus | None = None) -> list[OrgRegistration]:
        results: list[OrgRegistration] = []
        for path in sorted(self._orgs_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            org = OrgRegistration.from_dict(data)
            if status is not None and org.status != status:
                continue
            results.append(org)
        return results

    def update(self, org: OrgRegistration) -> None:
        self.save(org)


# ── Name normalization ──────────────────────────────────────────────────


def _normalize_org_name(raw: str) -> str:
    """Normalize an organization name for storage and lookup.

    1. Strip leading/trailing whitespace
    2. Convert to lowercase
    3. Replace spaces with hyphens
    4. Collapse multiple hyphens to single
    5. Remove any character that isn't alphanumeric or hyphen
    """
    name = raw.strip().lower()
    name = name.replace(" ", "-")
    name = name.replace("_", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name


# ── OrganizationRegistry ───────────────────────────────────────────────


class OrganizationRegistry:
    """Registry of organizations authorized to issue certificates.

    Organization names are unique.  Registration binds an org name to
    an issuer key fingerprint, preventing impersonation.
    """

    def __init__(self, store: OrgStore | None = None) -> None:
        """Create the registry.

        Args:
            store: Persistence backend.  Defaults to :class:`MemoryOrgStore`.
        """
        self._store: OrgStore = store or MemoryOrgStore()

    def register(
        self,
        name: str,
        issuer_fingerprint: str,
        *,
        contact_email: str | None = None,
    ) -> OrgRegistration:
        """Register a new organization.

        The *name* is normalized (lowercase, spaces to hyphens, stripped).
        Raises ``ValueError`` if the normalized name is already registered
        or if the name fails format validation.
        """
        normalized = _normalize_org_name(name)
        format_errors = _validate_name_format(normalized)
        if format_errors:
            raise ValueError(
                f"Invalid organization name '{name}' "
                f"(normalized: '{normalized}'): {'; '.join(format_errors)}"
            )
        existing = self._store.get(normalized)
        if existing is not None:
            raise ValueError(
                f"Organization '{normalized}' is already registered"
            )
        org = OrgRegistration(
            name=normalized,
            display_name=name.strip(),
            registered_at=_utcnow(),
            issuer_fingerprint=issuer_fingerprint,
            contact_email=contact_email,
        )
        self._store.save(org)
        return org

    def validate(self, name: str) -> ValidationResult:
        """Validate an org name (format and availability)."""
        normalized = _normalize_org_name(name)
        format_errors = _validate_name_format(normalized)
        if format_errors:
            return ValidationResult(
                valid=False, name=normalized, errors=format_errors,
            )
        existing = self._store.get(normalized)
        if existing is not None:
            return ValidationResult(
                valid=False,
                name=normalized,
                errors=[f"organization '{normalized}' is already registered"],
            )
        return ValidationResult(valid=True, name=normalized)

    def get(self, name: str) -> OrgRegistration | None:
        """Look up an org by normalized name."""
        normalized = _normalize_org_name(name)
        return self._store.get(normalized)

    def verify_issuer(self, name: str, issuer_fingerprint: str) -> bool:
        """Verify that a given issuer key is authorized for this org.

        Returns ``True`` if the fingerprint matches the registered org.
        Returns ``False`` if the org doesn't exist or fingerprint doesn't match.
        """
        org = self.get(name)
        if org is None:
            return False
        return org.issuer_fingerprint == issuer_fingerprint

    def list(self, status: OrgStatus | None = None) -> list[OrgRegistration]:
        """List all registered organizations."""
        return self._store.list(status=status)

    def suspend(self, name: str, reason: str) -> OrgRegistration:
        """Suspend an organization registration."""
        org = self.get(name)
        if org is None:
            raise KeyError(f"Organization not found: {name}")
        if org.status != OrgStatus.ACTIVE:
            raise ValueError(
                f"Cannot suspend organization in {org.status.name} state"
            )
        org.status = OrgStatus.SUSPENDED
        self._store.update(org)
        return org

    def revoke(self, name: str, reason: str) -> OrgRegistration:
        """Permanently revoke an organization registration."""
        org = self.get(name)
        if org is None:
            raise KeyError(f"Organization not found: {name}")
        if org.status == OrgStatus.REVOKED:
            raise ValueError("Organization is already revoked")
        org.status = OrgStatus.REVOKED
        self._store.update(org)
        return org
