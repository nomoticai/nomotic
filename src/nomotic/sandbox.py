"""Sandbox runtime — in-memory governance for CLI exploration.

This module provides the infrastructure for the developer journey CLI
commands: ``nomotic eval``, ``nomotic simulate``, ``nomotic scope set``,
``nomotic rule add``, ``nomotic config show``, and ``nomotic playground``.

Agent configurations (scope, boundaries, rules) are persisted to
``~/.nomotic/agents/<agent-id>.json`` so that ``nomotic eval`` can
load them without requiring a running server.

All governance evaluation happens in-memory via GovernanceRuntime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nomotic.authority import CertificateAuthority
from nomotic.keys import SigningKey
from nomotic.monitor import DriftConfig
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.store import FileCertificateStore, MemoryCertificateStore
from nomotic.types import Action, AgentContext, TrustProfile

__all__ = [
    "AgentConfig",
    "EthicalRuleSpec",
    "HumanOverrideSpec",
    "load_agent_config",
    "save_agent_config",
    "build_sandbox_runtime",
]


# ── Agent configuration persistence ────────────────────────────────────


@dataclass
class EthicalRuleSpec:
    """A serialisable ethical rule specification."""

    condition: str
    message: str
    name: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"type": "ethical", "condition": self.condition, "message": self.message, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EthicalRuleSpec:
        return cls(condition=d["condition"], message=d["message"], name=d.get("name", ""))

    def to_callable(self) -> Callable[[Action, AgentContext], tuple[bool, str]]:
        """Convert the condition string to a callable rule.

        Supported forms:
        - ``<param> <= <value>``  (e.g. ``amount <= 10000``)
        - ``<param> < <value>``
        - ``<param> >= <value>``
        - ``<param> > <value>``
        - ``<param> == <value>``
        - ``<param> != <value>``
        """
        condition = self.condition.strip()
        message = self.message
        ops = {"<=": lambda a, b: a <= b,
               ">=": lambda a, b: a >= b,
               "!=": lambda a, b: a != b,
               "==": lambda a, b: a == b,
               "<": lambda a, b: a < b,
               ">": lambda a, b: a > b}

        for op_str, op_fn in ops.items():
            if op_str in condition:
                parts = condition.split(op_str, 1)
                param_name = parts[0].strip()
                try:
                    threshold = float(parts[1].strip())
                except ValueError:
                    threshold = parts[1].strip()

                def _rule(
                    action: Action,
                    context: AgentContext,
                    _param: str = param_name,
                    _op: Any = op_fn,
                    _thresh: Any = threshold,
                    _msg: str = message,
                ) -> tuple[bool, str]:
                    val = action.parameters.get(_param)
                    if val is None:
                        return True, ""
                    try:
                        result = _op(float(val), float(_thresh))
                    except (TypeError, ValueError):
                        result = _op(val, _thresh)
                    if result:
                        return True, ""
                    return False, _msg

                return _rule
                break

        # Fallback: always pass
        def _always_pass(action: Action, context: AgentContext) -> tuple[bool, str]:
            return True, ""

        return _always_pass


@dataclass
class HumanOverrideSpec:
    """A serialisable human-override rule specification."""

    action: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"type": "human-override", "action": self.action, "message": self.message}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HumanOverrideSpec:
        return cls(action=d["action"], message=d["message"])


@dataclass
class AgentConfig:
    """Persistent configuration for an agent's governance envelope."""

    agent_id: str
    actions: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    ethical_rules: list[EthicalRuleSpec] = field(default_factory=list)
    human_overrides: list[HumanOverrideSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "scope": {
                "actions": self.actions,
                "boundaries": self.boundaries,
            },
            "rules": (
                [r.to_dict() for r in self.ethical_rules]
                + [r.to_dict() for r in self.human_overrides]
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentConfig:
        scope = d.get("scope", {})
        rules = d.get("rules", [])
        ethical = [EthicalRuleSpec.from_dict(r) for r in rules if r.get("type") == "ethical"]
        overrides = [HumanOverrideSpec.from_dict(r) for r in rules if r.get("type") == "human-override"]
        return cls(
            agent_id=d["agent_id"],
            actions=scope.get("actions", []),
            boundaries=scope.get("boundaries", []),
            ethical_rules=ethical,
            human_overrides=overrides,
        )


def _agents_dir(base: Path) -> Path:
    d = base / "agents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_agent_config(base: Path, agent_id: str) -> AgentConfig | None:
    """Load an agent's governance config from disk, or None if not found."""
    path = _agents_dir(base) / f"{agent_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return AgentConfig.from_dict(data)


def save_agent_config(base: Path, config: AgentConfig) -> Path:
    """Save an agent's governance config to disk."""
    path = _agents_dir(base) / f"{config.agent_id}.json"
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return path


def find_agent_cert_id(base: Path, agent_id: str) -> str | None:
    """Find the certificate ID for an agent by scanning cert files."""
    certs_dir = base / "certs"
    if not certs_dir.exists():
        return None
    for cert_file in certs_dir.glob("nmc-*.json"):
        try:
            data = json.loads(cert_file.read_text(encoding="utf-8"))
            if data.get("agent_id") == agent_id:
                return data.get("certificate_id")
        except (json.JSONDecodeError, KeyError):
            continue
    return None


# ── Sandbox runtime construction ────────────────────────────────────


def build_sandbox_runtime(
    agent_config: AgentConfig | None = None,
    agent_id: str = "",
    base_dir: Path | None = None,
    drift_config: DriftConfig | None = None,
) -> GovernanceRuntime:
    """Build an in-memory GovernanceRuntime with agent config applied.

    If *base_dir* is provided and contains certificates for the agent,
    they are loaded into the runtime's certificate authority.
    """
    if drift_config is None:
        drift_config = DriftConfig(window_size=50, check_interval=10, min_observations=10)

    runtime = GovernanceRuntime(
        config=RuntimeConfig(
            enable_fingerprints=True,
            drift_config=drift_config,
        )
    )

    if agent_config is not None:
        agent_id = agent_config.agent_id

        # Configure scope
        if agent_config.actions:
            runtime.configure_scope(
                agent_id,
                set(agent_config.actions),
                actor="cli",
                reason="Loaded from agent config",
            )

        # Configure boundaries
        if agent_config.boundaries:
            runtime.configure_boundaries(
                agent_id,
                set(agent_config.boundaries),
                actor="cli",
                reason="Loaded from agent config",
            )

        # Configure ethical rules
        for rule_spec in agent_config.ethical_rules:
            runtime.add_ethical_rule(
                rule_spec.to_callable(),
                actor="cli",
                reason=f"Ethical rule: {rule_spec.name or rule_spec.condition}",
                rule_name=rule_spec.name or rule_spec.condition,
            )

        # Configure human overrides
        if agent_config.human_overrides:
            actions = [ho.action for ho in agent_config.human_overrides]
            runtime.configure_human_override(
                *actions,
                actor="cli",
                reason="Human override rules from agent config",
            )

    return runtime


def apply_config_to_runtime(
    runtime: GovernanceRuntime,
    agent_config: AgentConfig,
) -> None:
    """Apply an AgentConfig to an already-created runtime."""
    agent_id = agent_config.agent_id

    if agent_config.actions:
        runtime.configure_scope(
            agent_id,
            set(agent_config.actions),
            actor="cli",
            reason="Configured via CLI",
        )

    if agent_config.boundaries:
        runtime.configure_boundaries(
            agent_id,
            set(agent_config.boundaries),
            actor="cli",
            reason="Configured via CLI",
        )

    for rule_spec in agent_config.ethical_rules:
        runtime.add_ethical_rule(
            rule_spec.to_callable(),
            actor="cli",
            reason=f"Ethical rule: {rule_spec.name or rule_spec.condition}",
            rule_name=rule_spec.name or rule_spec.condition,
        )

    if agent_config.human_overrides:
        actions = [ho.action for ho in agent_config.human_overrides]
        runtime.configure_human_override(
            *actions,
            actor="cli",
            reason="Human override rules from agent config",
        )


# ── Display helpers ─────────────────────────────────────────────────


def format_bar(value: float, width: int = 20) -> str:
    """Render a proportional bar chart segment."""
    filled = int(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def format_pct_bar(value: float, width: int = 40) -> str:
    """Render a percentage bar for distributions."""
    filled = int(value * width)
    return "\u2588" * filled
