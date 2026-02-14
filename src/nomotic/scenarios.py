"""Predefined simulation scenarios for ``nomotic simulate``.

Each scenario generates a sequence of actions that exercise different
aspects of the governance framework:

- ``normal``: Typical operations matching an agent's configured scope.
- ``drift``: Normal operations followed by a behavioural shift.
- ``violations``: Actions that trigger scope, boundary, and ethical vetoes.
- ``mixed``: A realistic mixture of allowed, denied, and edge-case actions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from nomotic.types import Action

__all__ = [
    "Scenario",
    "ScenarioAction",
    "BUILTIN_SCENARIOS",
    "generate_actions",
]


@dataclass
class ScenarioAction:
    """A single action specification within a scenario."""

    action_type: str
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A named simulation scenario."""

    name: str
    description: str
    phases: list[ScenarioPhase] = field(default_factory=list)


@dataclass
class ScenarioPhase:
    """One phase of a scenario — a distribution of actions to generate."""

    description: str
    count: int
    actions: list[ScenarioAction] = field(default_factory=list)
    weights: list[float] | None = None


def generate_actions(
    agent_id: str,
    scenario: Scenario,
    total_count: int | None = None,
) -> list[tuple[Action, str]]:
    """Generate a list of ``(Action, phase_description)`` pairs.

    If *total_count* is given and the scenario has exactly one phase,
    that phase's count is overridden. Otherwise the phase counts are
    used as-is.
    """
    result: list[tuple[Action, str]] = []

    for phase in scenario.phases:
        n = phase.count
        if total_count is not None and len(scenario.phases) == 1:
            n = total_count
        weights = phase.weights or [1.0] * len(phase.actions)
        for _ in range(n):
            spec = random.choices(phase.actions, weights=weights, k=1)[0]
            action = Action(
                agent_id=agent_id,
                action_type=spec.action_type,
                target=spec.target,
                parameters=dict(spec.parameters),
            )
            result.append((action, phase.description))

    return result


# ── Built-in scenarios ──────────────────────────────────────────────


def _normal_scenario() -> Scenario:
    return Scenario(
        name="normal",
        description="Normal operations — reads, queries, and occasional writes within scope",
        phases=[
            ScenarioPhase(
                description="normal operations",
                count=100,
                actions=[
                    ScenarioAction("read", "claims_db"),
                    ScenarioAction("read", "customer_records"),
                    ScenarioAction("query", "claims_db"),
                    ScenarioAction("query", "policy_db"),
                    ScenarioAction("write", "claims_db", {"claim_id": "CLM-1234", "amount": 5000}),
                    ScenarioAction("read", "policy_db"),
                    ScenarioAction("assess", "claims_db", {"claim_id": "CLM-5678", "amount": 2000}),
                ],
                weights=[25, 15, 15, 10, 10, 15, 10],
            ),
        ],
    )


def _drift_scenario() -> Scenario:
    return Scenario(
        name="drift",
        description="Normal operations followed by a sudden behavioural shift to mass deletes",
        phases=[
            ScenarioPhase(
                description="baseline (normal operations)",
                count=60,
                actions=[
                    ScenarioAction("read", "claims_db"),
                    ScenarioAction("read", "customer_records"),
                    ScenarioAction("query", "claims_db"),
                    ScenarioAction("write", "claims_db", {"claim_id": "CLM-100", "amount": 3000}),
                ],
                weights=[40, 20, 25, 15],
            ),
            ScenarioPhase(
                description="behavioural drift (mass deletes)",
                count=40,
                actions=[
                    ScenarioAction("delete", "claims_db"),
                    ScenarioAction("delete", "customer_records"),
                    ScenarioAction("delete", "policy_db"),
                    ScenarioAction("read", "claims_db"),
                ],
                weights=[40, 30, 20, 10],
            ),
        ],
    )


def _violations_scenario() -> Scenario:
    return Scenario(
        name="violations",
        description="Actions that trigger vetoes — out of scope, boundary breaches, ethical violations",
        phases=[
            ScenarioPhase(
                description="violation-heavy operations",
                count=50,
                actions=[
                    ScenarioAction("delete", "claims_db"),           # out of scope
                    ScenarioAction("read", "payroll_db"),            # boundary breach
                    ScenarioAction("read", "hr_records"),            # boundary breach
                    ScenarioAction("write", "claims_db", {"claim_id": "CLM-BIG", "amount": 25000}),  # ethical
                    ScenarioAction("read", "claims_db"),             # allowed
                    ScenarioAction("admin", "system_config"),        # out of scope + boundary
                ],
                weights=[20, 20, 15, 15, 20, 10],
            ),
        ],
    )


def _mixed_scenario() -> Scenario:
    return Scenario(
        name="mixed",
        description="A realistic mix of allowed, denied, and edge-case operations",
        phases=[
            ScenarioPhase(
                description="mixed operations",
                count=100,
                actions=[
                    ScenarioAction("read", "claims_db"),
                    ScenarioAction("query", "claims_db"),
                    ScenarioAction("write", "claims_db", {"claim_id": "CLM-200", "amount": 4000}),
                    ScenarioAction("read", "customer_records"),
                    ScenarioAction("read", "policy_db"),
                    ScenarioAction("delete", "claims_db"),           # out of scope
                    ScenarioAction("read", "payroll_db"),            # boundary breach
                    ScenarioAction("write", "claims_db", {"claim_id": "CLM-BIG", "amount": 50000}),  # ethical
                    ScenarioAction("assess", "claims_db", {"claim_id": "CLM-300", "amount": 8000}),
                ],
                weights=[20, 15, 10, 10, 10, 5, 5, 5, 20],
            ),
        ],
    )


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    "normal": _normal_scenario(),
    "drift": _drift_scenario(),
    "violations": _violations_scenario(),
    "mixed": _mixed_scenario(),
}
