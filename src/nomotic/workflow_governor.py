"""Workflow Governor — governs workflows as wholes, not just individual steps.

Individual action evaluation is necessary but insufficient for workflow governance.
A workflow where every step individually passes governance can still produce an
ungovernable outcome. The Workflow Governor catches what per-step evaluation misses:

1. **Cumulative risk** — ten medium-risk steps may constitute a high-risk workflow
2. **Ordering risks** — committing resources before resolving dependencies creates
   constraints
3. **Compound authority** — a sequence of authorized steps may achieve an
   unauthorized outcome
4. **Drift across steps** — an agent that gradually shifts behavior across a long
   workflow, where each step looks fine compared to the previous one but step 20
   looks nothing like step 1

The Workflow Governor does NOT replace per-action evaluation. It adds a
workflow-level governance layer that operates alongside it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nomotic.context_profile import (
    CompletedStep,
    ContextProfile,
    Dependency,
    PlannedStep,
    WorkflowContext,
)
from nomotic.types import Action, AgentContext

__all__ = [
    "CompoundAuthorityFlag",
    "ConsequenceProjector",
    "DependencyGraph",
    "DriftAcrossSteps",
    "OrderingConcern",
    "ProjectedRisk",
    "StepAssessment",
    "WorkflowGovernor",
    "WorkflowGovernorConfig",
    "WorkflowRiskAssessment",
    "WorkflowRiskFactor",
]


# ── Method category helpers ───────────────────────────────────────────────

# Transaction methods carry higher governance weight.
_TRANSACTION_METHODS = {
    "transfer", "refund", "charge", "reserve", "release",
    "reconcile", "purchase",
}

# Authority-expanding methods.
_AUTHORITY_METHODS = {
    "delegate", "escalate", "authorize", "elevate", "approve",
}

# Resource-consuming methods.
_RESOURCE_METHODS = {
    "reserve", "charge", "purchase", "transfer",
}

# Information-only methods (low downstream risk).
_INFO_METHODS = {
    "query", "read", "search", "find", "scan", "filter",
    "extract", "pull", "fetch",
}

# Security-sensitive methods.
_SECURITY_METHODS = {
    "authenticate", "authorize", "revoke", "elevate", "sign", "register",
}


def _method_base(method: str) -> str:
    """Extract the base method name (after the last dot)."""
    return method.rsplit(".", 1)[-1].lower() if method else ""


def _is_transaction(method: str) -> bool:
    base = _method_base(method)
    return base in _TRANSACTION_METHODS or "transaction" in method.lower()


def _is_authority_expanding(method: str) -> bool:
    base = _method_base(method)
    return base in _AUTHORITY_METHODS


def _is_resource_consuming(method: str) -> bool:
    base = _method_base(method)
    return base in _RESOURCE_METHODS


def _is_info_only(method: str) -> bool:
    base = _method_base(method)
    return base in _INFO_METHODS


def _is_security(method: str) -> bool:
    base = _method_base(method)
    return base in _SECURITY_METHODS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class WorkflowRiskFactor:
    """A specific risk factor contributing to workflow-level risk."""

    factor_type: str  # "cumulative_denials", "escalation_pattern", "trust_erosion",
    #   "method_concentration", "irreversibility_chain", "scope_boundary_pressure"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    contributing_steps: list[int]  # which step numbers contribute to this risk
    trend: str  # "new", "growing", "stable", "resolving"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_type": self.factor_type,
            "severity": self.severity,
            "description": self.description,
            "contributing_steps": self.contributing_steps,
            "trend": self.trend,
        }


@dataclass
class OrderingConcern:
    """A governance concern arising from step ordering within the workflow."""

    concern_type: str  # "commitment_before_dependency", "irreversible_before_verification",
    #   "authority_escalation_sequence", "resource_lock_chain"
    description: str
    step_causing: int  # the step that creates the concern
    steps_affected: list[int]  # downstream steps impacted
    severity: str  # "low", "medium", "high", "critical"
    mitigation: str  # what could address this concern

    def to_dict(self) -> dict[str, Any]:
        return {
            "concern_type": self.concern_type,
            "description": self.description,
            "step_causing": self.step_causing,
            "steps_affected": self.steps_affected,
            "severity": self.severity,
            "mitigation": self.mitigation,
        }


@dataclass
class CompoundAuthorityFlag:
    """Detected when a sequence of individually-authorized steps achieves an
    outcome that no single step was authorized for."""

    description: str
    methods_chained: list[str]  # sequence of methods creating compound authority
    steps_involved: list[int]
    resulting_capability: str  # what the chain collectively achieves
    severity: str  # "medium", "high", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "methods_chained": self.methods_chained,
            "steps_involved": self.steps_involved,
            "resulting_capability": self.resulting_capability,
            "severity": self.severity,
        }


@dataclass
class ProjectedRisk:
    """Forward-looking risk assessment for remaining workflow steps."""

    step_number: int
    method: str
    projected_risk_level: str  # "low", "medium", "high", "critical", "unknown"
    risk_description: str
    depends_on_current: bool  # does this future risk depend on current step's outcome?
    constraint_from_current: str | None  # how current step constrains this future step

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_number": self.step_number,
            "method": self.method,
            "projected_risk_level": self.projected_risk_level,
            "risk_description": self.risk_description,
            "depends_on_current": self.depends_on_current,
        }
        if self.constraint_from_current is not None:
            d["constraint_from_current"] = self.constraint_from_current
        return d


@dataclass
class WorkflowRiskAssessment:
    """Assessment of governance risk at the workflow level."""

    workflow_id: str
    cumulative_risk_score: float  # 0.0-1.0, 0=no risk, 1=maximum risk
    risk_trajectory: str  # "stable", "increasing", "decreasing", "accelerating"
    risk_factors: list[WorkflowRiskFactor] = field(default_factory=list)
    ordering_concerns: list[OrderingConcern] = field(default_factory=list)
    compound_authority_flags: list[CompoundAuthorityFlag] = field(default_factory=list)
    projected_risks: list[ProjectedRisk] = field(default_factory=list)
    recommendation: str = "continue"  # "continue", "caution", "pause_for_review", "halt"
    recommendation_reasoning: str = ""
    assessed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "cumulative_risk_score": round(self.cumulative_risk_score, 4),
            "risk_trajectory": self.risk_trajectory,
            "risk_factors": [f.to_dict() for f in self.risk_factors],
            "ordering_concerns": [c.to_dict() for c in self.ordering_concerns],
            "compound_authority_flags": [f.to_dict() for f in self.compound_authority_flags],
            "projected_risks": [r.to_dict() for r in self.projected_risks],
            "recommendation": self.recommendation,
            "recommendation_reasoning": self.recommendation_reasoning,
            "assessed_at": self.assessed_at,
        }


@dataclass
class StepAssessment:
    """Assessment of a specific step in its workflow context."""

    step_number: int
    workflow_risk_at_step: float  # cumulative risk up to this point
    step_specific_risks: list[WorkflowRiskFactor] = field(default_factory=list)
    ordering_concerns: list[OrderingConcern] = field(default_factory=list)
    compound_flags: list[CompoundAuthorityFlag] = field(default_factory=list)
    projected_risks: list[ProjectedRisk] = field(default_factory=list)
    unresolved_dependencies: list[str] = field(default_factory=list)
    commitment_depth: int = 0  # how many future steps this constrains
    recommended_additional_scrutiny: float = 0.0  # 0.0-1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "workflow_risk_at_step": round(self.workflow_risk_at_step, 4),
            "step_specific_risks": [f.to_dict() for f in self.step_specific_risks],
            "ordering_concerns": [c.to_dict() for c in self.ordering_concerns],
            "compound_flags": [f.to_dict() for f in self.compound_flags],
            "projected_risks": [r.to_dict() for r in self.projected_risks],
            "unresolved_dependencies": self.unresolved_dependencies,
            "commitment_depth": self.commitment_depth,
            "recommended_additional_scrutiny": round(self.recommended_additional_scrutiny, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class DriftAcrossSteps:
    """Assessment of behavioral drift across a workflow."""

    drift_detected: bool
    early_pattern: dict[str, float]  # method distribution from early steps
    recent_pattern: dict[str, float]  # method distribution from recent steps
    divergence: float  # JSD between early and recent
    description: str
    severity: str  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "early_pattern": self.early_pattern,
            "recent_pattern": self.recent_pattern,
            "divergence": round(self.divergence, 4),
            "description": self.description,
            "severity": self.severity,
        }


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class WorkflowGovernorConfig:
    """Configuration for the Workflow Governor."""

    cumulative_risk_threshold: float = 0.7  # above this, recommend pause_for_review
    cumulative_risk_halt_threshold: float = 0.9  # above this, recommend halt
    max_projection_depth: int = 10  # how far forward to project
    commitment_depth_warning: int = 5  # flag steps constraining > this many future steps
    compound_authority_detection: bool = True
    ordering_analysis: bool = True
    consequence_projection: bool = True
    drift_across_steps_detection: bool = True


# ── DependencyGraph ──────────────────────────────────────────────────────


class DependencyGraph:
    """Represents and analyzes the dependency structure of a workflow.

    Built from WorkflowContext.dependencies and WorkflowContext.steps_remaining.
    Internal representation: adjacency lists for forward and backward edges,
    typed by dependency_type.
    """

    def __init__(self) -> None:
        # forward[step] = [(to_step, dependency_type, description), ...]
        self._forward: dict[int, list[tuple[int, str, str]]] = {}
        # backward[step] = [(from_step, dependency_type, description), ...]
        self._backward: dict[int, list[tuple[int, str, str]]] = {}
        self._all_steps: set[int] = set()
        self._rollback_points: set[str] = set()

    @classmethod
    def from_workflow_context(cls, workflow: WorkflowContext) -> DependencyGraph:
        """Construct a dependency graph from a workflow context."""
        graph = cls()
        graph._rollback_points = set(workflow.rollback_points)

        # Register all known steps
        for step in workflow.steps_completed:
            graph._all_steps.add(step.step_number)
        for step in workflow.steps_remaining:
            graph._all_steps.add(step.step_number)
        # Include current step
        graph._all_steps.add(workflow.current_step)

        # Build adjacency lists from dependencies
        for dep in workflow.dependencies:
            graph._all_steps.add(dep.from_step)
            graph._all_steps.add(dep.to_step)

            fwd = graph._forward.setdefault(dep.from_step, [])
            fwd.append((dep.to_step, dep.dependency_type, dep.description))

            bwd = graph._backward.setdefault(dep.to_step, [])
            bwd.append((dep.from_step, dep.dependency_type, dep.description))

        return graph

    def get_dependencies(self, step: int) -> list[Dependency]:
        """What does this step depend on? (backward edges)."""
        result = []
        for from_step, dep_type, desc in self._backward.get(step, []):
            result.append(Dependency(
                from_step=from_step,
                to_step=step,
                dependency_type=dep_type,
                description=desc,
            ))
        return result

    def get_dependents(self, step: int) -> list[Dependency]:
        """What depends on this step? (forward edges)."""
        result = []
        for to_step, dep_type, desc in self._forward.get(step, []):
            result.append(Dependency(
                from_step=step,
                to_step=to_step,
                dependency_type=dep_type,
                description=desc,
            ))
        return result

    def get_constraint_chain(self, step: int) -> list[int]:
        """Follow the chain of 'constrains' dependencies forward from this step.

        Returns all steps transitively constrained by this step's outcome.
        """
        visited: set[int] = set()
        result: list[int] = []
        queue = [step]
        while queue:
            current = queue.pop(0)
            for to_step, dep_type, _ in self._forward.get(current, []):
                if dep_type == "constrains" and to_step not in visited:
                    visited.add(to_step)
                    result.append(to_step)
                    queue.append(to_step)
        return result

    def get_required_chain(self, step: int) -> list[int]:
        """Follow 'requires' dependencies backward.

        What must complete before this step?
        """
        visited: set[int] = set()
        result: list[int] = []
        queue = [step]
        while queue:
            current = queue.pop(0)
            for from_step, dep_type, _ in self._backward.get(current, []):
                if dep_type == "requires" and from_step not in visited:
                    visited.add(from_step)
                    result.append(from_step)
                    queue.append(from_step)
        return result

    def unresolved_dependencies(
        self, step: int, completed_steps: list[CompletedStep],
    ) -> list[Dependency]:
        """Which dependencies of this step haven't been satisfied yet?"""
        completed_numbers = {s.step_number for s in completed_steps}
        result = []
        for dep in self.get_dependencies(step):
            if dep.dependency_type == "requires" and dep.from_step not in completed_numbers:
                result.append(dep)
        return result

    def commitment_depth(self, step: int) -> int:
        """How many future steps are constrained by this step?

        High commitment depth = this step has far-reaching consequences.
        """
        return len(self.get_constraint_chain(step))

    def critical_path(self) -> list[int]:
        """The longest chain of 'requires' dependencies.

        Steps on the critical path have the most governance significance
        because delays cascade.
        """
        # Find all steps with no incoming 'requires' edges (roots)
        has_requires_incoming: set[int] = set()
        for step, edges in self._backward.items():
            for _, dep_type, _ in edges:
                if dep_type == "requires":
                    has_requires_incoming.add(step)

        roots = self._all_steps - has_requires_incoming
        if not roots:
            # Might be no requires deps at all, or a cycle
            roots = self._all_steps.copy()

        # BFS/DFS to find longest path through requires edges
        longest: list[int] = []
        for root in roots:
            path = self._longest_requires_path(root)
            if len(path) > len(longest):
                longest = path
        return longest

    def _longest_requires_path(self, start: int) -> list[int]:
        """Find longest path through 'requires' forward edges from start."""
        best: list[int] = [start]
        for to_step, dep_type, _ in self._forward.get(start, []):
            if dep_type == "requires":
                sub = self._longest_requires_path(to_step)
                candidate = [start] + sub
                if len(candidate) > len(best):
                    best = candidate
        return best

    def parallel_steps(self, step: int) -> list[int]:
        """Steps that can execute in parallel with this step.

        No dependency relationship in either direction.
        """
        # Find all steps reachable from or reaching this step
        connected: set[int] = {step}
        # Forward reachability
        queue = [step]
        while queue:
            current = queue.pop(0)
            for to_step, _, _ in self._forward.get(current, []):
                if to_step not in connected:
                    connected.add(to_step)
                    queue.append(to_step)
        # Backward reachability
        queue = [step]
        while queue:
            current = queue.pop(0)
            for from_step, _, _ in self._backward.get(current, []):
                if from_step not in connected:
                    connected.add(from_step)
                    queue.append(from_step)

        return [s for s in sorted(self._all_steps) if s not in connected]

    def is_reversible_to(
        self, step: int, completed_steps: list[CompletedStep],
    ) -> bool:
        """Can the workflow be reversed back to before this step?

        Checks rollback_points: all steps from `step` onward (that have been
        completed) must have rollback points.
        """
        completed_at_or_after = [
            s for s in completed_steps if s.step_number >= step
        ]
        if not completed_at_or_after:
            return True  # nothing to reverse
        for s in completed_at_or_after:
            if s.step_id not in self._rollback_points:
                return False
        return True


# ── ConsequenceProjector ─────────────────────────────────────────────────


class ConsequenceProjector:
    """Projects governance consequences of the current step onto future steps.

    This is NOT exhaustive enumeration of all orderings. It is targeted
    projection along dependency chains from the current step.
    """

    def __init__(self, max_depth: int = 10) -> None:
        self._max_depth = max_depth

    def project(
        self,
        current_step: int,
        action: Action,
        graph: DependencyGraph,
        workflow: WorkflowContext,
    ) -> list[ProjectedRisk]:
        """Project risks from the current step onto future constrained steps."""
        constraint_chain = graph.get_constraint_chain(current_step)
        if not constraint_chain:
            return []

        # Limit depth
        chain = constraint_chain[: self._max_depth]

        # Build a map of remaining steps for method/target lookup
        remaining_map: dict[int, PlannedStep] = {
            s.step_number: s for s in workflow.steps_remaining
        }

        method = action.action_type
        projections: list[ProjectedRisk] = []
        hop = 0

        for constrained_step_num in chain:
            hop += 1
            planned = remaining_map.get(constrained_step_num)
            step_method = planned.method if planned else "unknown"

            risk_level, risk_desc, constraint_desc = self._assess_downstream_risk(
                method, step_method, planned, hop,
            )

            projections.append(ProjectedRisk(
                step_number=constrained_step_num,
                method=step_method,
                projected_risk_level=risk_level,
                risk_description=risk_desc,
                depends_on_current=True,
                constraint_from_current=constraint_desc,
            ))

        return projections

    def _assess_downstream_risk(
        self,
        current_method: str,
        downstream_method: str,
        planned: PlannedStep | None,
        hop: int,
    ) -> tuple[str, str, str]:
        """Assess how a current step affects a downstream constrained step.

        Returns (risk_level, risk_description, constraint_description).
        """
        # Base risk from current step character
        is_irreversible = _is_transaction(current_method)
        is_resource = _is_resource_consuming(current_method)
        is_auth_expand = _is_authority_expanding(current_method)
        is_info = _is_info_only(current_method)

        if is_info:
            return (
                "low",
                f"Information step at hop {hop}; low downstream impact",
                "Information does not constrain future options",
            )

        # Risk amplification with hops
        hop_factor = min(1.0 + (hop - 1) * 0.1, 2.0)

        if is_irreversible:
            if hop <= 2:
                level = "high"
            elif hop <= 5:
                level = "medium"
            else:
                level = "low"
            return (
                level,
                f"Irreversible commitment constrains step {planned.step_number if planned else '?'} "
                f"(hop {hop}); options narrow with each constrained step",
                f"Irreversible action at current step locks in constraints "
                f"that propagate to this step (amplification factor: {hop_factor:.1f}x)",
            )

        if is_resource:
            level = "high" if hop <= 3 else "medium"
            return (
                level,
                f"Resource consumption constrains available resources at step "
                f"{planned.step_number if planned else '?'} (hop {hop})",
                f"Resource consumed at current step reduces availability for this step",
            )

        if is_auth_expand:
            level = "high" if hop <= 2 else "medium"
            return (
                level,
                f"Authority expansion at current step may grant elevated authority "
                f"at step {planned.step_number if planned else '?'} (hop {hop}); "
                f"check for compound authority",
                f"Authority expanded at current step propagates to this step",
            )

        # Default: moderate risk for constraining steps
        return (
            "medium" if hop <= 3 else "low",
            f"Current step constrains step {planned.step_number if planned else '?'} "
            f"(hop {hop})",
            f"General constraint propagation from current step",
        )


# ── Compound authority pattern templates ─────────────────────────────────

_COMPOUND_PATTERNS = [
    {
        "name": "scope_assembly",
        "sequence_matchers": [_is_info_only, _is_info_only, lambda m: not _is_info_only(m)],
        "condition": "targets_differ_on_reads",
        "severity": "high",
        "description": "Scope assembly: reading from multiple sources then acting on combined data",
        "capability": "Combined information access across separately-authorized sources",
    },
    {
        "name": "authority_ladder",
        "sequence_matchers": [_is_authority_expanding, _is_authority_expanding, lambda m: _is_transaction(m) or _method_base(m) in {"delete", "approve"}],
        "condition": "no_human_review_between",
        "severity": "critical",
        "description": "Authority ladder: sequential authority escalation without verification",
        "capability": "Effective authority elevation beyond individual step authorization",
    },
    {
        "name": "resource_aggregation",
        "sequence_matchers": [_is_resource_consuming, _is_resource_consuming, lambda m: _is_transaction(m) or _is_resource_consuming(m)],
        "condition": "cumulative_exceeds_per_action_limit",
        "severity": "high",
        "description": "Resource aggregation: cumulative resource consumption through multiple steps",
        "capability": "Aggregate resource consumption exceeding per-action limits",
    },
]


# ── WorkflowGovernor ─────────────────────────────────────────────────────


class WorkflowGovernor:
    """Governs workflows as wholes, not just individual steps.

    Thread-safe: all mutable state is accessed under a lock, and analysis
    methods are pure functions over the input data.
    """

    def __init__(self, config: WorkflowGovernorConfig | None = None) -> None:
        self.config = config or WorkflowGovernorConfig()
        self._projector = ConsequenceProjector(
            max_depth=self.config.max_projection_depth,
        )
        self._lock = threading.Lock()
        # Tracks cumulative risk history per workflow for trajectory calculation
        self._risk_history: dict[str, list[float]] = {}

    def assess_workflow(
        self, workflow_id: str, profile: ContextProfile,
    ) -> WorkflowRiskAssessment:
        """Full workflow risk assessment.

        Called periodically or on demand to get the overall governance picture.
        """
        workflow = profile.workflow
        if workflow is None:
            return WorkflowRiskAssessment(
                workflow_id=workflow_id,
                cumulative_risk_score=0.0,
                risk_trajectory="stable",
                recommendation="continue",
                recommendation_reasoning="No workflow context available",
            )

        # Build dependency graph
        graph = DependencyGraph.from_workflow_context(workflow)

        # Compute cumulative risk
        cumulative_risk = self._compute_cumulative_risk(workflow, graph)

        # Compute risk trajectory
        trajectory = self._compute_risk_trajectory(workflow_id, cumulative_risk)

        # Analyze ordering
        ordering_concerns: list[OrderingConcern] = []
        if self.config.ordering_analysis:
            ordering_concerns = self._analyze_ordering(graph, workflow)

        # Detect compound authority
        compound_flags: list[CompoundAuthorityFlag] = []
        if self.config.compound_authority_detection:
            compound_flags = self._detect_compound_authority(
                workflow.steps_completed, graph, workflow,
            )

        # Project risks for remaining steps
        projected_risks: list[ProjectedRisk] = []
        if self.config.consequence_projection and workflow.steps_remaining:
            # Create a dummy action for projection from current step
            current_method = ""
            for step in workflow.steps_remaining:
                if step.step_number == workflow.current_step:
                    current_method = step.method
                    break
            if current_method:
                dummy_action = Action(
                    action_type=current_method,
                    target="workflow_assessment",
                )
                projected_risks = self._projector.project(
                    workflow.current_step, dummy_action, graph, workflow,
                )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(workflow, graph)

        # Drift detection
        if self.config.drift_across_steps_detection:
            drift = self.detect_cross_step_drift(profile)
            if drift is not None and drift.drift_detected:
                risk_factors.append(WorkflowRiskFactor(
                    factor_type="cross_step_drift",
                    severity=drift.severity,
                    description=drift.description,
                    contributing_steps=[],
                    trend="growing",
                ))

        # Determine recommendation
        recommendation, reasoning = self._determine_recommendation(
            cumulative_risk, risk_factors, ordering_concerns,
            compound_flags, projected_risks,
        )

        return WorkflowRiskAssessment(
            workflow_id=workflow_id,
            cumulative_risk_score=cumulative_risk,
            risk_trajectory=trajectory,
            risk_factors=risk_factors,
            ordering_concerns=ordering_concerns,
            compound_authority_flags=compound_flags,
            projected_risks=projected_risks,
            recommendation=recommendation,
            recommendation_reasoning=reasoning,
        )

    def assess_step(
        self,
        step_number: int,
        action: Action,
        context: AgentContext,
        profile: ContextProfile,
    ) -> StepAssessment:
        """Assess a specific step in its workflow context.

        Called before per-action governance evaluation to provide
        workflow-level signals.
        """
        workflow = profile.workflow
        if workflow is None:
            return StepAssessment(
                step_number=step_number,
                workflow_risk_at_step=0.0,
                reasoning="No workflow context available",
            )

        graph = DependencyGraph.from_workflow_context(workflow)

        # Cumulative risk up to this point
        cumulative_risk = self._compute_cumulative_risk(workflow, graph)

        # Step-specific risk factors
        step_risks = self._identify_step_risks(step_number, action, workflow, graph)

        # Ordering concerns for this step
        ordering_concerns: list[OrderingConcern] = []
        if self.config.ordering_analysis:
            all_concerns = self._analyze_ordering(graph, workflow)
            ordering_concerns = [
                c for c in all_concerns
                if c.step_causing == step_number or step_number in c.steps_affected
            ]

        # Compound authority flags
        compound_flags: list[CompoundAuthorityFlag] = []
        if self.config.compound_authority_detection:
            compound_flags = self._detect_compound_authority(
                workflow.steps_completed, graph, workflow,
            )

        # Project risks from this step
        projected_risks: list[ProjectedRisk] = []
        if self.config.consequence_projection:
            projected_risks = self._projector.project(
                step_number, action, graph, workflow,
            )

        # Unresolved dependencies
        unresolved = graph.unresolved_dependencies(step_number, workflow.steps_completed)
        unresolved_descs = [
            f"Step {d.from_step} ({d.dependency_type}): {d.description}"
            for d in unresolved
        ]

        # Commitment depth
        depth = graph.commitment_depth(step_number)

        # Recommended additional scrutiny
        scrutiny = self._compute_scrutiny(
            cumulative_risk, step_risks, ordering_concerns,
            compound_flags, projected_risks, depth, unresolved,
        )

        # Build reasoning
        reasoning_parts: list[str] = []
        if cumulative_risk > 0.3:
            reasoning_parts.append(
                f"Cumulative workflow risk is {cumulative_risk:.2f}"
            )
        if step_risks:
            reasoning_parts.append(
                f"{len(step_risks)} step-specific risk factors identified"
            )
        if ordering_concerns:
            reasoning_parts.append(
                f"{len(ordering_concerns)} ordering concerns at this step"
            )
        if compound_flags:
            reasoning_parts.append(
                f"{len(compound_flags)} compound authority patterns detected"
            )
        if unresolved_descs:
            reasoning_parts.append(
                f"{len(unresolved_descs)} unresolved dependencies"
            )
        if depth > self.config.commitment_depth_warning:
            reasoning_parts.append(
                f"High commitment depth ({depth}): this step constrains "
                f"many future steps"
            )
        if not reasoning_parts:
            reasoning = "No significant workflow-level concerns at this step"
        else:
            reasoning = "; ".join(reasoning_parts)

        return StepAssessment(
            step_number=step_number,
            workflow_risk_at_step=cumulative_risk,
            step_specific_risks=step_risks,
            ordering_concerns=ordering_concerns,
            compound_flags=compound_flags,
            projected_risks=projected_risks,
            unresolved_dependencies=unresolved_descs,
            commitment_depth=depth,
            recommended_additional_scrutiny=scrutiny,
            reasoning=reasoning,
        )

    def record_step_outcome(
        self,
        workflow_id: str,
        step: CompletedStep,
        profile: ContextProfile,
    ) -> None:
        """Record the outcome of a step after governance evaluation and execution.

        Updates the workflow risk trajectory.
        """
        workflow = profile.workflow
        if workflow is None:
            return

        graph = DependencyGraph.from_workflow_context(workflow)
        cumulative_risk = self._compute_cumulative_risk(workflow, graph)

        with self._lock:
            history = self._risk_history.setdefault(workflow_id, [])
            history.append(cumulative_risk)
            # Keep bounded
            if len(history) > 100:
                self._risk_history[workflow_id] = history[-100:]

    def detect_cross_step_drift(
        self, profile: ContextProfile,
    ) -> DriftAcrossSteps | None:
        """Analyze whether the agent's behavior is gradually shifting.

        Compares method distribution from the first third of completed steps
        against the last third. If they differ significantly, returns a drift
        assessment.
        """
        workflow = profile.workflow
        if workflow is None:
            return None

        steps = workflow.steps_completed
        if len(steps) < 6:
            return None  # insufficient data

        # Split into thirds
        third = len(steps) // 3
        early = steps[:third]
        recent = steps[-third:]

        # Build method distributions
        early_dist = self._method_distribution(early)
        recent_dist = self._method_distribution(recent)

        # Jensen-Shannon divergence
        divergence = self._jsd(early_dist, recent_dist)

        # Thresholds
        if divergence < 0.1:
            return DriftAcrossSteps(
                drift_detected=False,
                early_pattern=early_dist,
                recent_pattern=recent_dist,
                divergence=divergence,
                description="Method distribution is consistent across workflow steps",
                severity="low",
            )

        if divergence < 0.3:
            severity = "medium"
            desc = (
                f"Moderate behavioral shift detected across workflow steps "
                f"(JSD={divergence:.3f}). Early steps focused on "
                f"{self._dominant_method(early_dist)}, recent steps shifted "
                f"toward {self._dominant_method(recent_dist)}"
            )
        elif divergence < 0.6:
            severity = "high"
            desc = (
                f"Significant behavioral shift detected (JSD={divergence:.3f}). "
                f"The agent's method usage has changed substantially between "
                f"early and recent workflow steps"
            )
        else:
            severity = "critical"
            desc = (
                f"Critical behavioral drift (JSD={divergence:.3f}). "
                f"Recent workflow steps look very different from early steps"
            )

        return DriftAcrossSteps(
            drift_detected=True,
            early_pattern=early_dist,
            recent_pattern=recent_dist,
            divergence=divergence,
            description=desc,
            severity=severity,
        )

    # ── Internal: Cumulative Risk ────────────────────────────────────────

    def _compute_cumulative_risk(
        self, workflow: WorkflowContext, graph: DependencyGraph,
    ) -> float:
        """Compute cumulative risk from completed steps."""
        if not workflow.steps_completed:
            return 0.0

        cumulative_risk = 0.0
        for step in workflow.steps_completed:
            step_risk = 1.0 - step.ucs  # invert: high UCS = low risk

            # Weight by step characteristics
            if _is_transaction(step.method):
                step_risk *= 1.5

            if step.verdict.upper() in ("DENY", "ESCALATE"):
                step_risk *= 2.0

            # Check reversibility
            if step.step_id not in workflow.rollback_points:
                step_risk *= 1.3

            cumulative_risk += step_risk

        # Normalize to 0.0-1.0
        normalizer = len(workflow.steps_completed) * 2.0
        cumulative_risk = min(1.0, cumulative_risk / normalizer)
        return cumulative_risk

    def _compute_risk_trajectory(
        self, workflow_id: str, current_risk: float,
    ) -> str:
        """Compute risk trajectory by comparing current vs recent risk."""
        with self._lock:
            history = self._risk_history.get(workflow_id, [])

        if len(history) < 3:
            return "stable"

        # Compare current risk to risk 3 steps ago
        past_risk = history[-3]
        diff = current_risk - past_risk

        if abs(diff) < 0.05:
            return "stable"
        elif 0.05 <= diff < 0.15:
            return "increasing"
        elif diff >= 0.15:
            return "accelerating"
        elif -0.15 < diff <= -0.05:
            return "decreasing"
        else:
            return "decreasing"

    # ── Internal: Risk Factor Identification ─────────────────────────────

    def _identify_risk_factors(
        self, workflow: WorkflowContext, graph: DependencyGraph,
    ) -> list[WorkflowRiskFactor]:
        """Identify workflow-level risk factors."""
        factors: list[WorkflowRiskFactor] = []

        if not workflow.steps_completed:
            return factors

        # Cumulative denials
        denied_steps = [
            s for s in workflow.steps_completed
            if s.verdict.upper() == "DENY"
        ]
        if len(denied_steps) >= 2:
            factors.append(WorkflowRiskFactor(
                factor_type="cumulative_denials",
                severity="high" if len(denied_steps) >= 3 else "medium",
                description=(
                    f"{len(denied_steps)} steps denied in this workflow. "
                    f"Repeated denials suggest persistent boundary-testing"
                ),
                contributing_steps=[s.step_number for s in denied_steps],
                trend="growing" if len(denied_steps) >= 3 else "new",
            ))

        # Escalation pattern
        escalated_steps = [
            s for s in workflow.steps_completed
            if s.verdict.upper() == "ESCALATE"
        ]
        if len(escalated_steps) >= 2:
            factors.append(WorkflowRiskFactor(
                factor_type="escalation_pattern",
                severity="medium",
                description=(
                    f"{len(escalated_steps)} escalations in this workflow. "
                    f"Frequent escalations may indicate scope misalignment"
                ),
                contributing_steps=[s.step_number for s in escalated_steps],
                trend="growing" if len(escalated_steps) >= 3 else "stable",
            ))

        # Trust erosion: check if UCS scores are declining
        ucs_scores = [s.ucs for s in workflow.steps_completed]
        if len(ucs_scores) >= 3:
            early_avg = sum(ucs_scores[:len(ucs_scores) // 2]) / max(1, len(ucs_scores) // 2)
            late_avg = sum(ucs_scores[len(ucs_scores) // 2:]) / max(1, len(ucs_scores) - len(ucs_scores) // 2)
            if early_avg - late_avg > 0.15:
                factors.append(WorkflowRiskFactor(
                    factor_type="trust_erosion",
                    severity="high" if early_avg - late_avg > 0.3 else "medium",
                    description=(
                        f"UCS scores declining across workflow: "
                        f"early avg {early_avg:.2f} → late avg {late_avg:.2f}"
                    ),
                    contributing_steps=[s.step_number for s in workflow.steps_completed],
                    trend="growing",
                ))

        # Method concentration: is the agent using a very narrow set of methods?
        method_counts: dict[str, int] = {}
        for s in workflow.steps_completed:
            base = _method_base(s.method)
            method_counts[base] = method_counts.get(base, 0) + 1
        if method_counts and len(workflow.steps_completed) >= 4:
            max_count = max(method_counts.values())
            concentration = max_count / len(workflow.steps_completed)
            if concentration > 0.7:
                dominant = max(method_counts, key=method_counts.get)  # type: ignore[arg-type]
                factors.append(WorkflowRiskFactor(
                    factor_type="method_concentration",
                    severity="medium",
                    description=(
                        f"Method concentration: {concentration:.0%} of steps use "
                        f"'{dominant}'. Unusual homogeneity may indicate repetitive probing"
                    ),
                    contributing_steps=[
                        s.step_number for s in workflow.steps_completed
                        if _method_base(s.method) == dominant
                    ],
                    trend="stable",
                ))

        # Irreversibility chain: consecutive non-reversible steps
        non_reversible_runs: list[list[int]] = []
        current_run: list[int] = []
        for s in workflow.steps_completed:
            if s.step_id not in workflow.rollback_points:
                current_run.append(s.step_number)
            else:
                if len(current_run) >= 3:
                    non_reversible_runs.append(current_run)
                current_run = []
        if len(current_run) >= 3:
            non_reversible_runs.append(current_run)

        for run in non_reversible_runs:
            factors.append(WorkflowRiskFactor(
                factor_type="irreversibility_chain",
                severity="high",
                description=(
                    f"{len(run)} consecutive non-reversible steps "
                    f"(steps {run[0]}-{run[-1]}). Recovery becomes "
                    f"increasingly difficult"
                ),
                contributing_steps=run,
                trend="growing",
            ))

        return factors

    def _identify_step_risks(
        self,
        step_number: int,
        action: Action,
        workflow: WorkflowContext,
        graph: DependencyGraph,
    ) -> list[WorkflowRiskFactor]:
        """Identify risks specific to a single step's position."""
        risks: list[WorkflowRiskFactor] = []

        # Check if this step has high commitment depth
        depth = graph.commitment_depth(step_number)
        if depth > self.config.commitment_depth_warning:
            risks.append(WorkflowRiskFactor(
                factor_type="high_commitment_depth",
                severity="high",
                description=(
                    f"Step {step_number} constrains {depth} future steps. "
                    f"This decision has far-reaching consequences"
                ),
                contributing_steps=[step_number],
                trend="new",
            ))

        # Check if this is a transaction on the critical path
        cpath = graph.critical_path()
        if step_number in cpath and _is_transaction(action.action_type):
            risks.append(WorkflowRiskFactor(
                factor_type="critical_path_transaction",
                severity="high",
                description=(
                    f"Step {step_number} is a transaction on the critical path. "
                    f"Failure here delays the entire workflow"
                ),
                contributing_steps=[step_number],
                trend="new",
            ))

        return risks

    # ── Internal: Ordering Analysis ──────────────────────────────────────

    def _analyze_ordering(
        self, graph: DependencyGraph, workflow: WorkflowContext,
    ) -> list[OrderingConcern]:
        """Scan completed and planned steps for ordering patterns that create
        governance risk."""
        concerns: list[OrderingConcern] = []
        completed_numbers = {s.step_number for s in workflow.steps_completed}
        all_step_methods: dict[int, str] = {}
        for s in workflow.steps_completed:
            all_step_methods[s.step_number] = s.method
        for s in workflow.steps_remaining:
            all_step_methods[s.step_number] = s.method

        # 1. Commitment before dependency
        for step in workflow.steps_completed:
            if _is_transaction(step.method) or step.step_id not in workflow.rollback_points:
                # This is a commitment step — check if it has unresolved requires deps
                deps = graph.get_dependencies(step.step_number)
                for dep in deps:
                    if dep.dependency_type == "requires" and dep.from_step not in completed_numbers:
                        concerns.append(OrderingConcern(
                            concern_type="commitment_before_dependency",
                            description=(
                                f"Step {step.step_number} ({step.method}) is a commitment "
                                f"that executed before required step {dep.from_step} completed"
                            ),
                            step_causing=step.step_number,
                            steps_affected=[dep.from_step],
                            severity="high",
                            mitigation=(
                                f"Ensure step {dep.from_step} completes before "
                                f"step {step.step_number} commits resources"
                            ),
                        ))

        # 2. Irreversible before verification
        for i, step in enumerate(workflow.steps_completed):
            if step.step_id not in workflow.rollback_points:
                # Check if any later completed step is a verification
                for later in workflow.steps_completed[i + 1:]:
                    later_base = _method_base(later.method)
                    if later_base in {"validate", "check", "verify", "evaluate"}:
                        # Check if there's a dependency linking them
                        deps = graph.get_dependencies(step.step_number)
                        from_steps = {d.from_step for d in deps}
                        if later.step_number in from_steps or any(
                            d.from_step == later.step_number
                            for d in graph.get_dependencies(step.step_number)
                        ):
                            concerns.append(OrderingConcern(
                                concern_type="irreversible_before_verification",
                                description=(
                                    f"Irreversible step {step.step_number} ({step.method}) "
                                    f"executed before verification step {later.step_number} "
                                    f"({later.method})"
                                ),
                                step_causing=step.step_number,
                                steps_affected=[later.step_number],
                                severity="high",
                                mitigation=(
                                    f"Run verification step {later.step_number} before "
                                    f"irreversible step {step.step_number}"
                                ),
                            ))

        # 3. Authority escalation sequence
        authority_runs: list[list[CompletedStep]] = []
        current_auth_run: list[CompletedStep] = []
        for step in workflow.steps_completed:
            if _is_authority_expanding(step.method):
                current_auth_run.append(step)
            else:
                base = _method_base(step.method)
                if base in {"validate", "check", "verify", "review"}:
                    # Verification breaks the sequence
                    if len(current_auth_run) >= 2:
                        authority_runs.append(current_auth_run)
                    current_auth_run = []
                # Non-authority, non-verification: keep accumulating if run exists
                elif not current_auth_run:
                    pass  # no run yet
        if len(current_auth_run) >= 2:
            authority_runs.append(current_auth_run)

        for run in authority_runs:
            concerns.append(OrderingConcern(
                concern_type="authority_escalation_sequence",
                description=(
                    f"{len(run)} authority-expanding steps in sequence "
                    f"(steps {', '.join(str(s.step_number) for s in run)}) "
                    f"without intervening verification"
                ),
                step_causing=run[0].step_number,
                steps_affected=[s.step_number for s in run[1:]],
                severity="critical" if len(run) >= 3 else "high",
                mitigation=(
                    f"Insert a verification or review step between "
                    f"authority-expanding operations"
                ),
            ))

        # 4. Resource lock chain
        resource_steps: list[CompletedStep] = []
        for step in workflow.steps_completed:
            if _is_resource_consuming(step.method):
                resource_steps.append(step)

        if len(resource_steps) >= 3:
            # Check if any resources were released between consumption
            release_steps = {
                s.step_number for s in workflow.steps_completed
                if _method_base(s.method) in {"release", "refund"}
            }
            # If fewer releases than consumptions - 1, we have accumulation
            if len(release_steps) < len(resource_steps) - 1:
                concerns.append(OrderingConcern(
                    concern_type="resource_lock_chain",
                    description=(
                        f"{len(resource_steps)} resource-consuming steps with only "
                        f"{len(release_steps)} releases. Resources accumulating "
                        f"without being freed"
                    ),
                    step_causing=resource_steps[0].step_number,
                    steps_affected=[s.step_number for s in resource_steps[1:]],
                    severity="high" if len(resource_steps) >= 4 else "medium",
                    mitigation=(
                        "Release consumed resources before acquiring more, "
                        "or verify aggregate consumption against resource boundaries"
                    ),
                ))

        return concerns

    # ── Internal: Compound Authority Detection ───────────────────────────

    def _detect_compound_authority(
        self,
        completed_steps: list[CompletedStep],
        graph: DependencyGraph,
        workflow: WorkflowContext,
    ) -> list[CompoundAuthorityFlag]:
        """Look for chains of individually-authorized methods that together
        achieve something beyond any single authorization."""
        flags: list[CompoundAuthorityFlag] = []

        if len(completed_steps) < 2:
            return flags

        # Check each pattern
        for pattern in _COMPOUND_PATTERNS:
            matches = self._match_compound_pattern(
                completed_steps, pattern, workflow,
            )
            flags.extend(matches)

        return flags

    def _match_compound_pattern(
        self,
        steps: list[CompletedStep],
        pattern: dict[str, Any],
        workflow: WorkflowContext,
    ) -> list[CompoundAuthorityFlag]:
        """Match a compound authority pattern against the step sequence.

        Steps don't need to be consecutive, just in order.
        """
        matchers = pattern["sequence_matchers"]
        flags: list[CompoundAuthorityFlag] = []

        # Sliding scan: find ordered subsequences matching the pattern
        n = len(matchers)
        if len(steps) < n:
            return flags

        # Try all combinations of n steps in order
        for i in range(len(steps)):
            if not matchers[0](steps[i].method):
                continue
            matched = [steps[i]]
            matcher_idx = 1
            for j in range(i + 1, len(steps)):
                if matcher_idx >= n:
                    break
                if matchers[matcher_idx](steps[j].method):
                    matched.append(steps[j])
                    matcher_idx += 1
            if matcher_idx < n:
                continue

            # Check condition
            condition = pattern["condition"]
            if condition == "targets_differ_on_reads":
                # Read steps should target different sources
                read_targets = {s.target for s in matched[:-1]}
                if len(read_targets) < 2:
                    continue

            elif condition == "no_human_review_between":
                # Check for review steps between matched steps
                min_step = matched[0].step_number
                max_step = matched[-1].step_number
                has_review = False
                for step in steps:
                    if min_step < step.step_number < max_step:
                        base = _method_base(step.method)
                        if base in {"validate", "check", "verify", "review"}:
                            has_review = True
                            break
                if has_review:
                    continue

            # No additional checks for cumulative_exceeds_per_action_limit
            # (heuristic match is sufficient)

            flags.append(CompoundAuthorityFlag(
                description=pattern["description"],
                methods_chained=[s.method for s in matched],
                steps_involved=[s.step_number for s in matched],
                resulting_capability=pattern["capability"],
                severity=pattern["severity"],
            ))

        return flags

    # ── Internal: Recommendation ─────────────────────────────────────────

    def _determine_recommendation(
        self,
        cumulative_risk: float,
        risk_factors: list[WorkflowRiskFactor],
        ordering_concerns: list[OrderingConcern],
        compound_flags: list[CompoundAuthorityFlag],
        projected_risks: list[ProjectedRisk],
    ) -> tuple[str, str]:
        """Determine overall recommendation and reasoning."""
        reasoning_parts: list[str] = []

        # Check halt threshold
        if cumulative_risk >= self.config.cumulative_risk_halt_threshold:
            reasoning_parts.append(
                f"Cumulative risk ({cumulative_risk:.2f}) exceeds halt threshold "
                f"({self.config.cumulative_risk_halt_threshold})"
            )
            return "halt", "; ".join(reasoning_parts)

        # Critical compound authority → halt
        critical_compounds = [
            f for f in compound_flags if f.severity == "critical"
        ]
        if critical_compounds:
            reasoning_parts.append(
                f"Critical compound authority detected: "
                f"{critical_compounds[0].description}"
            )
            return "halt", "; ".join(reasoning_parts)

        # Check pause threshold
        if cumulative_risk >= self.config.cumulative_risk_threshold:
            reasoning_parts.append(
                f"Cumulative risk ({cumulative_risk:.2f}) exceeds review threshold "
                f"({self.config.cumulative_risk_threshold})"
            )
            return "pause_for_review", "; ".join(reasoning_parts)

        # Critical risk factors → pause
        critical_factors = [f for f in risk_factors if f.severity == "critical"]
        if critical_factors:
            reasoning_parts.append(
                f"Critical risk factors: "
                f"{', '.join(f.factor_type for f in critical_factors)}"
            )
            return "pause_for_review", "; ".join(reasoning_parts)

        # Critical ordering concerns → pause
        critical_ordering = [c for c in ordering_concerns if c.severity == "critical"]
        if critical_ordering:
            reasoning_parts.append(
                f"Critical ordering concerns: "
                f"{', '.join(c.concern_type for c in critical_ordering)}"
            )
            return "pause_for_review", "; ".join(reasoning_parts)

        # High-severity factors → caution
        high_factors = [f for f in risk_factors if f.severity == "high"]
        high_ordering = [c for c in ordering_concerns if c.severity == "high"]
        if high_factors or high_ordering or compound_flags:
            reasons = []
            if high_factors:
                reasons.append(
                    f"{len(high_factors)} high-severity risk factors"
                )
            if high_ordering:
                reasons.append(
                    f"{len(high_ordering)} high-severity ordering concerns"
                )
            if compound_flags:
                reasons.append(
                    f"{len(compound_flags)} compound authority patterns"
                )
            reasoning_parts.extend(reasons)
            return "caution", "; ".join(reasoning_parts)

        # Moderate risk → caution
        if cumulative_risk > 0.4:
            reasoning_parts.append(
                f"Moderate cumulative risk ({cumulative_risk:.2f})"
            )
            return "caution", "; ".join(reasoning_parts)

        return "continue", "Workflow risk within acceptable bounds"

    # ── Internal: Scrutiny Computation ───────────────────────────────────

    def _compute_scrutiny(
        self,
        cumulative_risk: float,
        step_risks: list[WorkflowRiskFactor],
        ordering_concerns: list[OrderingConcern],
        compound_flags: list[CompoundAuthorityFlag],
        projected_risks: list[ProjectedRisk],
        commitment_depth: int,
        unresolved: list[Dependency],
    ) -> float:
        """Compute recommended additional scrutiny for a step (0.0-1.0)."""
        scrutiny = 0.0

        # Base from cumulative risk
        scrutiny += cumulative_risk * 0.3

        # Step-specific risks
        for risk in step_risks:
            if risk.severity == "critical":
                scrutiny += 0.3
            elif risk.severity == "high":
                scrutiny += 0.2
            elif risk.severity == "medium":
                scrutiny += 0.1

        # Ordering concerns
        for concern in ordering_concerns:
            if concern.severity == "critical":
                scrutiny += 0.25
            elif concern.severity == "high":
                scrutiny += 0.15

        # Compound authority
        for flag in compound_flags:
            if flag.severity == "critical":
                scrutiny += 0.3
            else:
                scrutiny += 0.15

        # High commitment depth
        if commitment_depth > self.config.commitment_depth_warning:
            scrutiny += 0.15

        # Unresolved dependencies
        if unresolved:
            scrutiny += 0.2

        # High projected risk
        high_projected = [
            r for r in projected_risks
            if r.projected_risk_level in ("high", "critical")
        ]
        if high_projected:
            scrutiny += 0.1 * min(len(high_projected), 3)

        return min(1.0, scrutiny)

    # ── Internal: Drift Detection Helpers ────────────────────────────────

    @staticmethod
    def _method_distribution(steps: list[CompletedStep]) -> dict[str, float]:
        """Build a normalized method distribution from steps."""
        counts: dict[str, int] = {}
        for step in steps:
            base = _method_base(step.method)
            counts[base] = counts.get(base, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
        """Jensen-Shannon divergence between two distributions."""
        if not p and not q:
            return 0.0
        if not p or not q:
            return 1.0

        # Union of all keys
        all_keys = set(p.keys()) | set(q.keys())

        # Build aligned distributions with smoothing
        eps = 1e-10
        p_vals = [p.get(k, 0.0) + eps for k in all_keys]
        q_vals = [q.get(k, 0.0) + eps for k in all_keys]

        # Normalize after smoothing
        p_sum = sum(p_vals)
        q_sum = sum(q_vals)
        p_vals = [v / p_sum for v in p_vals]
        q_vals = [v / q_sum for v in q_vals]

        # M = average
        m_vals = [(pi + qi) / 2.0 for pi, qi in zip(p_vals, q_vals)]

        # KL(P || M) and KL(Q || M)
        kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p_vals, m_vals))
        kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q_vals, m_vals))

        return (kl_pm + kl_qm) / 2.0

    @staticmethod
    def _dominant_method(dist: dict[str, float]) -> str:
        """Return the most common method in a distribution."""
        if not dist:
            return "unknown"
        return max(dist, key=dist.get)  # type: ignore[arg-type]
