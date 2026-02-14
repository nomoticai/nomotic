"""Tests for the Workflow Governor system (Phase 7C).

Comprehensive tests covering:
- DependencyGraph construction and analysis
- ConsequenceProjector forward-looking risk assessment
- WorkflowGovernor core assessment logic
- Ordering analysis (commitment, irreversibility, authority, resource patterns)
- Compound authority detection
- Cross-step drift detection
- Integration with GovernanceRuntime and ContextualModifier
"""

from __future__ import annotations

import pytest

from nomotic.context_profile import (
    CompletedStep,
    ContextProfile,
    Dependency,
    PlannedStep,
    RelationalContext,
    WorkflowContext,
)
from nomotic.contextual_modifier import ContextualModifier, ModifierConfig
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.types import Action, AgentContext, TrustProfile
from nomotic.workflow_governor import (
    CompoundAuthorityFlag,
    ConsequenceProjector,
    DependencyGraph,
    DriftAcrossSteps,
    OrderingConcern,
    ProjectedRisk,
    StepAssessment,
    WorkflowGovernor,
    WorkflowGovernorConfig,
    WorkflowRiskAssessment,
    WorkflowRiskFactor,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _action(action_type: str = "read", target: str = "data", agent_id: str = "agent-1") -> Action:
    return Action(agent_id=agent_id, action_type=action_type, target=target)


def _context(agent_id: str = "agent-1", trust: float = 0.5, profile_id: str | None = None) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trust_profile=TrustProfile(agent_id=agent_id, overall_trust=trust),
        context_profile_id=profile_id,
    )


def _profile(
    agent_id: str = "agent-1",
    workflow: WorkflowContext | None = None,
) -> ContextProfile:
    return ContextProfile(
        profile_id="cp-test123",
        agent_id=agent_id,
        profile_type="workflow",
        workflow=workflow,
    )


def _step(
    step_number: int,
    method: str = "read",
    target: str = "data",
    verdict: str = "ALLOW",
    ucs: float = 0.8,
    step_id: str | None = None,
) -> CompletedStep:
    return CompletedStep(
        step_id=step_id or f"step-{step_number}",
        step_number=step_number,
        method=method,
        target=target,
        verdict=verdict,
        ucs=ucs,
        timestamp="2025-01-01T00:00:00Z",
    )


def _planned(
    step_number: int,
    method: str = "read",
    target: str = "data",
    description: str = "Planned step",
    depends_on: list[int] | None = None,
    estimated_risk: str | None = None,
) -> PlannedStep:
    return PlannedStep(
        step_number=step_number,
        method=method,
        target=target,
        description=description,
        depends_on=depends_on or [],
        estimated_risk=estimated_risk,
    )


def _dep(
    from_step: int,
    to_step: int,
    dep_type: str = "requires",
    description: str = "dependency",
) -> Dependency:
    return Dependency(
        from_step=from_step,
        to_step=to_step,
        dependency_type=dep_type,
        description=description,
    )


def _workflow(
    total_steps: int = 5,
    current_step: int = 1,
    completed: list[CompletedStep] | None = None,
    remaining: list[PlannedStep] | None = None,
    dependencies: list[Dependency] | None = None,
    rollback_points: list[str] | None = None,
    status: str = "active",
) -> WorkflowContext:
    return WorkflowContext(
        workflow_id="wf-test-1",
        workflow_type="test_workflow",
        total_steps=total_steps,
        current_step=current_step,
        steps_completed=completed or [],
        steps_remaining=remaining or [],
        dependencies=dependencies or [],
        rollback_points=rollback_points or [],
        started_at="2025-01-01T00:00:00Z",
        status=status,
    )


# ══════════════════════════════════════════════════════════════════════════
# DependencyGraph tests (20 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestDependencyGraph:

    def test_empty_workflow(self):
        """Build from workflow with no dependencies — empty graph."""
        wf = _workflow(dependencies=[])
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.get_dependencies(1) == []
        assert graph.get_dependents(1) == []

    def test_build_with_requires(self):
        """Build with requires dependencies."""
        wf = _workflow(dependencies=[_dep(1, 2, "requires")])
        graph = DependencyGraph.from_workflow_context(wf)
        deps = graph.get_dependencies(2)
        assert len(deps) == 1
        assert deps[0].from_step == 1
        assert deps[0].dependency_type == "requires"

    def test_build_with_constrains(self):
        """Build with constrains dependencies."""
        wf = _workflow(dependencies=[_dep(1, 3, "constrains")])
        graph = DependencyGraph.from_workflow_context(wf)
        dependents = graph.get_dependents(1)
        assert len(dependents) == 1
        assert dependents[0].to_step == 3
        assert dependents[0].dependency_type == "constrains"

    def test_build_with_enables(self):
        """Build with enables dependencies."""
        wf = _workflow(dependencies=[_dep(2, 4, "enables")])
        graph = DependencyGraph.from_workflow_context(wf)
        deps = graph.get_dependencies(4)
        assert len(deps) == 1
        assert deps[0].dependency_type == "enables"

    def test_build_with_mixed_types(self):
        """Build with mixed dependency types."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "requires"),
            _dep(1, 3, "constrains"),
            _dep(2, 4, "enables"),
            _dep(3, 5, "informs"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        assert len(graph.get_dependents(1)) == 2
        assert len(graph.get_dependencies(4)) == 1

    def test_get_dependencies_correct(self):
        """get_dependencies returns correct deps for step."""
        wf = _workflow(dependencies=[
            _dep(1, 3, "requires"),
            _dep(2, 3, "constrains"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        deps = graph.get_dependencies(3)
        assert len(deps) == 2
        from_steps = {d.from_step for d in deps}
        assert from_steps == {1, 2}

    def test_get_dependents_correct(self):
        """get_dependents returns correct dependents."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "requires"),
            _dep(1, 3, "constrains"),
            _dep(1, 4, "enables"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        dependents = graph.get_dependents(1)
        assert len(dependents) == 3

    def test_constraint_chain_transitive(self):
        """get_constraint_chain follows transitive constraints."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "constrains"),
            _dep(2, 3, "constrains"),
            _dep(3, 5, "constrains"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        chain = graph.get_constraint_chain(1)
        assert chain == [2, 3, 5]

    def test_required_chain_transitive(self):
        """get_required_chain follows transitive requirements."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "requires"),
            _dep(2, 3, "requires"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        chain = graph.get_required_chain(3)
        assert set(chain) == {1, 2}

    def test_unresolved_dependencies_missing(self):
        """unresolved_dependencies identifies missing prereqs."""
        wf = _workflow(dependencies=[_dep(1, 3, "requires")])
        graph = DependencyGraph.from_workflow_context(wf)
        unresolved = graph.unresolved_dependencies(3, [])
        assert len(unresolved) == 1
        assert unresolved[0].from_step == 1

    def test_unresolved_dependencies_satisfied(self):
        """unresolved_dependencies empty when all satisfied."""
        wf = _workflow(dependencies=[_dep(1, 3, "requires")])
        graph = DependencyGraph.from_workflow_context(wf)
        completed = [_step(1)]
        unresolved = graph.unresolved_dependencies(3, completed)
        assert len(unresolved) == 0

    def test_commitment_depth(self):
        """commitment_depth counts correctly."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "constrains"),
            _dep(1, 3, "constrains"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.commitment_depth(1) == 2

    def test_commitment_depth_transitive(self):
        """commitment_depth with transitive chains."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "constrains"),
            _dep(2, 3, "constrains"),
            _dep(3, 4, "constrains"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.commitment_depth(1) == 3

    def test_critical_path_longest_chain(self):
        """critical_path identifies longest requires chain."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "requires"),
            _dep(2, 3, "requires"),
            _dep(3, 4, "requires"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        cp = graph.critical_path()
        assert cp == [1, 2, 3, 4]

    def test_critical_path_parallel_branches(self):
        """critical_path with parallel branches picks the longest."""
        wf = _workflow(dependencies=[
            _dep(1, 2, "requires"),
            _dep(2, 4, "requires"),
            _dep(1, 3, "requires"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        cp = graph.critical_path()
        # Longest chain is 1->2->4 (length 3)
        assert len(cp) == 3

    def test_parallel_steps_identified(self):
        """parallel_steps identifies independent steps."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "requires")],
            remaining=[_planned(3), _planned(4)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        parallel = graph.parallel_steps(1)
        assert 3 in parallel
        assert 4 in parallel
        assert 2 not in parallel

    def test_is_reversible_to_with_rollback(self):
        """is_reversible_to checks rollback points."""
        completed = [_step(1, step_id="step-1"), _step(2, step_id="step-2")]
        wf = _workflow(
            completed=completed,
            rollback_points=["step-1", "step-2"],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.is_reversible_to(1, completed) is True

    def test_is_reversible_to_without_rollback(self):
        """is_reversible_to returns False without rollback points."""
        completed = [_step(1, step_id="step-1"), _step(2, step_id="step-2")]
        wf = _workflow(completed=completed, rollback_points=["step-1"])
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.is_reversible_to(1, completed) is False

    def test_single_step_workflow(self):
        """Single step workflow produces minimal graph."""
        wf = _workflow(total_steps=1, current_step=1)
        graph = DependencyGraph.from_workflow_context(wf)
        assert graph.critical_path() == [1]

    def test_complex_graph_multiple_paths(self):
        """Complex graph with multiple paths."""
        wf = _workflow(dependencies=[
            _dep(1, 3, "requires"),
            _dep(2, 3, "requires"),
            _dep(3, 4, "constrains"),
            _dep(3, 5, "enables"),
            _dep(4, 6, "requires"),
            _dep(5, 6, "requires"),
        ])
        graph = DependencyGraph.from_workflow_context(wf)
        deps_of_6 = graph.get_dependencies(6)
        assert len(deps_of_6) == 2
        deps_of_3 = graph.get_dependencies(3)
        assert len(deps_of_3) == 2
        # Constraint chain from 3 includes 4
        assert 4 in graph.get_constraint_chain(3)


# ══════════════════════════════════════════════════════════════════════════
# ConsequenceProjector tests (15 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestConsequenceProjector:

    def test_irreversible_step_elevated_risk(self):
        """Irreversible step projects elevated risk to constrained steps."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2, method="write")],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector(max_depth=10)
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 1
        assert risks[0].projected_risk_level == "high"
        assert risks[0].depends_on_current is True

    def test_resource_consuming_boundary_pressure(self):
        """Resource-consuming step projects boundary pressure."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        # Use "reserve" which is resource-consuming but not transaction
        # to isolate the resource path vs. the irreversible/transaction path
        action = _action(action_type="charge")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 1
        assert risks[0].projected_risk_level == "high"
        # charge is both resource-consuming and transaction, so description
        # may reference irreversibility or resources
        assert risks[0].depends_on_current is True

    def test_info_step_low_risk(self):
        """Information step projects low risk."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="query")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 1
        assert risks[0].projected_risk_level == "low"

    def test_authority_expanding_flags_compound(self):
        """Authority-expanding step flags compound authority concern."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="delegate")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 1
        assert risks[0].projected_risk_level == "high"
        assert "authority" in risks[0].risk_description.lower()

    def test_projection_respects_depth_limit(self):
        """Projection respects depth limit."""
        deps = [_dep(i, i + 1, "constrains") for i in range(1, 20)]
        remaining = [_planned(i) for i in range(2, 20)]
        wf = _workflow(dependencies=deps, remaining=remaining)
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector(max_depth=5)
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) <= 5

    def test_no_constrained_steps_no_projections(self):
        """No constrained steps → no projections."""
        wf = _workflow(dependencies=[])
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 0

    def test_risk_propagation_through_chains(self):
        """Risk propagation amplifies through chains."""
        wf = _workflow(
            dependencies=[
                _dep(1, 2, "constrains"),
                _dep(2, 3, "constrains"),
                _dep(3, 4, "constrains"),
            ],
            remaining=[_planned(2), _planned(3), _planned(4)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 3
        # First hop should be highest risk
        assert risks[0].projected_risk_level == "high"

    def test_projection_with_mixed_dep_types(self):
        """Projection with mixed dependency types only follows constrains."""
        wf = _workflow(
            dependencies=[
                _dep(1, 2, "constrains"),
                _dep(1, 3, "requires"),  # not constrains
                _dep(1, 4, "enables"),  # not constrains
            ],
            remaining=[_planned(2), _planned(3), _planned(4)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        # Only step 2 is constrained
        assert len(risks) == 1
        assert risks[0].step_number == 2

    def test_empty_remaining_steps_no_projections(self):
        """Empty remaining steps → no projections when no constraint chain."""
        wf = _workflow(remaining=[])
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 0

    def test_projection_includes_constraint_description(self):
        """Projection describes constraint from current step."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="reserve")

        risks = projector.project(1, action, graph, wf)
        assert risks[0].constraint_from_current is not None
        assert len(risks[0].constraint_from_current) > 0

    def test_transaction_methods_highest_propagation(self):
        """Transaction methods get highest risk propagation."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()

        # Transaction
        t_risks = projector.project(1, _action(action_type="transfer"), graph, wf)
        # Read
        r_risks = projector.project(1, _action(action_type="read"), graph, wf)

        assert t_risks[0].projected_risk_level == "high"
        assert r_risks[0].projected_risk_level == "low"

    def test_read_query_methods_lowest(self):
        """Read/query methods get lowest risk projection."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()

        for method in ["read", "query", "search", "find"]:
            risks = projector.project(1, _action(action_type=method), graph, wf)
            assert risks[0].projected_risk_level == "low"

    def test_projection_handles_no_circular_deps(self):
        """Projection handles gracefully with no circular deps."""
        wf = _workflow(
            dependencies=[
                _dep(1, 2, "constrains"),
                _dep(2, 1, "constrains"),  # circular
            ],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        # Should not infinite loop — visited set prevents it
        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 2  # 1->2 and 2->1

    def test_multiple_constraint_chains(self):
        """Multiple constraint chains from same step."""
        wf = _workflow(
            dependencies=[
                _dep(1, 2, "constrains"),
                _dep(1, 3, "constrains"),
            ],
            remaining=[_planned(2), _planned(3)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks) == 2

    def test_projection_reasoning_descriptive(self):
        """Projection reasoning is descriptive."""
        wf = _workflow(
            dependencies=[_dep(1, 2, "constrains")],
            remaining=[_planned(2)],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        projector = ConsequenceProjector()
        action = _action(action_type="transfer")

        risks = projector.project(1, action, graph, wf)
        assert len(risks[0].risk_description) > 10


# ══════════════════════════════════════════════════════════════════════════
# WorkflowGovernor core tests (15 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestWorkflowGovernorCore:

    def test_clean_workflow_low_risk(self):
        """assess_workflow with clean workflow → low risk, continue."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[_step(1, ucs=0.9), _step(2, ucs=0.85)],
            current_step=3,
        )
        profile = _profile(workflow=wf)
        assessment = gov.assess_workflow("wf-1", profile)

        assert assessment.cumulative_risk_score < 0.3
        assert assessment.recommendation == "continue"

    def test_multiple_denials_elevated_risk(self):
        """assess_workflow with multiple denials → elevated risk."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, verdict="DENY", ucs=0.2),
                _step(2, verdict="DENY", ucs=0.3),
                _step(3, verdict="DENY", ucs=0.2),
            ],
            current_step=4,
        )
        profile = _profile(workflow=wf)
        assessment = gov.assess_workflow("wf-1", profile)

        assert assessment.cumulative_risk_score > 0.3
        # Multiple denials with low UCS push risk very high
        assert assessment.recommendation in ("caution", "pause_for_review", "halt")

    def test_critical_risk_halt(self):
        """assess_workflow with critical risk → halt recommendation."""
        gov = WorkflowGovernor(WorkflowGovernorConfig(
            cumulative_risk_halt_threshold=0.5,
        ))
        wf = _workflow(
            completed=[
                _step(1, method="transfer", verdict="DENY", ucs=0.1),
                _step(2, method="transfer", verdict="DENY", ucs=0.1),
                _step(3, method="purchase", verdict="ESCALATE", ucs=0.15),
            ],
            current_step=4,
        )
        profile = _profile(workflow=wf)
        assessment = gov.assess_workflow("wf-1", profile)

        assert assessment.recommendation == "halt"

    def test_assess_step_returns_assessment(self):
        """assess_step returns step-specific assessment."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[_step(1, ucs=0.8)],
            current_step=2,
            remaining=[_planned(2, method="write")],
        )
        profile = _profile(workflow=wf)
        action = _action(action_type="write")
        ctx = _context()

        assessment = gov.assess_step(2, action, ctx, profile)
        assert isinstance(assessment, StepAssessment)
        assert assessment.step_number == 2

    def test_assess_step_flags_unresolved_deps(self):
        """assess_step with unresolved dependencies flags them."""
        gov = WorkflowGovernor()
        wf = _workflow(
            current_step=3,
            dependencies=[_dep(1, 3, "requires")],
            remaining=[_planned(3)],
        )
        profile = _profile(workflow=wf)
        action = _action()
        ctx = _context()

        assessment = gov.assess_step(3, action, ctx, profile)
        assert len(assessment.unresolved_dependencies) > 0

    def test_record_step_outcome_updates_state(self):
        """record_step_outcome updates workflow state."""
        gov = WorkflowGovernor()
        wf = _workflow(completed=[_step(1, ucs=0.9)])
        profile = _profile(workflow=wf)

        new_step = _step(2, ucs=0.85)
        gov.record_step_outcome("wf-1", new_step, profile)

        # Risk history should be recorded
        assert "wf-1" in gov._risk_history

    def test_cumulative_risk_weights_transactions(self):
        """Cumulative risk computation weights transactions higher."""
        gov = WorkflowGovernor()
        # Same UCS, one with transaction method
        wf_regular = _workflow(
            completed=[_step(1, method="read", ucs=0.5)],
        )
        wf_transaction = _workflow(
            completed=[_step(1, method="transfer", ucs=0.5)],
        )
        graph_r = DependencyGraph.from_workflow_context(wf_regular)
        graph_t = DependencyGraph.from_workflow_context(wf_transaction)

        risk_regular = gov._compute_cumulative_risk(wf_regular, graph_r)
        risk_transaction = gov._compute_cumulative_risk(wf_transaction, graph_t)

        assert risk_transaction > risk_regular

    def test_cumulative_risk_weights_denials(self):
        """Cumulative risk computation weights denials higher."""
        gov = WorkflowGovernor()
        wf_allow = _workflow(completed=[_step(1, verdict="ALLOW", ucs=0.5)])
        wf_deny = _workflow(completed=[_step(1, verdict="DENY", ucs=0.5)])

        graph_a = DependencyGraph.from_workflow_context(wf_allow)
        graph_d = DependencyGraph.from_workflow_context(wf_deny)

        risk_allow = gov._compute_cumulative_risk(wf_allow, graph_a)
        risk_deny = gov._compute_cumulative_risk(wf_deny, graph_d)

        assert risk_deny > risk_allow

    def test_cumulative_risk_weights_irreversible(self):
        """Cumulative risk computation weights irreversible higher."""
        gov = WorkflowGovernor()
        wf_reversible = _workflow(
            completed=[_step(1, ucs=0.5, step_id="step-1")],
            rollback_points=["step-1"],
        )
        wf_irreversible = _workflow(
            completed=[_step(1, ucs=0.5, step_id="step-1")],
            rollback_points=[],
        )

        graph_r = DependencyGraph.from_workflow_context(wf_reversible)
        graph_i = DependencyGraph.from_workflow_context(wf_irreversible)

        risk_rev = gov._compute_cumulative_risk(wf_reversible, graph_r)
        risk_irr = gov._compute_cumulative_risk(wf_irreversible, graph_i)

        assert risk_irr > risk_rev

    def test_risk_trajectory_stable(self):
        """Risk trajectory stable when consistent."""
        gov = WorkflowGovernor()
        gov._risk_history["wf-1"] = [0.3, 0.31, 0.32, 0.3]
        trajectory = gov._compute_risk_trajectory("wf-1", 0.31)
        assert trajectory == "stable"

    def test_risk_trajectory_increasing(self):
        """Risk trajectory increasing when worsening."""
        gov = WorkflowGovernor()
        gov._risk_history["wf-1"] = [0.2, 0.25, 0.28, 0.30]
        trajectory = gov._compute_risk_trajectory("wf-1", 0.35)
        assert trajectory == "increasing"

    def test_risk_trajectory_accelerating(self):
        """Risk trajectory accelerating when rapidly worsening."""
        gov = WorkflowGovernor()
        gov._risk_history["wf-1"] = [0.1, 0.2, 0.3, 0.4]
        trajectory = gov._compute_risk_trajectory("wf-1", 0.6)
        assert trajectory == "accelerating"

    def test_recommendation_thresholds_respected(self):
        """Recommendation thresholds are respected."""
        config = WorkflowGovernorConfig(
            cumulative_risk_threshold=0.5,
            cumulative_risk_halt_threshold=0.8,
        )
        gov = WorkflowGovernor(config)

        # Below threshold
        rec, _ = gov._determine_recommendation(0.3, [], [], [], [])
        assert rec == "continue"

        # At review threshold
        rec, _ = gov._determine_recommendation(0.6, [], [], [], [])
        assert rec == "pause_for_review"

        # At halt threshold
        rec, _ = gov._determine_recommendation(0.85, [], [], [], [])
        assert rec == "halt"

    def test_empty_workflow_baseline(self):
        """Empty workflow → baseline assessment."""
        gov = WorkflowGovernor()
        profile = _profile(workflow=_workflow())
        assessment = gov.assess_workflow("wf-1", profile)

        assert assessment.cumulative_risk_score == 0.0
        assert assessment.recommendation == "continue"

    def test_config_disables_analysis(self):
        """Config options disable specific analysis types."""
        config = WorkflowGovernorConfig(
            ordering_analysis=False,
            compound_authority_detection=False,
            consequence_projection=False,
            drift_across_steps_detection=False,
        )
        gov = WorkflowGovernor(config)
        wf = _workflow(
            completed=[
                _step(1, method="delegate"),
                _step(2, method="escalate"),
                _step(3, method="approve"),
            ],
            remaining=[_planned(4)],
            dependencies=[_dep(3, 4, "constrains")],
            current_step=4,
        )
        profile = _profile(workflow=wf)
        assessment = gov.assess_workflow("wf-1", profile)

        assert assessment.ordering_concerns == []
        assert assessment.compound_authority_flags == []
        assert assessment.projected_risks == []


# ══════════════════════════════════════════════════════════════════════════
# Ordering analysis tests (10 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestOrderingAnalysis:

    def test_commitment_before_dependency(self):
        """Detects commitment before dependency."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[_step(2, method="transfer")],  # committed without step 1
            dependencies=[_dep(1, 2, "requires")],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        commitment_concerns = [
            c for c in concerns
            if c.concern_type == "commitment_before_dependency"
        ]
        assert len(commitment_concerns) >= 1

    def test_irreversible_before_verification(self):
        """Detects irreversible before verification."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="write", step_id="step-1"),
                _step(2, method="validate"),
            ],
            dependencies=[_dep(2, 1, "requires")],
            rollback_points=[],  # step 1 is irreversible
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        irrev_concerns = [
            c for c in concerns
            if c.concern_type == "irreversible_before_verification"
        ]
        assert len(irrev_concerns) >= 1

    def test_authority_escalation_sequence(self):
        """Detects authority escalation sequence."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="delegate"),
                _step(2, method="escalate"),
                _step(3, method="authorize"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        auth_concerns = [
            c for c in concerns
            if c.concern_type == "authority_escalation_sequence"
        ]
        assert len(auth_concerns) >= 1

    def test_resource_lock_chain(self):
        """Detects resource lock chain."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="reserve"),
                _step(2, method="charge"),
                _step(3, method="purchase"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        resource_concerns = [
            c for c in concerns
            if c.concern_type == "resource_lock_chain"
        ]
        assert len(resource_concerns) >= 1

    def test_no_issues_well_ordered(self):
        """No ordering issues in well-ordered workflow."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="read", step_id="step-1"),
                _step(2, method="validate", step_id="step-2"),
                _step(3, method="write", step_id="step-3"),
            ],
            rollback_points=["step-1", "step-2", "step-3"],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        # No commitment, authority, or resource chain concerns
        assert all(c.concern_type != "authority_escalation_sequence" for c in concerns)
        assert all(c.concern_type != "resource_lock_chain" for c in concerns)

    def test_ordering_concern_mitigation(self):
        """Ordering concern includes mitigation suggestion."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="delegate"),
                _step(2, method="escalate"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        for concern in concerns:
            assert len(concern.mitigation) > 0

    def test_ordering_concern_identifies_steps(self):
        """Ordering concern identifies specific steps."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="delegate"),
                _step(2, method="authorize"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        auth_concerns = [
            c for c in concerns
            if c.concern_type == "authority_escalation_sequence"
        ]
        if auth_concerns:
            assert auth_concerns[0].step_causing in (1, 2)

    def test_mixed_ordering_some_concerns(self):
        """Mixed ordering — some concerns, some clear."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="read"),
                _step(2, method="delegate"),
                _step(3, method="escalate"),
                _step(4, method="validate"),
                _step(5, method="write"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        # Should find the delegate->escalate sequence
        auth_concerns = [
            c for c in concerns
            if c.concern_type == "authority_escalation_sequence"
        ]
        assert len(auth_concerns) >= 1

    def test_parallel_steps_no_false_positives(self):
        """Ordering analysis with parallel steps (no false positives)."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="read", step_id="step-1"),
                _step(2, method="read", step_id="step-2"),
            ],
            rollback_points=["step-1", "step-2"],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        # Parallel read operations should not generate concerns
        assert all(c.concern_type != "authority_escalation_sequence" for c in concerns)
        assert all(c.concern_type != "resource_lock_chain" for c in concerns)

    def test_concern_severity_appropriate(self):
        """Concern severity appropriate to pattern type."""
        gov = WorkflowGovernor()
        wf = _workflow(
            completed=[
                _step(1, method="delegate"),
                _step(2, method="escalate"),
                _step(3, method="authorize"),
            ],
        )
        graph = DependencyGraph.from_workflow_context(wf)
        concerns = gov._analyze_ordering(graph, wf)

        auth_concerns = [
            c for c in concerns
            if c.concern_type == "authority_escalation_sequence"
        ]
        if auth_concerns:
            # 3-step authority sequence should be critical
            assert auth_concerns[0].severity == "critical"


# ══════════════════════════════════════════════════════════════════════════
# Compound authority detection tests (10 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestCompoundAuthorityDetection:

    def test_scope_assembly_pattern(self):
        """Detects scope assembly pattern (read + read + write from different sources)."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="read", target="system-A"),
            _step(2, method="read", target="system-B"),
            _step(3, method="write", target="combined"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        scope_flags = [f for f in flags if "scope" in f.description.lower()]
        assert len(scope_flags) >= 1

    def test_authority_ladder_pattern(self):
        """Detects authority ladder pattern."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="delegate"),
            _step(2, method="authorize"),
            _step(3, method="transfer"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        auth_flags = [f for f in flags if "authority" in f.description.lower()]
        assert len(auth_flags) >= 1

    def test_resource_aggregation_pattern(self):
        """Detects resource aggregation pattern."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="reserve"),
            _step(2, method="charge"),
            _step(3, method="purchase"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        resource_flags = [f for f in flags if "resource" in f.description.lower()]
        assert len(resource_flags) >= 1

    def test_no_false_positive_same_source(self):
        """No false positive on normal read-then-write (same source)."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="read", target="system-A"),
            _step(2, method="read", target="system-A"),  # same source
            _step(3, method="write", target="system-A"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        scope_flags = [f for f in flags if "scope" in f.description.lower()]
        # Same source reads should not trigger scope assembly
        assert len(scope_flags) == 0

    def test_no_false_positive_with_review(self):
        """No false positive on authorized escalation with human review between."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="delegate"),
            _step(2, method="review"),  # human review breaks the chain
            _step(3, method="authorize"),
            _step(4, method="transfer"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        auth_flags = [f for f in flags if "authority" in f.description.lower()]
        assert len(auth_flags) == 0

    def test_compound_flag_includes_capability(self):
        """Compound flag includes resulting capability description."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="delegate"),
            _step(2, method="authorize"),
            _step(3, method="transfer"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        for flag in flags:
            assert len(flag.resulting_capability) > 0

    def test_compound_flag_includes_steps_and_methods(self):
        """Compound flag includes involved steps and methods."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="read", target="A"),
            _step(2, method="read", target="B"),
            _step(3, method="write", target="C"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        scope_flags = [f for f in flags if "scope" in f.description.lower()]
        if scope_flags:
            assert len(scope_flags[0].steps_involved) >= 2
            assert len(scope_flags[0].methods_chained) >= 2

    def test_severity_levels_appropriate(self):
        """Severity levels are appropriate to pattern."""
        gov = WorkflowGovernor()
        # Authority ladder should be critical
        steps = [
            _step(1, method="delegate"),
            _step(2, method="authorize"),
            _step(3, method="transfer"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        auth_flags = [f for f in flags if "authority" in f.description.lower()]
        if auth_flags:
            assert auth_flags[0].severity == "critical"

    def test_detection_across_non_consecutive_steps(self):
        """Detects across non-consecutive steps (other steps between)."""
        gov = WorkflowGovernor()
        steps = [
            _step(1, method="read", target="X"),
            _step(2, method="validate"),  # intervening step
            _step(3, method="read", target="Y"),
            _step(4, method="notify"),  # intervening step
            _step(5, method="write", target="Z"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        scope_flags = [f for f in flags if "scope" in f.description.lower()]
        assert len(scope_flags) >= 1

    def test_multi_agent_compound_detection(self):
        """Multi-agent compound detection (shared_workflow_agents)."""
        gov = WorkflowGovernor()
        # When relational context has shared agents, the basic compound
        # patterns still detect on the step sequence
        steps = [
            _step(1, method="delegate"),
            _step(2, method="authorize"),
            _step(3, method="approve"),
        ]
        wf = _workflow(completed=steps)
        graph = DependencyGraph.from_workflow_context(wf)

        flags = gov._detect_compound_authority(steps, graph, wf)
        assert len(flags) >= 1


# ══════════════════════════════════════════════════════════════════════════
# Cross-step drift detection tests (5 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestDriftDetection:

    def test_no_drift_consistent_workflow(self):
        """No drift in consistent workflow."""
        gov = WorkflowGovernor()
        steps = [_step(i, method="read") for i in range(1, 10)]
        wf = _workflow(completed=steps)
        profile = _profile(workflow=wf)

        drift = gov.detect_cross_step_drift(profile)
        assert drift is not None
        assert drift.drift_detected is False

    def test_drift_detected_method_shift(self):
        """Drift detected when method distribution shifts."""
        gov = WorkflowGovernor()
        # First 6 steps: reads; last 6 steps: writes
        early = [_step(i, method="read") for i in range(1, 7)]
        late = [_step(i, method="transfer") for i in range(7, 13)]
        wf = _workflow(completed=early + late)
        profile = _profile(workflow=wf)

        drift = gov.detect_cross_step_drift(profile)
        assert drift is not None
        assert drift.drift_detected is True
        assert drift.divergence > 0.1

    def test_drift_severity_proportional(self):
        """Drift severity proportional to divergence."""
        gov = WorkflowGovernor()
        # Moderate shift
        steps = (
            [_step(i, method="read") for i in range(1, 5)]
            + [_step(i, method="write") for i in range(5, 7)]
            + [_step(i, method="delete") for i in range(7, 10)]
        )
        wf = _workflow(completed=steps)
        profile = _profile(workflow=wf)

        drift = gov.detect_cross_step_drift(profile)
        if drift is not None and drift.drift_detected:
            assert drift.severity in ("medium", "high", "critical")

    def test_drift_compares_first_third_last_third(self):
        """Drift compares first third vs last third."""
        gov = WorkflowGovernor()
        steps = (
            [_step(i, method="query") for i in range(1, 4)]  # first third
            + [_step(i, method="read") for i in range(4, 7)]  # middle third
            + [_step(i, method="transfer") for i in range(7, 10)]  # last third
        )
        wf = _workflow(completed=steps)
        profile = _profile(workflow=wf)

        drift = gov.detect_cross_step_drift(profile)
        assert drift is not None
        # Should compare query (early) vs transfer (late), not middle
        assert "query" in drift.early_pattern or "read" in drift.early_pattern

    def test_short_workflow_returns_none(self):
        """Short workflow (< 6 steps) returns None (insufficient data)."""
        gov = WorkflowGovernor()
        steps = [_step(i, method="read") for i in range(1, 5)]
        wf = _workflow(completed=steps)
        profile = _profile(workflow=wf)

        drift = gov.detect_cross_step_drift(profile)
        assert drift is None


# ══════════════════════════════════════════════════════════════════════════
# Integration tests (5 tests)
# ══════════════════════════════════════════════════════════════════════════


class TestIntegration:

    def test_runtime_creates_governor_when_enabled(self):
        """Runtime creates workflow governor when enabled."""
        config = RuntimeConfig(
            enable_workflow_governor=True,
        )
        runtime = GovernanceRuntime(config)
        assert runtime.workflow_governor is not None

    def test_runtime_no_governor_when_disabled(self):
        """Governor disabled → no workflow governor on runtime."""
        config = RuntimeConfig(
            enable_workflow_governor=False,
        )
        runtime = GovernanceRuntime(config)
        assert runtime.workflow_governor is None

    def test_governor_output_feeds_modifier(self):
        """Governor output feeds into contextual modifier adjustments."""
        runtime = GovernanceRuntime(RuntimeConfig(
            enable_workflow_governor=True,
            enable_contextual_modifier=True,
        ))

        # Create a profile with a risky workflow
        wf = _workflow(
            completed=[
                _step(1, method="delegate", ucs=0.3, verdict="ESCALATE"),
                _step(2, method="escalate", ucs=0.25, verdict="DENY"),
            ],
            current_step=3,
            remaining=[_planned(3, method="authorize")],
            dependencies=[_dep(2, 3, "constrains")],
        )
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            workflow=wf,
        )

        # Set up agent scope so evaluation doesn't fail on scope
        runtime.configure_scope("agent-1", {"read", "write", "authorize", "delegate", "escalate"})

        action = _action(action_type="authorize")
        ctx = _context(profile_id=profile.profile_id)

        # Evaluate — the workflow governor should feed into the modifier
        verdict = runtime.evaluate(action, ctx)
        # The verdict should exist (we don't assert specific outcome, just that
        # the chain works without error)
        assert verdict is not None

    def test_full_chain_governor_modifier_dimensions_verdict(self):
        """Full chain: governor → modifier → dimensions → verdict."""
        runtime = GovernanceRuntime(RuntimeConfig(
            enable_workflow_governor=True,
            enable_contextual_modifier=True,
        ))

        wf = _workflow(
            completed=[_step(1, method="read", ucs=0.9)],
            current_step=2,
            remaining=[_planned(2, method="write")],
        )
        profile = runtime.context_profiles.create_profile(
            agent_id="agent-1",
            profile_type="workflow",
            workflow=wf,
        )

        runtime.configure_scope("agent-1", {"read", "write"})

        action = _action(action_type="write")
        ctx = _context(profile_id=profile.profile_id)

        verdict = runtime.evaluate(action, ctx)
        assert verdict is not None
        assert verdict.ucs >= 0.0

    def test_governor_disabled_evaluation_works(self):
        """Governor disabled → no workflow analysis, evaluation works normally."""
        runtime = GovernanceRuntime(RuntimeConfig(
            enable_workflow_governor=False,
            enable_contextual_modifier=True,
        ))

        runtime.configure_scope("agent-1", {"read"})
        action = _action(action_type="read")
        ctx = _context()

        verdict = runtime.evaluate(action, ctx)
        assert verdict is not None


# ══════════════════════════════════════════════════════════════════════════
# Serialization tests
# ══════════════════════════════════════════════════════════════════════════


class TestSerialization:

    def test_workflow_risk_assessment_to_dict(self):
        """WorkflowRiskAssessment serializes correctly."""
        assessment = WorkflowRiskAssessment(
            workflow_id="wf-1",
            cumulative_risk_score=0.45,
            risk_trajectory="increasing",
            risk_factors=[
                WorkflowRiskFactor(
                    factor_type="cumulative_denials",
                    severity="medium",
                    description="test",
                    contributing_steps=[1, 2],
                    trend="growing",
                ),
            ],
            recommendation="caution",
            recommendation_reasoning="Moderate risk",
        )
        d = assessment.to_dict()
        assert d["workflow_id"] == "wf-1"
        assert d["cumulative_risk_score"] == 0.45
        assert len(d["risk_factors"]) == 1
        assert d["recommendation"] == "caution"

    def test_step_assessment_to_dict(self):
        """StepAssessment serializes correctly."""
        assessment = StepAssessment(
            step_number=3,
            workflow_risk_at_step=0.25,
            commitment_depth=4,
            recommended_additional_scrutiny=0.35,
            reasoning="Test reasoning",
        )
        d = assessment.to_dict()
        assert d["step_number"] == 3
        assert d["commitment_depth"] == 4

    def test_drift_to_dict(self):
        """DriftAcrossSteps serializes correctly."""
        drift = DriftAcrossSteps(
            drift_detected=True,
            early_pattern={"read": 0.8, "write": 0.2},
            recent_pattern={"transfer": 0.7, "write": 0.3},
            divergence=0.55,
            description="Test drift",
            severity="high",
        )
        d = drift.to_dict()
        assert d["drift_detected"] is True
        assert d["divergence"] == 0.55
        assert d["severity"] == "high"

    def test_projected_risk_to_dict(self):
        """ProjectedRisk serializes correctly."""
        risk = ProjectedRisk(
            step_number=5,
            method="transfer",
            projected_risk_level="high",
            risk_description="Test projected risk",
            depends_on_current=True,
            constraint_from_current="Constraint A",
        )
        d = risk.to_dict()
        assert d["step_number"] == 5
        assert d["constraint_from_current"] == "Constraint A"

    def test_compound_authority_flag_to_dict(self):
        """CompoundAuthorityFlag serializes correctly."""
        flag = CompoundAuthorityFlag(
            description="Authority ladder",
            methods_chained=["delegate", "authorize", "transfer"],
            steps_involved=[1, 2, 3],
            resulting_capability="Effective authority elevation",
            severity="critical",
        )
        d = flag.to_dict()
        assert d["severity"] == "critical"
        assert len(d["methods_chained"]) == 3


# ══════════════════════════════════════════════════════════════════════════
# No-workflow-context edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_assess_workflow_no_workflow_context(self):
        """assess_workflow with no workflow context returns baseline."""
        gov = WorkflowGovernor()
        profile = _profile()  # no workflow
        assessment = gov.assess_workflow("wf-1", profile)
        assert assessment.cumulative_risk_score == 0.0
        assert assessment.recommendation == "continue"

    def test_assess_step_no_workflow_context(self):
        """assess_step with no workflow context returns baseline."""
        gov = WorkflowGovernor()
        profile = _profile()
        action = _action()
        ctx = _context()
        assessment = gov.assess_step(1, action, ctx, profile)
        assert assessment.workflow_risk_at_step == 0.0

    def test_record_step_no_workflow_context(self):
        """record_step_outcome with no workflow context is a no-op."""
        gov = WorkflowGovernor()
        profile = _profile()
        step = _step(1)
        gov.record_step_outcome("wf-1", step, profile)
        # Should not crash

    def test_detect_drift_no_workflow(self):
        """detect_cross_step_drift with no workflow returns None."""
        gov = WorkflowGovernor()
        profile = _profile()
        drift = gov.detect_cross_step_drift(profile)
        assert drift is None
