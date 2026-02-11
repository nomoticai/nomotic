# The 13 Governance Dimensions

Every action is evaluated across all 13 dimensions simultaneously. Each dimension produces a `DimensionScore` with:

- **score**: 0.0 (maximum governance concern) to 1.0 (no concern)
- **weight**: How much this dimension influences the UCS
- **veto**: If `True`, overrides all other scores and forces Tier 1 denial
- **confidence**: How certain the dimension is about its assessment (0.0-1.0)
- **reasoning**: Human-readable explanation of the score

Dimensions are independent. They evaluate the same action but from different perspectives and do not see each other's scores.

---

## Dimension 1: Scope Compliance

**Class**: `ScopeCompliance`
**Weight**: 1.5
**Can Veto**: Yes

Is the action within the agent's authorized scope?

### What It Checks

Each agent has a set of allowed action types. Scope compliance checks whether the requested action type appears in that set. This is the most fundamental governance check — an agent cannot do what it is not allowed to do.

### Configuration

```python
scope = runtime.registry.get("scope_compliance")

# Allow specific action types
scope.configure_agent_scope("agent-1", {"read", "write", "query"})

# Allow everything (use with caution)
scope.configure_agent_scope("agent-2", {"*"})
```

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| Action type in allowed set | 1.0 | No |
| Wildcard `*` in allowed set | 1.0 | No |
| No scope defined for agent | 0.5 | No |
| Action type not in allowed set | 0.0 | **Yes** |

### Notes

- No scope defined returns 0.5 (caution) rather than 0.0 (veto). This allows unscoped agents to operate with reduced confidence rather than being completely blocked. Configure explicit scopes for production agents.
- The wildcard `*` allows all action types. This bypasses scope checking entirely.

---

## Dimension 2: Authority Verification

**Class**: `AuthorityVerification`
**Weight**: 1.5
**Can Veto**: Yes

Does the agent have authority for this specific action?

### What It Checks

Scope says what *types* of actions are allowed. Authority says whether *this specific action on this specific target* is permitted. Authority checks are custom callables that receive the action and context.

### Configuration

```python
auth = runtime.registry.get("authority_verification")

# Add a check: agent can only write to targets they own
auth.add_authority_check(
    lambda action, ctx: action.target in get_owned_targets(ctx.agent_id)
)

# Add another check: no actions on archived targets
auth.add_authority_check(
    lambda action, ctx: not is_archived(action.target)
)
```

All checks must pass. Any single failure is a veto.

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| No checks configured | 0.7 | No |
| All checks pass | 1.0 | No |
| Any check fails | 0.0 | **Yes** |

---

## Dimension 3: Resource Boundaries

**Class**: `ResourceBoundaries`
**Weight**: 1.2
**Can Veto**: Yes

Are rate, concurrency, and cost limits respected?

### What It Checks

Three resource constraints evaluated together:
1. **Rate**: Actions per minute against `max_actions_per_minute`
2. **Concurrency**: Active simultaneous actions against `max_concurrent_actions`
3. **Cost**: Per-action cost and total accumulated cost against limits

### Configuration

```python
from nomotic.dimensions import ResourceLimits

resources = runtime.registry.get("resource_boundaries")
resources._limits = ResourceLimits(
    max_actions_per_minute=30,
    max_concurrent_actions=5,
    max_cost_per_action=100.0,
    max_total_cost=10000.0,
)
```

Action cost is read from `action.parameters.get("cost", 0.0)`.

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| Rate limit exceeded | 0.0 | **Yes** |
| Concurrency limit exceeded | 0.0 | **Yes** |
| Per-action cost exceeded | 0.0 | **Yes** |
| Total cost exceeded | 0.0 | **Yes** |
| Within all limits | `1.0 - max(rate_ratio, conc_ratio) * 0.5` | No |

The graduated score means actions approaching limits get lower scores before actually hitting them. At 80% of the rate limit, the score is 0.6 rather than 1.0. This gives the UCS engine early warning.

### Defaults

| Parameter | Default |
|-----------|---------|
| `max_actions_per_minute` | 60 |
| `max_concurrent_actions` | 10 |
| `max_cost_per_action` | Infinity |
| `max_total_cost` | Infinity |

---

## Dimension 4: Behavioral Consistency

**Class**: `BehavioralConsistency`
**Weight**: 1.0
**Can Veto**: No

Does this action match the agent's established behavior patterns?

### What It Checks

Tracks which action types each agent has used. A new agent performing its first action type has no baseline. An established agent suddenly using a novel action type raises moderate concern. An agent using a familiar action type gets full marks.

### Scoring

| Condition | Score |
|-----------|-------|
| First action (no baseline) | 0.7 |
| Action type seen before | 1.0 |
| Novel action type | 0.5 |

### Notes

- This dimension cannot veto. Novel behavior is suspicious but not automatically disqualifying.
- The baseline is per-agent. Agent A's history does not affect Agent B's scoring.
- Novel action types are added to the baseline after evaluation. The second time an agent uses a type, it scores 1.0.

---

## Dimension 5: Cascading Impact

**Class**: `CascadingImpact`
**Weight**: 1.3
**Can Veto**: No

What are the downstream consequences of this action?

### What It Checks

Classifies actions by their potential to trigger chains of effects. The classification is based on keywords in the action type.

### Scoring

| Action Type Contains | Impact Level | Score |
|---------------------|-------------|-------|
| `delete`, `deploy`, `publish`, `broadcast`, `transfer` | High | 0.3 |
| `update`, `modify`, `write`, `send`, `create` | Medium | 0.6 |
| Anything else | Low | 0.9 |

### Notes

- Matching is case-insensitive and uses substring matching. `delete_user` matches `delete`. `bulk_transfer` matches `transfer`.
- This dimension cannot veto. High cascading impact lowers the UCS but doesn't block alone. Combined with other low-scoring dimensions, it contributes to denial.

---

## Dimension 6: Stakeholder Impact

**Class**: `StakeholderImpact`
**Weight**: 1.2
**Can Veto**: No

Who is affected by this action and how?

### What It Checks

Two mechanisms:
1. **Sensitive targets**: Explicitly marked targets get the lowest score
2. **External indicators**: Targets containing keywords like "customer", "user", "external", "public", or "client" get a medium-low score

### Configuration

```python
stakeholder = runtime.registry.get("stakeholder_impact")
stakeholder.mark_sensitive("production_db", "payment_gateway", "user_pii")
```

### Scoring

| Condition | Score |
|-----------|-------|
| Target marked sensitive | 0.2 |
| Target contains external keywords | 0.4 |
| Internal target | 0.9 |

---

## Dimension 7: Incident Detection

**Class**: `IncidentDetection`
**Weight**: 1.5
**Can Veto**: Yes

Does this action match known failure or attack patterns?

### What It Checks

Two layers of detection:
1. **Built-in**: Detects 5+ identical consecutive actions in recent history (repetitive pattern)
2. **Custom patterns**: User-supplied detectors that receive the action and context

### Configuration

```python
incident = runtime.registry.get("incident_detection")

# Detect privilege escalation patterns
def detect_escalation(action, context):
    recent = context.action_history[-10:]
    if any(r.action.action_type == "elevate_permissions" for r in recent):
        if action.action_type in ("delete", "transfer"):
            return 0.05  # Very low score = high concern
    return None  # No match

incident.add_pattern(detect_escalation)
```

Custom detectors return a score (0.0-1.0) if the pattern matches, or `None` if it doesn't. The worst (lowest) score across all detectors is used.

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| No patterns matched | 1.0 | No |
| Repetitive pattern (5+ identical) | 0.2 | No |
| Custom pattern matched | Detector's score | **Yes** if score <= 0.1 |

---

## Dimension 8: Isolation Integrity

**Class**: `IsolationIntegrity`
**Weight**: 1.4
**Can Veto**: Yes

Are containment boundaries maintained?

### What It Checks

Each agent operates within defined boundaries. Actions targeting resources outside those boundaries are vetoed. This prevents agents from accessing other agents' resources or modifying shared state without coordination.

### Configuration

```python
isolation = runtime.registry.get("isolation_integrity")
isolation.set_boundaries("agent-1", {"db_orders", "db_inventory", "cache_local"})
isolation.set_boundaries("agent-2", {"db_users", "db_auth"})
```

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| No boundaries defined | 0.6 | No |
| Target within boundaries | 1.0 | No |
| Wildcard `*` in boundaries | 1.0 | No |
| Empty target | 1.0 | No |
| Target outside boundaries | 0.0 | **Yes** |

---

## Dimension 9: Temporal Compliance

**Class**: `TemporalCompliance`
**Weight**: 0.8
**Can Veto**: Yes

Is the timing of this action appropriate?

### What It Checks

Two temporal constraints:
1. **Time windows**: Some action types are only allowed during specific hours
2. **Minimum intervals**: Some action types must wait a minimum time between executions

### Configuration

```python
temporal = runtime.registry.get("temporal_compliance")

# Deployments only between 9am and 5pm
temporal.set_time_window("deploy", start_hour=9, end_hour=17)

# Minimum 1 hour between deployments per agent
temporal.set_min_interval("deploy", seconds=3600)
```

Time windows support wrap-around (e.g., `start_hour=22, end_hour=6` for overnight windows).

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| No constraints defined | 1.0 | No |
| Outside time window | 0.0 | **Yes** |
| Min interval not met | 0.1 | **Yes** |
| All constraints satisfied | 1.0 | No |

---

## Dimension 10: Precedent Alignment

**Class**: `PrecedentAlignment`
**Weight**: 0.7
**Can Veto**: No

Is this consistent with past governance decisions?

### What It Checks

Examines the agent's action history for similar actions (same action type). If similar actions were frequently denied (>50%), the current action gets a low score. Consistent precedent of approval gets a high score.

### Scoring

| Condition | Score |
|-----------|-------|
| No action history | 0.8 |
| No precedent for this action type | 0.7 |
| Similar actions frequently denied (>50%) | 0.3 |
| Consistent with past approvals | 0.9 |

### Notes

- This dimension cannot veto. Precedent is informational, not authoritative.
- Low weight (0.7) means precedent influences the UCS but doesn't dominate.

---

## Dimension 11: Transparency

**Class**: `Transparency`
**Weight**: 0.6
**Can Veto**: No

Is the action auditable and explainable?

### What It Checks

Whether the action carries sufficient metadata for auditing. Missing fields reduce the score.

### Scoring

| Missing Field | Penalty |
|---------------|---------|
| `action_type` | -0.3 |
| `target` | -0.2 |
| `agent_id` | -0.2 |
| `parameters` (for non-read actions) | -0.1 |

A fully transparent action with all fields scores 1.0. An action missing `action_type` and `target` scores 0.5. The minimum possible score is 0.0 (all fields missing on a non-read action: -0.3 -0.2 -0.2 -0.1 = -0.8, clamped to 0.0).

### Notes

- Read/list/query actions are not penalized for missing parameters, since reads often don't need them.
- This dimension cannot veto. Opaque actions lower the UCS but aren't blocked.
- Lowest weight (0.6) among all dimensions. Missing metadata is a concern but not as serious as scope violations or ethical issues.

---

## Dimension 12: Human Override

**Class**: `HumanOverride`
**Weight**: 2.0 (highest, tied with Ethical Alignment)
**Can Veto**: Yes

Is human intervention required or requested?

### What It Checks

Three triggers for human override:
1. **Required action types**: Actions explicitly configured to always need human approval
2. **Low trust**: Agents with trust below 0.3 require human approval for everything
3. **Pre-approval**: Actions that have been explicitly approved by a human

### Configuration

```python
human = runtime.registry.get("human_override")

# These action types always need human approval
human.require_human_for("deploy", "delete_user", "transfer_funds")

# Approve a specific action (after human reviews it)
human.approve(action.id)
```

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| Already human-approved | 1.0 | No |
| Action type requires human | 0.0 | **Yes** |
| Agent trust < 0.3 | 0.0 | **Yes** |
| No override needed | 1.0 | No |

### Special Tier 1 Behavior

When the *only* veto comes from human override (no other dimension vetoed), Tier 1 returns `ESCALATE` instead of `DENY`. This allows the action to be queued for human review rather than permanently rejected. Call `human.approve(action.id)` and re-evaluate to proceed.

---

## Dimension 13: Ethical Alignment

**Class**: `EthicalAlignment`
**Weight**: 2.0 (highest, tied with Human Override)
**Can Veto**: Yes

Does the action meet ethical constraints?

### What It Checks

Runs a series of ethical rules against the action. Each rule is a callable that returns `(passes: bool, reason: str)`. All rules must pass.

### Configuration

```python
ethics = runtime.registry.get("ethical_alignment")

# No actions that affect minors' data
ethics.add_rule(lambda a, c: (
    "minor" not in a.target.lower(),
    "Cannot operate on minor data without special authorization"
))

# No automated termination actions
ethics.add_rule(lambda a, c: (
    a.action_type != "terminate_account",
    "Account termination requires human decision"
))
```

### Scoring

| Condition | Score | Veto |
|-----------|-------|------|
| No rules configured | 0.8 | No |
| All rules pass | 1.0 | No |
| Any rule fails | 0.0 | **Yes** |

### Notes

- Ethical rules are hard constraints. They cannot be overridden by high trust, other dimension scores, or Tier 3 deliberation. A veto is final.
- The 0.8 default when no rules are configured reflects uncertainty. Configure explicit rules for production.
- Rules are evaluated in order. The first failure stops evaluation (short-circuit).

---

## Dimension Weight Summary

Ordered by influence on the UCS:

| Weight | Dimensions |
|--------|-----------|
| **2.0** | Human Override, Ethical Alignment |
| **1.5** | Scope Compliance, Authority Verification, Incident Detection |
| **1.4** | Isolation Integrity |
| **1.3** | Cascading Impact |
| **1.2** | Resource Boundaries, Stakeholder Impact |
| **1.0** | Behavioral Consistency |
| **0.8** | Temporal Compliance |
| **0.7** | Precedent Alignment |
| **0.6** | Transparency |

The weight hierarchy reflects governance priorities:
- Human authority and ethical constraints dominate
- Security boundaries (scope, authority, incident, isolation) are heavily weighted
- Impact assessment (cascading, stakeholder, resources) has moderate weight
- Behavioral and temporal analysis provides context
- Precedent and transparency are informational

## Custom Dimensions

To add a custom dimension, subclass `GovernanceDimension`:

```python
from nomotic.dimensions import GovernanceDimension
from nomotic.types import Action, AgentContext, DimensionScore

class ComplianceCheck(GovernanceDimension):
    name = "compliance_check"
    weight = 1.5
    can_veto = True

    def evaluate(self, action: Action, context: AgentContext) -> DimensionScore:
        if violates_regulation(action):
            return DimensionScore(
                dimension_name=self.name,
                score=0.0,
                weight=self.weight,
                veto=True,
                reasoning="Regulatory violation detected",
            )
        return DimensionScore(
            dimension_name=self.name,
            score=1.0,
            weight=self.weight,
            reasoning="Compliant",
        )

runtime.registry.register(ComplianceCheck())
```
