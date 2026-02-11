# Configuration Reference

Every tunable parameter in the Nomotic governance framework, with defaults and explanations.

## RuntimeConfig

Top-level configuration passed to `GovernanceRuntime`.

```python
from nomotic.runtime import GovernanceRuntime, RuntimeConfig
from nomotic.trust import TrustConfig

runtime = GovernanceRuntime(config=RuntimeConfig(
    allow_threshold=0.7,
    deny_threshold=0.3,
    trust_influence=0.2,
    trust_config=TrustConfig(...),
    max_history_per_agent=1000,
))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_threshold` | float | 0.7 | UCS at or above this value → `ALLOW` at Tier 2 |
| `deny_threshold` | float | 0.3 | UCS at or below this value → `DENY` at Tier 2 |
| `trust_influence` | float | 0.2 | How much agent trust shifts the UCS (0.0 = no effect, 1.0 = trust dominates) |
| `trust_config` | TrustConfig | (see below) | Trust calibration parameters |
| `max_history_per_agent` | int | 1000 | Maximum action records stored per agent before oldest are dropped |

### Threshold Tuning

The gap between `allow_threshold` and `deny_threshold` defines the ambiguity zone that falls through to Tier 3.

| Setting | Effect |
|---------|--------|
| Narrow gap (e.g., 0.55/0.45) | More actions decided at Tier 2, fewer at Tier 3. Faster but less nuanced. |
| Wide gap (e.g., 0.8/0.2) | More actions fall to Tier 3 for deliberation. Slower but more careful. |
| Equal (e.g., 0.5/0.5) | No ambiguity zone. Everything is either allowed or denied at Tier 2. |
| Inverted (allow > deny always) | Normal operation. Allow threshold must be greater than deny threshold. |

---

## TrustConfig

Configuration for the trust calibration system.

```python
from nomotic.trust import TrustConfig

config = TrustConfig(
    success_increment=0.01,
    violation_decrement=0.05,
    interrupt_decrement=0.03,
    baseline_trust=0.5,
    decay_rate=0.01,
    min_trust=0.05,
    max_trust=0.95,
    ramp_up_actions=100,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `success_increment` | float | 0.01 | Trust increase per successful action (verdict or completion) |
| `violation_decrement` | float | 0.05 | Trust decrease per governance violation (denied action) |
| `interrupt_decrement` | float | 0.03 | Trust decrease per interrupted action |
| `baseline_trust` | float | 0.5 | Starting trust for new agents; idle trust decays toward this |
| `decay_rate` | float | 0.01 | Trust drift toward baseline per hour of inactivity |
| `min_trust` | float | 0.05 | Absolute trust floor (trust never drops below this) |
| `max_trust` | float | 0.95 | Absolute trust ceiling (trust never exceeds this) |
| `ramp_up_actions` | int | 100 | Reference: number of successes to reach max from baseline |

### Trust Dynamics

**Building trust**: At the default `success_increment` of 0.01, it takes 45 successful actions to go from baseline (0.5) to max (0.95).

**Losing trust**: At the default `violation_decrement` of 0.05, it takes 9 violations to go from baseline (0.5) to min (0.05).

**Recovery ratio**: 5:1. Each violation requires 5 successful actions to recover the same trust.

**Trust events and their effects**:

| Event | Trust Change | When |
|-------|-------------|------|
| Verdict: ALLOW | +0.01 | After governance evaluation |
| Verdict: DENY | -0.05 | After governance evaluation |
| Completion: success | +0.005 | After action executes |
| Completion: interrupted | -0.03 | After action is interrupted |
| Time decay | ±0.01/hour toward baseline | Before each evaluation |

**Per-dimension trust** is also tracked:
- On DENY: dimensions that vetoed or scored < 0.3 lose 0.05 trust
- On ALLOW: dimensions that scored > 0.7 gain 0.005 trust

### Tuning Trust

| Goal | Adjustment |
|------|------------|
| Faster trust building | Increase `success_increment` |
| Harsher violation penalty | Increase `violation_decrement` |
| More forgiving of interrupts | Decrease `interrupt_decrement` |
| Higher starting trust | Increase `baseline_trust` |
| Faster trust decay when idle | Increase `decay_rate` |
| Allow full trust to be reached | Set `max_trust` to 1.0 |
| Never fully untrust | Increase `min_trust` |

---

## UCS Engine

```python
from nomotic.ucs import UCSEngine

engine = UCSEngine(trust_influence=0.2)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trust_influence` | float | 0.2 | How much agent trust modulates the UCS. At 0.2, trust can shift the score by ±10% (0.2 * distance from neutral). |

### UCS Computation Details

The UCS is computed in four stages:

**Stage 1: Veto Check**
Any dimension with `veto=True` → UCS = 0.0 immediately.

**Stage 2: Weighted Average**
```
base = sum(score_i * weight_i * confidence_i) / sum(weight_i * confidence_i)
```

**Stage 3: Trust Modulation**
```
delta = (trust.overall_trust - 0.5) * trust_influence
modulated = base + delta
```

At default `trust_influence=0.2`:
- Trust 0.0 → shifts UCS by -0.10
- Trust 0.5 → no shift
- Trust 1.0 → shifts UCS by +0.10

**Stage 4: Floor Drag**
```
if min_score < 0.2:
    drag = (0.2 - min_score) * 0.3
    modulated -= drag
```

If the lowest individual dimension score is 0.1, the drag is `(0.2 - 0.1) * 0.3 = 0.03`. If the lowest is 0.0, the drag is `0.2 * 0.3 = 0.06`.

This prevents high scores from masking one dangerously low score.

---

## Resource Limits

```python
from nomotic.dimensions import ResourceLimits

limits = ResourceLimits(
    max_actions_per_minute=60,
    max_concurrent_actions=10,
    max_cost_per_action=float("inf"),
    max_total_cost=float("inf"),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_actions_per_minute` | int | 60 | Maximum actions in any rolling 60-second window |
| `max_concurrent_actions` | int | 10 | Maximum simultaneously executing actions |
| `max_cost_per_action` | float | Infinity | Maximum cost of a single action (from `action.parameters["cost"]`) |
| `max_total_cost` | float | Infinity | Maximum cumulative cost across all actions |

---

## Tier 3 Deliberation Thresholds

These are hardcoded in `TierThreeDeliberator` and govern its built-in decision logic:

| Threshold | Value | Effect |
|-----------|-------|--------|
| High trust cutoff | 0.7 | Trust above this → benefit of the doubt for borderline UCS |
| Borderline UCS cutoff | 0.5 | UCS above this (with high trust) → ALLOW |
| Low trust cutoff | 0.4 | Trust below this → ESCALATE |
| Critical dimension weight | 1.3 | Dimensions with weight >= this are "critical" |
| Critical score threshold | 0.4 | Critical dimensions scoring below this → MODIFY |

To override Tier 3 logic, add a custom deliberator:

```python
tier3 = runtime.tier_three

def my_deliberator(action, context, scores, ucs):
    # Return a Verdict to override, or None to fall through
    if is_special_case(action):
        return Verdict.ALLOW
    return None

tier3.add_deliberator(my_deliberator)
```

Custom deliberators run before built-in logic. The first to return a `Verdict` wins.

---

## Dimension Weights

All weights can be changed by accessing the dimension instance:

```python
dim = runtime.registry.get("cascading_impact")
dim.weight = 2.0  # Increase cascading impact's influence
```

Default weights:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| human_override | 2.0 | Human authority is paramount |
| ethical_alignment | 2.0 | Ethics are non-negotiable |
| scope_compliance | 1.5 | Fundamental access control |
| authority_verification | 1.5 | Specific authorization |
| incident_detection | 1.5 | Security pattern matching |
| isolation_integrity | 1.4 | Containment enforcement |
| cascading_impact | 1.3 | Downstream risk assessment |
| resource_boundaries | 1.2 | Resource protection |
| stakeholder_impact | 1.2 | Who gets affected |
| behavioral_consistency | 1.0 | Baseline behavioral check |
| temporal_compliance | 0.8 | Timing constraints |
| precedent_alignment | 0.7 | Historical consistency |
| transparency | 0.6 | Audit readiness |

---

## Human Override Thresholds

```python
human = runtime.registry.get("human_override")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `_trust_threshold` | 0.3 | Agents with trust below this require human approval for all actions |

---

## Full Example: Production Configuration

```python
from nomotic import GovernanceRuntime
from nomotic.dimensions import ResourceLimits
from nomotic.runtime import RuntimeConfig
from nomotic.trust import TrustConfig

runtime = GovernanceRuntime(config=RuntimeConfig(
    # Tighter thresholds: less ambiguity
    allow_threshold=0.75,
    deny_threshold=0.35,
    # Trust has moderate influence
    trust_influence=0.15,
    # Conservative trust calibration
    trust_config=TrustConfig(
        success_increment=0.005,     # Slower trust building
        violation_decrement=0.08,    # Harsher penalties (16:1 ratio)
        interrupt_decrement=0.05,    # Interrupts are serious
        baseline_trust=0.3,          # Start with less trust
        min_trust=0.01,              # Near-zero floor
        max_trust=0.90,              # Can't reach full trust
    ),
    max_history_per_agent=5000,
))

# Scope: agents can only do what they're explicitly allowed to do
scope = runtime.registry.get("scope_compliance")
scope.configure_agent_scope("order-agent", {"read_order", "update_status", "query_inventory"})
scope.configure_agent_scope("comms-agent", {"send_email", "send_sms", "read_template"})

# Isolation: agents can only access their own resources
isolation = runtime.registry.get("isolation_integrity")
isolation.set_boundaries("order-agent", {"db_orders", "db_inventory"})
isolation.set_boundaries("comms-agent", {"db_templates", "email_service", "sms_service"})

# Resource limits
resources = runtime.registry.get("resource_boundaries")
resources._limits = ResourceLimits(
    max_actions_per_minute=30,
    max_concurrent_actions=5,
    max_cost_per_action=50.0,
    max_total_cost=5000.0,
)

# Temporal: deployments only during business hours, 30-min cooldown
temporal = runtime.registry.get("temporal_compliance")
temporal.set_time_window("deploy", start_hour=9, end_hour=17)
temporal.set_min_interval("deploy", seconds=1800)

# Human override for sensitive actions
human = runtime.registry.get("human_override")
human.require_human_for("delete_account", "refund_payment", "deploy")

# Sensitive targets
stakeholder = runtime.registry.get("stakeholder_impact")
stakeholder.mark_sensitive("payment_gateway", "user_pii", "production_db")

# Ethical rules
ethics = runtime.registry.get("ethical_alignment")
ethics.add_rule(lambda a, c: (
    a.parameters.get("amount", 0) <= 10000,
    "Transactions over $10,000 require manual processing"
))
ethics.add_rule(lambda a, c: (
    "test" not in a.target.lower() or a.action_type == "read",
    "Non-read actions on test resources are blocked in production"
))
```
