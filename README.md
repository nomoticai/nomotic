# Nomotic

Runtime governance framework for agentic AI. Laws for agents, enforced continuously.

Nomotic provides the architectural layer between "the agent can act" and "the agent should act." It evaluates every action across 13 governance dimensions simultaneously, computes a unified confidence score, and — critically — maintains mechanical authority to interrupt actions mid-execution.

Most governance frameworks operate before or after execution. Nomotic operates *during* execution. If you cannot stop it, you do not control it.

## Quickstart

```python
from nomotic import (
    Action,
    AgentContext,
    GovernanceRuntime,
    TrustProfile,
    Verdict,
)

# Create the runtime — all 13 dimensions, three evaluation tiers,
# interruption authority, and trust calibration are initialized.
runtime = GovernanceRuntime()

# Configure what the agent is allowed to do
scope = runtime.registry.get("scope_compliance")
scope.configure_agent_scope("agent-1", {"read", "write", "query"})

# Create an action the agent wants to perform
action = Action(
    agent_id="agent-1",
    action_type="write",
    target="customer_records",
    parameters={"field": "email", "value": "new@example.com"},
)

# Create the agent's context
context = AgentContext(
    agent_id="agent-1",
    trust_profile=TrustProfile(agent_id="agent-1"),
)

# Evaluate the action through the full governance pipeline
verdict = runtime.evaluate(action, context)

print(f"Verdict: {verdict.verdict.name}")  # ALLOW, DENY, MODIFY, ESCALATE, or SUSPEND
print(f"UCS: {verdict.ucs:.3f}")           # 0.0-1.0 unified confidence
print(f"Tier: {verdict.tier}")             # Which tier decided (1, 2, or 3)
print(f"Time: {verdict.evaluation_time_ms:.1f}ms")
```

## Execution with Interruption Rights

The governance pipeline does not end at the verdict. For approved actions, the runtime provides execution handles that allow governance to intervene mid-stream.

```python
if verdict.verdict == Verdict.ALLOW:
    # Register the action for monitored execution
    handle = runtime.begin_execution(
        action,
        context,
        rollback=lambda: undo_write(action),  # Called if interrupted
    )

    # The execution layer cooperates by checking for interrupts
    for record in records_to_process:
        if handle.check_interrupt():
            break  # Governance has halted this action
        process(record)

    # On normal completion, update trust and history
    runtime.complete_execution(action.id, context)
```

Governance can interrupt at any time, from any thread:

```python
# Interrupt a single action
runtime.interrupt_action(action.id, reason="Anomaly detected in write pattern")

# Interrupt everything an agent is doing
from nomotic import InterruptScope
runtime.interrupt_action(action.id, reason="Agent compromised", scope=InterruptScope.AGENT)

# Emergency: interrupt all running actions globally
runtime.interrupt_action(action.id, reason="System-wide halt", scope=InterruptScope.GLOBAL)
```

## The Governance Pipeline

Every action passes through this pipeline:

```
                                          ┌─────────────────────────────┐
                                          │     13 Dimensions           │
                                          │  (evaluated simultaneously) │
                                          └──────────┬──────────────────┘
                                                     │
                                                     ▼
┌──────────┐    ┌────────────────────┐    ┌─────────────────────┐
│  Action   │───▶│  Trust Time Decay   │───▶│  Tier 1: Veto Gate  │──── Veto? ──▶ DENY/ESCALATE
└──────────┘    └────────────────────┘    └─────────┬───────────┘
                                                     │ No veto
                                                     ▼
                                          ┌──────────────────────┐
                                          │  UCS Computation      │
                                          │  (weighted + trust)   │
                                          └─────────┬────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Tier 2: Thresholds   │──── Clear? ──▶ ALLOW/DENY
                                          │  allow=0.7  deny=0.3  │
                                          └─────────┬────────────┘
                                                     │ Ambiguous
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Tier 3: Deliberation │──▶ ALLOW/DENY/MODIFY/ESCALATE
                                          │  (trust + history)    │
                                          └──────────────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Trust Calibration    │
                                          │  (update from verdict)│
                                          └──────────────────────┘
```

## The 13 Governance Dimensions

Every action is evaluated across all 13 dimensions simultaneously. Each dimension produces an independent score (0.0 = maximum concern, 1.0 = no concern). Dimensions with veto authority can halt an action regardless of other scores.

| # | Dimension | Weight | Veto | What It Checks |
|---|-----------|--------|------|----------------|
| 1 | Scope Compliance | 1.5 | Yes | Is the action within the agent's authorized scope? |
| 2 | Authority Verification | 1.5 | Yes | Does the agent have specific authority for this action? |
| 3 | Resource Boundaries | 1.2 | Yes | Are rate, concurrency, and cost limits respected? |
| 4 | Behavioral Consistency | 1.0 | No | Does this match the agent's established patterns? |
| 5 | Cascading Impact | 1.3 | No | What are the downstream consequences? |
| 6 | Stakeholder Impact | 1.2 | No | Who is affected and how sensitive are they? |
| 7 | Incident Detection | 1.5 | Yes | Does this match known failure or attack patterns? |
| 8 | Isolation Integrity | 1.4 | Yes | Are containment boundaries maintained? |
| 9 | Temporal Compliance | 0.8 | Yes | Is the timing appropriate? |
| 10 | Precedent Alignment | 0.7 | No | Is this consistent with past governance decisions? |
| 11 | Transparency | 0.6 | No | Is the action auditable and explainable? |
| 12 | Human Override | **2.0** | Yes | Is human intervention required? |
| 13 | Ethical Alignment | **2.0** | Yes | Does the action meet ethical constraints? |

See [docs/dimensions.md](docs/dimensions.md) for detailed documentation of each dimension.

## Trust Calibration

Trust is not binary. It is calibrated continuously:

- **Building trust is slow**: +0.01 per successful action
- **Losing trust is fast**: -0.05 per violation (5:1 ratio)
- **Interruption costs trust**: -0.03 per interrupt
- **Trust decays**: Drifts toward baseline (0.5) when idle
- **Trust is bounded**: Floor at 0.05, ceiling at 0.95

Trust feeds back into every governance decision:
- Lower trust shifts UCS scores downward
- Very low trust (<0.3) triggers mandatory human review
- Trust influences Tier 3 deliberation outcomes

See [docs/configuration.md](docs/configuration.md) for all tunable parameters.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architectural design, including:

- Why governance must operate synchronously with execution
- The temporal gap problem and how interruption rights solve it
- The trust-control relationship
- Design decisions and tradeoffs

## Requirements

- Python 3.11+
- Zero runtime dependencies

## License

MIT
