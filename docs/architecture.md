# Architecture

## The Problem

Most AI governance operates on the wrong timescale.

Pre-execution governance evaluates requests before they begin. Post-execution governance reviews outcomes after they complete. The actual runtime — where actions occur, consequences accumulate, and failures cascade — remains ungoverned.

This is the temporal gap. Agents act in milliseconds. Humans review in minutes or hours. Governance that operates at human speed cannot govern systems that operate at machine speed.

Nomotic closes this gap by making governance a runtime participant, not a gatekeeper at the entrance or an auditor at the exit.

## Design Principles

### Governance Has Mechanical Authority Over Execution

Not advisory input. Not logging and alerting. Actual authority to stop an action mid-stream, before it completes, before consequences become irreversible.

This means the governance layer operates in parallel with execution. It has real-time visibility into what the agent is doing. It has the technical capability to intervene. And it has the authority to override.

### Simultaneous Multi-Dimensional Evaluation

Every action is assessed across 13 independent dimensions at once. No single dimension captures the full governance picture. Scope compliance says whether the agent *can* do something. Ethical alignment says whether it *should*. Cascading impact says what happens *if* it does. All 13 must agree.

### Three-Tier Decision Cascade

Not every action needs the same depth of evaluation:

- **Tier 1 (Deterministic)**: Binary pass/fail on hard boundaries. Microsecond decisions. A scope violation doesn't need weighted analysis — it needs a veto.
- **Tier 2 (Weighted)**: The Unified Confidence Score combines all dimension signals with weights, trust modulation, and contextual factors. Handles the bulk of decisions.
- **Tier 3 (Deliberative)**: For the ambiguous middle. When Tier 2 can't decide, Tier 3 applies deeper analysis — trust trajectory, historical precedent, worst-case assessment.

The cascade means simple cases are decided instantly, complex cases get proportionally more analysis, and nothing falls through without a decision.

### Trust is Earned, Not Granted

Trust starts at baseline (0.5) and moves based on observed behavior. The asymmetry is intentional:

- Building trust: +0.01 per success
- Losing trust: -0.05 per violation
- Recovery from one violation requires five successful actions

This means a new agent must demonstrate consistent good behavior before governance relaxes. An agent with a history of violations faces increasingly strict governance. Trust is the feedback loop that makes governance adaptive rather than static.

### Interruption Enables Deployment

The counterintuitive insight: organizations deploy AI more confidently when they can stop it. The ability to intervene mid-execution reduces the trust required to begin execution. You don't need to believe the agent will handle every situation correctly — you need to believe you can catch and stop it when it doesn't.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GovernanceRuntime                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Dimension    │  │   UCS    │  │  Three   │  │   Interrupt   │  │
│  │  Registry     │  │  Engine  │  │  Tiers   │  │   Authority   │  │
│  │  (13 dims)    │  │          │  │  1/2/3   │  │               │  │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│         │               │             │                 │          │
│         ▼               ▼             ▼                 ▼          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Trust Calibrator                          │   │
│  │              (continuous feedback loop)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### GovernanceRuntime

The single entry point. Every action passes through `runtime.evaluate()`. The runtime orchestrates the full pipeline:

1. Apply time decay to the agent's trust profile
2. Evaluate all 13 dimensions simultaneously
3. Pass dimension scores through Tier 1 (deterministic gate)
4. If Tier 1 doesn't decide, compute the UCS and pass through Tier 2
5. If Tier 2 doesn't decide, pass through Tier 3 (deliberation)
6. Record the verdict and update trust calibration
7. For approved actions, provide execution handles with interrupt capability

The runtime holds references to all components and coordinates their interaction. It also maintains per-agent action history (capped at 1,000 records per agent) for precedent analysis and pattern detection.

### DimensionRegistry

Manages the 13 governance dimensions. The `create_default()` factory instantiates all dimensions with default configuration. Dimensions can be individually retrieved via `registry.get("dimension_name")` for configuration.

`evaluate_all()` runs every dimension against the action and returns a list of `DimensionScore` objects. Dimensions are evaluated independently — they don't see each other's scores.

### UCS Engine

The Unified Confidence Score is not a simple weighted average. The computation:

1. **Check vetoes**: Any veto forces UCS to 0.0.
2. **Weighted average**: `sum(score * weight * confidence) / sum(weight * confidence)`. Each dimension's weight reflects its governance importance. Each dimension's confidence reflects how certain it is about its assessment.
3. **Trust modulation**: Shifts the score based on agent trust. A trust influence of 0.2 means trust can move the score by ±10% (0.2 * distance from 0.5).
4. **Floor drag**: A very low individual score (below 0.2) drags the overall UCS down by `(0.2 - min_score) * 0.3`. One badly-scoring dimension cannot be completely averaged away.
5. **Clamping**: Final result bounded to [0.0, 1.0].

The floor drag is a safety mechanism. Without it, twelve high scores could mask one dangerous score. The drag ensures that extreme governance concerns are felt in the final number even when other dimensions are fine.

### Three-Tier Cascade

**Tier 1: TierOneGate**

Checks only for vetoes. Any dimension that has `can_veto=True` and returns `veto=True` in its score triggers an immediate decision. No scoring, no weighing.

Special case: if the only veto comes from `human_override`, the verdict is `ESCALATE` rather than `DENY`. This allows the action to be approved by a human rather than permanently rejected.

**Tier 2: TierTwoEvaluator**

Takes the UCS and compares it against two thresholds:
- UCS >= 0.7 (allow threshold) → `ALLOW`
- UCS <= 0.3 (deny threshold) → `DENY`
- Between 0.3 and 0.7 → ambiguous, pass to Tier 3

The gap between thresholds is the ambiguity zone. Actions that land here are neither clearly safe nor clearly dangerous. They need deeper analysis.

**Tier 3: TierThreeDeliberator**

Handles the ambiguous cases with contextual reasoning:

1. Run custom deliberators (if registered). These are application-specific logic.
2. High trust (>0.7) + borderline UCS (>0.5) → `ALLOW`. An agent with a strong track record gets the benefit of the doubt.
3. Low trust (<0.4) → `ESCALATE`. An agent without a track record (or with a bad one) gets human review.
4. Any critical dimension (weight >= 1.3) scoring below 0.4 → `MODIFY`. The action can proceed but with reduced scope and confirmation required.
5. Default → `ALLOW`. If nothing is clearly wrong and Tier 3 can't find a reason to deny, the action proceeds.

### InterruptAuthority

The enforcement mechanism. The interrupt authority:

- **Registers** actions when execution begins, returning an `ExecutionHandle`
- **Tracks** active executions by action ID, agent ID, and workflow ID
- **Interrupts** at four granularities: single action, all actions by an agent, all actions in a workflow, or everything globally
- **Executes rollback** functions when available
- **Records** interrupt history for auditing
- **Runs monitors** — continuous governance checks during execution

The `ExecutionHandle` is the mechanical link between governance and execution. The execution layer must cooperate by calling `handle.check_interrupt()` at safe points. This cooperative model means:

- Execution retains control of *when* interrupts take effect (at safe checkpoints)
- Governance retains control of *whether* interrupts happen (the authority to signal)
- The system handles partial completion gracefully (rollback + state)

The alternative — forcibly terminating execution — creates state corruption. The cooperative model is a deliberate design choice.

### TrustCalibrator

The adaptive feedback loop. Updates trust profiles based on three events:

1. **Governance verdict**: Denied actions decrease trust. Allowed actions increase it slightly.
2. **Action completion**: Successful completion reinforces trust. Interruption decreases it.
3. **Time decay**: Idle trust drifts toward baseline. A once-trusted agent that hasn't been active gradually returns to neutral.

Per-dimension trust is also tracked. An agent that consistently triggers concerns on one dimension (e.g., cascading impact) will have its trust lowered specifically for that dimension, even if other dimensions are fine.

## Data Flow

### Evaluation Flow

```
Action + AgentContext
    │
    ├──▶ TrustCalibrator.apply_time_decay()
    │       │
    │       ▼
    │    Updated TrustProfile
    │
    ├──▶ DimensionRegistry.evaluate_all()
    │       │
    │       ▼
    │    13 DimensionScores
    │
    ├──▶ TierOneGate.evaluate(scores)
    │       │
    │       ├── Veto found ──▶ DENY/ESCALATE (verdict recorded)
    │       │
    │       ▼ No veto
    │
    ├──▶ UCSEngine.compute(scores, trust)
    │       │
    │       ▼
    │    UCS float
    │
    ├──▶ TierTwoEvaluator.evaluate(scores, ucs)
    │       │
    │       ├── UCS >= 0.7 ──▶ ALLOW (verdict recorded)
    │       ├── UCS <= 0.3 ──▶ DENY (verdict recorded)
    │       │
    │       ▼ Ambiguous (0.3 < UCS < 0.7)
    │
    └──▶ TierThreeDeliberator.evaluate(scores, ucs)
            │
            ▼
         ALLOW/DENY/MODIFY/ESCALATE (verdict recorded)
```

### Execution Flow

```
Approved Action
    │
    ├──▶ InterruptAuthority.register_execution()
    │       │
    │       ▼
    │    ExecutionHandle
    │
    ├──▶ Execution begins
    │       │
    │       ├── handle.check_interrupt() at each safe point
    │       │       │
    │       │       ├── Not interrupted ──▶ Continue
    │       │       │
    │       │       └── Interrupted ──▶ Stop + rollback
    │       │
    │       ▼
    │    Execution completes
    │
    └──▶ GovernanceRuntime.complete_execution()
            │
            ├── InterruptAuthority.complete_execution()
            ├── TrustCalibrator.record_completion()
            └── Action history updated
```

### Interrupt Flow

```
Governance detects problem
    │
    ├──▶ InterruptAuthority.interrupt(action_id, scope)
    │       │
    │       ├── Resolve targets (action/agent/workflow/global)
    │       │
    │       ├── For each target:
    │       │       ├── Signal interrupt (set threading.Event)
    │       │       ├── Execute rollback if available
    │       │       └── Record interrupt
    │       │
    │       ▼
    │    Execution layer sees interrupt at next check_interrupt()
    │
    └──▶ TrustCalibrator.record_completion(interrupted=True)
            │
            ▼
         Trust decreased (-0.03)
```

## Design Decisions

### Why 13 Dimensions?

Fewer dimensions collapse distinct concerns into ambiguous aggregates. "Is this action safe?" is not one question — it's at least 13: Is it in scope? Is it authorized? Does it respect resource limits? Is it consistent with behavior? What are the downstream effects? Who is affected? Does it match incident patterns? Are boundaries maintained? Is the timing right? Is it consistent with precedent? Is it transparent? Does a human need to review? Is it ethical?

Collapsing these into fewer dimensions forces tradeoffs between concerns that should be independently assessed. A scope-compliant action can still violate ethical constraints. A well-timed action can still have unacceptable cascading impact.

### Why Vetoes?

Weighted scoring alone has a fundamental flaw: enough high scores can mask a critical failure. If 12 dimensions score 1.0 and one scores 0.0, a weighted average might produce 0.92 — well above the allow threshold. But that zero-score dimension might represent a hard ethical constraint.

Vetoes solve this. Certain dimensions have absolute authority: scope compliance, authority verification, ethical alignment, human override. Their "no" is final regardless of what other dimensions say.

### Why Cooperative Interruption?

The interrupt authority signals interrupts through `threading.Event`. The execution layer must call `handle.check_interrupt()` to see the signal. Why not forcibly terminate?

1. **State management**: Forcible termination leaves partial state. A database write interrupted mid-transaction, a file half-written, a message partially sent. The cooperative model lets execution reach a safe checkpoint.
2. **Rollback**: The execution layer knows what to undo. The governance layer knows *that* to undo but not *how*. The rollback function bridges this gap.
3. **Granularity**: Different checkpoints in execution have different costs. Some can be interrupted cheaply, others are in critical sections. The execution layer knows which is which.

The tradeoff is latency. A misbehaving execution layer that never calls `check_interrupt()` cannot be stopped cooperatively. For this case, the system tracks execution handles and their `started_at` timestamps — monitoring can detect stalled executions and escalate to process-level intervention.

### Why Asymmetric Trust?

Building trust takes 5x longer than losing it. One violation costs as much trust as five successes earn. This reflects reality:

- One security breach can outweigh years of clean operation
- One data leak can destroy trust that took thousands of successful transactions to build
- Recovery from governance failure requires disproportionate evidence of improvement

The alternative — symmetric trust — treats mistakes and successes equally. This leads to oscillating trust profiles that don't reflect the actual cost of governance failures.

### Why Zero Dependencies?

Governance is infrastructure. It runs in the critical path of every action. External dependencies add:

- Supply chain risk (compromised dependency = compromised governance)
- Startup latency
- Version conflicts with the host application
- Failure modes that governance cannot control

The framework uses only Python standard library. It runs anywhere Python 3.11 runs.
