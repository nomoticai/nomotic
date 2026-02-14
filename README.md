# Nomotic

Runtime governance framework for agentic AI. Laws for agents, enforced continuously.

Nomotic provides the architectural layer between "the agent can act" and "the agent should act." It evaluates every action across 13 governance dimensions simultaneously, governs agent reasoning before action occurs, and maintains mechanical authority to interrupt actions mid-execution.

Most governance frameworks operate before or after execution. Nomotic operates *during* execution. If you cannot stop it, you do not control it.

## What Nomotic Does

Nomotic governs AI agents at every level:

**What agents are allowed to do** — scope compliance, authority envelopes, explicit permission boundaries.

**What agents actually do** — behavioral fingerprints, drift detection, pattern monitoring across thousands of actions.

**How agents reason about what to do** — the Nomotic Protocol: structured reasoning artifacts that externalize agent deliberation in a governable format.

**Whether reasoning holds up** — structural evaluation of reasoning completeness, authority claims, decision-action alignment, uncertainty calibration.

**Proof that governance occurred** — signed JWT governance tokens, full audit trails, configuration provenance, accountability chains tracing every decision to human authority.

**Contextual awareness** — ten types of situational context that adjust governance posture based on workflow state, operational conditions, delegation chains, and environmental signals.

**Workflow-level governance** — dependency analysis, consequence projection, compound authority detection, and cumulative risk tracking across multi-step agent workflows.

**Ethical transparency** — outcome equity analysis, bias detection in governance rules, ethical reasoning evaluation, and cross-dimensional signal detection. Nomotic surfaces patterns. Organizations define criteria. Humans make judgment calls.

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
# interruption authority, trust calibration, audit trail, contextual
# modifier, and workflow governor are initialized.
runtime = GovernanceRuntime()

# Configure what the agent is allowed to do
runtime.configure_scope(
    agent_id="agent-1",
    scope={"read", "write", "query"},
    actor="admin@acme.com",
    reason="Initial agent deployment",
)

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

## The Nomotic Protocol

The Nomotic Protocol makes agent reasoning visible, structured, and governable. Agents externalize their reasoning as structured artifacts. Governance evaluates the reasoning before action occurs. A signed token proves evaluation took place.

```python
from nomotic import (
    ReasoningArtifact,
    ProtocolEvaluator,
    GovernanceToken,
)

# Agent produces a structured reasoning artifact
artifact = ReasoningArtifact(
    identity={"agent_id": "cs-agent-47", "envelope_id": "env-returns-gold"},
    task={
        "goal": "Process return request for order #ORD-88421",
        "origin": "user_request",
        "constraints_identified": [
            {"type": "policy", "description": "Standard return limit is $500",
             "source": "policy://returns/standard-limit"},
            {"type": "authority", "description": "Gold-tier authority extends to $1500",
             "source": "envelope://env-returns-gold"},
        ],
    },
    reasoning={
        "factors": [
            {"id": "f1", "type": "constraint", "description": "Amount exceeds standard limit",
             "source": "policy://returns/standard-limit",
             "assessment": "Conditional authority required",
             "influence": "decisive", "confidence": 1.0},
            {"id": "f2", "type": "context", "description": "Customer is Gold tier, $127K lifetime spend",
             "source": "data://customer/profile",
             "assessment": "Qualifies for conditional authority",
             "influence": "decisive", "confidence": 0.99},
        ],
        "alternatives_considered": [
            {"method": "deny", "context": "Cite standard $500 limit",
             "reason_rejected": "Customer qualifies for Gold-tier exception"},
            {"method": "escalate", "context": "Route to human reviewer",
             "reason_rejected": "Amount within conditional authority envelope"},
        ],
    },
    decision={
        "intended_action": {"method": "approve", "target": "order/ORD-88421",
                           "context": "Gold-tier return exception",
                           "parameters": {"amount": 800.00}},
        "justifications": [
            {"factor_id": "f1", "explanation": "Amount requires conditional authority"},
            {"factor_id": "f2", "explanation": "Gold tier activates conditional envelope"},
        ],
        "authority_claim": {"envelope_type": "conditional",
                           "conditions_met": ["Customer tier: Gold", "Amount within $500-$1500"]},
    },
    uncertainty={
        "unknowns": [{"description": "Product condition unverified",
                      "impact": "Does not affect authority at this amount"}],
        "assumptions": [{"description": "Customer tier data is current",
                        "basis": "Real-time profile system",
                        "risk_if_wrong": "Conditional authority may not apply"}],
        "overall_confidence": 0.88,
    },
)

# Governance evaluates the reasoning
evaluator = ProtocolEvaluator(runtime=runtime, signing_key="your-secret-key")
response = evaluator.evaluate_full(artifact)

print(f"Verdict: {response.verdict}")  # PROCEED, REVISE, ESCALATE, or DENY
print(f"Token: {response.token}")       # Signed JWT governance token
print(f"Completeness: {response.assessment.completeness.score}")
print(f"Alignment: {response.assessment.alignment.score}")
```

The governance token is a signed JWT that execution environments validate before permitting actions — the same pattern as OAuth tokens for authorization:

```
Agent reasons → Submits artifact → Governance evaluates → Token issued
                                                              ↓
                                         Agent acts with token attached
                                                              ↓
                                      Execution environment validates token
```

### Method Taxonomy

Every action in the protocol is identified by a standardized **method** — a single word that preserves agent intent throughout the governance lifecycle. 84 methods across 10 categories:

| Category | Methods | Governance Profile |
|----------|---------|-------------------|
| Data | `query`, `read`, `write`, `update`, `delete`, `archive`, `restore`, `export`, `import` | Standard data governance |
| Retrieval | `fetch`, `search`, `find`, `scan`, `filter`, `extract`, `pull` | Lower weight, primarily audit |
| Decision | `approve`, `deny`, `escalate`, `recommend`, `classify`, `prioritize`, `evaluate`, `validate`, `check`, `rank`, `predict` | Elevated — agent making determinations |
| Communication | `notify`, `request`, `respond`, `reply`, `broadcast`, `subscribe`, `publish`, `send`, `call` | Moderate, stakeholder emphasis |
| Orchestration | `schedule`, `assign`, `delegate`, `invoke`, `retry`, `cancel`, `pause`, `resume`, `route`, `run`, `start`, `open` | Cascading impact emphasis |
| Transaction | `transfer`, `refund`, `charge`, `reserve`, `release`, `reconcile`, `purchase` | Highest governance weight |
| Security | `authenticate`, `authorize`, `revoke`, `elevate`, `sign`, `register` | Critical, veto-capable |
| System | `configure`, `deploy`, `monitor`, `report`, `log`, `audit`, `sync` | Operational governance |
| Generation | `generate`, `create`, `summarize`, `transform`, `translate`, `normalize`, `merge`, `link`, `map`, `make` | Moderate, ethical emphasis |
| Control | `set`, `take`, `show`, `turn`, `break`, `submit` | Context-dependent |

Methods serve governance, behavioral fingerprinting, token binding, and API design — one vocabulary, end to end.

### Protocol Flows

| Flow | When to Use | Token |
|------|------------|-------|
| **Full Deliberation** | High-stakes, irreversible, or novel actions | Single-use, 60s lifetime |
| **Summary** | Routine actions within established authority | Class-scope, 15m lifetime |
| **Post-Hoc** | Latency-sensitive, low-risk, reversible | No token (retroactive assessment) |

## The Governance Pipeline

Every action passes through this pipeline:

```
                                     ┌─────────────────────────────────┐
                                     │  Context Profile (10 types)     │
                                     │  Workflow │ Situational │ etc.  │
                                     └──────────────┬──────────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────────┐
                                     │  Contextual Modifier             │
                                     │  (adjusts weights per-evaluation)│
                                     └──────────────┬───────────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────────┐
                                     │  Workflow Governor               │
                                     │  (sequence + dependency analysis)│
                                     └──────────────┬───────────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────────┐
                                     │      13 Dimensions              │
                                     │  (evaluated simultaneously)     │
                                     └──────────────┬──────────────────┘
                                                    │
                                                    ▼
┌──────────┐    ┌────────────────────┐    ┌─────────────────────┐
│  Action   │───▶│  Trust Time Decay   │───▶│  Tier 1: Veto Gate  │──── Veto? ──▶ DENY/ESCALATE
└──────────┘    └────────────────────┘    └─────────┬───────────┘
                                                    │ No veto
                                                    ▼
                                     ┌──────────────────────────┐
                                     │  UCS Computation          │
                                     │  (weighted + trust)       │
                                     └──────────┬───────────────┘
                                                │
                                                ▼
                                     ┌──────────────────────────┐
                                     │  Tier 2: Thresholds       │──── Clear? ──▶ ALLOW/DENY
                                     │  allow=0.7  deny=0.3      │
                                     └──────────┬───────────────┘
                                                │ Ambiguous
                                                ▼
                                     ┌──────────────────────────┐
                                     │  Tier 3: Deliberation     │──▶ ALLOW/DENY/MODIFY/ESCALATE
                                     │  (trust + history)        │
                                     └──────────┬───────────────┘
                                                │
                                     ┌──────────▼───────────────┐
                                     │  Cross-Dimensional        │
                                     │  Signal Detection         │
                                     └──────────┬───────────────┘
                                                │
                                     ┌──────────▼───────────────┐
                                     │  Trust Calibration        │
                                     │  Behavioral Fingerprint   │
                                     │  Audit Trail              │
                                     └──────────────────────────┘
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
- **Drift erodes trust**: -0.002 to -0.04 scaled by drift severity and confidence
- **Trust decays**: Drifts toward baseline (0.5) when idle
- **Trust recovers**: When drift normalizes, trust stops eroding
- **Trust is bounded**: Floor at 0.05, ceiling at 0.95

Trust feeds back into every governance decision:
- Lower trust shifts UCS scores downward
- Very low trust (<0.3) triggers mandatory human review
- Trust influences Tier 3 deliberation outcomes
- Trust trajectory (rising, falling, volatile, stable) provides historical context
- Reasoning quality contributes to trust over time

## Behavioral Intelligence

Nomotic builds a behavioral fingerprint for each agent from observed governance verdicts:

- **Four distributions**: action types, targets, temporal patterns, outcomes
- **Ten archetype priors**: analyst, communicator, processor, administrator, etc.
- **JSD-based drift detection**: Jensen-Shannon divergence measures how far current behavior deviates from baseline
- **Sliding windows**: recent behavior compared against established patterns
- **Drift severity tiers**: low, moderate, high, critical — each with corresponding trust erosion rates
- **Alert generation**: drift alerts with deduplication and acknowledgment tracking

The behavioral loop is fully operational: observe → detect drift → adjust trust → trust affects governance → verdicts observed → cycle continues.

## Transparency & Accountability

Every governance decision is fully auditable:

- **Audit Trail**: Structured, queryable log of every governance decision with full dimension score snapshots, trust state, drift state, context codes, and human-readable justification narratives
- **43 Context Codes** across 10 categories for structured event classification
- **Configuration Provenance**: Every rule change records who changed what, when, why, with ticket references — Git for governance rules
- **Owner Engagement Tracking**: Monitors whether the human responsible for an agent reviews alerts, acknowledges drift, and approves overrides
- **User Activity Classification**: Tracks user interaction patterns (normal, boundary testing, suspicious) without storing raw content
- **Governance Tokens**: Signed JWTs proving that reasoning was evaluated and action was approved, with full chain from token → evaluation → reasoning → agent → human authority

## Contextual Governance

Ten types of context adjust governance posture based on the situation:

| Context Type | What It Captures |
|-------------|-----------------|
| **Workflow** | Step position, dependencies, completed/remaining steps, rollback points |
| **Situational** | Origin (user request, scheduled, agent-initiated), operational mode, urgency |
| **Relational** | Delegation chains, multi-agent coordination, compound capability detection |
| **Temporal** | Operational state, recent events, time pressure, environmental conditions |
| **Historical** | Trust trajectory, recent verdicts, scope changes, reasoning quality trends |
| **Input** | Structured summary of what was requested (privacy-preserving, no raw content) |
| **Output** | What was produced, cumulative impact, reversibility |
| **External** | Market data, threat intelligence, regulatory alerts, system status |
| **Meta** | Evaluation count, revise/denial history, governance load |
| **Feedback** | User satisfaction signals, human override history, downstream outcomes |

The **Contextual Modifier** reads context profiles and produces per-evaluation weight adjustments. Agent-initiated actions increase Human Override weight. Unresolved dependencies produce critical signals. Falling trust increases all dimension weights. Adversarial input patterns elevate security dimensions. All adjustments are temporary (per-evaluation), auditable, and configurable.

## Workflow Governance

The **Workflow Governor** tracks multi-step workflows as governance objects:

- **Dependency Graph**: Structural analysis of step relationships — requires, constrains, enables, informs
- **Consequence Projection**: Forward-looking risk assessment along dependency chains (depth-limited, not exhaustive enumeration)
- **Ordering Analysis**: Detects commitment before dependency, irreversible before verification, authority escalation sequences, resource lock chains
- **Compound Authority Detection**: Identifies when individually-authorized steps achieve an unauthorized outcome — scope assembly, authority ladders, resource aggregation
- **Cumulative Risk Tracking**: Aggregate risk across workflow steps with trajectory analysis (stable, increasing, accelerating)
- **Cross-Step Drift Detection**: Monitors whether agent behavior gradually shifts across a long workflow

## Ethical Governance Infrastructure

Nomotic provides transparency for ethical accountability, not ethical judgment. Organizations define criteria. Nomotic evaluates against them. Humans make the calls.

- **Outcome Equity Analyzer**: Examines governance decision patterns across populations. Organizations define protected attributes and disparity thresholds. The analyzer surfaces statistical disparities and proxy discrimination signals. It does not label outcomes as "biased" — it presents data for human evaluation.
- **Anonymization Policy**: Configurable rules for which attributes are visible to agents for which methods. Gender hidden during returns processing but visible during healthcare. Contextual, not blanket.
- **Bias Detection Engine**: Examines governance rules themselves for structural bias potential — neutral rules on non-uniform populations, proxy variables, threshold cliff effects, asymmetric authority.
- **Ethical Reasoning Evaluator**: Scores agent reasoning on stakeholder consideration, harm awareness, fairness consideration, alternative equity, and uncertainty honesty. Structural checks, not semantic judgment.
- **Cross-Dimensional Signal Detector**: Detects governance patterns that emerge from dimension interactions — discriminatory compliance, empathetic exploitation, invisible walls, trust-authority mismatches. Eight built-in patterns, extensible with organization-specific patterns.

## Integration

### SDK

Agents integrate governance through the SDK:

```python
from nomotic.sdk import GovernedAgent

agent = GovernedAgent(
    agent_id="agent-1",
    governance_url="https://governance.acme.com",
    certificate_path="certs/agent-1.pem",
)

# Every request goes through governance
response = agent.request("POST", "https://api.acme.com/orders", json=order_data)
```

### Middleware

Services validate governance through middleware:

```python
# FastAPI
from nomotic.middleware import NomoticFastAPI

app = FastAPI()
NomoticFastAPI(app, validation_level="local_ca")

# Flask
from nomotic.middleware import NomoticFlask

app = Flask(__name__)
NomoticFlask(app, validation_level="headers")
```

### REST API

Full governance API with endpoints for:

- Action evaluation and reasoning submission
- Governance token validation and introspection
- Audit trail queries and summaries
- Configuration provenance
- Owner engagement and user activity
- Context profile management
- Workflow assessment and dependency analysis
- Equity reports and bias assessments
- Cross-dimensional signal detection
- Schema and version negotiation

## Architecture

The governance pipeline chain:

```
Context Profile → Contextual Modifier → Workflow Governor →
  13 Dimensions (simultaneous) → UCS Engine → Three-Tier Cascade →
    Cross-Dimensional Signals → Trust Calibration →
      Behavioral Fingerprint → Audit Trail
```

For approved actions:
```
Governance Token → Execution with Interruption Rights →
  Completion → Trust Update → Fingerprint Update → Audit Record
```

For the Nomotic Protocol:
```
Agent Reasoning Artifact → Structural Evaluation →
  Ethical Reasoning Assessment → Dimensional Evaluation →
    Governance Response → Signed JWT Token →
      Execution Environment Validates Token
```

See [docs/architecture.md](docs/architecture.md) for the full architectural design.

## Four Layers of Governance

Nomotic implements a multi-modal governance model:

**Layer 1 — Deterministic Law.** Hard boundaries. Vetoes. Binary scope checks. The lines that don't move. Tier 1 of the evaluation cascade.

**Layer 2 — Structural Authority.** Conditional envelopes that define where agents exercise judgment. Authority that expands or contracts based on conditions, context, and trust.

**Layer 3 — Evaluative Governance.** Pattern-based, retrospective, trust-calibrated. Behavioral fingerprints, drift detection, and outcome equity analysis evaluate whether aggregate decisions stay within acceptable bounds.

**Layer 4 — Behavioral Governance.** Reasoning evaluation. Structured deliberation protocol. The agent's reasoning process becomes a governable, auditable artifact. Governance engages with *how agents think*, not just what they do.

## Design Principles

**Governance as Architecture** — Built in, not bolted on. Governance is a design decision, not a compliance layer.

**Pre-Action Authorization** — Governance exists before action, not after. The Nomotic Protocol evaluates reasoning before execution occurs.

**Explicit Authority Boundaries** — Authority is delegated, never assumed. Every permission traces to a human decision.

**Verifiable Trust** — Trust is earned through evidence, not assumed from capability. Building trust takes 5x longer than losing it.

**Ethical Justification** — Actions must be justifiable, not merely executable. The reasoning protocol requires agents to externalize and justify their decisions.

**Accountable Governance** — Every rule has an owner. Every authorization traces to a responsible human. The provenance log and accountability chain make this mechanical, not aspirational.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.11+
- Zero runtime dependencies

## Project Status

| Phase | Component | Status | Tests |
|-------|-----------|--------|-------|
| 1-2 | Core Runtime (13 dimensions, UCS, tiers, trust, interrupts) | ✅ Complete | ~100 |
| 3 | Integration (SDK, middleware, framework adapters) | ✅ Complete | ~50 |
| 4 | Behavioral Intelligence (fingerprints, drift, trust trajectory) | ✅ Complete | ~115 |
| 5 | Transparency & Accountability (audit, provenance, context codes) | ✅ Complete | ~150 |
| 6 | Nomotic Protocol (reasoning artifacts, governance tokens, method taxonomy) | ✅ Complete | ~120 |
| 7A | Context Profile Schema (10 context types) | ✅ Complete | ~60 |
| 7B | Contextual Modifier (per-evaluation weight adjustment) | ✅ Complete | ~70 |
| 7C | Workflow Governor (dependency analysis, consequence projection) | ✅ Complete | ~80 |
| 8 | Ethical Governance (equity, bias, cross-dimensional signals) | 🔄 In Progress | ~135 |

**Total: 1,000+ tests passing. Zero runtime dependencies.**

## Documentation

- [Architecture](docs/architecture.md) — full architectural design, component interactions, design decisions
- [13 Dimensions](docs/dimensions.md) — detailed documentation of each governance dimension
- [Configuration](docs/configuration.md) — all tunable parameters with defaults and explanations
- [Nomotic Protocol Specification](docs/nomotic-protocol-spec.md) — the protocol definition for reasoning governance
- [Reasoning Artifact Schema](schemas/reasoning-artifact.v0.1.0.schema.json) — JSON Schema for reasoning artifacts
- [Governance Response Schema](schemas/governance-response.v0.1.0.schema.json) — JSON Schema for governance responses
- [Governance Token Schema](schemas/governance-token.v0.1.0.schema.json) — JSON Schema for JWT governance tokens
- [Context Profile Schema](schemas/context-profile.v0.1.0.schema.json) — JSON Schema for context profiles

## About Nomotic AI

Nomotic AI is the governance counterpart to agentic AI. The term derives from the Greek word *nomos* (νόμος), meaning law, rule, or governance.

Where agentic AI asks: *What can this system do?*
Nomotic AI asks: *What should this system do, and under what laws?*

Agentic AI is characterized by four verbs: perceive, reason, plan, act.
Nomotic AI is characterized by four verbs: govern, authorize, trust, evaluate.

Neither is complete without the other. Actions without laws are chaos. Laws without actions are inert. Effective AI deployment requires both.

For the full conceptual foundation, see the [Nomotic AI position paper](docs/NomoticAI-Paper-v3.pdf).

## License

MIT
