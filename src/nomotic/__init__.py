"""Nomotic: Runtime governance framework for agentic AI."""

__version__ = "0.1.0"

from nomotic.types import (
    Action,
    AgentContext,
    GovernanceVerdict,
    TrustProfile,
    Verdict,
)
from nomotic.dimensions import DimensionRegistry, GovernanceDimension
from nomotic.ucs import UCSEngine
from nomotic.tiers import TierOneGate, TierTwoEvaluator, TierThreeDeliberator
from nomotic.interrupt import InterruptAuthority, InterruptScope
from nomotic.trust import TrustCalibrator
from nomotic.runtime import GovernanceRuntime
from nomotic.keys import SigningKey, VerifyKey
from nomotic.certificate import AgentCertificate, CertStatus, CertVerifyResult, LiveVerifyResult
from nomotic.authority import CertificateAuthority
from nomotic.store import CertificateStore, MemoryCertificateStore, FileCertificateStore
from nomotic.registry import (
    ArchetypeDefinition,
    ArchetypeRegistry,
    FileOrgStore,
    MemoryOrgStore,
    OrganizationRegistry,
    OrgRegistration,
    OrgStatus,
    OrgStore,
    ValidationResult,
    ZoneValidator,
)
from nomotic.sdk import GovernedAgent, GovernedResponse, CertificateLoadError, GovernedRequestError
from nomotic.executor import GovernedToolExecutor, ExecutionResult
from nomotic.middleware import NomoticGateway, GatewayConfig, GatewayResult
from nomotic.fingerprint import BehavioralFingerprint, TemporalPattern
from nomotic.priors import ArchetypePrior, TemporalProfile, PriorRegistry
from nomotic.observer import FingerprintObserver
from nomotic.drift import DriftCalculator, DriftScore
from nomotic.window import SlidingWindow
from nomotic.monitor import DriftMonitor, DriftConfig, DriftAlert
from nomotic.trajectory import TrustTrajectory, TrustEvent
from nomotic.context import ContextCode, CODES
from nomotic.audit import AuditRecord, AuditTrail, build_justification, verify_chain
from nomotic.id_registry import AgentIdRegistry
from nomotic.revocation import create_revocation_seal, load_revocation_seal
from nomotic.audit_store import AuditStore, PersistentAuditRecord
from nomotic.provenance import ProvenanceRecord, ProvenanceLog
from nomotic.accountability import (
    OwnerActivity,
    OwnerActivityLog,
    UserActivityTracker,
    UserStats,
)
from nomotic.types import UserContext
from nomotic.protocol import (
    METHODS,
    METHOD_CATEGORIES,
    PROTOCOL_VERSION,
    Assessment,
    AuthorityClaim,
    Condition,
    Denial,
    Escalation,
    GovernanceResponseData,
    Guidance,
    IntendedAction,
    ProtocolVerdict,
    ReasoningArtifact,
    ResponseMetadata,
    validate_artifact,
)
from nomotic.token import GovernanceToken, TokenClaims, TokenValidationResult, TokenValidator
from nomotic.evaluator import (
    EthicalReasoningAssessment,
    EthicalReasoningConfig,
    EvaluatorConfig,
    PostHocAssessment,
    ProtocolEvaluator,
)
from nomotic.equity import (
    AnonymizationPolicy,
    AnonymizationRule,
    DisparityFinding,
    EquityAnalyzer,
    EquityConfig,
    EquityReport,
    EquityThreshold,
    GroupOutcome,
    ProtectedAttribute,
    ProxyAlert,
)
from nomotic.bias import (
    BiasDetector,
    GovernanceBiasReport,
    RuleBiasAssessment,
    StructuralConcern,
)
from nomotic.cross_dimensional import (
    CROSS_DIMENSIONAL_PATTERNS,
    CrossDimensionalDetector,
    CrossDimensionalReport,
    CrossDimensionalSignal,
)
from nomotic.contextual_modifier import (
    ContextConstraint,
    ContextModification,
    ContextRiskSignal,
    ContextualModifier,
    ModifierConfig,
    WeightAdjustment,
)
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
from nomotic.human_drift import (
    HumanAuditStore,
    HumanDriftCalculator,
    HumanDriftMonitor,
    HumanDriftResult,
    HumanInteractionEvent,
    HumanInteractionProfile,
)
from nomotic.context_profile import (
    CompletedStep,
    CompoundMethod,
    ContextProfile,
    ContextProfileManager,
    DelegationLink,
    Dependency,
    ExternalContext,
    ExternalSignal,
    FeedbackContext,
    FeedbackRecord,
    HistoricalContext,
    InputContext,
    MetaContext,
    OutcomeRecord,
    OutputContext,
    OutputRecord,
    OverrideRecord,
    PlannedStep,
    RecentVerdict,
    RelationalContext,
    SituationalContext,
    TemporalContext,
    TemporalEvent,
    WorkflowContext,
)

__all__ = [
    "Action",
    "AgentContext",
    "GovernanceVerdict",
    "TrustProfile",
    "Verdict",
    "DimensionRegistry",
    "GovernanceDimension",
    "UCSEngine",
    "TierOneGate",
    "TierTwoEvaluator",
    "TierThreeDeliberator",
    "InterruptAuthority",
    "InterruptScope",
    "TrustCalibrator",
    "GovernanceRuntime",
    "SigningKey",
    "VerifyKey",
    "AgentCertificate",
    "CertStatus",
    "CertVerifyResult",
    "LiveVerifyResult",
    "CertificateAuthority",
    "CertificateStore",
    "MemoryCertificateStore",
    "FileCertificateStore",
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
    "GovernedAgent",
    "GovernedResponse",
    "CertificateLoadError",
    "GovernedRequestError",
    "GovernedToolExecutor",
    "ExecutionResult",
    "NomoticGateway",
    "GatewayConfig",
    "GatewayResult",
    "BehavioralFingerprint",
    "TemporalPattern",
    "ArchetypePrior",
    "TemporalProfile",
    "PriorRegistry",
    "FingerprintObserver",
    "DriftCalculator",
    "DriftScore",
    "SlidingWindow",
    "DriftMonitor",
    "DriftConfig",
    "DriftAlert",
    "TrustTrajectory",
    "TrustEvent",
    "ContextCode",
    "CODES",
    "AuditRecord",
    "AuditTrail",
    "build_justification",
    "verify_chain",
    "AgentIdRegistry",
    "create_revocation_seal",
    "load_revocation_seal",
    "AuditStore",
    "PersistentAuditRecord",
    "ProvenanceRecord",
    "ProvenanceLog",
    "OwnerActivity",
    "OwnerActivityLog",
    "UserActivityTracker",
    "UserStats",
    "UserContext",
    # Nomotic Protocol (Phase 6)
    "METHODS",
    "METHOD_CATEGORIES",
    "PROTOCOL_VERSION",
    "Assessment",
    "AuthorityClaim",
    "Condition",
    "Denial",
    "Escalation",
    "GovernanceResponseData",
    "Guidance",
    "IntendedAction",
    "ProtocolVerdict",
    "ReasoningArtifact",
    "ResponseMetadata",
    "validate_artifact",
    "GovernanceToken",
    "TokenClaims",
    "TokenValidationResult",
    "TokenValidator",
    "EthicalReasoningAssessment",
    "EthicalReasoningConfig",
    "EvaluatorConfig",
    "PostHocAssessment",
    "ProtocolEvaluator",
    # Equity Analysis (Phase 8)
    "AnonymizationPolicy",
    "AnonymizationRule",
    "DisparityFinding",
    "EquityAnalyzer",
    "EquityConfig",
    "EquityReport",
    "EquityThreshold",
    "GroupOutcome",
    "ProtectedAttribute",
    "ProxyAlert",
    # Bias Detection (Phase 8)
    "BiasDetector",
    "GovernanceBiasReport",
    "RuleBiasAssessment",
    "StructuralConcern",
    # Cross-Dimensional Signals (Phase 8)
    "CROSS_DIMENSIONAL_PATTERNS",
    "CrossDimensionalDetector",
    "CrossDimensionalReport",
    "CrossDimensionalSignal",
    # Contextual Modifier (Phase 7B)
    "ContextConstraint",
    "ContextModification",
    "ContextRiskSignal",
    "ContextualModifier",
    "ModifierConfig",
    "WeightAdjustment",
    # Workflow Governor (Phase 7C)
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
    # Human Drift Detection (Bidirectional Oversight)
    "HumanAuditStore",
    "HumanDriftCalculator",
    "HumanDriftMonitor",
    "HumanDriftResult",
    "HumanInteractionEvent",
    "HumanInteractionProfile",
    # Context Profiles (Phase 7A)
    "CompletedStep",
    "CompoundMethod",
    "ContextProfile",
    "ContextProfileManager",
    "DelegationLink",
    "Dependency",
    "ExternalContext",
    "ExternalSignal",
    "FeedbackContext",
    "FeedbackRecord",
    "HistoricalContext",
    "InputContext",
    "MetaContext",
    "OutcomeRecord",
    "OutputContext",
    "OutputRecord",
    "OverrideRecord",
    "PlannedStep",
    "RecentVerdict",
    "RelationalContext",
    "SituationalContext",
    "TemporalContext",
    "TemporalEvent",
    "WorkflowContext",
]
