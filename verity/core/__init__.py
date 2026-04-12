"""
verity.core
===========
The Verity engine core. Import from here, not from submodules.
"""

from verity.core.types import (
    DEFAULT_DECAY_PARAMETERS,
    # REMEMBER
    AuditEvent,
    # Enumerations
    AuditEventType,
    # Type aliases
    AuditRef,
    CheckpointDecision,
    # GOVERN
    CheckpointResult,
    Completeness,
    # Consent
    ConsentRecord,
    ConsentRef,
    ContextBundle,
    # Request / response
    ContextRequest,
    DataClassification,
    # Scientific types
    DecayParameters,
    EntityId,
    ExclusionNote,
    ModuleId,
    # Module interface
    ModuleManifest,
    ProposedAction,
    ProvenanceRef,
    # RELATE
    RelateResult,
    SessionId,
    # Session
    SessionState,
    ThreeAxisWeight,
    TrustSource,
    # Facts and edges
    TypedFact,
    WeightedEdge,
)

__all__ = [
    # Type aliases
    "AuditRef",
    "ConsentRef",
    "EntityId",
    "ModuleId",
    "ProvenanceRef",
    "SessionId",
    # Enumerations
    "AuditEventType",
    "CheckpointDecision",
    "Completeness",
    "DataClassification",
    "TrustSource",
    # Scientific types
    "DecayParameters",
    "ThreeAxisWeight",
    "DEFAULT_DECAY_PARAMETERS",
    # Module interface
    "ModuleManifest",
    # Consent
    "ConsentRecord",
    # Facts and edges
    "TypedFact",
    "WeightedEdge",
    # Request / response
    "ContextRequest",
    "ContextBundle",
    "ExclusionNote",
    # GOVERN
    "CheckpointResult",
    "ProposedAction",
    # RELATE
    "RelateResult",
    # REMEMBER
    "AuditEvent",
    # Session
    "SessionState",
]
