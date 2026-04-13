"""
verity
======
Trustworthy context for AI agents.
Embedded semantic reasoning for regulated industries.

Quick start:
    import asyncio
    from verity import Engine

    async def main():
        engine = await Engine.start()

        async with engine.session(consent_ref="consent:abc123") as s:
            await s.ingest("patient reported fatigue and low mood")
            context = await s.context(
                query="recent observations",
                purpose="clinical_decision_support",
            )
            print(context.agent_prompt)
            print(f"Uncertainty: {context.uncertainty:.0%}")
            print(f"Audit ID:    {context.audit_id}")

    asyncio.run(main())

The full public API is available from this top-level import.
No need to import from submodules directly.
"""

from verity.core.engine import Engine, Session
from verity.core.exceptions import (
    BackendNotAvailableError,
    CanaryError,
    ClassificationError,
    ConsentExpiredError,
    ConsentRequiredError,
    ConsentRevokedError,
    CrisisBarrierError,
    EngineNotStartedError,
    GraphStoreError,
    ModuleError,
    ModuleNotFoundError,
    PrinciplesError,
    PurposeMismatchError,
    SessionClosedError,
    SignatureError,
    ValidationError,
    VerityError,
)
from verity.core.profiles import (
    DEVELOPER,
    ENTERPRISE,
    PERSONAL,
    PROFESSIONAL,
    EngineProfile,
    get_profile,
)
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

__version__ = "0.1.0"
__license__ = "Apache-2.0"
__author__  = "Verity Contributors"

__all__ = [
    # Engine
    "Engine",
    "Session",
    # Profiles
    "EngineProfile",
    "get_profile",
    "PERSONAL",
    "DEVELOPER",
    "PROFESSIONAL",
    "ENTERPRISE",
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
    # Exceptions
    "VerityError",
    "PrinciplesError",
    "SignatureError",
    "CanaryError",
    "ConsentRequiredError",
    "ConsentExpiredError",
    "ConsentRevokedError",
    "PurposeMismatchError",
    "CrisisBarrierError",
    "ValidationError",
    "ClassificationError",
    "GraphStoreError",
    "BackendNotAvailableError",
    "ModuleError",
    "ModuleNotFoundError",
    "EngineNotStartedError",
    "SessionClosedError",
    # Version
    "__version__",
]
