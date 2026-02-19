"""Memory configurations and schemas."""

# V2 modules - deterministic, memory-first relational reasoning
# Optional exports - importing these does not affect v1 behavior
try:
    from nanobot.memory.types_v2 import (
        TruthValue,
        RelationType,
        QueryResult,
        IngestionStatus,
        IngestionResult,
    )
    from nanobot.memory.config_v2 import (
        CONF_THRESHOLD,
        RANKING_POLICY,
        SUPERLATIVE_POLICY,
        UNKNOWN_PAIRWISE_MESSAGE,
        UNKNOWN_SUPERLATIVE_MESSAGE,
        UNKNOWN_INCONSISTENT_MESSAGE,
        MEMORY_FIRST_ENFORCED,
    )
    from nanobot.memory.relational_cache_v2 import RelationalCacheV2
    from nanobot.memory.relation_extractor_v2 import RelationExtractionEngineV2
    from nanobot.memory.memory_first_reasoner_v2 import MemoryFirstReasonerV2
    
    __all__ = [
        # V2 types
        'TruthValue',
        'RelationType',
        'QueryResult',
        'IngestionStatus',
        'IngestionResult',
        # V2 config
        'CONF_THRESHOLD',
        'RANKING_POLICY',
        'SUPERLATIVE_POLICY',
        'UNKNOWN_PAIRWISE_MESSAGE',
        'UNKNOWN_SUPERLATIVE_MESSAGE',
        'UNKNOWN_INCONSISTENT_MESSAGE',
        'MEMORY_FIRST_ENFORCED',
        # V2 modules
        'RelationalCacheV2',
        'RelationExtractionEngineV2',
        'MemoryFirstReasonerV2',
    ]
except ImportError:
    # V2 modules not available or dependencies missing
    __all__ = []

