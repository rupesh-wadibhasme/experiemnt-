"""
Configuration for OData query validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .metadata_analyzer import EntityProfile


@dataclass
class ValidatorConfig:
    """
    Configuration for OData query validation rules.

    Attributes
    ----------
    max_retrieval_limit : int
        Maximum allowed value for $top parameter (default: 1000).
    warn_retrieval_threshold : int
        Issue warning if $top exceeds this value (default: 500).
    max_expand_depth : int
        Maximum allowed $expand nesting depth (default: 3).
    max_filter_complexity : int
        Maximum allowed filter expression complexity score (default: 100).
    max_cost_score : int
        Query is invalid if estimated cost exceeds this value (default: 80).
    warn_cost_threshold : int
        Issue warning if estimated cost exceeds this value (default: 60).
    allow_batch : bool
        Whether $batch operations are allowed (default: False).
    require_top_limit : bool
        Whether queries must include a $top parameter (default: True).
    enable_cost_estimation : bool
        Whether to calculate query cost scores (default: True).
    """

    # Retrieval limits (fallback when no entity profile is available)
    max_retrieval_limit: int = 1000
    warn_retrieval_threshold: int = 500

    # Query complexity limits
    max_expand_depth: int = 3
    max_filter_complexity: int = 100

    # Cost score limits
    max_cost_score: int = 80
    warn_cost_threshold: int = 60

    # Feature flags
    allow_batch: bool = False
    require_top_limit: bool = True
    enable_cost_estimation: bool = True

    # Per-entity profiles loaded from OData $metadata (optional).
    # When present, entity-specific $top limits and cost width factors
    # override the flat max_retrieval_limit / warn_retrieval_threshold above.
    entity_profiles: Dict[str, "EntityProfile"] = field(default_factory=dict)

    @classmethod
    def for_production(cls) -> ValidatorConfig:
        """Production configuration with strict limits."""
        return cls(
            max_retrieval_limit=500,
            warn_retrieval_threshold=250,
            max_expand_depth=2,
            max_filter_complexity=50,
            max_cost_score=80,
            warn_cost_threshold=60,
            require_top_limit=True,
            allow_batch=False,
        )
