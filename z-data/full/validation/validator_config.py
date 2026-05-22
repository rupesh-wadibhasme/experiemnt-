"""
Configuration for OData query validator.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    allow_batch : bool
        Whether $batch operations are allowed (default: False).
    require_top_limit : bool
        Whether queries must include a $top parameter (default: True).
    enable_cost_estimation : bool
        Whether to calculate query cost scores (default: True).
    """

    # Retrieval limits
    max_retrieval_limit: int = 1000
    warn_retrieval_threshold: int = 500

    # Query complexity limits
    max_expand_depth: int = 3
    max_filter_complexity: int = 100

    # Feature flags
    allow_batch: bool = False
    require_top_limit: bool = True
    enable_cost_estimation: bool = True

    @classmethod
    def for_production(cls) -> ValidatorConfig:
        """Production configuration with strict limits."""
        return cls(
            max_retrieval_limit=500,
            warn_retrieval_threshold=250,
            max_expand_depth=2,
            max_filter_complexity=50,
            require_top_limit=True,
            allow_batch=False,
        )
