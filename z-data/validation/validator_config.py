"""
Configuration for OData query validator.

Supports environment-specific settings through class instantiation
or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Set


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
    allow_count : bool
        Whether $count queries are allowed (default: True).
    allow_batch : bool
        Whether $batch operations are allowed (default: False).
    allowed_http_methods : Set[str]
        Allowed HTTP methods for OData queries (default: {"GET"}).
    blocked_operations : Set[str]
        OData operations that should be blocked (default: POST, PATCH, PUT, DELETE).
    require_top_limit : bool
        Whether queries must include a $top parameter (default: True).
    enable_cost_estimation : bool
        Whether to calculate query cost scores (default: True).
    strict_schema_validation : bool
        Whether to strictly validate entity/property names against schema (default: True).
    """
    
    # Retrieval limits
    max_retrieval_limit: int = 1000
    warn_retrieval_threshold: int = 500
    
    # Query complexity limits
    max_expand_depth: int = 3
    max_filter_complexity: int = 100
    
    # Feature flags
    allow_count: bool = True
    allow_batch: bool = False
    require_top_limit: bool = True
    enable_cost_estimation: bool = True
    strict_schema_validation: bool = True
    
    # Security settings
    allowed_http_methods: Set[str] = None
    blocked_operations: Set[str] = None
    
    def __post_init__(self):
        """Initialize default sets if not provided."""
        if self.allowed_http_methods is None:
            self.allowed_http_methods = {"GET"}
        
        if self.blocked_operations is None:
            # These operations modify data - we only allow read operations
            self.blocked_operations = {"POST", "PATCH", "PUT", "DELETE", "MERGE"}
    
    @classmethod
    def from_env(cls, prefix: str = "ODATA_VALIDATOR_") -> ValidatorConfig:
        """
        Create configuration from environment variables.
        
        Parameters
        ----------
        prefix : str
            Prefix for environment variable names (default: "ODATA_VALIDATOR_").
        
        Environment Variables
        ---------------------
        ODATA_VALIDATOR_MAX_RETRIEVAL_LIMIT : int
        ODATA_VALIDATOR_WARN_RETRIEVAL_THRESHOLD : int
        ODATA_VALIDATOR_MAX_EXPAND_DEPTH : int
        ODATA_VALIDATOR_MAX_FILTER_COMPLEXITY : int
        ODATA_VALIDATOR_REQUIRE_TOP_LIMIT : bool (0/1, true/false)
        ODATA_VALIDATOR_STRICT_SCHEMA_VALIDATION : bool
        
        Returns
        -------
        ValidatorConfig
            Configuration instance populated from environment.
        
        Examples
        --------
        >>> os.environ['ODATA_VALIDATOR_MAX_RETRIEVAL_LIMIT'] = '5000'
        >>> config = ValidatorConfig.from_env()
        >>> config.max_retrieval_limit
        5000
        """
        def get_int(key: str, default: int) -> int:
            value = os.getenv(f"{prefix}{key}")
            return int(value) if value is not None else default
        
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(f"{prefix}{key}")
            if value is None:
                return default
            return value.lower() in ("1", "true", "yes", "on")
        
        return cls(
            max_retrieval_limit=get_int("MAX_RETRIEVAL_LIMIT", 1000),
            warn_retrieval_threshold=get_int("WARN_RETRIEVAL_THRESHOLD", 500),
            max_expand_depth=get_int("MAX_EXPAND_DEPTH", 3),
            max_filter_complexity=get_int("MAX_FILTER_COMPLEXITY", 100),
            require_top_limit=get_bool("REQUIRE_TOP_LIMIT", True),
            enable_cost_estimation=get_bool("ENABLE_COST_ESTIMATION", True),
            strict_schema_validation=get_bool("STRICT_SCHEMA_VALIDATION", True),
            allow_count=get_bool("ALLOW_COUNT", True),
            allow_batch=get_bool("ALLOW_BATCH", False),
        )
    
    @classmethod
    def for_development(cls) -> ValidatorConfig:
        """
        Create a permissive configuration suitable for development environments.
        
        Returns
        -------
        ValidatorConfig
            Development configuration with relaxed limits.
        """
        return cls(
            max_retrieval_limit=5000,
            warn_retrieval_threshold=1000,
            max_expand_depth=5,
            max_filter_complexity=200,
            require_top_limit=False,
            strict_schema_validation=False,
        )
    
    @classmethod
    def for_production(cls) -> ValidatorConfig:
        """
        Create a strict configuration suitable for production environments.
        
        Returns
        -------
        ValidatorConfig
            Production configuration with strict limits.
        """
        return cls(
            max_retrieval_limit=500,
            warn_retrieval_threshold=250,
            max_expand_depth=2,
            max_filter_complexity=50,
            require_top_limit=True,
            strict_schema_validation=True,
            allow_batch=False,
        )
