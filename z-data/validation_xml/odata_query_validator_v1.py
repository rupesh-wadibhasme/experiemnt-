"""
OData query validator - v1 (simplified).

Covers two checks:
  1. Empty query guard
  2. $top limit enforcement
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .validation_result import ValidationIssue, ValidationResult, ValidationSeverity
from .validator_config import ValidatorConfig

LOGGER = logging.getLogger(__name__)


class ODataQueryValidator:
    """
    Stateless validator for OData queries.

    Parameters
    ----------
    config : ValidatorConfig, optional
        Validation configuration. If not provided, uses default settings.
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()

    def validate(
        self,
        query: str,
        entity_schemas: Optional[Dict[str, Any]] = None,
        http_method: str = "GET",
    ) -> ValidationResult:
        """
        Validate an OData query.

        Parameters
        ----------
        query : str
            The OData query string to validate.
        entity_schemas : Dict[str, Any], optional
            Unused in v1; accepted for interface compatibility.
        http_method : str, default "GET"
            Accepted for interface compatibility; unused in v1.

        Returns
        -------
        ValidationResult
            Structured validation result with issues and metadata.
        """
        issues: List[ValidationIssue] = []

        # Check 1: empty query
        if not query or query.strip() == "":
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="generation",
                message="No OData query was generated",
                suggestion="Check query generation logic and prompts",
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                query=query or "",
                validation_metadata={"step_failed": "query_generation"},
            )

        parsed_query = self._parse_query(query)

        # Check 2: $top limit
        issues.extend(self._validate_retrieval_limits(parsed_query))

        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        row_count = None
        if "$top" in parsed_query:
            try:
                row_count = int(parsed_query["$top"])
            except ValueError:
                row_count = None

        return ValidationResult(
            is_valid=is_valid,
            issues=sorted(issues, key=lambda x: (
                0 if x.severity == ValidationSeverity.CRITICAL else
                1 if x.severity == ValidationSeverity.WARNING else 2
            )),
            query=query,
            estimated_row_count=row_count,
            validation_metadata={
                "parsed_params": list(parsed_query.keys()),
            },
        )

    def _parse_query(self, query: str) -> Dict[str, str]:
        """Parse OData query string into a parameter dict."""
        if "?" in query:
            _, query_string = query.split("?", 1)
        else:
            query_string = query

        parsed = {}
        if query_string:
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    parsed[key] = value

        return parsed

    def _validate_retrieval_limits(self, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """Enforce $top presence and maximum value."""
        issues = []

        if self.config.require_top_limit and "$top" not in parsed_query:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="limits",
                message="Query does not specify $top limit",
                suggestion=f"Add $top parameter (max: {self.config.max_retrieval_limit})",
            ))

        if "$top" in parsed_query:
            try:
                top_value = int(parsed_query["$top"])

                if top_value <= 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="limits",
                        message=f"Invalid $top value: {top_value} (must be positive)",
                        query_segment=f"$top={top_value}",
                        suggestion="Use a positive integer for $top",
                    ))
                elif top_value > self.config.max_retrieval_limit:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="limits",
                        message=f"$top value {top_value} exceeds maximum allowed: {self.config.max_retrieval_limit}",
                        query_segment=f"$top={top_value}",
                        suggestion=f"Reduce $top to {self.config.max_retrieval_limit} or less",
                    ))
                elif top_value > self.config.warn_retrieval_threshold:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="limits",
                        message=f"$top value {top_value} is high (threshold: {self.config.warn_retrieval_threshold})",
                        query_segment=f"$top={top_value}",
                        suggestion="Consider reducing the limit for better performance",
                    ))

            except ValueError:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="limits",
                    message=f"Invalid $top value: '{parsed_query['$top']}' (must be an integer)",
                    query_segment=f"$top={parsed_query['$top']}",
                    suggestion="Use a numeric value for $top",
                ))

        return issues
