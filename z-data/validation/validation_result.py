"""
Validation result model for OData query validation.

This module defines the structured output returned by the ODataQueryValidator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    
    CRITICAL = "critical"  # Query cannot be executed (syntax errors, unsafe operations)
    WARNING = "warning"    # Query can execute but may have issues (high cost, no limits)
    INFO = "info"          # Informational notices (optimization suggestions)


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue found in an OData query.
    
    Attributes
    ----------
    severity : ValidationSeverity
        How severe this issue is (critical/warning/info).
    category : str
        Issue category: 'syntax', 'safety', 'cost', 'limits', 'schema', 'security'.
    message : str
        Human-readable description of the issue.
    query_segment : Optional[str]
        The specific part of the query that caused this issue.
    suggestion : Optional[str]
        Recommended fix or action to resolve the issue.
    """
    
    severity: ValidationSeverity
    category: str
    message: str
    query_segment: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """Format the issue for logging/display."""
        parts = [f"[{self.severity.upper()}] {self.category}: {self.message}"]
        if self.query_segment:
            parts.append(f"  → Query segment: {self.query_segment}")
        if self.suggestion:
            parts.append(f"  → Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """
    Complete validation result for an OData query.
    
    Attributes
    ----------
    is_valid : bool
        Overall validation status. False if any CRITICAL issues exist.
    issues : List[ValidationIssue]
        All validation issues found, ordered by severity.
    query : str
        The original query that was validated.
    estimated_cost_score : Optional[float]
        Estimated query complexity/cost (0-100 scale, higher = more expensive).
    estimated_row_count : Optional[int]
        Estimated number of rows that will be retrieved (if determinable).
    validation_metadata : dict
        Additional metadata about the validation process.
    """
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    query: str = ""
    estimated_cost_score: Optional[float] = None
    estimated_row_count: Optional[int] = None
    validation_metadata: dict = field(default_factory=dict)
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def get_error_message(self) -> Optional[str]:
        """
        Get a formatted error message for critical issues.
        Returns None if validation passed.
        """
        if self.is_valid:
            return None
        
        critical = self.critical_issues
        if not critical:
            return None
        
        lines = ["OData query validation failed:"]
        for issue in critical:
            lines.append(f"  • {issue.message}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization/logging."""
        return {
            "is_valid": self.is_valid,
            "query": self.query,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "query_segment": issue.query_segment,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
            "estimated_cost_score": self.estimated_cost_score,
            "estimated_row_count": self.estimated_row_count,
            "metadata": self.validation_metadata,
        }
