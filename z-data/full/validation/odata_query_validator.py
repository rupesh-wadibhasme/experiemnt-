"""
OData query validator module.

Provides comprehensive validation for OData queries including:
- Syntax validation
- Safety checks (prevent mutations)
- Cost estimation
- Limit enforcement
- Security vulnerability detection
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from .validation_result import ValidationIssue, ValidationResult, ValidationSeverity
from .validator_config import ValidatorConfig

LOGGER = logging.getLogger(__name__)


class ODataQueryValidator:
    """
    Stateless validator for OData queries.
    
    Validates queries against configurable rules and returns structured results.
    All methods are stateless.
    
    Parameters
    ----------
    config : ValidatorConfig, optional
        Validation configuration. If not provided, uses default settings.
    
    """
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        """Initialize the validator with optional configuration."""
        self.config = config or ValidatorConfig()
        
        # Compile regex patterns once for performance
        self._unsafe_patterns = self._compile_unsafe_patterns()
    
    def validate(self, query: str) -> ValidationResult:
        """
        Validate an OData query comprehensively.

        Parameters
        ----------
        query : str
            The OData query string to validate (may include entity set name).
            Examples: "Users?$top=10" or "$top=10&$filter=Age gt 18"

        Returns
        -------
        ValidationResult
            Structured validation result with issues and metadata.
        """
        issues: List[ValidationIssue] = []
        
        # Check if query was generated at all
        if not query or query.strip() == "":
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="generation",
                message="No OData query was generated",
                suggestion="Check query generation logic and prompts"
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                query=query or "",
                validation_metadata={"step_failed": "query_generation"}
            )
        
        # Parse the query
        parsed_query = self._parse_query(query)
        
        # Run all validation checks
        issues.extend(self._validate_syntax(query, parsed_query))
        issues.extend(self._validate_safety(query, parsed_query))
        issues.extend(self._validate_retrieval_limits(parsed_query))
        issues.extend(self._validate_expand_depth(parsed_query))
        issues.extend(self._validate_filter_complexity(parsed_query))
        
        # Cost estimation
        cost_score = None
        if self.config.enable_cost_estimation:
            cost_score = self._estimate_cost(parsed_query)
            issues.extend(self._validate_cost(cost_score))
        
        # Estimate row count if $top is specified
        row_count = parsed_query.get("$top")
        if row_count:
            try:
                row_count = int(row_count)
            except ValueError:
                row_count = None
        
        # Determine overall validity (no critical issues)
        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=sorted(issues, key=lambda x: (
                0 if x.severity == ValidationSeverity.CRITICAL else
                1 if x.severity == ValidationSeverity.WARNING else 2
            )),
            query=query,
            estimated_cost_score=cost_score,
            estimated_row_count=row_count,
            validation_metadata={
                "parsed_params": list(parsed_query.keys()),
            }
        )
    
    def _parse_query(self, query: str) -> Dict[str, str]:
        """
        Parse OData query string into components.
        
        Parameters
        ----------
        query : str
            Full OData query (may include entity set).
        
        Returns
        -------
        Dict[str, str]
            Dictionary of query parameters (e.g., {'$top': '10', '$filter': '...'})
        """
        # Handle both full URLs and query strings
        if "?" in query:
            _, query_string = query.split("?", 1)
        else:
            query_string = query
        
        # Parse query parameters
        parsed = {}
        if query_string:
            # Handle OData parameters (including system query options starting with $)
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    parsed[key] = value
        
        return parsed
    
    def _validate_syntax(self, query: str, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """Validate basic OData syntax."""
        issues = []
        
        # Check for common syntax errors
        
        # Unmatched parentheses in filter
        if "$filter" in parsed_query:
            filter_expr = parsed_query["$filter"]
            if filter_expr.count("(") != filter_expr.count(")"):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="syntax",
                    message="Unmatched parentheses in $filter expression",
                    query_segment=f"$filter={filter_expr}",
                    suggestion="Check filter expression for balanced parentheses"
                ))
        
        # Check for invalid parameter names (should start with $ for system query options)
        valid_params = {
            "$select", "$filter", "$orderby", "$top", "$skip", "$count",
            "$expand", "$search", "$apply", "$compute", "$index"
        }
        
        for param in parsed_query.keys():
            if param.startswith("$") and param not in valid_params:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="syntax",
                    message=f"Unknown OData system query option: {param}",
                    query_segment=param,
                    suggestion=f"Valid options: {', '.join(sorted(valid_params))}"
                ))
        
        # Check for empty parameter values
        for param, value in parsed_query.items():
            if not value or value.strip() == "":
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="syntax",
                    message=f"Empty value for parameter: {param}",
                    query_segment=f"{param}=",
                    suggestion=f"Provide a value for {param} or remove the parameter"
                ))
        
        return issues
    
    def _validate_safety(self, query: str, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """
        Validate that the query contains no unsafe operations.
        
        Checks for:
        - Mutation keywords (DELETE, UPDATE, INSERT, etc.)
        - $batch operations (if disabled)
        - Injection patterns
        """
        issues = []
        
        query_upper = query.upper()
        
        # Check for mutation keywords
        mutation_keywords = [
            "DELETE", "UPDATE", "INSERT", "MERGE", "PATCH",
            "CREATE", "DROP", "ALTER", "TRUNCATE"
        ]
        
        for keyword in mutation_keywords:
            # Look for the keyword as a whole word (not substring)
            if re.search(rf"\b{keyword}\b", query_upper):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="safety",
                    message=f"Detected unsafe operation keyword: {keyword}",
                    query_segment=keyword,
                    suggestion="Only SELECT/read operations are allowed"
                ))
        
        # Check for $batch if disabled
        if not self.config.allow_batch and "$batch" in query.lower():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="safety",
                message="$batch operations are disabled",
                query_segment="$batch",
                suggestion="Execute queries individually"
            ))
        
        # Check for potential SQL injection patterns
        for pattern in self._unsafe_patterns:
            if pattern.search(query):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="security",
                    message="Detected potential SQL injection pattern",
                    query_segment=pattern.pattern,
                    suggestion="Remove suspicious patterns from query"
                ))
        
        # Check for script injection in filter strings
        if "$filter" in parsed_query:
            filter_expr = parsed_query["$filter"]
            script_patterns = ["<script", "javascript:", "onerror=", "onload="]
            for pattern in script_patterns:
                if pattern in filter_expr.lower():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="security",
                        message=f"Detected potential script injection: {pattern}",
                        query_segment=f"$filter={filter_expr}",
                        suggestion="Remove script-like patterns from filter"
                    ))
        
        return issues
    
    def _validate_retrieval_limits(self, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """Validate $top parameter and retrieval limits."""
        issues = []
        
        # Check if $top is required but missing
        if self.config.require_top_limit and "$top" not in parsed_query:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="limits",
                message="Query does not specify $top limit",
                suggestion=f"Add $top parameter (max: {self.config.max_retrieval_limit})"
            ))
        
        # Validate $top value if present
        if "$top" in parsed_query:
            try:
                top_value = int(parsed_query["$top"])
                
                if top_value <= 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="limits",
                        message=f"Invalid $top value: {top_value} (must be positive)",
                        query_segment=f"$top={top_value}",
                        suggestion="Use a positive integer for $top"
                    ))
                
                elif top_value > self.config.max_retrieval_limit:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="limits",
                        message=f"$top value {top_value} exceeds maximum allowed: {self.config.max_retrieval_limit}",
                        query_segment=f"$top={top_value}",
                        suggestion=f"Reduce $top to {self.config.max_retrieval_limit} or less"
                    ))
                
                elif top_value > self.config.warn_retrieval_threshold:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="limits",
                        message=f"$top value {top_value} is high (threshold: {self.config.warn_retrieval_threshold})",
                        query_segment=f"$top={top_value}",
                        suggestion="Consider reducing the limit for better performance"
                    ))
            
            except ValueError:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="limits",
                    message=f"Invalid $top value: {parsed_query['$top']} (must be an integer)",
                    query_segment=f"$top={parsed_query['$top']}",
                    suggestion="Use a numeric value for $top"
                ))
        
        return issues
    
    def _validate_expand_depth(self, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """Validate $expand nesting depth."""
        issues = []
        
        if "$expand" not in parsed_query:
            return issues
        
        expand_expr = parsed_query["$expand"]
        depth = self._calculate_expand_depth(expand_expr)
        
        if depth > self.config.max_expand_depth:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="cost",
                message=f"$expand depth {depth} exceeds maximum: {self.config.max_expand_depth}",
                query_segment=f"$expand={expand_expr}",
                suggestion=f"Reduce nesting depth to {self.config.max_expand_depth} or less"
            ))
        
        return issues
    
    def _validate_filter_complexity(self, parsed_query: Dict[str, str]) -> List[ValidationIssue]:
        """Validate $filter expression complexity."""
        issues = []
        
        if "$filter" not in parsed_query:
            return issues
        
        filter_expr = parsed_query["$filter"]
        complexity = self._calculate_filter_complexity(filter_expr)
        
        if complexity > self.config.max_filter_complexity:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="cost",
                message=f"Filter complexity score {complexity} exceeds maximum allowed: {self.config.max_filter_complexity}",
                query_segment=f"$filter={filter_expr}",
                suggestion="Simplify the filter expression or break into multiple queries"
            ))
        
        return issues
    
    def _estimate_cost(self, parsed_query: Dict[str, str]) -> float:
        """
        Estimate query execution cost on a 0-100 scale.
        
        Factors considered:
        - Number of records ($top)
        - Expand depth
        - Filter complexity
        - Number of selected fields
        - Sorting requirements
        
        Returns
        -------
        float
            Cost score from 0 (cheap) to 100 (expensive)
        """
        cost = 0.0
        
        # Base cost for retrieval count
        if "$top" in parsed_query:
            try:
                top = int(parsed_query["$top"])
                # Scale: 100 records = 10 points, 1000 records = 30 points
                cost += min(30, (top / 100) * 3)
            except ValueError:
                pass
        else:
            # No limit = potentially expensive
            cost += 40
        
        # Expand depth cost
        if "$expand" in parsed_query:
            depth = self._calculate_expand_depth(parsed_query["$expand"])
            # Each level of expansion adds significant cost
            cost += depth * 15
        
        # Filter complexity cost
        if "$filter" in parsed_query:
            complexity = self._calculate_filter_complexity(parsed_query["$filter"])
            # Scale filter complexity to 0-25 points
            cost += min(25, complexity / 4)
        
        # Sorting cost
        if "$orderby" in parsed_query:
            # Sorting adds moderate cost
            cost += 10
        
        # $count operations are relatively cheap
        if "$count" in parsed_query and parsed_query["$count"].lower() == "true":
            cost += 5
        
        return min(100.0, cost)
    
    def _validate_cost(self, cost_score: float) -> List[ValidationIssue]:
        """Block queries above 80; warn between 60 and 80."""
        issues = []

        if cost_score > 80:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="cost",
                message=f"Query cost score {cost_score:.1f}/100 exceeds maximum allowed (80)",
                suggestion="Consider adding filters, reducing $top, or limiting $expand depth"
            ))
        elif cost_score > 60:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="cost",
                message=f"Query has moderate cost: {cost_score:.1f}/100 (limit: 80)",
                suggestion="Monitor performance and consider optimizations if slow"
            ))

        return issues
    
    def _calculate_expand_depth(self, expand_expr: str) -> int:
        """
        Calculate the maximum nesting depth of $expand.
        
        Examples:
        - "Orders" -> depth 1
        - "Orders($expand=OrderItems)" -> depth 2
        - "Orders($expand=OrderItems($expand=Product))" -> depth 3
        """
        # Count nested $expand keywords
        depth = 1  # Base expand
        
        # Simple heuristic: count $expand occurrences within the expression
        nested_expands = expand_expr.count("$expand")
        return depth + nested_expands
    
    def _calculate_filter_complexity(self, filter_expr: str) -> int:
        """
        Calculate filter expression complexity score.
        
        Factors:
        - Number of logical operators (and, or, not)
        - Number of comparison operators
        - Number of function calls
        - Nesting depth
        """
        complexity = 0
        
        # Logical operators
        complexity += filter_expr.lower().count(" and ") * 2
        complexity += filter_expr.lower().count(" or ") * 2
        complexity += filter_expr.lower().count(" not ") * 1
        
        # Comparison operators
        for op in ["eq", "ne", "gt", "ge", "lt", "le"]:
            complexity += filter_expr.lower().count(f" {op} ") * 1
        
        # Function calls (common OData functions)
        functions = [
            "contains", "startswith", "endswith", "length", "substring",
            "tolower", "toupper", "trim", "year", "month", "day",
            "hour", "minute", "second"
        ]
        for func in functions:
            complexity += filter_expr.lower().count(f"{func}(") * 3
        
        # Parentheses nesting adds complexity
        complexity += filter_expr.count("(") * 2
        
        return complexity
    
    def _compile_unsafe_patterns(self) -> List[re.Pattern]:
        """
        Compile regex patterns for detecting potential injection attacks.
        
        Returns
        -------
        List[re.Pattern]
            Compiled regex patterns for unsafe content.
        """
        patterns = [
            # SQL injection patterns
            re.compile(r"';.*--", re.IGNORECASE),  # SQL comment injection
            re.compile(r"union\s+select", re.IGNORECASE),  # UNION injection
            re.compile(r";\s*drop\s+table", re.IGNORECASE),  # DROP table
            re.compile(r";\s*exec\s*\(", re.IGNORECASE),  # Exec injection
            re.compile(r"xp_cmdshell", re.IGNORECASE),  # Command execution
            
            # Suspicious patterns
            re.compile(r"<script.*?>", re.IGNORECASE),  # Script tags
            re.compile(r"javascript:", re.IGNORECASE),  # JavaScript protocol
        ]
        
        return patterns
