"""
State definition for the Text-to-OData LangGraph agent.

The state is the single source of truth shared across all nodes in the graph.
Each node reads from and writes back to this typed dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated
import operator


class ODataQueryAttempt(TypedDict):
    """Records a single query-generation / validation cycle."""

    query: str
    validation_error: Optional[str]
    attempt_number: int


class TextToODataState(TypedDict):
    """
    Shared state for the Text-to-OData agent graph.

    Fields
    ------
    user_question : str
        The raw natural-language question from the caller.
    tenant_id : str
        Tenant identifier used for routing to the correct OData endpoint.
    user_id : str
        Authenticated user identifier; used for row-level-security claims.
    sid : str
        Integrity application session identifier passed through to the OData JWT.
    current_page_url : str
        UI context hint – the Integrity page that triggered the request.

    intent : Optional[str]
        Classified intent / topic extracted from the question.
    relevant_entities : List[str]
        OData entity-set names identified as relevant to the question.
    entity_schemas : Dict[str, Any]
        Fetched $metadata schemas keyed by entity-set name.

    odata_query : Optional[str]
        The most recently generated OData query string.
    query_attempts : List[ODataQueryAttempt]
        Full history of query-generation and validation attempts (for retry loop).
    retry_count : int
        Number of validation-fix cycles completed so far.
    max_retries : int
        Maximum allowed validation-fix cycles before falling back to error handling.
    is_query_valid : bool
        Whether the current ``odata_query`` passed validation.

    raw_odata_response : Optional[Any]
        The raw JSON/dict payload returned by the OData endpoint.
    formatted_answer : Optional[str]
        The final human-readable answer to be returned to the user.

    error : Optional[str]
        Holds an error message if any node fails; triggers error-handling path.
    """

    # ── Caller context ────────────────────────────────────────────────────────
    user_question: str
    tenant_id: str
    user_id: str
    sid: str
    current_page_url: str

    # ── Intent / schema discovery ─────────────────────────────────────────────
    intent: Optional[str]
    relevant_entities: List[str]
    entity_schemas: Dict[str, Any]

    # ── Query generation & validation loop ───────────────────────────────────
    odata_query: Optional[str]
    query_attempts: Annotated[List[ODataQueryAttempt], operator.add]  # append-only
    retry_count: int
    max_retries: int
    is_query_valid: bool

    # ── FAQ short-circuit ─────────────────────────────────────────────────────
    faq_answer: Optional[str]

    # ── Execution & response ──────────────────────────────────────────────────
    raw_odata_response: Optional[Any]
    formatted_answer: Optional[str]

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]
