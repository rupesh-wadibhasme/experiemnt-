"""
Node implementations for the Text-to-OData LangGraph agent.

Each public method on ``TextToODataNodes`` maps 1-to-1 to a node in the
StateGraph.  Methods receive the current ``TextToODataState`` and return a
*partial* state dict with only the fields they update.

Architecture flow (see graph.py for wiring):

    START
      │
      ▼
  classify_intent          ← parse the question, extract intent & filters
      │
      ▼
  select_entities          ← pick relevant OData entity sets
      │
      ▼
  fetch_entity_schemas     ← retrieve $metadata for each selected entity set
      │
      ▼
  generate_odata_query     ← LLM generates the OData query string
      │
      ▼
  validate_query           ── [valid]   ──► execute_query
      │                                         │
      └── [invalid, retries left] ──► fix_query ┘ (loop back to validate)
      │
      └── [invalid, max retries]  ──► handle_error
                                           │
                                           ▼
                                         END
      execute_query
          │
          ▼
      format_response
          │
          ▼
        END
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from models.chat_model import ChatModel

from .prompts import (
    INTENT_PROMPT,
    ENTITY_SELECTION_PROMPT,
    QUERY_GENERATION_PROMPT,
    QUERY_FIX_PROMPT,
    RESPONSE_FORMAT_PROMPT,
)
from .state import ODataQueryAttempt, TextToODataState
from .validation import ODataQueryValidator, ValidatorConfig  # -> New

LOGGER = logging.getLogger(__name__)


class TextToODataNodes:
    """
    Encapsulates all LangGraph node functions for the Text-to-OData agent.

    Parameters
    ----------
    chat_model : ChatModel
        ChatModel instance; ``base_model()`` is called internally to obtain
        the underlying ``AzureChatOpenAI`` used by all LLM-powered nodes.
    odata_base_url : str
        Base URL of the OData service, e.g. ``https://odata.example.com/v1``.
    available_entities : list[str]
        Full list of OData entity-set names exposed by the service.
    validator_config : ValidatorConfig, optional  # -> New
        Validation configuration. Defaults to production settings if not provided.  # -> New
    """

    def __init__(
        self,
        chat_model: ChatModel,
        odata_base_url: str,
        available_entities: list[str],
        validator_config: ValidatorConfig = None,  # -> New
    ) -> None:
        self._llm = chat_model.base_model()
        self._odata_base_url = odata_base_url
        self._available_entities = available_entities
        self._validator = ODataQueryValidator(  # -> New
            config=validator_config or ValidatorConfig.for_production()  # -> New
        )  # -> New

    # ââ Node 1: check_faq ââââââââââââââââââââââââââââââââââââââââââââââââââââ

    async def check_faq(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Check whether the user's question can be answered directly from a
        pre-built FAQ knowledge base, bypassing the full OData query generation pipeline.
        Runs first so that simple FAQ questions never enter the OData query generation flow.

        Reads  : state["user_question"]
        Writes : state["faq_answer"]
        """
        raise NotImplementedError

    # ââ Node 2: classify_intent âââââââââââââââââââââââââââââââââââââââââââââââ

    async def classify_intent(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Classify the user's intent and extract structured metadata
        (intent label, time period, filter dimensions).

        Reads  : state["user_question"]
        Writes : state["intent"]
        """
        raise NotImplementedError

    # ââ Node 3: select_entities âââââââââââââââââââââââââââââââââââââââââââââââ

    async def select_entities(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Identify which OData entity sets are relevant to the classified intent.

        Reads  : state["intent"], state["user_question"]
        Writes : state["relevant_entities"]
        """
        raise NotImplementedError

    # ââ Node 4: fetch_entity_schemas ââââââââââââââââââââââââââââââââââââââââââ

    async def fetch_entity_schemas(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Retrieve the $metadata / schema definition for each selected entity set.
        Results are cached per tenant to avoid redundant HTTP calls.

        Reads  : state["relevant_entities"], state["tenant_id"]
        Writes : state["entity_schemas"]
        """
        raise NotImplementedError

    # ââ Node 5: generate_odata_query ââââââââââââââââââââââââââââââââââââââââââ

    async def generate_odata_query(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Use the LLM to generate an OData v4 query string from the question
        and the fetched entity schemas.

        Reads  : state["user_question"], state["intent"], state["entity_schemas"]
        Writes : state["odata_query"], state["query_attempts"]
        """
        raise NotImplementedError

    # ââ Node 6: validate_query ââââââââââââââââââââââââââââââââââââââââââââââââ

    async def validate_query(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Validate the generated OData query syntactically and semantically
        against the fetched entity schemas.

        Reads  : state["odata_query"]
        Writes : state["is_query_valid"], state["query_attempts"] (last entry)
        """
        # -> New (entire method body below)
        odata_query = state.get("odata_query")

        if not odata_query:
            LOGGER.error("No OData query to validate")
            query_attempts = state.get("query_attempts", [])
            if query_attempts:
                query_attempts[-1]["validation_error"] = "No OData query was generated"
            else:
                query_attempts = [{
                    "query": "",
                    "validation_error": "No OData query was generated",
                    "attempt_number": state.get("retry_count", 0) + 1,
                }]
            return {
                "is_query_valid": False,
                "query_attempts": query_attempts,
                "error": "Query generation failed - no query produced",
            }

        try:
            validation_result = self._validator.validate(query=odata_query)

            LOGGER.info("Validation complete: is_valid=%s", validation_result.is_valid)
            for issue in validation_result.issues:
                log_level = (
                    logging.ERROR if issue.severity.value == "critical"
                    else logging.WARNING if issue.severity.value == "warning"
                    else logging.INFO
                )
                LOGGER.log(log_level, "Validation: %s - %s", issue.category, issue.message)

            query_attempts = state.get("query_attempts", [])
            if query_attempts:
                query_attempts[-1]["validation_error"] = (
                    validation_result.get_error_message()
                    if not validation_result.is_valid
                    else None
                )
            else:
                query_attempts = [{
                    "query": odata_query,
                    "validation_error": (
                        validation_result.get_error_message()
                        if not validation_result.is_valid
                        else None
                    ),
                    "attempt_number": state.get("retry_count", 0) + 1,
                }]

            return {
                "is_query_valid": validation_result.is_valid,
                "query_attempts": query_attempts,
            }

        except Exception as e:
            LOGGER.exception("Unexpected error during query validation: %s", e)
            query_attempts = state.get("query_attempts", [])
            if query_attempts:
                query_attempts[-1]["validation_error"] = f"Validation error: {str(e)}"
            else:
                query_attempts = [{
                    "query": odata_query,
                    "validation_error": f"Validation error: {str(e)}",
                    "attempt_number": state.get("retry_count", 0) + 1,
                }]
            return {
                "is_query_valid": False,
                "query_attempts": query_attempts,
                "error": f"Validation system error: {str(e)}",
            }

    # ââ Node 7: fix_query âââââââââââââââââââââââââââââââââââââââââââââââââââââ

    async def fix_query(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Ask the LLM to repair the invalid OData query using the validation
        error message as feedback.

        Reads  : state["odata_query"], state["query_attempts"],
                 state["entity_schemas"], state["user_question"]
        Writes : state["odata_query"], state["retry_count"], state["query_attempts"]
        """
        raise NotImplementedError

    # ââ Node 8: execute_query âââââââââââââââââââââââââââââââââââââââââââââââââ

    async def execute_query(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Execute the validated OData query against the live OData endpoint.
        Attaches tenant / user claims to the request for row-level security.

        Reads  : state["odata_query"], state["tenant_id"],
                 state["user_id"], state["sid"]
        Writes : state["raw_odata_response"]
        """
        raise NotImplementedError

    # ââ Node 9: format_response âââââââââââââââââââââââââââââââââââââââââââââââ

    async def format_response(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Use the LLM to convert the raw OData JSON payload into a clear,
        human-readable answer.

        Reads  : state["user_question"], state["raw_odata_response"]
        Writes : state["formatted_answer"]
        """
        raise NotImplementedError

    # ââ Node 10: handle_error âââââââââââââââââââââââââââââââââââââââââââââââââ

    async def handle_error(self, state: TextToODataState) -> Dict[str, Any]:
        """
        Produce a user-friendly error message when the agent cannot generate
        a valid query within the allowed retry budget or when an unrecoverable
        exception occurs.

        Reads  : state["error"], state["query_attempts"]
        Writes : state["formatted_answer"]
        """
        raise NotImplementedError

    # ââ Conditional edge helpers ââââââââââââââââââââââââââââââââââââââââââââââ

    @staticmethod
    def route_after_faq(state: TextToODataState) -> str:
        """
        Routing function called after ``check_faq``.

        Returns
        -------
        "answered"     â FAQ produced an answer â skip OData pipeline, go to format_response.
        "continue"     â No FAQ match â proceed to select_entities.
        "handle_error" â Node set an error â jump to error handler.
        """
        if state.get("error"):
            return "handle_error"
        if state.get("faq_answer"):
            return "answered"
        return "continue"

    @staticmethod
    def route_after_validation(state: TextToODataState) -> str:
        """
        Routing function called after ``validate_query``.

        Returns
        -------
        "execute_query"  â query is valid â proceed to execution.
        "fix_query"      â query is invalid and retries remain.
        "handle_error"   â query is invalid and retry budget is exhausted.
        """
        if state["is_query_valid"]:
            return "execute_query"
        if state["retry_count"] < state["max_retries"]:
            return "fix_query"
        return "handle_error"

    @staticmethod
    def route_after_any_node(state: TextToODataState) -> str:
        """
        Generic guard: if a node set ``state["error"]``, short-circuit to
        ``handle_error``; otherwise continue normally.

        Returns
        -------
        "continue"      â no error, proceed.
        "handle_error"  â an error was set, jump to error handler.
        """
        if state.get("error"):
            return "handle_error"
        return "continue"
