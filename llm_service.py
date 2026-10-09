"""Portkey-backed LLM and OpenAI Responses web-search helpers."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import httpx
import truststore
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI


load_dotenv(dotenv_path=".env", override=True)

from portkey_profiles import (
    CredentialType,
    model_profile,
    normalize_credential_type,
)

PORTKEY_BASE_URL = (
    os.getenv("PORTKEY_BASE_URL") or "https://portkey.bain.dev/v1"
).strip()
PORTKEY_LLM_MODEL = os.getenv("PORTKEY_LLM_MODEL", "").strip()

LLM_MAX_RETRIES = max(0, int(os.getenv("LLM_MAX_RETRIES", "1")))
LLM_REQUEST_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
)
LLM_WALL_TIMEOUT_SECONDS = max(
    1.0,
    float(
        os.getenv(
            "LLM_WALL_TIMEOUT_SECONDS",
            str(LLM_REQUEST_TIMEOUT_SECONDS + 30.0),
        )
    ),
)
WEB_SEARCH_WALL_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("WEB_SEARCH_WALL_TIMEOUT_SECONDS", "600"))
)
WEB_SEARCH_POLL_INTERVAL_SECONDS = max(
    0.5, float(os.getenv("WEB_SEARCH_POLL_INTERVAL_SECONDS", "3"))
)
WEB_SEARCH_MAX_TOOL_CALLS = max(
    1, int(os.getenv("WEB_SEARCH_MAX_TOOL_CALLS", "2"))
)

_session_api_key = ContextVar("portkey_api_key", default=None)
_session_credential_type: ContextVar[CredentialType] = ContextVar(
    "portkey_credential_type", default="personal"
)


def set_api_key(api_key: str | None) -> None:
    _session_api_key.set((api_key or "").strip() or None)


def set_credential_type(credential_type: str | None) -> None:
    _session_credential_type.set(normalize_credential_type(credential_type))


def set_runtime_credentials(
    api_key: str | None,
    credential_type: str | None = "personal",
) -> None:
    set_api_key(api_key)
    set_credential_type(credential_type)


def get_api_key() -> str | None:
    return _session_api_key.get() or os.getenv("PORTKEY_API_KEY")


def get_credential_type() -> CredentialType:
    return _session_credential_type.get()


def get_llm_model() -> str:
    return model_profile(get_credential_type()).llm_model


def get_portkey_provider() -> str:
    """Return the provider slug required by the Portkey gateway."""
    credential_type = get_credential_type()
    environment_name = (
        "PORTKEY_PROVIDER"
        if credential_type == "personal"
        else "CASE_PORTKEY_PROVIDER"
    )
    configured = os.getenv(environment_name, "").strip()
    if configured:
        return configured

    model = get_llm_model()
    if model.startswith("@") and "/" in model:
        return model.split("/", 1)[0]
    raise ValueError(
        f"{environment_name} is required because model {model!r} does not "
        "contain a Portkey provider slug."
    )


def get_portkey_headers() -> dict[str, str]:
    return {"x-portkey-provider": get_portkey_provider()}


async def generate_text_async(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Generate text without blocking the event loop."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("Portkey API key is not configured.")

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    try:
        async with httpx.AsyncClient(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        ) as request_client:
            llm = ChatOpenAI(
                base_url=PORTKEY_BASE_URL,
                model=get_llm_model(),
                api_key=api_key,
                default_headers=get_portkey_headers(),
                http_async_client=request_client,
                max_retries=LLM_MAX_RETRIES,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
            async with asyncio.timeout(LLM_WALL_TIMEOUT_SECONDS):
                response = await llm.ainvoke(messages)
    except TimeoutError as exc:
        raise TimeoutError(
            "LLM request exceeded the "
            f"{LLM_WALL_TIMEOUT_SECONDS:g}s wall-clock limit."
        ) from exc

    return (
        response.content.strip()
        if isinstance(response.content, str)
        else str(response.content).strip()
    )


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Synchronous compatibility entry point for non-async callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(generate_text_async(prompt, system_prompt))
    raise RuntimeError(
        "generate_text() cannot run inside an active event loop; "
        "await generate_text_async() instead."
    )


NEWS_RESULTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The non-empty article headline.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A non-empty 2-3 sentence summary.",
                    },
                    "url": {
                        "type": "string",
                        "description": "The non-empty direct article URL.",
                    },
                },
                "required": ["title", "summary", "url"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["articles"],
    "additionalProperties": False,
}


class WebSearchTimeoutError(TimeoutError):
    """Timeout that retains the resumable background Response ID."""

    def __init__(self, message: str, response_id: str | None = None) -> None:
        super().__init__(message)
        self.response_id = response_id


async def search_news_async(
    query: str,
    start_date: str,
    end_date: str,
    include_usage: bool = False,
    response_id: str | None = None,
    on_response_created: Callable[[str], None] | None = None,
) -> list[dict[str, str]] | dict[str, Any]:
    """Create or resume a web search and return ten compact news records."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("Portkey API key is not configured.")

    prompt = (
        f"Search the live web for: {query}\n"
        f"Use news published from {start_date} through {end_date}, inclusive. "
        "Return exactly ten distinct, genuinely relevant news or analysis "
        "articles. Prefer primary reporting and authoritative publications. "
        "Each summary must be factual, self-contained, and 2-3 sentences. "
        "Every URL must be a direct link to the specific article. Never use "
        "a publication homepage, topic page, category page, tag page, search "
        "page, or the same URL for multiple records. Verify that all ten "
        "URL strings are distinct before returning the response. Title, "
        "summary, and URL must all be non-empty for every record."
    )

    response = None
    try:
        async with httpx.AsyncClient(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        ) as request_client:
            client = AsyncOpenAI(
                base_url=PORTKEY_BASE_URL,
                api_key=api_key,
                default_headers=get_portkey_headers(),
                http_client=request_client,
                max_retries=LLM_MAX_RETRIES,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
            async with asyncio.timeout(WEB_SEARCH_WALL_TIMEOUT_SECONDS):
                if response_id:
                    response = await client.responses.retrieve(
                        response_id,
                        include=["web_search_call.action.sources"],
                    )
                else:
                    response = await client.responses.create(
                        model=get_llm_model(),
                        background=True,
                        store=True,
                        reasoning={"effort": "low"},
                        tools=[
                            {
                                "type": "web_search",
                                "search_context_size": "medium",
                                "external_web_access": True,
                            }
                        ],
                        tool_choice="required",
                        max_tool_calls=WEB_SEARCH_MAX_TOOL_CALLS,
                        include=["web_search_call.action.sources"],
                        input=prompt,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "news_search_results",
                                "strict": True,
                                "schema": NEWS_RESULTS_SCHEMA,
                            }
                        },
                    )
                    if on_response_created:
                        on_response_created(response.id)
                while response.status in {"queued", "in_progress"}:
                    await asyncio.sleep(WEB_SEARCH_POLL_INTERVAL_SECONDS)
                    response = await client.responses.retrieve(
                        response.id,
                        include=["web_search_call.action.sources"],
                    )
    except TimeoutError as exc:
        response_id = getattr(response, "id", None)
        response_note = (
            f" Background response ID: {response_id}."
            if response_id
            else ""
        )
        raise WebSearchTimeoutError(
            "Web-search request exceeded the "
            f"{WEB_SEARCH_WALL_TIMEOUT_SECONDS:g}s wall-clock limit."
            f"{response_note} Rerun to resume this background response.",
            response_id=response_id,
        ) from exc

    if response.status != "completed":
        detail = getattr(response, "error", None) or getattr(
            response, "incomplete_details", None
        )
        raise RuntimeError(
            f"Web-search response ended with status {response.status!r}: "
            f"{detail}"
        )

    payload = json.loads(response.output_text)
    articles = payload.get("articles", [])
    if len(articles) != 10:
        raise ValueError(
            f"Expected exactly 10 web-search results; received {len(articles)}."
        )

    normalized = []
    incomplete_count = 0
    for article in articles:
        record = {
            key: " ".join(str(article.get(key, "")).split())
            for key in ("title", "summary", "url")
        }
        if not all(record.values()):
            incomplete_count += 1
            continue
        normalized.append(record)
    if incomplete_count:
        print(
            f"  discarded {incomplete_count} incomplete web-search "
            "record(s); the repair pass will replace them"
        )
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    response_output = getattr(response, "output", None) or []
    usage_record = {
        "response_id": getattr(response, "id", None),
        "model": get_llm_model(),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(
            getattr(input_details, "cached_tokens", 0) or 0
        ),
        "cache_write_tokens": int(
            getattr(input_details, "cache_write_tokens", 0) or 0
        ),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "web_search_calls": sum(
            1
            for item in response_output
            if (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            == "web_search_call"
        ),
    }
    if include_usage:
        return {"articles": normalized, "usage": usage_record}
    return normalized
