"""Portkey model aliases associated with each supported credential type."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast


CredentialType = Literal["personal", "case"]

DEFAULT_PERSONAL_LLM_MODEL = "@personal-openai/gpt-5.4"
DEFAULT_PERSONAL_EMBED_MODEL = "@personal-openai/text-embedding-3-large"
DEFAULT_CASE_LLM_MODEL = "@openai---enr-di-1bd14b/gpt-5.4"
DEFAULT_CASE_EMBED_MODEL = "@openai---enr-di-1bd14b/text-embedding-3-large"


@dataclass(frozen=True)
class PortkeyModelProfile:
    llm_model: str
    embedding_model: str


def _configured_value(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    return environ.get(name, "").strip() or default


def load_model_profiles(
    environ: Mapping[str, str] = os.environ,
) -> dict[CredentialType, PortkeyModelProfile]:
    return {
        "personal": PortkeyModelProfile(
            llm_model=_configured_value(
                environ, "PORTKEY_LLM_MODEL", DEFAULT_PERSONAL_LLM_MODEL
            ),
            embedding_model=_configured_value(
                environ,
                "PORT_KEY_EMBEDDING_MODEL",
                DEFAULT_PERSONAL_EMBED_MODEL,
            ),
        ),
        "case": PortkeyModelProfile(
            llm_model=_configured_value(
                environ, "CASE_LLM_MODEL", DEFAULT_CASE_LLM_MODEL
            ),
            embedding_model=_configured_value(
                environ, "CASE_EMBEDDING_MODEL", DEFAULT_CASE_EMBED_MODEL
            ),
        ),
    }


MODEL_PROFILES = load_model_profiles()
PERSONAL_LLM_MODEL = MODEL_PROFILES["personal"].llm_model
PERSONAL_EMBED_MODEL = MODEL_PROFILES["personal"].embedding_model
CASE_LLM_MODEL = MODEL_PROFILES["case"].llm_model
CASE_EMBED_MODEL = MODEL_PROFILES["case"].embedding_model


def normalize_credential_type(value: str | None) -> CredentialType:
    normalized = (value or "personal").strip().casefold()
    if normalized not in MODEL_PROFILES:
        raise ValueError("Credential type must be 'personal' or 'case'.")
    return cast(CredentialType, normalized)


def model_profile(value: str | None) -> PortkeyModelProfile:
    return MODEL_PROFILES[normalize_credential_type(value)]
