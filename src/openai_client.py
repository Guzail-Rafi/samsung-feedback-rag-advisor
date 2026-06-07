import os
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from tracing_utils import sanitize_trace_inputs, sanitize_trace_outputs, tracing_enabled


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_ALIASES = {
    "gpt-5.4-min": "gpt-5.4-mini",
}
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_LAST_LLM_METADATA = ContextVar("last_llm_metadata", default={})
_LAST_LLM_METADATA_GLOBAL = {}

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )
except ImportError as error:
    OpenAI = None
    OPENAI_IMPORT_ERROR = error
else:
    OPENAI_IMPORT_ERROR = None

try:
    from ollama import Client as OllamaClient
except ImportError as error:
    OllamaClient = None
    OLLAMA_IMPORT_ERROR = error
else:
    OLLAMA_IMPORT_ERROR = None


def env_bool(name, default=False):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_openai_model():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    return MODEL_ALIASES.get(model, model)


def get_llama_model():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def get_ollama_base_url():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()


def llama_fallback_enabled():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return env_bool("LLM_FALLBACK_ENABLED", default=True)


def get_last_llm_metadata():
    metadata = _LAST_LLM_METADATA.get()
    return dict(metadata or _LAST_LLM_METADATA_GLOBAL)


def set_last_llm_metadata(provider, model, fallback_used=False, fallback_reason=None):
    global _LAST_LLM_METADATA_GLOBAL

    metadata = {
        "provider": provider,
        "model": model,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }
    _LAST_LLM_METADATA.set(metadata)
    _LAST_LLM_METADATA_GLOBAL = metadata
    return metadata


def is_openai_fallback_error(error):
    if isinstance(error, (AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError)):
        return True

    if isinstance(error, InternalServerError):
        return True

    if isinstance(error, APIStatusError):
        return error.status_code in {401, 408, 409, 429} or error.status_code >= 500

    return isinstance(error, ValueError) and "OPENAI_API_KEY" in str(error)


def fallback_reason(error):
    status_code = getattr(error, "status_code", None)
    return f"{error.__class__.__name__}" + (
        f" (HTTP {status_code})" if status_code is not None else ""
    )


def get_openai_client():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    if OpenAI is None:
        if llama_fallback_enabled():
            return None
        raise ImportError(
            "The openai package is required. Install it with: python -m pip install openai"
        ) from OPENAI_IMPORT_ERROR

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        if llama_fallback_enabled():
            return None
        raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")

    client = OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "1")),
    )

    if tracing_enabled():
        return wrap_openai(client)

    return client


@traceable(
    name="Local Llama Generation",
    run_type="llm",
    tags=["ollama", "llama", "fallback"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def generate_llama_response(messages, temperature=0.2, max_completion_tokens=700):
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    if OllamaClient is None:
        raise ImportError(
            "The ollama package is required. Install it with: python -m pip install ollama"
        ) from OLLAMA_IMPORT_ERROR

    client = OllamaClient(host=get_ollama_base_url())
    response = client.chat(
        model=get_llama_model(),
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": max_completion_tokens,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        },
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
    )
    content = response.message.content

    if not content:
        raise RuntimeError("The local Llama model returned an empty response.")

    return content


def generate_chat_response(client, messages, temperature=0.2, max_completion_tokens=700):
    openai_error = None

    if client is not None:
        try:
            response = client.responses.create(
                model=get_openai_model(),
                input=messages,
                temperature=temperature,
                max_output_tokens=max_completion_tokens,
            )
            set_last_llm_metadata("openai", get_openai_model())
            return response.output_text
        except Exception as error:
            if not is_openai_fallback_error(error):
                raise
            openai_error = error
    else:
        openai_error = ValueError("OPENAI_API_KEY is unavailable.")

    if not llama_fallback_enabled():
        raise openai_error

    try:
        answer = generate_llama_response(messages, temperature, max_completion_tokens)
    except Exception as llama_error:
        raise RuntimeError(
            "OpenAI failed and the local Llama fallback was unavailable. "
            f"OpenAI: {fallback_reason(openai_error)}; "
            f"Llama: {llama_error.__class__.__name__}."
        ) from llama_error

    set_last_llm_metadata(
        "ollama",
        get_llama_model(),
        fallback_used=True,
        fallback_reason=fallback_reason(openai_error),
    )
    return answer
