import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
MODEL_ALIASES = {
    "gpt-5.4-min": "gpt-5.4-mini",
}
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

try:
    from openai import OpenAI
except ImportError as error:
    OpenAI = None
    OPENAI_IMPORT_ERROR = error
else:
    OPENAI_IMPORT_ERROR = None


def get_openai_model():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    return MODEL_ALIASES.get(model, model)


def get_openai_client():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    if OpenAI is None:
        raise ImportError(
            "The openai package is required. Install it with: python -m pip install openai"
        ) from OPENAI_IMPORT_ERROR

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")

    return OpenAI(api_key=api_key)


def generate_chat_response(client, messages, temperature=0.2, max_completion_tokens=700):
    response = client.responses.create(
        model=get_openai_model(),
        input=messages,
        temperature=temperature,
        max_output_tokens=max_completion_tokens,
    )

    return response.output_text
