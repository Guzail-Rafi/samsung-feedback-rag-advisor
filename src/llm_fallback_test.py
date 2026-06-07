import httpx
from openai import APIConnectionError
from unittest.mock import MagicMock, patch

from agent_router import route_intent
from langchain_router import route_with_langchain
from openai_client import generate_chat_response, get_last_llm_metadata


def main():
    print("Simulating unavailable OpenAI answer generation...")
    answer = generate_chat_response(
        client=None,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Local Llama fallback works",
            }
        ],
        temperature=0,
        max_completion_tokens=30,
    )
    generation_metadata = get_last_llm_metadata()

    print("Generation provider:", generation_metadata["provider"])
    print("Generation model:", generation_metadata["model"])
    print("Generation fallback used:", generation_metadata["fallback_used"])
    print("Generation answer:", answer.strip())

    if generation_metadata["provider"] != "ollama" or not generation_metadata["fallback_used"]:
        raise RuntimeError("Generation did not use the local Llama fallback.")

    print("\nSimulating unavailable OpenAI LangChain router...")
    router_query = "What should Samsung do to improve customer satisfaction?"
    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )

    with patch(
        "langchain_router.get_openai_router_chain",
        side_effect=connection_error,
    ):
        decision = route_with_langchain(router_query)

    print("Router provider:", decision["router_provider"])
    print("Router model:", decision["router_model"])
    print("Router fallback used:", decision["router_fallback_used"])
    print("Selected agent:", decision["selected_agent"])

    if decision["router_provider"] != "ollama" or not decision["router_fallback_used"]:
        raise RuntimeError("Routing did not use the local Llama fallback.")

    if decision["normalized_query"] != router_query:
        raise RuntimeError("Llama routing did not preserve the original retrieval query.")

    print("\nSimulating both LLM routers being unavailable...")

    with patch(
        "langchain_router.route_with_langchain",
        side_effect=RuntimeError("Both LLM providers unavailable"),
    ):
        final_fallback = route_intent("What percentage of comments are negative?")

    print("Final router provider:", final_fallback["router_provider"])
    print("Final routing method:", final_fallback["routing_method"])
    print("Selected agent:", final_fallback["selected_agent"])

    if final_fallback["router_provider"] != "deterministic_rules":
        raise RuntimeError("Routing did not use deterministic rules as the final fallback.")

    if final_fallback["selected_agent"] != "sentiment_agent":
        raise RuntimeError("The deterministic fallback selected the wrong agent.")

    print("\nVerifying application errors do not activate Llama...")
    broken_client = MagicMock()
    broken_client.responses.create.side_effect = TypeError("Application bug")

    with patch("openai_client.generate_llama_response") as llama_generation:
        try:
            generate_chat_response(
                broken_client,
                [{"role": "user", "content": "This should not reach Llama."}],
            )
        except TypeError:
            pass
        else:
            raise RuntimeError("An application error was incorrectly hidden by fallback.")

        llama_generation.assert_not_called()

    with (
        patch(
            "langchain_router.get_openai_router_chain",
            side_effect=ValueError("Application bug"),
        ),
        patch("langchain_router.get_llama_router_chain") as llama_router,
    ):
        try:
            route_with_langchain("This should not reach the Llama router.")
        except ValueError:
            pass
        else:
            raise RuntimeError("A router application error was incorrectly hidden by fallback.")

        llama_router.assert_not_called()

    print("Application errors correctly bypass Llama fallback.")
    print("\nOpenAI-to-Llama failover verified successfully.")


if __name__ == "__main__":
    main()
