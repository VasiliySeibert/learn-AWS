"""
SPECIFICATION, do not remove this documentation block.
This .py script contains the utils functions to use the local lm studio server for LLM prompting.

"""

import re
from openai import OpenAI
from platform_utils import get_lm_studio_url

# LM Studio runs an OpenAI-compatible server
# URL is automatically configured based on environment (WSL vs macOS)
LM_STUDIO_BASE_URL = get_lm_studio_url()

# Model identifier for nvidia nemotron-3-nano (reasoning model)
# Can be overridden with LM_STUDIO_MODEL environment variable
import os
NEMOTRON_MODEL = os.getenv("LM_STUDIO_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")


def get_available_models() -> list[str]:
    """
    Fetch the list of available models from LM Studio.

    Returns:
        List of model IDs available in LM Studio
    """
    try:
        import requests
        response = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            print(f"Failed to fetch models from LM Studio: {response.status_code}")
            return [NEMOTRON_MODEL]  # Return default if fetch fails
    except Exception as e:
        print(f"Error fetching models from LM Studio: {e}")
        return [NEMOTRON_MODEL]  # Return default if connection fails


def _strip_reasoning(response: str) -> str:
    """
    Remove the <think>...</think> reasoning block from the response.

    Args:
        response: The raw LLM response containing reasoning tags

    Returns:
        The response with reasoning stripped out
    """
    # Handle both cases: with <think> tag or without (just </think>)
    # First try: remove <think>...</think> block
    stripped = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    # Second: remove everything before </think> if opening tag was missing
    stripped = re.sub(r"^.*?</think>\s*", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def prompt_nemotron(prompt: str, system_message: str = None, include_reasoning: bool = False, model: str = None) -> str:
    """
    Send a prompt to the nvidia/nemotron-3-nano model via LM Studio.

    Args:
        prompt: The user prompt to send to the LLM
        system_message: Optional system message to set context
        include_reasoning: If True, include the <think>...</think> reasoning in the response.
                          If False (default), strip the reasoning and return only the answer.
        model: Optional model identifier. If None, uses NEMOTRON_MODEL default.

    Returns:
        The LLM's response as a string
    """
    client = OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio"  # LM Studio doesn't require a real API key
    )

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model or NEMOTRON_MODEL,
        messages=messages,
        temperature=0.7
    )

    result = response.choices[0].message.content

    if include_reasoning:
        return result
    else:
        return _strip_reasoning(result)


if __name__ == "__main__":
    # Test the function
    print("Testing LM Studio connection with nemotron model...\n")

    print("=" * 50)
    print("WITHOUT reasoning:")
    print("=" * 50)
    response = prompt_nemotron(
        prompt="What is AWS EC2 in one sentence?",
        system_message="You are a helpful AWS expert.",
        include_reasoning=False
    )
    print(response)

    # print("\n" + "=" * 50)
    # print("WITH reasoning:")
    # print("=" * 50)
    # response = prompt_nemotron(
    #     prompt="What is AWS EC2 in one sentence?",
    #     system_message="You are a helpful AWS expert.",
    #     include_reasoning=True
    # )
    # print(response)