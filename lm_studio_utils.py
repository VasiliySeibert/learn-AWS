"""
SPECIFICATION, do not remove this documentation block.
This .py script contains the utils functions to use the local lm studio server for LLM prompting.

"""

import re
from openai import OpenAI

# LM Studio runs an OpenAI-compatible server on localhost:1234
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Model identifier for nvidia nemotron-3-nano (reasoning model)
NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"


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


def prompt_nemotron(prompt: str, system_message: str = None, include_reasoning: bool = False) -> str:
    """
    Send a prompt to the nvidia/nemotron-3-nano model via LM Studio.

    Args:
        prompt: The user prompt to send to the LLM
        system_message: Optional system message to set context
        include_reasoning: If True, include the <think>...</think> reasoning in the response.
                          If False (default), strip the reasoning and return only the answer.

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
        model=NEMOTRON_MODEL,
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