from typing import List, Dict
from typing import Optional

from helpers.token_utils import estimate_token_count


def calculate_remaining_tokens(
        user_input: str,
        knowledge: Optional[str],
        behavior: Optional[str],
        context_window: int,
        max_tokens: int,
        user_role: str = "user"
) -> int:
    """
    Estimate how many tokens are left for history, given the current prompt context.

    Args:
        user_input: The user's input message.
        knowledge: Knowledge prompt content.
        behavior: Behavior prompt content.
        context_window: Max context size (input + output).
        max_tokens: Max output length.
        user_role: Used for calculating token size for user input.

    Returns:
        Remaining token space for history.
    """
    prompt_parts = []

    if knowledge:
        prompt_parts.append({"role": "system", "content": knowledge})

    if behavior:
        prompt_parts.append({"role": "system", "content": behavior})

    if user_input:
        prompt_parts.append({"role": user_role, "content": user_input})

    used_tokens = estimate_token_count(prompt_parts)
    return max(0, context_window - max_tokens - used_tokens)


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Converts a list of memory messages into a readable string
    suitable for LLM input as history.

    Each message will be formatted like:
        user: Hello
        assistant: Hi there

    Args:
        history: List of memory messages, each with 'role' and 'content'.

    Returns:
        A formatted string representing the conversation history.
    """
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history if 'role' in msg and 'content' in msg)
