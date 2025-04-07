# token_utils.py

def estimate_token_count(prompt: list[dict]) -> int:
    """
    Estimate the number of tokens in the prompt.

    This is a rough estimation using the rule of thumb:
      1 token ≈ 4 characters.

    Note: For more accurate results, consider using a dedicated tokenizer
    (e.g., OpenAI's tiktoken library).

    Args:
        prompt (list[dict]): A list of message dicts (with a 'content' key).

    Returns:
        int: The estimated token count.
    """
    total_chars = sum(len(message.get("content", "")) for message in prompt)
    # Using integer division to get an approximate token count.
    return total_chars // 4
