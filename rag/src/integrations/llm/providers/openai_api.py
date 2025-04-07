import os
from typing import List, Dict, Optional

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional for OpenAI: discourages repeating exact phrases.
frequency_penalty = 0.0
# Optional for OpenAI: encourages discussion of new topics rather than staying on familiar ones.
presence_penalty = 0.0


def openai_call(
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None
) -> str:
    """Call the OpenAI API with extended options."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"
