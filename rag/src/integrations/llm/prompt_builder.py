from typing import Optional, List, Dict


def build_prompt(
        user_input: Optional[str],
        knowledge: Optional[str],
        history: Optional[str]  # ← include_history is removed
) -> List[Dict[str, str]]:
    """Construct the full prompt as a list of messages."""
    prompt = []

    if knowledge:
        prompt.append({"role": "system", "content": knowledge})

    if history:
        prompt.append({"role": "system", "content": f"This is the conversation history:\n{history}"})

    if user_input:
        prompt.append({"role": "user", "content": user_input})

    return prompt
