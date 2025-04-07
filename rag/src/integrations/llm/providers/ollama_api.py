import ollama


def ollama_call(
        messages,
        model="llama3.2",
        format=None,  # e.g. "json", "text", "md"
        options=None,  # dict: {"temperature": 0.7, "top_p": 0.9, ...}
        stream=False,  # True to stream output
        keep_alive="5m",  # Duration to keep model loaded (e.g., "5m", "0" for unload)
):
    """
    Call a local Ollama model with a structured chat message.

    Parameters:
        messages (list): [{"role": "user", "content": "Your prompt"}]
        model (str): Model name (must be pulled via `ollama run <model>` first)
        format (str|None): Optional format like "json", "text", or "markdown"
        options (dict|None): Advanced generation options (see below)
        stream (bool): Whether to stream the response
        keep_alive (str): How long to keep the model in memory ("5m", "0", "inf")

    Returns:
        str: Response content or error message
    """
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            format=format,
            options=options,
            stream=stream,
            keep_alive=keep_alive,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API error: {e}"
