import ollama


def ollama_call(
        prompt
):
    try:
        messages = [{"role": "user", "content": prompt}]
        model = "llama3.2"
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API error: {e}"
