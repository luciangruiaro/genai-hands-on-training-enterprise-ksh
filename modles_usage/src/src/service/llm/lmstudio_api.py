import requests


def lmstudio_call(prompt: str, port: int = 1234) -> str:
    try:
        res = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "llama-3.2-3b-instruct",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LM Studio error: {e}"
