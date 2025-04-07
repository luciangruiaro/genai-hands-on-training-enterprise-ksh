import requests
from llama_cpp import Llama

LLAMA_SERVER_URL = "http://localhost:8080/completion"
MODEL_PATH = r"llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"

mode = "rest"  # cli rest bindings


def llamacpp_call(prompt, temperature) -> str:
    if mode == "rest":
        return call_llama_rest(prompt, temperature)
    if mode == "bindings":
        return call_llama_bindings(prompt)
    return "Invalid llama.cpp mode."


def call_llama_rest(prompt: str, temperature) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": 200,
        "temperature": temperature,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1
    }
    print(f"[INFO] Sending prompt to llama-server at {LLAMA_SERVER_URL}")
    try:
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get("content", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Failed to query llama-server: {e}"


def call_llama_bindings(prompt: str, temperature) -> str:
    llm = Llama(
        model_path=MODEL_PATH,
        n_threads=6,
        n_ctx=2048
    )
    response = llm(
        prompt=prompt,
        max_tokens=150,
        temperature=temperature,
        top_k=40,
        top_p=0.9
    )
    return response["choices"][0]["text"]
