import subprocess
import threading
from typing import List
import requests
from llama_cpp import Llama

LLAMA_CLI = r"C:\Users\lugr\Desktop\lgro\git\llama.cpp\build-x64-windows-msvc-release\bin\llama-cli.exe"
MODEL_PATH = r"C:\Users\lugr\Desktop\lgro\git\llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"
LLAMA_SERVER_URL = "http://localhost:8080/completion"

mode = "rest"  # cli rest bindings


def call_llamacpp(prompt: str) -> str:
    if mode == "cli":
        return call_llama_cli_live(prompt)
    if mode == "rest":
        return call_llama_server(prompt)
    if mode == "bindings":
        return call_llama_bindings(prompt)
    return "Invalid llama.cpp mode."


def call_llama_cli_live(prompt: str, timeout: int = 15, live_output: bool = True) -> str:
    command: List[str] = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--n-predict", "150",
        "--temp", "0.8",
        "--threads", "6"
    ]
    print(f"[INFO] Running llama-cli with prompt: {prompt!r}")
    result: List[str] = []

    def read_output(proc: subprocess.Popen, output: List[str]):
        try:
            for line in iter(proc.stdout.readline, ''):
                if line:
                    if live_output:
                        print(line, end="")
                    output.append(line)
        except Exception as e:
            output.append(f"\n[ERROR] Output read error: {e}")

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            reader_thread = threading.Thread(target=read_output, args=(proc, result))
            reader_thread.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[WARN] Timeout after {timeout}s. Terminating...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("[WARN] Forced kill.")
                    proc.kill()

            reader_thread.join()

    except FileNotFoundError:
        return "[ERROR] llama-cli not found. Check the path."
    except Exception as e:
        return f"[ERROR] Unexpected exception: {e}"

    return ''.join(result).strip()


def call_llama_server(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": 200,
        "temperature": 0.7,
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


def call_llama_bindings(prompt: str) -> str:
    llm = Llama(
        model_path=MODEL_PATH,
        n_threads=6,
        n_ctx=2048
    )
    response = llm(
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_k=40,
        top_p=0.9
    )
    return response["choices"][0]["text"]
