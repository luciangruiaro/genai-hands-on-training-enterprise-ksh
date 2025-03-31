# 🧠 LLaMA.cpp Quick Reference

Guide to working with [llama.cpp](https://github.com/ggml-org/llama.cpp) using local models and different run modes.

---

## 📦 Model Repository

- Primary model source: [TheBloke on Hugging Face](https://huggingface.co/TheBloke)
- Format: `GGUF` (optimized for `llama.cpp`)

### What GGUF actually is:

A **standardized binary file format** created by the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) community to:

- Package LLM weights + metadata + tokenizer into a **single portable file**
- Support **multiple model architectures** (LLaMA, Mistral, Falcon, etc.)
- Enable **fast inference** on CPU with quantization (INT4/INT8)

#### GGUF = Grokking Good Unified Format (standardized binary file format created by the llama.cpp community)

---

## 🔍 What's inside a `.gguf` file?

- Model weights (quantized or not)
- Tokenizer (`tokenizer.json`, `tokenizer.model`, or vocab)
- Metadata (architecture, hyperparams, format version)
- Optional: LoRA adapters, chat templates, etc.

---

## Benefits of GGUF

| Feature           | Why it matters                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| 🧳 Self-contained | No need for separate tokenizer/config files                                    |
| 📉 Quantized      | Super compact (runs on CPU easily)                                             |
| ⚙️ Standardized   | Works across tools like `llama.cpp`, `llama-cpp-python`, `LM Studio`, `Ollama` |
| 🚀 Fast loading   | Designed for fast streaming & mmap access                                      |

---

## llama.cpp Run Modes

`llama.cpp` supports **four main run modes**:

| Mode            | Description                                                                                                |
|-----------------|------------------------------------------------------------------------------------------------------------|
| `main.exe`      | CLI tool for direct inference — run prompts and receive outputs                                            |
| `server.exe`    | Starts a local REST API server for programmatic interaction                                                |
| `quantize.exe`  | Converts full-size `.bin` models into smaller, faster **quantized** `.gguf` models                         |
| Python bindings | Use via `llama-cpp-python`, linking against the compiled native shared library (`llama.dll` or `llama.so`) |

### Key Paths

| Item        | Path                                            |
|-------------|-------------------------------------------------|
| Models      | `\llama.cpp\models`                             |
| Executables | `\llama.cpp\build-x64-windows-msvc-release\bin` |

- *Build path may differ, depending on the system architecture

---

## GPU Acceleration

To build `llama.cpp` with GPU support (CUDA):

```bash
cmake -DLLAMA_CUBLAS=ON ...
```

---

## `llama-cli` Parameters (Cheat Sheet)

Here are the most useful CLI flags for `llama-cli.exe`:

| Flag                   | Description                             | What it actually means (simple)                                                                |
|------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------|
| `-m <path>`            | **Required**: path to the `.gguf` model | Tells the program which model file to load for generating text                                 |
| `-p "<prompt>"`        | Prompt string for inference             | What you want the model to respond to (your question or command)                               |
| `--n-predict <N>`      | Max number of tokens to generate        | How long the response should be (in tokens, not words)                                         |
| `--temp <float>`       | Temperature (creativity level)          | Controls randomness: lower = focused/deterministic, higher = more creative                     |
| `--top-k <int>`        | Top-K sampling                          | Picks from the top *K* most likely words instead of all possible ones                          |
| `--top-p <float>`      | Top-P (nucleus) sampling                | Chooses from the smallest group of words whose combined probability is ≥ *p* (e.g. 0.9)        |
| `--repeat-penalty <f>` | Penalty for repeated tokens             | Prevents repeating the same word/phrase too often                                              |
| `--repeat-last-n <n>`  | Context length for repetition penalty   | Applies the penalty to the last *n* tokens (e.g. last sentence)                                |
| `--ctx-size <int>`     | Context window size (e.g., 2048, 4096)  | How much text the model can "remember" in one go                                               |
| `--threads <n>`        | CPU threads to use                      | How many processor cores to use for generating text                                            |
| `--prompt-cache`       | Enable prompt caching                   | Speeds up repeat runs by caching the prompt’s result                                           |
| `--color`              | Colorize terminal output                | Adds colors to output (easier to read)                                                         |
| `--log-disable`        | Suppress logs                           | Turns off system logs to reduce noise in the terminal                                          |
| `--seed <int>`         | Set random seed (reproducibility)       | Ensures you get the same output each time with the same prompt                                 |
| `--file <file.txt>`    | Load prompt from a file                 | Reads your prompt from a text file instead of typing it inline                                 |
| `--interactive`        | Enter interactive chat after response   | Lets you keep chatting with the model instead of exiting after one answer                      |
| `--chatml`             | Use ChatML prompt formatting            | Uses a chat-specific format (needed for some chat-trained models like Mistral or LLaMA-2 Chat) |

---

## Example: Full CLI Command

```powershell
.\llama-cli.exe ^
  -m "C:\llama.cpp\models\mistral-7b-instruct-v0.1.Q4_0.gguf" ^
  -p "What is the meaning of life?" ^
  --n-predict 200 ^
  --temp 0.8 ^
  --top-k 40 ^
  --top-p 0.9 ^
  --repeat-penalty 1.1 ^
  --threads 8 ^
  --ctx-size 4096 ^
  --color
```

💡 Tip: In PowerShell, use `^` for multiline commands (similar to `\` on Linux/macOS).

