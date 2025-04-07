# LL Performance

This document summarizes insights and experiments related to **local LLM inference performance**, focusing on tools like `llama.cpp`, `Ollama`, and `LM Studio`. It serves both as an introduction to key concepts and a benchmark reference.

---

## 🔍 What Influences LLM Inference Time?

### General Formula:
```
T_total ≈ T_prompt_processing + (T_per_token * N)
```
- `T_total`: total inference time
- `T_prompt_processing`: time needed to tokenize and embed the prompt
- `T_per_token`: average time per generated token
- `N`: number of tokens in the output

This model is an approximation and performance may vary depending on:
- Model size and quantization (e.g., LLaMA 3B vs 7B)
- Hardware (CPU vs GPU, Apple M1/M2 Metal, etc.)
- Number of tokens generated
- Sampling strategy (temperature, top-k, top-p)
- Memory management (caching, kv-store reuse, etc.)

---

## 🎲 Sampling & Temperature – Technical Explanation

During inference, each next-token prediction comes from a probability distribution over the model’s vocabulary. To make generation controllable, we apply:

### 🔸 Temperature
```
p'_i = exp(log(p_i) / T)
```
- `T < 1`: sharpens distribution → more deterministic
- `T = 1`: default, unmodified distribution
- `T > 1`: flattens distribution → more diverse, but also more chaotic

### 🔸 Sampling Strategies
- **Greedy**: pick the highest-probability token
- **Top-k**: sample randomly among top `k` tokens
- **Top-p (nucleus)**: sample among smallest set of tokens whose cumulative probability exceeds `p` (e.g., 0.9)

---

## 📦 Prompt Caching – How It Works

### 🔹 llama.cpp
- No prompt caching between sessions by default
- Each prompt is processed from scratch unless reused inside the same context window

### 🔹 Ollama
- **Yes**, it caches prompts internally to accelerate repeated inference
- Uses **kv-caching** (key-value attention cache) and **token reuse** from previous runs (even between sessions)
- You may observe sub-second latency on repeated prompts

### 🔹 LM Studio
- LM Studio may reuse memory/cache during active sessions
- Behavior varies based on backend model (GGUF/transformers) and settings (e.g., streaming mode)

---

## 📊 Output Parameters for Performance Assessment

### Key Metrics (across tools):
- `prompt_eval_time`: time to embed/process prompt
- `eval_time`: total time for generating response
- `tokens/sec`: generation speed
- `n_tokens`: output length
- `num_predict`: maximum tokens to generate

---

## 🔎 Tool-Specific Performance Outputs

### 🔹 OpenAI (via API)
- `usage.total_tokens`, `prompt_tokens`, `completion_tokens`
- No raw timing exposed, but response latency observable via client

### 🔹 Gemini (Google)
- Output includes `candidates`, no explicit timing
- Streaming mode available; performance depends on latency per chunk

### 🔹 LLaMA (via llama.cpp)
- Use `--log-timing` to get:
  - `prompt eval time`
  - `generation time`
  - `tokens per second`

### 🔹 Mistral / Gemma (via Ollama)
- Output logs include:
  - `tokens/sec`
  - Total time
  - May reuse cached context for short prompts

### 🔹 LM Studio
- GUI tools visualize:
  - Prompt time
  - Generation speed
- CLI/API usage can output tokens/sec if backend supports it

---

## 📌 Conclusions

- **Caching** plays a major role in perceived speed — especially in Ollama
- **Short outputs** are generated faster; linear relation with output length
- **Temperature and sampling strategy** directly affect determinism vs creativity
- **Performance should be benchmarked using consistent prompts and settings**
