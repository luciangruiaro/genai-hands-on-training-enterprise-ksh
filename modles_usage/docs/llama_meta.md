# 🐪 LLaMA CLI – Setup & Model Access Guide

This guide walks you through installing the LLaMA CLI, downloading models from Meta, and authenticating with Hugging
Face.

---

## Install the LLaMA CLI

Install `llama-stack` in your preferred Python environment:

```bash
pip install llama-stack
```

> 💡 Already installed? Update to the latest version:

```bash
pip install -U llama-stack
```

---

## List Available Models

List the latest LLaMA models available for download:

```bash
llama model list
```

To view **all versions**, including older ones:

```bash
llama model list --show-all
```

---

## ️ Download a Model from Meta

Choose and download a model by ID (example: `Llama3.2-3B`):

```bash
llama model download --source meta --model-id Llama3.2-3B
```

When prompted, **paste your unique Meta access URL**:

```
https://llama3-2-lightweight.llamameta.net/*?Policy=...&Signature=...
```

> 📌 You receive this URL after requesting access on Meta’s
> official [LLaMA Downloads Page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

---

## 🔐 Authenticate via Hugging Face

To access LLaMA models hosted on Hugging Face:

1. Log in using the CLI:

```bash
huggingface-cli login
```

2. Get your access token here:  
   👉 https://huggingface.co/settings/tokens

3. If you haven’t yet, request access to Meta’s gated models:  
   👉 https://huggingface.co/meta-llama/Meta-Llama-3-8B
