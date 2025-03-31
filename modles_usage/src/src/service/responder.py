from typing import Dict

from service.llm import openai_api, gemini_api, ollama_api, lmstudio_api, llamacpp_api

RESPONSE_TEMPLATE = "Hello {prompt}!"

# Set your provider here (ideally move to a config later)
llm_provider = "llamacpp"  # Options: openai, gemini, ollama, lmstudio, llamacpp


def llm_ask(prompt: str) -> str:
    if llm_provider == "openai":
        return openai_api.openai_call(prompt)
    elif llm_provider == "gemini":
        return gemini_api.gemini_call(prompt)
    elif llm_provider == "ollama":
        return ollama_api.ollama_call([{"role": "user", "content": prompt}], model="llama3.2")
    elif llm_provider == "lmstudio":
        return lmstudio_api.lmstudio_call(prompt)
    elif llm_provider == "llamacpp":
        return llamacpp_api.call_llamacpp(prompt)
    else:
        return "Error: Unknown LLM provider."


def generate_response(prompt: str) -> Dict:
    llm_response = llm_ask(prompt)
    dummy_response = RESPONSE_TEMPLATE.format(prompt=prompt)

    return {
        "llm_provider": llm_provider,
        "llm_response": llm_response,
        "dummy_response": dummy_response
    }
