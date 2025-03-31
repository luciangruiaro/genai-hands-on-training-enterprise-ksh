from service.llm import ollama_api, llamacpp_api, lmstudio_api
from service.llm.gemini_api import gemini_call
from service.llm.openai_api import openai_call
from service.llm.py_hf import hf_call

RESPONSE_TEMPLATE = "Hello {prompt}!"

llm_provider = "pyhf"  # Options: openai, gemini, ollama, lmstudio, llamacpp


def llm_ask(prompt: str) -> str:
    if llm_provider == "openai":
        return openai_call(prompt)
    elif llm_provider == "gemini":
        return gemini_call(prompt)
    elif llm_provider == "ollama":
        return ollama_api.ollama_call(prompt)
    elif llm_provider == "llamacpp":
        return llamacpp_api.call_llamacpp(prompt)
    elif llm_provider == "lmstudio":
        return lmstudio_api.lmstudio_call(prompt)
    elif llm_provider == "pyhf":
        return hf_call(prompt)


def generate_response(message):
    llm_response = llm_ask(message)
    dummy_response = RESPONSE_TEMPLATE.format(prompt=message)

    return {
        # "dummy_response": dummy_response,
        "llm_provider": llm_provider,
        "llm_response": llm_response
    }
