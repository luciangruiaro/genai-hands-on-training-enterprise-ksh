import json
from typing import Optional

from helpers.logger import setup_logger
from helpers.token_utils import estimate_token_count
from integrations.llm.prompt_builder import build_prompt
from integrations.llm.providers.gemini_api import gemini_call
from integrations.llm.providers.llamacpp_api import llamacpp_call
from integrations.llm.providers.lmstudio_api import lmstudio_call
from integrations.llm.providers.ollama_api import ollama_call
from integrations.llm.providers.openai_api import openai_call

logger = setup_logger("app")

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_WINDOW = 10000  # 10k tokens ≈ 7500 words
DEFAULT_MAX_TOKENS = 10000  # 1000 tokens ≈ 750 words
DEFAULT_TOP_P = 1.0


class LLMClient:
    def __init__(self, config):
        llm_config = config.get("llm_config", {})

        self.provider: str = llm_config.get("provider", DEFAULT_PROVIDER)
        self.model: str = llm_config.get("model", DEFAULT_MODEL)
        self.temperature: float = llm_config.get("temperature", DEFAULT_TEMPERATURE)
        self.context_window: int = llm_config.get("context_window", DEFAULT_CONTEXT_WINDOW)
        self.max_tokens: int = llm_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.top_p: float = llm_config.get("top_p", DEFAULT_TOP_P)

        self.handlers = {
            "openai": lambda prompt, model, temperature: openai_call(prompt, model, temperature, self.max_tokens,
                                                                     self.top_p),
            "gemini": lambda prompt, model, temperature: gemini_call(str(prompt), model, temperature),
            "ollama": lambda prompt, model, _: ollama_call(prompt, model),
            "llamacpp": lambda prompt, *_: llamacpp_call(str(prompt)),
            "lmstudio": lambda prompt, *_: lmstudio_call(str(prompt)),
        }

    def ask(
            self,
            user_input: Optional[str] = None,
            knowledge: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            history: Optional[str] = None
    ):
        """Main method to send the prompt to the selected LLM provider."""
        model = model or self.model
        temperature = temperature or self.temperature

        prompt = build_prompt(user_input=user_input, knowledge=knowledge, history=history)

        estimated_input_tokens = estimate_token_count(prompt)
        # Warn if the combined tokens (input prompt + expected output) might exceed the context window.
        if estimated_input_tokens + self.max_tokens > self.context_window:
            logger.warning(
                "The combined prompt and max_tokens may exceed the model's context window. "
                "Consider trimming your prompt or reducing max_tokens."
            )

        logger.info(f"Calling LLM ({self.provider}, model: {model}) with prompt: {user_input}")
        logger.debug(f"Full prompt: {json.dumps(prompt, indent=2)}")

        handler = self.handlers.get(self.provider)
        if handler:
            return handler(prompt, model, temperature)
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            return "Error: Unsupported LLM provider."
