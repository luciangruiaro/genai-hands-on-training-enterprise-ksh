import os

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_call(prompt: str, model, temperature: float = None) -> str:
    try:
        # Configure Generation Parameters
        generation_config_params = {}
        if temperature is not None:
            # Add validation if needed (e.g., 0.0 <= temperature <= 1.0)
            generation_config_params["temperature"] = temperature
        generation_config_params["max_output_tokens"] = 2048
        generation_config_params["top_p"] = 0.9
        generation_config_params["top_k"] = 40
        generation_config = GenerationConfig(**generation_config_params) if generation_config_params else None
        # 4. Set Safety Settings (Optional but Recommended)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        response = model.generate_content(
            prompt,  # todo
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text.strip() if response and response.text else ""
    except Exception as e:
        return f"Gemini error: {e}"
