import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


def gemini_call(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else ""
    except Exception as e:
        return f"Gemini error: {e}"
