import os
from openai import OpenAI

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=LLM_API_KEY)
model = "gpt-3.5-turbo"
temperature = 0


def openai_call(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"
