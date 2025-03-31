from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "tiiuae/falcon-rw-1b"

model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "how are you?"


def hf_call(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
