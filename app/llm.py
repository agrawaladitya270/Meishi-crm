from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.eos_token_id

def parse_fields_llm(text):

    prompt = f"""
Return ONLY valid JSON with keys:
Name
Company
Title
Email
Phone

Text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    start = result.find("{")
    end = result.rfind("}") + 1

    try:
        return json.loads(result[start:end])
    except:
        return {}
