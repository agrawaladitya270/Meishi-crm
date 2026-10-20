"""Optional LLM fallback that fills fields the rules missed (off by default: MEISHI_LLM=on).

Uses a small local chat model through Hugging Face transformers. Its answers are only accepted
if they appear in the OCR text, so the model cannot invent values; filled fields are always
sent to review.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

MODEL = os.getenv("MEISHI_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

PROMPT = """Extract these fields from the Japanese business card text: {fields}.
Copy values exactly as written in the text. Use "" if a field is not present.
Return only JSON.

Text:
{text}"""


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    return tok, model


def fill_missing(result: dict, ocr_text: str) -> dict:
    missing = [k for k, v in result["fields"].items() if not v]
    tok, model = _load()
    messages = [{"role": "user", "content": PROMPT.format(fields=", ".join(missing), text=ocr_text)}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    out = model.generate(inputs, max_new_tokens=200, do_sample=False)
    reply = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    compact = ocr_text.replace(" ", "")
    for k in missing:
        value = str(data.get(k, "")).strip()
        if value and value.replace(" ", "") in compact:   # only accept text that exists on the card
            result["fields"][k] = value
            result["confidence"][k] = 0.5                  # below the threshold: always reviewed
    return result
