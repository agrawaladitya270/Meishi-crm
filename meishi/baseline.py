"""The original notebook logic (v0), kept unchanged so the evaluation can compare against it.

The optional TinyLlama fallback from the notebook is not included here; see meishi/llm.py.
"""
import re

import pytesseract
from PIL import Image


def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="jpn+eng")
    return text


COMPANY_KEYWORDS = ["株式会社", "有限会社", "Inc", "Co", "Ltd", "LLC"]

TITLE_KEYWORDS = [
    "代表取締役", "取締役", "社長", "副社長",
    "部長", "課長", "主任",
    "マネージャー", "Manager", "Director",
    "CEO", "CTO", "CFO",
]


def is_company_line(line):
    return any(keyword in line for keyword in COMPANY_KEYWORDS)


def is_title_line(line):
    return any(keyword in line for keyword in TITLE_KEYWORDS)


def looks_like_japanese_name(line):
    kanji = re.findall(r"[一-龯]", line)
    kana = re.findall(r"[ぁ-んァ-ン]", line)
    if len(line) <= 12 and (len(kanji) >= 2 or len(kana) >= 3):
        if not is_company_line(line) and not is_title_line(line):
            return True
    return False


def parse_fields_rule_based(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\+?\d[\d\- ]{8,}\d"
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    name = company = title = ""
    for line in lines:
        if not company and is_company_line(line):
            company = line
        elif not title and is_title_line(line):
            title = line
        elif not name and looks_like_japanese_name(line):
            name = line
    return {
        "Name": name,
        "Company": company,
        "Title": title,
        "Email": email.group(0) if email else "",
        "Phone": phone.group(0) if phone else "",
    }


def extract(image_path):
    return parse_fields_rule_based(extract_text(image_path))
