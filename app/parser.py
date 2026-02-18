import re
from app.llm import parse_fields_llm

COMPANY_KEYWORDS = ["株式会社","有限会社","Inc","Co","Ltd","LLC"]
TITLE_KEYWORDS = [
    "代表取締役","取締役","社長","副社長",
    "部長","課長","主任",
    "マネージャー","Manager","Director",
    "CEO","CTO","CFO"
]

def is_company_line(line):
    return any(k in line for k in COMPANY_KEYWORDS)

def is_title_line(line):
    return any(k in line for k in TITLE_KEYWORDS)

def looks_like_japanese_name(line):
    kanji = re.findall(r"[一-龯]", line)
    kana = re.findall(r"[ぁ-んァ-ン]", line)
    return len(line) <= 12 and (len(kanji)>=2 or len(kana)>=3)

def parse_fields_rule_based(text):

    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.search(r"\+?\d[\d\- ]{8,}\d", text)

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    name = ""
    company = ""
    title = ""

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
        "Phone": phone.group(0) if phone else ""
    }

def parse_fields(text):
    data = parse_fields_rule_based(text)

    if not data["Name"] or not data["Company"] or not data["Title"]:
        llm_data = parse_fields_llm(text)
        for key in data:
            if not data[key] and key in llm_data:
                data[key] = llm_data[key]

    return data
