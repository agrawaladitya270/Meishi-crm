import re
from app.llm import extract_with_llm

def parse_fields(text: str):

    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.search(r"\+?\d[\d\- ]{8,}\d", text)

    data = {
        "Name": "",
        "Company": "",
        "Title": "",
        "Email": email.group(0) if email else "",
        "Phone": phone.group(0) if phone else ""
    }

    llm_data = extract_with_llm(text)

    for key in data:
        if key in llm_data and not data[key]:
            data[key] = llm_data[key]

    return data
