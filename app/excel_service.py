import os
from openpyxl import Workbook, load_workbook
from datetime import datetime

EXCEL_PATH = "data/master.xlsx"

HEADERS = [
    "Name","Company","Title","Email","Phone",
    "Created_at","Updated_at","Source"
]

def ensure_excel():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(EXCEL_PATH):
        wb = Workbook()
        ws = wb.active
        ws.append(HEADERS)
        wb.save(EXCEL_PATH)

def upsert_contact(data, source):

    ensure_excel()

    wb = load_workbook(EXCEL_PATH)
    ws = wb.active

    email = data.get("Email")
    phone = data.get("Phone")

    found = None

    for row in ws.iter_rows(min_row=2):
        if email and row[3].value == email:
            found = row
            break
        if phone and row[4].value == phone:
            found = row
            break

    now = datetime.utcnow().isoformat()

    if found:
        if not found[0].value:
            found[0].value = data["Name"]
        if not found[1].value:
            found[1].value = data["Company"]
        if not found[2].value:
            found[2].value = data["Title"]
        found[6].value = now
        status = "updated"
    else:
        ws.append([
            data["Name"],
            data["Company"],
            data["Title"],
            data["Email"],
            data["Phone"],
            now,
            now,
            source
        ])
        status = "inserted"

    wb.save(EXCEL_PATH)
    return status
