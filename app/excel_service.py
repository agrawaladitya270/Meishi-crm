from openpyxl import load_workbook
from datetime import datetime

EXCEL_PATH = "data/master.xlsx"

def upsert_contact(data: dict, source: str):

    wb = load_workbook(EXCEL_PATH)
    ws = wb.active

    email = data.get("Email")
    phone = data.get("Phone")

    found_row = None

    for row in ws.iter_rows(min_row=2):
        existing_email = row[3].value
        existing_phone = row[4].value

        if email and existing_email == email:
            found_row = row
            break
        if phone and existing_phone == phone:
            found_row = row
            break

    now = datetime.utcnow().isoformat()

    if found_row:
        if not found_row[0].value:
            found_row[0].value = data["Name"]
        if not found_row[1].value:
            found_row[1].value = data["Company"]
        if not found_row[2].value:
            found_row[2].value = data["Title"]

        found_row[6].value = now
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
