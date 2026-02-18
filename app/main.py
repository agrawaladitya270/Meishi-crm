from fastapi import FastAPI, UploadFile, File
import os
import shutil

from app.ocr import extract_text
from app.parser import parse_fields
from app.excel_service import upsert_contact

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(path)
    data = parse_fields(text)
    status = upsert_contact(data, file.filename)

    return {
        "extracted": data,
        "status": status
    }
