"""HTTP API.  Run:  uvicorn meishi.api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from . import __version__
from .pipeline import extract
from .store import upsert_contact

MAX_BYTES = 10 * 1024 * 1024
CONTACTS_PATH = os.getenv("MEISHI_CONTACTS", "data/contacts.xlsx")
UI_PAGE = Path(__file__).with_name("ui.html")

app = FastAPI(title="Meishi OCR CRM", version=__version__)


def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "Not a supported image file") from None


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_PAGE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


# plain `def` endpoints run in a thread pool, so CPU-heavy OCR doesn't block the server
@app.post("/extract")
def extract_card(file: UploadFile = File(...)) -> dict:
    result = extract(_read_image(file))
    result.pop("ocr_lines", None)
    return result


@app.post("/contacts")
def add_contact(file: UploadFile = File(...)) -> dict:
    result = extract(_read_image(file))
    status = upsert_contact(CONTACTS_PATH, result["fields"], source=file.filename or "upload")
    return {"status": status, "fields": result["fields"], "needs_review": result["needs_review"]}
