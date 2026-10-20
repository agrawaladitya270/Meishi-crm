# ---- build stage: install Python packages into a separate folder ----
FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime stage: Tesseract with Japanese + the app, running as a non-root user ----
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-jpn libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app
COPY --from=build /install /usr/local
WORKDIR /app
COPY meishi ./meishi
RUN mkdir -p data && chown app:app data
USER app
ENV MEISHI_CONTACTS=/app/data/contacts.xlsx
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "meishi.api:app", "--host", "0.0.0.0", "--port", "8000"]
