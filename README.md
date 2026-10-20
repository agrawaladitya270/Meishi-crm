# Meishi OCR CRM

[![ci](https://github.com/agrawaladitya270/Meishi-crm/actions/workflows/ci.yml/badge.svg)](https://github.com/agrawaladitya270/Meishi-crm/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Reads **name, company, title, email and phone** from photos or scans of Japanese and English
business cards (名刺), and saves them to an Excel contacts sheet. Runs on CPU with Tesseract, no paid APIs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ui-dark.png">
  <img alt="The upload page reading a phone photo of a Japanese business card" src="docs/ui-light.png" width="820">
</picture>

- **87.6% field accuracy** on 150 held-out test cards, up from 47.7% for the original notebook
- **About 0.5 s per card** on one CPU core
- Web upload page, JSON API, and a contacts sheet that updates existing rows instead of adding duplicates
- Docker image, and CI that fails the build if accuracy drops below 80%

## How it works

```
image ──► find card + flatten perspective ──► fix lighting ──► upscale ──► Tesseract (jpn+eng)
      ──► lines with size + confidence ──► field rules ──► (optional local LLM fill) ──► JSON / Excel
```

## Quick start

**Docker** (easiest)

```bash
docker compose up --build
```

**Linux**

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
pip install -r requirements.txt
uvicorn meishi.api:app --port 8000
```

**macOS:** `brew install tesseract tesseract-lang`, then the last two Linux lines.

**Windows** (PowerShell)

```powershell
winget install UB-Mannheim.TesseractOCR
# The installer ships English only. Put English + Japanese data in a folder you can write to:
$td = "$env:LOCALAPPDATA\tessdata"; New-Item -ItemType Directory -Force $td | Out-Null
Copy-Item "C:\Program Files\Tesseract-OCR\tessdata\*.traineddata" $td
Invoke-WebRequest -UseBasicParsing https://github.com/tesseract-ocr/tessdata_fast/raw/main/jpn.traineddata -OutFile "$td\jpn.traineddata"

python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Each time you start it:
$env:PATH = "C:\Program Files\Tesseract-OCR;" + $env:PATH
$env:TESSDATA_PREFIX = "$env:LOCALAPPDATA\tessdata"
.\.venv\Scripts\python -m uvicorn meishi.api:app --port 8000
```

Then open **http://localhost:8000**.

## Use it

**In the browser:** open http://localhost:8000, drop in a card image (or paste one), and click
**Read card** or **Read & save to contacts**. Fields that could not be found are flagged for review.

**Through the API:**

| Endpoint | What it does |
|---|---|
| `POST /extract` | Image in, fields out as JSON |
| `POST /contacts` | Same, and saves the contact to the Excel sheet. A row with the same email or phone is updated, not duplicated |
| `GET /health` | Status and version |
| `GET /docs` | Interactive API docs |

```bash
curl -F "file=@samples/scan_en.jpg" localhost:8000/extract
```

```json
{
  "fields": {
    "Name": "Tomoko Suzuki",
    "Company": "Hikari Denshi Inc.",
    "Title": "Marketing Dept. Senior Engineer",
    "Email": "suzuki@hikari-d.com",
    "Phone": "+81-52-895-6146"
  },
  "confidence": {"Name": 0.915, "Company": 0.917, "Title": 0.92, "Email": 0.925, "Phone": 0.915},
  "needs_review": [],
  "seconds": 0.506
}
```

Images can be up to 10 MB. Settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MEISHI_CONTACTS` | `data/contacts.xlsx` | Where contacts are saved |
| `MEISHI_LLM` | `off` | `on` lets a small local LLM fill fields the rules missed. Needs `pip install transformers torch accelerate` |
| `MEISHI_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Model used when `MEISHI_LLM=on` |

## Results

Test set: **150 synthetic cards** (half clean scans, half phone-photo style), never used for tuning.
Rules and settings were tuned on a separate 60-card dev set. Field scoring is strict (see
`eval/run_eval.py`): exact match after Unicode normalization; phone = same digits.

| Step (each row adds one change) | Mean field accuracy | Cards with all 5 fields right | Photos | Scans |
|---|---|---|---|---|
| v0 — original notebook (Tesseract defaults + first rules) | 47.7% | 9.3% | 31.6% | 63.4% |
| v1 — new field parser | 69.2% | 32.7% | 51.9% | 86.1% |
| v2 — + card detection and perspective correction | 79.9% | 42.7% | 72.4% | 87.1% |
| v3 — + lighting correction | 81.2% | 40.0% | 75.4% | 86.8% |
| **v4 — + upscaling to 1200 px and Tesseract layout mode 4 (default)** | **87.6%** | **55.3%** | **83.2%** | **91.8%** |

Per field (v4): company 95.3%, phone 94.7%, title 88.0%, email 84.0%, name 76.0%.
Speed: median 0.57 s per card, p95 0.69 s, on one CPU core.

### What the new parser fixed
- The original rules took the **postal code as the phone number** and **department names as the
  person's name**. The phone parser now reads the labels (TEL / 電話 / FAX / 携帯 / Mobile), skips FAX,
  and validates digit counts.
- Tesseract puts spaces between Japanese characters (`山 田 太 郎`); text is NFKC-normalized and those
  spaces are removed.
- The name is chosen as the **tallest** name-like line (names are the biggest text on a card).
- Emails are repaired for common OCR damage: commas for dots, spaces inside the address, `.cojp`.

### Findings that changed the design
- **OCR confidence does not predict correctness.** On the dev set, precision of auto-accepted fields
  stayed at ~93% whatever the confidence threshold, so confidence-based auto-approval was dropped.
  Only empty fields are flagged for review. Field validators (e.g. phone area code vs address
  prefecture) are the next step.
- **Upscaling helps small text but hurts the big name text.** A second OCR pass at native size for
  the name raised name accuracy on dev (+6.7 points) but only +0.7 on test while doubling the time,
  so it is off by default (`Config(name_pass=True)` to enable).

### Limits — read before quoting these numbers
- The cards are **synthetic** (generated with real Japanese fonts and realistic noise). Real cards
  have logos, vertical text and unusual layouts. Add your own real cards to `data/real/` with
  `labels.jsonl` in the same format and run the same script.
- The optional LLM fallback (`MEISHI_LLM=on`) is **not** included in the numbers above.

## Evaluate

```bash
python eval/gen_cards.py --out data/dev  --n 60  --seed 1 --split dev
python eval/gen_cards.py --out data/test --n 150 --seed 2 --split test
python eval/run_eval.py --data data/test --configs v0 v1 v2 v3 default --out results/test.json
```
The test split uses different seeds and includes titles, departments, companies and cities that
never appear in the dev split. CI regenerates 40 cards and fails the build if mean field accuracy
drops below 80%.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

## Project layout

```
meishi/   ocr.py (card detection, lighting, OCR lines) · parse.py (field rules) · pipeline.py
          baseline.py (original v0 logic, for comparison) · llm.py (optional) · store.py · api.py
          ui.html (upload page)
eval/     gen_cards.py (synthetic dataset) · run_eval.py (metrics, ablations, CI gate)
tests/    parser unit tests, API tests
results/  saved evaluation runs (dev and test)
docs/     screenshots
```

## License

[MIT](LICENSE) © 2026 Aditya Agrawal
