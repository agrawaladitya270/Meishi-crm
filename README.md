# Meishi OCR CRM

Extracts **name, company, title, email and phone** from photos or scans of Japanese and English
business cards (名刺), and saves them to a contacts sheet. Runs on CPU, no paid APIs.

```
image ──► find card + flatten perspective ──► fix lighting ──► upscale ──► Tesseract (jpn+eng)
      ──► lines with size + confidence ──► field rules ──► (optional local LLM fill) ──► JSON / Excel
```

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

## Run

```bash
# local
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
pip install -r requirements.txt
uvicorn meishi.api:app --port 8000

# or Docker
docker compose up --build

# use it
curl -F "file=@samples/photo_ja.jpg" localhost:8000/extract     # fields as JSON
curl -F "file=@samples/photo_ja.jpg" localhost:8000/contacts    # extract + save to data/contacts.xlsx
```

## Evaluate

```bash
python eval/gen_cards.py --out data/dev  --n 60  --seed 1 --split dev
python eval/gen_cards.py --out data/test --n 150 --seed 2 --split test
python eval/run_eval.py --data data/test --configs v0 v1 v2 v3 default --out results/test.json
```
The test split uses different seeds and includes titles, departments, companies and cities that
never appear in the dev split. CI regenerates 40 cards and fails the build if mean field accuracy
drops below 80%.

## Project layout

```
meishi/   ocr.py (card detection, lighting, OCR lines) · parse.py (field rules) · pipeline.py
          baseline.py (original v0 logic, for comparison) · llm.py (optional) · store.py · api.py
eval/     gen_cards.py (synthetic dataset) · run_eval.py (metrics, ablations, CI gate)
tests/    parser unit tests, API tests
results/  saved evaluation runs (dev and test)
```
