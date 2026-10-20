"""Rule-based field extraction from OCR lines, with a confidence score per field."""
from __future__ import annotations

import re

from .ocr import CJK, Line

FIELDS = ("Name", "Company", "Title", "Email", "Phone")

COMPANY_RE = re.compile(
    r"(株式会社|有限会社|合同会社|合資会社|\(株\)|\(有\)|\bCo\.?,?\s*Ltd\b|\bLtd\b|\bInc\b|"
    r"\bCorporation\b|\bCorp\b|\bK\.?K\.|\bLLC\b|\bGmbH\b)", re.I)
# OCR often misreads 会 in 株式会社 (as 雲, 亡 ...); no other character fits there
COMPANY_FIX = re.compile(r"(株式|有限|合同|合資)[^\s会]社")

# Common Japanese and English job titles (general knowledge, not taken from the test data)
TITLES_JA = ["代表取締役社長", "代表取締役", "取締役社長", "取締役", "執行役員", "副社長", "社長", "会長",
             "本部長", "副本部長", "事業部長", "部長", "次長", "課長", "係長", "室長", "所長", "支店長",
             "主任研究員", "上席研究員", "研究員", "主任", "主査", "主事",
             "マネージャー", "マネジャー", "リーダー", "ディレクター", "エンジニア", "コンサルタント",
             "スペシャリスト", "プロデューサー", "デザイナー", "アナリスト", "担当"]
TITLES_EN = ["CEO", "CTO", "CFO", "COO", "CIO", "President", "Vice President", "VP", "Chairman",
             "Director", "Manager", "Engineer", "Developer", "Scientist", "Researcher", "Consultant",
             "Specialist", "Analyst", "Designer", "Lead", "Head", "Officer", "Representative",
             "Partner", "Associate", "Executive"]
TITLE_RE = re.compile("|".join(sorted(TITLES_JA, key=len, reverse=True))
                      + r"|\b(" + "|".join(TITLES_EN) + r")\b", re.I)

DEPT_RE = re.compile(r"(部|課|室|本部|センター|グループ|チーム|支店|事業所|工場|"
                     r"Division|Dept\.?|Department|Center|Centre|Group|Team|Office)$", re.I)
ADDRESS_RE = re.compile(r"(〒|\d{3}-\d{4}|[都道府県].{0,8}[市区町村郡]|\b\w+-(ku|shi|cho|machi)\b|"
                        r"\bJapan\b|\d+-\d+-\d+)", re.I)
CONTACT_RE = re.compile(r"(@|https?:|www\.|\.co\.jp|\.com\b|TEL|FAX|電話|携帯|Mobile|Phone|"
                        r"\d{2,4}-\d{2,4}-\d{3,4})", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+81[\s-]?\(?0?\)?\d{1,4}|\(?0\d{1,4}\)?)[\s-]?\d{1,4}[\s-]?\d{3,4}")
FAX_LABEL = re.compile(r"(fax|pax|ファックス|ファクス|\bF\b)\W*$", re.I)  # OCR often reads FAX as PAX
MOBILE_LABEL = re.compile(r"(携帯|mobile|mob|\bM\b)\W*$", re.I)
JA_NAME_RE = re.compile(rf"[{CJK}]{{2,8}}")
HIRAGANA_ONLY = re.compile(r"[぀-ゟ]+")  # a misread logo, not a name (names have kanji/katakana)
EN_NAME_RE = re.compile(r"[A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+){1,2}")


def _is_other_field(text: str) -> bool:
    return bool(COMPANY_RE.search(text) or TITLE_RE.search(text) or DEPT_RE.search(text)
                or ADDRESS_RE.search(text) or CONTACT_RE.search(text))


LABEL_WORDS = {"mail", "email", "e-mail"}
TLD_FIXES = [(re.compile(r"\.cojp\b"), ".co.jp"), (re.compile(r"\.(?:ip|jo)$"), ".jp")]


def repair_email_line(text: str) -> str:
    """Undo common OCR damage in an email line: commas for dots, stray spaces, lost dots."""
    if "@" not in text:
        return text
    left, right = text.replace(",", ".").split("@", 1)
    left = re.sub(r"\s*\.\s*", ".", left.rstrip())
    right = re.sub(r"\s+", "", right)                      # a domain never contains spaces
    tokens = re.split(r"[\s:：]+", left)
    # the label glued onto the address when OCR drops the space: "e-mailtaro@" -> "taro@"
    local = re.sub(r"^e-?mail(?=\w)", "", tokens[-1], flags=re.I) if tokens else ""
    # a dot read as a space: "taro yamada@" -> "taro.yamada@" (not for labels like "Mail");
    # next to _ or - the space is just noise: "r_ madan@" -> "r_madan@"
    if len(tokens) >= 2 and re.fullmatch(r"[a-z0-9_-]+", tokens[-2]) and tokens[-2] not in LABEL_WORDS:
        sep = "" if tokens[-2][-1] in "_-" or local[:1] in ("_", "-") else "."
        local = f"{tokens[-2]}{sep}{local}"
    email = f"{local}@{right}"
    for pattern, repl in TLD_FIXES:
        email = pattern.sub(repl, email)
    return email


def find_email(lines: list[Line]) -> tuple[str, float]:
    for ln in lines:
        m = EMAIL_RE.search(repair_email_line(ln.text))
        if m:
            return m.group(0).lower().rstrip("."), ln.conf
    return "", 0.0


def find_phone(lines: list[Line]) -> tuple[str, float]:
    """Prefer the main TEL number; skip FAX; fall back to a mobile number."""
    found: dict[str, tuple[str, float]] = {}
    for ln in lines:
        for m in PHONE_RE.finditer(ln.text):
            digits = re.sub(r"\D", "", m.group(0))
            if not 9 <= len(digits) <= 12:
                continue
            prefix = ln.text[max(0, m.start() - 12):m.start()]
            kind = "fax" if FAX_LABEL.search(prefix) else "mobile" if MOBILE_LABEL.search(prefix) else "tel"
            found.setdefault(kind, (m.group(0).strip(), ln.conf))
    for kind in ("tel", "mobile"):
        if kind in found:
            return found[kind]
    return "", 0.0


def find_company(lines: list[Line]) -> tuple[str, float]:
    for ln in lines:
        text = COMPANY_FIX.sub(r"\1会社", ln.text)
        if COMPANY_RE.search(text) and not CONTACT_RE.search(text) and not ADDRESS_RE.search(text):
            return text, ln.conf
    return "", 0.0


def find_title(lines: list[Line]) -> tuple[str, float]:
    for ln in lines:
        if (TITLE_RE.search(ln.text) and not COMPANY_RE.search(ln.text)
                and not CONTACT_RE.search(ln.text) and not ADDRESS_RE.search(ln.text)):
            return ln.text, ln.conf
    return "", 0.0


def _ja_name_like(text: str) -> bool:
    t = text.replace(" ", "")
    return bool(JA_NAME_RE.fullmatch(t)) and not HIRAGANA_ONLY.fullmatch(t)


def find_name(lines: list[Line]) -> tuple[str, float]:
    """The name is usually the biggest text on the card: pick the tallest name-like line."""
    ja = [ln for ln in lines if _ja_name_like(ln.text) and not _is_other_field(ln.text)]
    if ja:
        best = max(ja, key=lambda ln: ln.height)
        return best.text, best.conf
    en = [ln for ln in lines if EN_NAME_RE.fullmatch(ln.text) and not _is_other_field(ln.text)]
    if en:
        best = max(en, key=lambda ln: ln.height)
        return best.text, best.conf
    return "", 0.0


def parse_lines(lines: list[Line]) -> dict:
    """Return {"fields": {...}, "confidence": {...}} for the five fields."""
    found = {
        "Name": find_name(lines),
        "Company": find_company(lines),
        "Title": find_title(lines),
        "Email": find_email(lines),
        "Phone": find_phone(lines),
    }
    return {
        "fields": {k: v[0] for k, v in found.items()},
        "confidence": {k: round(v[1], 3) for k, v in found.items()},
    }
