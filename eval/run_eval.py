"""Evaluate extraction pipelines on a labeled card set.

Field scoring (after Unicode NFKC normalization and removing spaces):
  Name, Company: exact match.  Email: exact match (lower case).
  Phone: same digits.  Title: the true title appears in the predicted line (a department
  prefix on the same line is allowed, up to 20 extra characters).

Usage:
  python eval/run_eval.py --data data/dev --configs v0 v1 v2 --out results/dev.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from meishi import baseline  # noqa: E402
from meishi.pipeline import Config, extract  # noqa: E402

FIELDS = ["Name", "Company", "Title", "Email", "Phone"]

TESS = {"orient": False, "reader": "tesseract"}  # Tesseract, no orientation step, as in v1-v5
CONFIGS = {
    "v0": None,  # original notebook rules (no LLM)
    "v1": Config(detect_card=False, flatten_light=False, target_width=0, psm=3, **TESS),  # new parser only
    "v2": Config(detect_card=True, flatten_light=False, target_width=0, psm=3, **TESS),   # + card detection
    "v3": Config(detect_card=True, flatten_light=True, target_width=0, psm=3, **TESS),    # + light flattening
    "v4_1200": Config(detect_card=True, flatten_light=True, target_width=1200, psm=3, **TESS),
    "v4_1500": Config(detect_card=True, flatten_light=True, target_width=1500, psm=3, **TESS),
    "v4_1800": Config(detect_card=True, flatten_light=True, target_width=1800, psm=3, **TESS),
}
CONFIGS["v4"] = Config(**TESS)                                  # the previous default
CONFIGS["v6"] = Config(reader="tesseract")                      # + orientation model
CONFIGS["default"] = Config()                                   # the shipped setting (+ PaddleOCR reader)
CONFIGS["final"] = Config(name_pass=True, **TESS)               # + native-size name pass
CONFIGS["final_no_name_pass"] = Config(name_pass=False, **TESS)
CONFIGS["final_name_by_conf"] = Config(name_pass=True, name_pass_wins=False, **TESS)
for _psm in (4, 6, 11):
    for _w in (0, 1200, 1500):
        CONFIGS[f"v5_w{_w}_psm{_psm}"] = Config(detect_card=True, flatten_light=True, target_width=_w, psm=_psm,
                                                **TESS)


def _n(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or "")).lower()


def correct(field: str, pred: str, gt: str) -> bool:
    if not pred:
        return False
    if field == "Phone":
        return re.sub(r"\D", "", pred) == re.sub(r"\D", "", gt)
    if field == "Title":
        return _n(gt) in _n(pred) and len(_n(pred)) <= len(_n(gt)) + 20
    return _n(pred) == _n(gt)


def run(data: Path, name: str, limit: int | None = None, rotate: int = 0) -> list[dict]:
    labels = [json.loads(line) for line in open(data / "labels.jsonl", encoding="utf-8")][:limit]
    rows = []
    for lab in labels:
        path = data / lab["file"]
        t0 = time.perf_counter()
        if CONFIGS[name] is None:  # v0 reads the file itself, so --rotate does not apply
            fields, conf = baseline.extract(str(path)), {f: 1.0 for f in FIELDS}
        else:
            r = extract(Image.open(path).rotate(rotate, expand=True), CONFIGS[name])
            fields, conf = r["fields"], r["confidence"]
        rows.append({"id": lab["id"], "kind": lab["kind"], "lang": lab["lang"],
                     "seconds": time.perf_counter() - t0,
                     "pred": fields, "conf": conf,
                     "ok": {f: correct(f, fields.get(f, ""), lab[f]) for f in FIELDS}})
    return rows


def summarize(rows: list[dict]) -> dict:
    def acc(rs, f):
        return round(sum(r["ok"][f] for r in rs) / len(rs), 3) if rs else None
    out = {"n": len(rows),
           "field_acc": {f: acc(rows, f) for f in FIELDS},
           "mean_field_acc": round(sum(sum(r["ok"].values()) for r in rows) / (5 * len(rows)), 3),
           "all_fields_correct": round(sum(all(r["ok"].values()) for r in rows) / len(rows), 3)}
    for key in ("kind", "lang"):
        for val in sorted({r[key] for r in rows}):
            rs = [r for r in rows if r[key] == val]
            out[f"mean_field_acc_{val}"] = round(sum(sum(r["ok"].values()) for r in rs) / (5 * len(rs)), 3)
    secs = sorted(r["seconds"] for r in rows)
    out["p50_s"] = round(statistics.median(secs), 2)
    out["p95_s"] = round(secs[min(len(secs) - 1, int(0.95 * len(secs)))], 2)
    return out


def routing(rows: list[dict], thresholds=None) -> list[dict]:
    """Coverage and precision of auto-accepted fields at each confidence threshold."""
    thresholds = thresholds or [round(0.5 + 0.05 * i, 2) for i in range(10)]
    res = []
    for t in thresholds:
        acc = [(r["ok"][f]) for r in rows for f in FIELDS if r["pred"].get(f) and r["conf"][f] >= t]
        res.append({"threshold": t, "coverage": round(len(acc) / (5 * len(rows)), 3),
                    "precision": round(sum(acc) / len(acc), 3) if acc else None})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=["v0", "v1"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="turn every card counter-clockwise first, to test sideways photos")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-mean-acc", type=float, help="exit 1 if the last config scores below this")
    args = ap.parse_args()
    report = {}
    for name in args.configs:
        rows = run(args.data, name, args.limit, args.rotate)
        report[name] = {"summary": summarize(rows), "routing": routing(rows) if name != "v0" else None,
                        "rows": rows}
        print(name, json.dumps(report[name]["summary"], ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    last = report[args.configs[-1]]["summary"]["mean_field_acc"]
    if args.min_mean_acc is not None and last < args.min_mean_acc:
        sys.exit(f"mean field accuracy {last} is below the gate {args.min_mean_acc}")


if __name__ == "__main__":
    main()
