"""End-to-end extraction: preprocess -> OCR -> rules -> (optional LLM fill) -> review flags."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from PIL import Image

from .ocr import ocr_lines, paddle_lines, preprocess
from .parse import FIELDS, find_name, parse_lines


@dataclass
class Config:
    detect_card: bool = True
    flatten_light: bool = True
    target_width: int = 1200
    orient: bool = True               # turn sideways / upside-down cards upright (PaddleOCR model)
    reader: str = "paddle"            # "paddle" (PP-OCRv5) or "tesseract"
    psm: int = 4                      # Tesseract layout mode
    # Second OCR pass at native size for the (large) name text. Off by default: it helped on the
    # dev set but added only +0.7 points on the test set while doubling the time per card.
    name_pass: bool = False
    name_pass_wins: bool = True
    use_llm: bool = os.getenv("MEISHI_LLM", "off") == "on"


def extract(image: Image.Image, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    img = preprocess(image, cfg.detect_card, cfg.flatten_light, cfg.target_width, cfg.orient)
    lines = paddle_lines(img) if cfg.reader == "paddle" else ocr_lines(img, psm=cfg.psm)
    result = parse_lines(lines)
    if cfg.name_pass and cfg.target_width:
        native = ocr_lines(preprocess(image, cfg.detect_card, cfg.flatten_light, 0, cfg.orient), psm=cfg.psm)
        name, conf = find_name(native)
        if name and (cfg.name_pass_wins or conf >= result["confidence"]["Name"]):
            result["fields"]["Name"], result["confidence"]["Name"] = name, round(conf, 3)
    # OCR confidence turned out NOT to predict correctness (precision stayed ~93% at every
    # threshold on the dev set), so only fields the OCR rules could not find are flagged for
    # review. They stay flagged when the LLM fills them in, as llm.py promises.
    missing = [f for f in FIELDS if not result["fields"][f]]
    if cfg.use_llm and missing:
        from .llm import fill_missing  # imported lazily: the model is large and optional
        result = fill_missing(result, "\n".join(ln.text for ln in lines))
    result["needs_review"] = missing
    result["ocr_lines"] = [ln.text for ln in lines]
    result["seconds"] = round(time.perf_counter() - t0, 3)
    return result
