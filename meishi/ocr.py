"""Image preprocessing and OCR into lines with confidence and size."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u3005\u3006\u3012\u30fc"


@dataclass
class Line:
    text: str
    conf: float      # mean word confidence, 0-1
    height: float    # median word-box height in px (a proxy for font size)


def normalize(text: str) -> str:
    """NFKC (full-width to half-width etc.) and remove the spaces OCR puts between CJK characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def find_card(bgr: np.ndarray) -> np.ndarray:
    """Find the card in a photo and warp it flat. Returns the input if no clear card is found."""
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr
    c = max(contours, key=cv2.contourArea)
    if not 0.2 * h * w < cv2.contourArea(c) < 0.95 * h * w:
        return bgr  # nothing card-like, or the card already fills the frame
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    pts = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
    tl, tr, br, bl = pts = _order(np.asarray(pts, dtype=np.float32))
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(pts, dst), (width, height))


def preprocess(img: Image.Image, detect_card: bool = True, flatten_light: bool = True,
               target_width: int = 1800) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if detect_card:
        bgr = find_card(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if target_width and gray.shape[1] < target_width:
        f = target_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if flatten_light:  # divide by a blurred copy to remove uneven lighting
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=31)
        gray = cv2.divide(gray, bg, scale=245)
    return Image.fromarray(gray)


def ocr_lines(img: Image.Image, psm: int = 3) -> list[Line]:
    data = pytesseract.image_to_data(img, lang="jpn+eng", config=f"--oem 1 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)
    lines = []
    for idx in groups.values():
        text = normalize(" ".join(data["text"][i] for i in idx))
        if text:
            conf = float(np.mean([float(data["conf"][i]) for i in idx])) / 100
            height = float(np.median([data["height"][i] for i in idx]))
            lines.append(Line(text, conf, height))
    return lines
