"""Image preprocessing and OCR into lines with confidence and size."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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


def _card_corners(mask: np.ndarray) -> np.ndarray | None:
    """Corners of the biggest blob in a mask, or None if nothing card-sized is there."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if not 0.2 * h * w < cv2.contourArea(c) < 0.95 * h * w:
        return None  # nothing card-like, or the card already fills the frame
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    return approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(c))


def _card_corners_from_edges(bgr: np.ndarray) -> np.ndarray | None:
    """For a white card on a light desk, where brightness alone can't separate them: trace its outline.
    Works at ~1000 px so the blur and edge thresholds behave the same for any camera resolution."""
    s = min(1.0, 1000 / max(bgr.shape[:2]))
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 20, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    area = small.shape[0] * small.shape[1]
    if len(approx) != 4 or not 0.2 * area < cv2.contourArea(hull) < 0.95 * area:
        return None
    return approx.reshape(-1, 2) / s


def find_card(bgr: np.ndarray) -> np.ndarray:
    """Find the card in a photo and warp it flat. Returns the input if no clear card is found."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    pts = _card_corners(th)
    if pts is None:
        pts = _card_corners_from_edges(bgr)
    if pts is None:
        return bgr
    tl, tr, br, bl = pts = _order(np.asarray(pts, dtype=np.float32))
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(pts, dst), (width, height))


# PaddleOCR's document-orientation classifier (PP-LCNet_x1_0_doc_ori, Apache-2.0), ONNX export,
# pinned to one revision and checked by hash. Downloaded once (6.8 MB) into MODEL_DIR.
ORIENT_URL = ("https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/resolve/"
              "7330ab7039123e46af2dc03154b9969aa412c61d/inference.onnx")
ORIENT_SHA256 = "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"
MODEL_DIR = Path(os.getenv("MEISHI_MODEL_DIR", Path.home() / ".cache" / "meishi"))


@lru_cache(maxsize=1)
def _orientation_model():
    import onnxruntime as ort
    path = MODEL_DIR / "PP-LCNet_x1_0_doc_ori.onnx"
    if not path.exists():
        data = urllib.request.urlopen(ORIENT_URL, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != ORIENT_SHA256:
            raise RuntimeError(f"orientation model from {ORIENT_URL} does not match its expected hash")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").replace(path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # the export carries unused weights; don't warn about them on every start
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def upright(bgr: np.ndarray) -> np.ndarray:
    """Turn a sideways or upside-down card the right way up. On the dev set turned 0/90/180/270
    degrees, the model got all 240 right."""
    h, w = bgr.shape[:2]
    f = 256 / min(h, w)  # the model's own preprocessing: short side 256, centre crop 224, ImageNet norm
    img = cv2.resize(bgr, (round(w * f), round(h * f)))
    y, x = (img.shape[0] - 224) // 2, (img.shape[1] - 224) // 2
    img = (img[y:y + 224, x:x + 224] / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    sess = _orientation_model()
    scores = sess.run(None, {sess.get_inputs()[0].name: img.transpose(2, 0, 1)[None].astype(np.float32)})[0]
    k = int(np.argmax(scores))  # classes 0/90/180/270 = how far the card is turned clockwise
    return np.ascontiguousarray(np.rot90(bgr, k)) if k else bgr


def preprocess(img: Image.Image, detect_card: bool = True, flatten_light: bool = True,
               target_width: int = 1800, orient: bool = False) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if detect_card:
        bgr = find_card(bgr)
    if orient:
        bgr = upright(bgr)
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


@lru_cache(maxsize=1)
def _paddle_reader():
    # imported lazily; RapidOCR downloads the PP-OCRv5 mobile models (~20 MB) on first use
    from rapidocr import ModelType, OCRVersion, RapidOCR
    return RapidOCR(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
                            "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
                            "Global.log_level": "error"})


def _join_rows(items: list[tuple]) -> list[Line]:
    """PaddleOCR returns one box per phrase. Join phrases side by side in the same row and of the same
    size ("マダン" + "リティック"); small text beside big text (a department next to the name) stays apart."""
    rows: list[list[tuple]] = []
    for it in sorted(items, key=lambda it: it[0]):  # it = (x0, y0, x1, y1, text, score)
        h, cy = it[3] - it[1], (it[1] + it[3]) / 2
        for row in rows:
            last = row[-1]
            lh, lcy = last[3] - last[1], (last[1] + last[3]) / 2
            if (abs(cy - lcy) < 0.35 * min(h, lh) and max(h, lh) < 1.4 * min(h, lh)
                    and it[0] - last[2] < 2 * max(h, lh)):
                row.append(it)
                break
        else:
            rows.append([it])
    rows.sort(key=lambda r: ((r[0][1] + r[0][3]) / 2, r[0][0]))
    return [Line(normalize(" ".join(it[4] for it in r)), float(np.mean([it[5] for it in r])),
                 float(np.median([it[3] - it[1] for it in r]))) for r in rows]


def paddle_lines(img: Image.Image) -> list[Line]:
    """Read the card with PaddleOCR's PP-OCRv5 models (Chinese, Japanese, English) through RapidOCR."""
    res = _paddle_reader()(cv2.cvtColor(np.array(img.convert("L")), cv2.COLOR_GRAY2BGR))
    if res.boxes is None:
        return []
    items = []
    for box, text, score in zip(res.boxes, res.txts, res.scores, strict=True):
        b = np.asarray(box)
        items.append((b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max(), text, float(score)))
    return [ln for ln in _join_rows(items) if ln.text]
