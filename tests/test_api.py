import io
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from meishi.api import app

client = TestClient(app)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ui_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_rejects_non_images():
    r = client.post("/extract", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")
def test_extract_simple_card():
    img = Image.new("RGB", (1050, 600), "white")
    d = ImageDraw.Draw(img)
    font = lambda size: ImageFont.truetype(FONT, size)  # noqa: E731
    d.text((60, 50), "株式会社テストシステム", font=font(34), fill="black")
    d.text((60, 130), "営業部 部長", font=font(24), fill="black")
    d.text((60, 180), "山田 太郎", font=font(64), fill="black")
    d.text((60, 460), "TEL 03-1234-5678", font=font(22), fill="black")
    d.text((60, 500), "taro.yamada@test-sys.co.jp", font=font(22), fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post("/extract", files={"file": ("card.png", buf.getvalue(), "image/png")})
    fields = r.json()["fields"]
    assert fields["Phone"] == "03-1234-5678"
    assert fields["Email"] == "taro.yamada@test-sys.co.jp"
