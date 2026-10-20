"""Generate a synthetic Japanese/English business-card dataset with ground-truth labels.

Each card gets a random layout, font and content (name, company, title, department,
address, phone, fax, email, URL). Half the cards stay clean ("scan"); the other half are
degraded to look like phone photos: placed on a background, rotated, tilted, unevenly lit,
blurred, noisy, downscaled and JPEG-compressed.

The dev and test splits use different seeds, and the test split also draws some values
(titles, departments, companies, cities) that never appear in dev, so rules tuned on dev
are checked on unseen data.

--style corporate makes cards modelled on a real phone photo instead: a sideways URL banner, logo
and tagline, departments beside a big (often katakana) name with romaji above it, an office label,
TEL(代)/FAX, and a licence-number footer, photographed on a light desk and often turned sideways.

Usage:  python eval/gen_cards.py --out data/test --n 200 --seed 2 --split test
        python eval/gen_cards.py --out data/corporate --n 60 --seed 4 --split test --style corporate
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc", 0),
    ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", 0),
    ("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", 0),
    # Windows ships these instead (used only where the Linux fonts are missing)
    ("C:/Windows/Fonts/YuGothR.ttc", 0),
    ("C:/Windows/Fonts/YuGothM.ttc", 0),
    ("C:/Windows/Fonts/YuGothB.ttc", 0),
    ("C:/Windows/Fonts/msgothic.ttc", 0),
]
FONTS = [f for f in FONTS if Path(f[0]).exists()] or FONTS  # skip fonts this machine lacks

SURNAMES = [("佐藤", "Sato"), ("鈴木", "Suzuki"), ("高橋", "Takahashi"), ("田中", "Tanaka"),
            ("伊藤", "Ito"), ("渡辺", "Watanabe"), ("山本", "Yamamoto"), ("中村", "Nakamura"),
            ("小林", "Kobayashi"), ("加藤", "Kato"), ("吉田", "Yoshida"), ("山田", "Yamada"),
            ("佐々木", "Sasaki"), ("山口", "Yamaguchi"), ("松本", "Matsumoto"), ("井上", "Inoue"),
            ("木村", "Kimura"), ("林", "Hayashi"), ("清水", "Shimizu"), ("森", "Mori"),
            ("池田", "Ikeda"), ("橋本", "Hashimoto"), ("阿部", "Abe"), ("石川", "Ishikawa"),
            ("前田", "Maeda"), ("藤田", "Fujita"), ("小川", "Ogawa"), ("岡田", "Okada"),
            ("後藤", "Goto"), ("長谷川", "Hasegawa")]
GIVEN = [("太郎", "Taro"), ("健太", "Kenta"), ("翔太", "Shota"), ("大輔", "Daisuke"),
         ("直樹", "Naoki"), ("浩二", "Koji"), ("誠", "Makoto"), ("拓也", "Takuya"),
         ("美咲", "Misaki"), ("陽子", "Yoko"), ("恵", "Megumi"), ("花子", "Hanako"),
         ("由美", "Yumi"), ("彩", "Aya"), ("健一", "Kenichi"), ("和也", "Kazuya"),
         ("亮", "Ryo"), ("真由美", "Mayumi"), ("愛", "Ai"), ("智子", "Tomoko")]

COMPANIES_SHARED = [("東亜システム", "Toa Systems", "toa-sys"), ("未来テクノロジー", "Mirai Technology", "mirai-tech"),
                    ("日本データサービス", "Nippon Data Service", "nds"), ("光電子工業", "Hikari Denshi", "hikari-d"),
                    ("さくらネットワーク", "Sakura Network", "sakura-net"), ("青葉商事", "Aoba Trading", "aoba"),
                    ("北斗ソリューションズ", "Hokuto Solutions", "hokuto-sol"), ("大和精機", "Yamato Seiki", "yamato-sk"),
                    ("新星電機", "Shinsei Denki", "shinsei-e"), ("緑川化学", "Midorikawa Chemical", "midori-chem")]
COMPANIES_TEST_ONLY = [("東邦物流", "Toho Logistics", "toho-log"), ("富士見企画", "Fujimi Planning", "fujimi"),
                       ("瀬戸内マテリアル", "Setouchi Materials", "setouchi-m"), ("ひかり医療機器", "Hikari Medical", "hikari-med")]

TITLES_JA_SHARED = ["代表取締役社長", "取締役", "部長", "課長", "主任", "マネージャー", "エンジニア", "営業担当"]
TITLES_JA_TEST_ONLY = ["係長", "主任研究員", "プロジェクトマネージャー", "執行役員"]
TITLES_EN_SHARED = ["CEO", "CTO", "Director", "Sales Manager", "Senior Engineer", "Engineer"]
TITLES_EN_TEST_ONLY = ["Product Manager", "Consultant"]

DEPTS_JA_SHARED = ["営業部", "開発本部", "技術部 システム課", "人事部", "マーケティング部"]
DEPTS_JA_TEST_ONLY = ["研究開発センター", "品質保証部", "海外事業室"]
DEPTS_EN = ["Sales Division", "R&D Center", "Engineering Dept.", "Marketing Dept."]

# (Japanese address, English address, postal code, area code)
CITIES_SHARED = [("東京都千代田区丸の内", "Marunouchi, Chiyoda-ku, Tokyo", "100-0005", "03"),
                 ("大阪府大阪市北区梅田", "Umeda, Kita-ku, Osaka", "530-0001", "06"),
                 ("兵庫県神戸市中央区三宮町", "Sannomiya-cho, Chuo-ku, Kobe", "650-0021", "078"),
                 ("愛知県名古屋市中村区名駅", "Meieki, Nakamura-ku, Nagoya", "450-0002", "052")]
CITIES_TEST_ONLY = [("福岡県福岡市博多区博多駅前", "Hakataekimae, Hakata-ku, Fukuoka", "812-0011", "092"),
                    ("京都府京都市下京区烏丸通", "Karasuma-dori, Shimogyo-ku, Kyoto", "600-8216", "075")]

# --style corporate
CORP_TAGLINES = ["TECHNICAL SOLUTION", "SYSTEM INTEGRATION", "ENGINEERING SERVICE", "SMART FACTORY"]
CORP_DEPTS = [("設計開発部", "情報処理課"), ("営業本部", "第一営業課"), ("技術部", "システム課"),
              ("管理本部", "総務課"), ("製造部", "品質管理課")]
OFFICES = ["本社", "東京事業所", "大阪事業所", "名古屋営業所", "神戸事業所"]
# foreign names written in katakana (family name first), with the romaji printed above each part
KATAKANA_SURNAMES = [("シャルマ", "SHARMA"), ("グエン", "NGUYEN"), ("ガルシア", "GARCIA"), ("リー", "LEE"),
                     ("ミラー", "MILLER"), ("パテル", "PATEL")]
KATAKANA_GIVEN = [("アミット", "AMIT"), ("プリヤ", "PRIYA"), ("ミン", "MINH"), ("アンナ", "ANNA"),
                  ("ダニエル", "DANIEL"), ("ラフル", "RAHUL")]


def pools(split: str) -> dict:
    test = split == "test"
    return {
        "companies": COMPANIES_SHARED + (COMPANIES_TEST_ONLY if test else []),
        "titles_ja": TITLES_JA_SHARED + (TITLES_JA_TEST_ONLY if test else []),
        "titles_en": TITLES_EN_SHARED + (TITLES_EN_TEST_ONLY if test else []),
        "depts_ja": DEPTS_JA_SHARED + (DEPTS_JA_TEST_ONLY if test else []),
        "cities": CITIES_SHARED + (CITIES_TEST_ONLY if test else []),
    }


def phone_number(rng: random.Random, area: str, mobile: bool = False) -> str:
    if mobile:
        return f"{rng.choice(['090', '080', '070'])}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    if len(area) == 2:
        return f"{area}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    return f"{area}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def make_content(rng: random.Random, p: dict, lang: str) -> dict:
    sur, given = rng.choice(SURNAMES), rng.choice(GIVEN)
    core_ja, core_en, slug = rng.choice(p["companies"])
    city_ja, city_en, postal, area = rng.choice(p["cities"])
    domain = slug + rng.choice([".co.jp", ".jp", ".com"])
    mobile_only = rng.random() < 0.15
    tel = phone_number(rng, area, mobile=mobile_only)
    fax = phone_number(rng, area) if (not mobile_only and rng.random() < 0.5) else None
    email_user = rng.choice([f"{given[1].lower()}.{sur[1].lower()}", f"{sur[1].lower()}",
                             f"{given[1][0].lower()}-{sur[1].lower()}"])
    email = f"{email_user}@{domain}"
    block = f"{rng.randint(1, 5)}-{rng.randint(1, 20)}-{rng.randint(1, 30)}"
    if lang == "ja":
        company = rng.choice([f"株式会社{core_ja}", f"{core_ja}株式会社", f"株式会社{core_ja}",
                              f"有限会社{core_ja}", f"合同会社{core_ja}"])
        name = f"{sur[0]} {given[0]}"
        title = rng.choice(p["titles_ja"])
        dept = rng.choice(p["depts_ja"]) if rng.random() < 0.7 else None
        address = f"〒{postal} {city_ja}{block}"
        romaji = f"{given[1]} {sur[1]}" if rng.random() < 0.5 else None
        tel_label = rng.choice(["TEL", "Tel.", "電話", "TEL:"]) if not mobile_only else rng.choice(["携帯", "Mobile"])
        fax_label = rng.choice(["FAX", "Fax.", "FAX:"])
    else:
        company = core_en + rng.choice([" Co., Ltd.", " Inc.", " Corporation", " K.K."])
        name = rng.choice([f"{given[1]} {sur[1]}", f"{given[1]} {sur[1].upper()}"])
        title = rng.choice(p["titles_en"])
        dept = rng.choice(DEPTS_EN) if rng.random() < 0.6 else None
        address = f"{block} {city_en} {postal}, Japan"
        romaji = None
        if not mobile_only:
            tel = "+81-" + tel[1:]
        tel_label = rng.choice(["Tel", "TEL", "T:", "Phone"]) if not mobile_only else "Mobile"
        fax_label = rng.choice(["Fax", "F:"])
        if fax:
            fax = "+81-" + fax[1:]
    return {
        "Name": name, "Company": company, "Title": title, "Email": email, "Phone": tel,
        "_dept": dept, "_address": address, "_romaji": romaji, "_fax": fax,
        "_tel_label": tel_label, "_fax_label": fax_label,
        "_url": f"https://www.{domain}" if rng.random() < 0.5 else None,
        "_email_label": rng.choice(["E-mail:", "Email:", "Mail:", ""]),
    }


def make_corporate(rng: random.Random, p: dict) -> dict:
    core_ja, _, slug = rng.choice(p["companies"])
    city_ja, _, postal, area = rng.choice(p["cities"])
    domain = slug + rng.choice([".co.jp", ".com"])
    if rng.random() < 0.5:
        sur, given = rng.choice(KATAKANA_SURNAMES), rng.choice(KATAKANA_GIVEN)
    else:
        (sur_ja, sur_en), (given_ja, given_en) = rng.choice(SURNAMES), rng.choice(GIVEN)
        sur, given = (sur_ja, sur_en.upper()), (given_ja, given_en.upper())
    pref = rng.randint(10, 47)
    return {
        "Name": f"{sur[0]} {given[0]}",
        "Company": rng.choice([f"{core_ja}株式会社", f"株式会社{core_ja}"]),
        "Title": rng.choice(p["titles_ja"]) if rng.random() < 0.5 else "",  # often departments only
        "Email": f"{given[1][0].lower()}_{sur[1].lower()}@{domain}",
        "Phone": phone_number(rng, area),
        "_romaji": (sur[1], given[1]), "_depts": rng.choice(CORP_DEPTS), "_fax": phone_number(rng, area),
        "_url": f"https://{domain}", "_tagline": rng.choice(CORP_TAGLINES),
        "_address": f"{rng.choice(OFFICES)} 〒{postal} {city_ja}"
                    f"{rng.randint(1, 5)}丁目{rng.randint(1, 20)}番{rng.randint(1, 30)}号",
        "_building": f"{rng.choice(['駅前', '中央', '第一', 'センター'])}ビル{rng.randint(2, 15)}階",
        "_licence": f"一般労働者派遣事業 派{pref}-{rng.randint(100000, 999999)} / "
                    f"有料職業紹介事業 {pref}-ユ-{rng.randint(10000, 99999):06d}",
    }


def render_corporate(rng: random.Random, c: dict) -> Image.Image:
    W, H = 1050, 600
    paper = tuple(rng.randint(245, 255) for _ in range(3))
    img = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(img)
    accent = rng.choice([(35, 55, 140), (20, 90, 60), (150, 30, 40)])
    ink = (25, 25, 25)
    font = rng.choice(FONTS)

    def put(text, xy, size, fill=ink, max_w=None):
        d.text(xy, text, font=fit_font(d, text, font, size, max_w or W - xy[0] - 40), fill=fill)

    # URL banner down the left edge, printed sideways
    band = 90
    strip = Image.new("RGB", (H, band), accent)
    sd = ImageDraw.Draw(strip)
    sd.text((40, 18), c["_url"], font=fit_font(sd, c["_url"], font, 44, H - 80), fill="white")
    img.paste(strip.rotate(90, expand=True), (0, 0))
    # logo, tagline, company
    d.ellipse([140, 40, 205, 100], outline=accent, width=12)
    d.ellipse([180, 40, 245, 100], outline=accent, width=12)
    put(c["_tagline"], (265, 55), 26, fill=accent)
    put(c["Company"], (140, 115), 46)
    # departments (and maybe a title) on the left, the big name to the right with romaji above
    for i, line in enumerate(t for t in (*c["_depts"], c["Title"]) if t):
        put(line, (140, 240 + 35 * i), 27, max_w=245)
    x = 400
    for part, roman in zip(c["Name"].split(" "), c["_romaji"], strict=True):
        name_font = fit_font(d, part, font, 62, 260)
        put(roman, (x + 8, 210), 20)
        d.text((x, 240), part, font=name_font, fill=ink)
        x += d.textlength(part, font=name_font) + 55
    # office address, building, phone line, email, licence footer
    put(c["_address"], (140, 375), 22, max_w=700)
    put(c["_building"], (250, 407), 22)
    put(f"TEL {c['Phone']}(代)    FAX {c['_fax']}", (250, 439), 22, max_w=560)
    put(f"e-mail {c['Email']}", (250, 471), 22, max_w=560)
    put(c["_licence"], (125, 548), 19, max_w=W - 165)
    # QR-code-like block and a round mark
    cell, qx, qy = 6, 870, 300
    grid = [[rng.random() < 0.5 for _ in range(21)] for _ in range(21)]
    for r0, c0 in ((0, 0), (0, 14), (14, 0)):  # the three finder squares
        for r in range(7):
            for col in range(7):
                grid[r0 + r][c0 + col] = r in (0, 6) or col in (0, 6) or (2 <= r <= 4 and 2 <= col <= 4)
    for r in range(21):
        for col in range(21):
            if grid[r][col]:
                d.rectangle([qx + col * cell, qy + r * cell, qx + col * cell + cell - 1, qy + r * cell + cell - 1],
                            fill=ink)
    d.ellipse([860, 450, 950, 540], outline=accent, width=5)
    put("P", (888, 462), 52, fill=accent)
    return img


def fit_font(draw, text, path_idx, size, max_w):
    path, idx = path_idx
    while size > 12:
        font = ImageFont.truetype(path, size, index=idx)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 2
    return ImageFont.truetype(path, size, index=idx)


def render(rng: random.Random, c: dict, lang: str) -> tuple[Image.Image, str]:
    W, H = 1050, 600
    bg = tuple(rng.randint(242, 255) for _ in range(3))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    ink = rng.choice([(20, 20, 20), (45, 45, 45), (25, 40, 90)])
    body_font, name_font = rng.choice(FONTS), rng.choice(FONTS)
    layout = rng.choice(["left", "center", "band"])
    x0 = 60
    if layout == "band":
        band = tuple(rng.randint(30, 160) for _ in range(3))
        d.rectangle([0, 0, 170, H], fill=band)
        x0 = 210
    max_w = W - x0 - 50

    def put(text, y, size, font_choice=None, align=None):
        font = fit_font(d, text, font_choice or body_font, size, max_w)
        w = d.textlength(text, font=font)
        x = (W - w) / 2 if (align or ("center" if layout == "center" else "left")) == "center" else x0
        d.text((x, y), text, font=font, fill=ink)

    y = 45
    put(c["Company"], y, rng.randint(28, 36))
    y += 70
    title_line = f"{c['_dept']} {c['Title']}" if (c["_dept"] and rng.random() < 0.5) else c["Title"]
    if c["_dept"] and title_line == c["Title"]:
        put(c["_dept"], y, rng.randint(20, 24))
        y += 34
    put(title_line, y, rng.randint(22, 26))
    y += 42
    put(c["Name"], y, rng.randint(52, 68), font_choice=name_font)
    y += 88
    if c["_romaji"]:
        put(c["_romaji"], y, rng.randint(18, 22))
    yb = H - 175
    put(c["_address"], yb, rng.randint(18, 21))
    phone_line = f"{c['_tel_label']} {c['Phone']}"
    if c["_fax"]:
        if rng.random() < 0.6:
            phone_line += f"   {c['_fax_label']} {c['_fax']}"
            put(phone_line, yb + 34, rng.randint(18, 21))
        else:
            put(phone_line, yb + 34, rng.randint(18, 21))
            put(f"{c['_fax_label']} {c['_fax']}", yb + 66, rng.randint(18, 21))
            yb += 32
    else:
        put(phone_line, yb + 34, rng.randint(18, 21))
    put(f"{c['_email_label']} {c['Email']}".strip(), yb + 68, rng.randint(18, 21))
    if c["_url"]:
        put(c["_url"], yb + 102, rng.randint(17, 20))
    return img, layout


def photo_degrade(rng: random.Random, card: Image.Image, light_desk: bool = False) -> Image.Image:
    """Make a clean card look like a phone photo. light_desk: a pale desk (hard to tell from a white
    card) and full phone-camera resolution."""
    W, H = card.size
    m = 140
    desk = rng.randint(195, 225) if light_desk else rng.randint(40, 140)
    bgc = np.array([desk] * 3) + np.array([rng.randint(-15, 15) for _ in range(3)])
    canvas = np.full((H + 2 * m, W + 2 * m, 3), np.clip(bgc, 0, 255), dtype=np.uint8)
    canvas[m:m + H, m:m + W] = np.array(card)
    h, w = canvas.shape[:2]
    j = lambda s: rng.uniform(-s, s)  # noqa: E731
    src = np.float32([[m, m], [m + W, m], [m + W, m + H], [m, m + H]])
    dst = np.float32([[m + j(25), m + j(25)], [m + W + j(25), m + j(25)],
                      [m + W + j(25), m + H + j(25)], [m + j(25), m + H + j(25)]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, M, (w, h), borderValue=tuple(int(v) for v in np.clip(bgc, 0, 255)))
    R = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), 1.0)
    out = cv2.warpAffine(out, R, (w, h), borderValue=tuple(int(v) for v in np.clip(bgc, 0, 255)))
    grad = np.linspace(rng.uniform(0.72, 0.9), rng.uniform(0.95, 1.08), w)[None, :, None]
    if rng.random() < 0.5:
        grad = grad[:, ::-1]
    out = np.clip(out.astype(np.float32) * grad + rng.gauss(0, rng.uniform(3, 9)), 0, 255).astype(np.uint8)
    img = Image.fromarray(out).filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.3)))
    scale = rng.uniform(1.3, 2.0) if light_desk else rng.uniform(0.55, 0.8)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--style", choices=["classic", "corporate"], default="classic")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    p = pools(args.split)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(args.n):
            if args.style == "corporate":
                lang, c = "ja", make_corporate(rng, p)
                card, layout = render_corporate(rng, c), "corporate"
                kind = "photo" if rng.random() < 0.7 else "scan"
                img = card
                if kind == "photo":  # like the real photo: often taken sideways
                    img = photo_degrade(rng, card, light_desk=True).rotate(rng.choice([0, 90, 270]), expand=True)
            else:
                lang = "en" if rng.random() < 0.2 else "ja"
                c = make_content(rng, p, lang)
                card, layout = render(rng, c, lang)
                kind = "photo" if rng.random() < 0.5 else "scan"
                img = photo_degrade(rng, card) if kind == "photo" else card
            fname = f"card_{i:04d}.jpg"
            img.save(out / "images" / fname, quality=rng.randint(55, 85) if kind == "photo" else 92)
            label = {k: v for k, v in c.items() if not k.startswith("_")}
            label.update({"id": i, "file": f"images/{fname}", "kind": kind, "lang": lang, "layout": layout})
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} cards to {out}")


if __name__ == "__main__":
    main()
