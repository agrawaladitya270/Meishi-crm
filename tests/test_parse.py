from meishi.ocr import Line, normalize
from meishi.parse import find_company, find_email, find_name, find_phone, find_title, repair_email_line


def L(text: str, height: float = 30, conf: float = 0.9) -> Line:
    return Line(normalize(text), conf, height)


def test_normalize_removes_spaces_between_japanese_characters():
    assert normalize("山 田  太 郎") == "山田太郎"
    assert normalize("ＴＥＬ　０３") == "TEL 03"


def test_phone_skips_fax_and_postal_code():
    lines = [L("〒530-0001 大阪府大阪市北区梅田1-2-3"), L("FAX 06-1111-2222  TEL 06-3333-4444")]
    assert find_phone(lines)[0] == "06-3333-4444"


def test_phone_falls_back_to_mobile():
    assert find_phone([L("携帯 090-1234-5678")])[0] == "090-1234-5678"


def test_phone_international_format():
    assert find_phone([L("Tel +81-3-1234-5678")])[0] == "+81-3-1234-5678"


def test_email_repairs_common_ocr_damage():
    assert repair_email_line("misaki, hashimoto@mi rai-tech. jp") == "misaki.hashimoto@mirai-tech.jp"
    assert repair_email_line("ikeda@hikari-d.cojp") == "ikeda@hikari-d.co.jp"
    assert find_email([L("E-mail: Yumi.Ito@toa-sys.com")])[0] == "yumi.ito@toa-sys.com"
    assert repair_email_line("e-mail k_ tanaka@abc.co.jp") == "k_tanaka@abc.co.jp"


def test_name_is_the_tallest_name_like_line_not_the_department():
    lines = [L("株式会社東亜システム", 40), L("マーケティング部", 30), L("取締役", 30),
             L("伊藤 由美", 90), L("Yumi Ito", 20)]
    assert find_name(lines)[0] == "伊藤由美"


def test_english_name():
    lines = [L("Hikari Denshi Inc.", 40), L("Sales Division", 25), L("Tomoko SUZUKI", 70)]
    assert find_name(lines)[0] == "Tomoko SUZUKI"


def test_company_and_title():
    lines = [L("Mail: taro@abc.co.jp"), L("合同会社未来テクノロジー"), L("開発本部 主任研究員")]
    assert find_company(lines)[0] == "合同会社未来テクノロジー"
    assert "主任研究員" in find_title(lines)[0]


def test_company_suffix_with_misread_character():
    assert find_company([L("TECHNICAL SOLUTION"), L("東亜工業株式雲社")])[0] == "東亜工業株式会社"


def test_logo_read_as_hiragana_is_not_the_name():
    assert find_name([L("のの", 150), L("スミス ジョン", 130)])[0] == "スミスジョン"


def test_engineering_department_is_not_a_title():
    assert find_title([L("Engineering Dept.")])[0] == ""
