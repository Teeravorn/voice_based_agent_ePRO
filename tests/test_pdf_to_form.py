"""เทสต์ tools/pdf_to_form.py

ส่วนแรกทดสอบการแก้ข้อความภาษาไทยโดยไม่ต้องใช้ PDF ส่วนหลังรันกับ forms/DASH_Thai.pdf
จริง และใช้ QuickDASH ที่คนตรวจแล้วเป็นเฉลย (QuickDASH 11 ข้อเป็นเซ็ตย่อยของ DASH)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pdf_to_form import (  # noqa: E402
    ERRATA,
    build_form,
    join_lines,
    parse,
    restore_sara_am,
    validate,
)

PDF_PATH = ROOT / "forms" / "DASH_Thai.pdf"
QUICKDASH_PATH = ROOT / "forms" / "quickdash_th.json"


# --- การแก้สระอำ ----------------------------------------------------------------


@pytest.mark.parametrize("broken,fixed", [
    ("ท าอาหาร", "ทำอาหาร"),
    ("ค าถาม", "คำถาม"),
    ("ข้อจ ากัด", "ข้อจำกัด"),
    ("ขัดห้องน ้า", "ขัดห้องน้ำ"),       # ช่องว่างมาก่อนวรรณยุกต์
    ("อาบน ้า", "อาบน้ำ"),
    ("เข็มต า", "เข็มตำ"),
])
def test_restores_sara_am(broken, fixed):
    assert restore_sara_am(broken) == fixed


@pytest.mark.parametrize("text", [
    "ทาสี",                 # "ทา" เป็นคำจริง ไม่มีช่องว่าง ต้องไม่ถูกแตะ
    "ไม่ยาก",
    "โดยทั่วไป ท่าน",       # ช่องว่างจริงระหว่างคำ ตัวที่ตามมาไม่ใช่สระอา
    "ปิดแน่น หรือที่",
])
def test_leaves_valid_text_alone(text):
    assert restore_sara_am(text) == text


# --- การต่อบรรทัด ---------------------------------------------------------------


def test_joins_wrapped_thai_without_space():
    """ภาษาไทยไม่เว้นวรรคระหว่างคำ บรรทัดที่ถูกตัดกลางประโยคต้องต่อกันตรง ๆ"""
    assert join_lines(["ของท่านรบกวน", "การทำกิจกรรม"]) == "ของท่านรบกวนการทำกิจกรรม"


def test_joins_after_punctuation_with_space():
    assert join_lines(["(เช่น ตีกอล์ฟ,", "ตีเทนนิส ฯลฯ)"]) == "(เช่น ตีกอล์ฟ, ตีเทนนิส ฯลฯ)"


# --- รันกับ PDF จริง -------------------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    return json.loads(QUICKDASH_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parsed():
    pymupdf = pytest.importorskip("pymupdf")
    with pymupdf.open(PDF_PATH) as doc:
        return parse(doc)


@pytest.fixture(scope="module")
def built(parsed, reference):
    form, review = build_form(parsed, reference, apply_errata=False, pdf_path=PDF_PATH)
    return form, review


def _items(form):
    return {item["id"]: item for section in form["sections"] for item in section["items"]}


def test_extracts_every_section(built):
    form, _ = built
    counts = {section["id"]: len(section["items"]) for section in form["sections"]}
    assert counts == {"core": 30, "work": 4, "sports": 4}


def test_passes_its_own_validation(built, reference):
    form, _ = built
    assert validate(form, reference) == []


def test_finds_every_quickdash_item(built, reference):
    """ถ้าข้อความดึงมาเพี้ยน จะหา QuickDASH บางข้อใน DASH ไม่เจอ"""
    form, _ = built
    linked = {item.get("quickdash_id") for item in _items(form).values()}
    expected = {item["id"] for item in reference["sections"][0]["items"]}
    assert expected <= linked


def test_quickdash_items_sit_at_their_dash_positions(built):
    """ตำแหน่งตามมาตรฐาน DASH: QuickDASH ข้อ 1-11 = DASH ข้อ 1,7,10,14,16,18,22,23,24,26,29"""
    form, _ = built
    by_quickdash = {item["quickdash_id"]: item["id"]
                    for item in _items(form).values() if "quickdash_id" in item}
    assert by_quickdash == {
        "q1": "d1", "q2": "d7", "q3": "d10", "q4": "d14", "q5": "d16", "q6": "d18",
        "q7": "d22", "q8": "d23", "q9": "d24", "q10": "d26", "q11": "d29",
    }


def test_no_broken_sara_am_left(built):
    form, _ = built
    for item in _items(form).values():
        assert restore_sara_am(item["text_th"]) == item["text_th"], item["id"]


def test_merged_and_split_labels_are_separated(built):
    """'ไม่มีข้อจำกัดมีเล็กน้อย' เป็น span เดียวใน PDF และ 'มีข้อจำกัด'/'มาก' อยู่คนละบรรทัด"""
    form, _ = built
    labels = [option["label_th"] for option in form["scales"]["limitation"]["options"]]
    assert labels == ["ไม่มีข้อจำกัด", "มีเล็กน้อย", "มีปานกลาง", "มีข้อจำกัดมาก", "ทำไม่ได้เลย"]


def test_scales_assigned_by_section(built):
    form, _ = built
    items = _items(form)
    assert {items[f"d{n}"]["scale"] for n in range(1, 22)} == {"difficulty"}
    assert items["d22"]["scale"] == "interference"
    assert items["d23"]["scale"] == "limitation"
    assert {items[f"d{n}"]["scale"] for n in range(24, 29)} == {"severity"}
    assert items["d29"]["scale"] == "sleep_difficulty"
    assert items["d30"]["scale"] == "agreement"
    assert {items[f"w{n}"]["scale"] for n in range(1, 5)} == {"difficulty"}


def test_paper_only_instruction_is_dropped(built):
    form, _ = built
    assert all("วงกลม" not in item["text_th"] for item in _items(form).values())


def test_new_scale_is_flagged_not_invented(built):
    """scale ที่ไม่มีในฟอร์มอ้างอิงต้องเว้น description ว่างและแจ้งให้คนเขียน ห้ามแต่งเอง"""
    form, review = built
    agreement = form["scales"]["agreement"]
    assert agreement["needs_review"] is True
    assert all(option["description_th"] == "" for option in agreement["options"])
    assert any("agreement" in note for note in review)


def test_source_typo_kept_by_default_and_reported(built):
    form, review = built
    assert ERRATA["d24"][0] in _items(form)["d24"]["text_th"]
    assert any(note.startswith("d24") for note in review)


def test_errata_applied_on_request(parsed, reference):
    form, _ = build_form(parsed, reference, apply_errata=True, pdf_path=PDF_PATH)
    items = _items(form)
    assert ERRATA["d24"][0] not in items["d24"]["text_th"]
    assert items["d24"]["text_th"] == "โดยทั่วไป ท่านมีอาการปวดแขน, ไหล่หรือมือ หรือไม่"


def test_output_loads_with_load_form(built, tmp_path):
    from extractor import load_form

    form, _ = built
    path = tmp_path / "dash_th.json"
    path.write_text(json.dumps(form, ensure_ascii=False), encoding="utf-8")
    loaded = load_form(path)
    assert len(loaded.items) == 38
    assert loaded.scale_for(loaded.item("d24")).id == "severity"
