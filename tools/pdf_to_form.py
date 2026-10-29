"""สร้างไฟล์ฟอร์ม JSON จาก PDF แบบสอบถาม DASH ฉบับภาษาไทย

    python tools/pdf_to_form.py
    python tools/pdf_to_form.py --pdf forms/DASH_Thai.pdf --out forms/dash_th.json
    python tools/pdf_to_form.py --apply-errata

ผลลัพธ์อยู่ในโครงสร้างเดียวกับ ``forms/quickdash_th.json`` จึงโหลดด้วย
``extractor.forms.load_form`` และใช้กับ extractor ทุกตัวได้ทันที

## PDF ให้อะไร และไม่ให้อะไร

PDF เป็นแหล่งความจริงของ **โครงสร้างและข้อความคำถาม** ได้แก่ ข้อความทุกข้อ ลำดับ
ป้ายของตัวเลือกแต่ละ scale และการแบ่งชุดคำถาม (หลัก / การทำงาน / กีฬา)

แต่ PDF **ไม่มี** ฟิลด์ที่คนเขียนเพิ่มสำหรับระบบเสียง ซึ่งฟิลด์ที่สำคัญที่สุดคือ
``description_th`` ของแต่ละตัวเลือก (extractor ใช้คำอธิบายนี้เทียบกับสิ่งที่ผู้ป่วยเล่า
ไม่ใช่ป้ายสั้น ๆ) ฟิลด์พวกนี้ดึงมาจากฟอร์มอ้างอิง (``--reuse`` ค่าตั้งต้นคือ
``quickdash_th.json``) เมื่อจับคู่ได้ ส่วนที่จับคู่ไม่ได้จะถูกเว้นว่างแล้วรายงานให้คนเขียนเพิ่ม

QuickDASH 11 ข้อเป็นเซ็ตย่อยของ DASH 30 ข้อ จึงใช้เป็นเฉลยตรวจผลการดึงข้อความได้ด้วย

## ปัญหาภาษาไทยใน PDF นี้

ฟอนต์ Cordia ในไฟล์นี้ map สระอำไม่ถูก ทุก library (pdftotext, pypdf, PyMuPDF) จึงได้
"ท าอาหาร" แทน "ทำอาหาร" สิ่งที่เหลือไว้แทนสระอำคือช่องว่างธรรมดาหนึ่งตัว ซึ่งแก้กลับได้
อย่างปลอดภัยเพราะในภาษาไทยจริง สระอาไม่มีทางตามหลังช่องว่าง (ขึ้นต้นคำไม่ได้)
ดู ``restore_sara_am``

ป้ายตัวเลือกบางอันถูกตัดคนละบรรทัด ("ยากปาน" / "กลาง") และบางอันติดกันเป็น span
เดียวคร่อมสองคอลัมน์ ("ไม่มีข้อจำกัดมีเล็กน้อย") จึงต้องอ่านระดับตัวอักษรแล้วใช้พิกัด x
ตัดสินว่าแต่ละตัวอยู่คอลัมน์ไหน ดู ``read_scale_header``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

# --- ภาษาไทย ---------------------------------------------------------------

_THAI_CONSONANT = "ก-ฮ"
_THAI_TONE = "่-๋"
_SARA_AA = "า"
_SARA_AM = "ำ"

# พยัญชนะ + ช่องว่าง + วรรณยุกต์ (ถ้ามี) + สระอา  ->  พยัญชนะ + วรรณยุกต์ + สระอำ
# "ท าอาหาร" -> "ทำอาหาร" , "น ้า" -> "น้ำ"
_BROKEN_SARA_AM = re.compile(f"([{_THAI_CONSONANT}]) ([{_THAI_TONE}]?){_SARA_AA}")


def restore_sara_am(text: str) -> str:
    """คืนสระอำที่ฟอนต์ใน PDF map ไม่ได้

    ต้องเรียก**ก่อน**ตัดช่องว่างใด ๆ เสมอ เพราะช่องว่างคือตัวบอกตำแหน่งของสระอำ
    """
    return _BROKEN_SARA_AM.sub(lambda m: m.group(1) + m.group(2) + _SARA_AM, text)


def join_lines(lines: list[str]) -> str:
    """ต่อข้อความที่ถูกตัดหลายบรรทัดเป็นบรรทัดเดียว

    ภาษาไทยไม่เว้นวรรคระหว่างคำ ข้อความที่ถูกตัดกลางประโยคจึงต้องต่อกันตรง ๆ
    ("...ของท่านรบกวน" + "การทำกิจกรรม...") ยกเว้นตอนบรรทัดก่อนจบด้วยเครื่องหมาย
    ที่ตามด้วยช่องว่างอยู่แล้ว ("ตีกอล์ฟ," + "ตีเทนนิส") ตรงกับที่ QuickDASH ฉบับที่คน
    ตรวจแล้วเขียนไว้ทุกข้อ
    """
    out = ""
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if out and (out[-1] in ",;:)" or line[0] == "("):
            out += " "
        out += line
    return re.sub(r"\s+", " ", out).strip()


# คำสั่งสำหรับแบบฟอร์มกระดาษ ไม่มีความหมายเมื่ออ่านออกเสียงให้ผู้ป่วยฟัง
_PAPER_ONLY = re.compile(r"\s*\(วงกลมตัวเลข\)\s*")


# --- โครงสร้างข้อมูลระหว่างทาง -------------------------------------------------


@dataclass
class Line:
    page: int
    y: float
    x0: float
    chars: list[dict]

    @property
    def text(self) -> str:
        """ข้อความของบรรทัด พร้อมช่องว่างระหว่างคำที่ PDF เก็บเป็นระยะห่าง

        PDF นี้เก็บวรรคระหว่างคำบางที่เป็นระยะห่างระหว่างตัวอักษร ไม่ใช่ตัวอักษรช่องว่าง
        ("ปิดแน่น หรือที่" ใน rawdict ไม่มีช่องว่างอยู่เลย) จึงต้องแทรกเองเมื่อสองตัวอักษร
        ห่างกันเกินสัดส่วนของขนาดฟอนต์ สระและวรรณยุกต์ที่ลอยทับพยัญชนะมีระยะติดลบ
        จึงไม่ถูกแทรกช่องว่างผิดที่
        """
        out = []
        previous = None
        for ch in self.chars:
            if previous is not None:
                gap = ch["bbox"][0] - previous["bbox"][2]
                height = previous["bbox"][3] - previous["bbox"][1]
                if gap > _WORD_GAP_RATIO * height:
                    out.append(" ")
            out.append(ch["c"])
            previous = ch
        # ยุบช่องว่างซ้อนก่อนแก้สระอำ เพราะตัวแทนสระอำคือช่องว่างตัวเดียวพอดี
        return restore_sara_am(re.sub(r"\s+", " ", "".join(out))).strip()


# ระยะห่างระหว่างตัวอักษร (เทียบกับความสูงตัวอักษร) ที่ถือว่าเป็นการเว้นวรรค
_WORD_GAP_RATIO = 0.2


@dataclass
class ParsedItem:
    section_id: str
    number: int
    lines: list[str] = field(default_factory=list)
    labels: tuple[str, ...] = ()
    option_count: int = 0

    @property
    def text(self) -> str:
        return _PAPER_ONLY.sub(" ", join_lines(self.lines)).strip()


# --- อ่าน PDF -----------------------------------------------------------------

# ข้อความคำถามเริ่มที่ขอบซ้าย ส่วนป้ายตัวเลือกและตัวเลข 1-5 อยู่ฝั่งขวาของหน้า
_LEFT_MAX_X = 120
_RIGHT_MIN_X = 310

_ITEM_START = re.compile(r"^(\d{1,2})\.\s*(.*)$")

# หัวชุดคำถามเสริมในหน้าสุดท้าย คำถามในแต่ละชุดเริ่มนับ 1 ใหม่
_SECTION_MARKERS = (
    ("work", "ชุดคำถามเกี่ยวกับการทำงาน"),
    ("sports", "ชุดคำถามเกี่ยวกับการเล่นกีฬา"),
)


def read_lines(doc: pymupdf.Document) -> list[Line]:
    """อ่านทุกบรรทัดพร้อมตัวอักษรแต่ละตัว เรียงตามหน้าแล้วบนลงล่าง"""
    lines = []
    for page_index, page in enumerate(doc):
        for block in page.get_text("rawdict")["blocks"]:
            for raw in block.get("lines", []):
                chars = [ch for span in raw["spans"] for ch in span["chars"]]
                if not chars or not "".join(ch["c"] for ch in chars).strip():
                    continue
                # ข้ามข้อความที่หมุน (สูตรคิดคะแนนท้ายหน้า 3 พิมพ์แนวตั้ง)
                if abs(raw.get("dir", (1, 0))[1]) > 0.01:
                    continue
                lines.append(Line(page_index, raw["bbox"][1], raw["bbox"][0], chars))
    lines.sort(key=lambda line: (line.page, line.y, line.x0))
    return lines


def is_digit_line(line: Line) -> bool:
    return line.x0 >= _RIGHT_MIN_X and line.text in {"1", "2", "3", "4", "5", "6", "7"}


def column_centers(lines: list[Line]) -> list[float]:
    """หาตำแหน่ง x ของคอลัมน์ตัวเลข 1-5 จากตัวเลขที่พิมพ์อยู่จริงในหน้า

    ไม่ hardcode เพราะถ้า PDF เปลี่ยนขนาดหน้าหรือ layout ตัวเลขจะเลื่อน
    """
    by_digit: dict[int, list[float]] = {}
    for line in lines:
        if is_digit_line(line):
            ch = line.chars[0]
            by_digit.setdefault(int(line.text), []).append((ch["bbox"][0] + ch["bbox"][2]) / 2)
    if not by_digit:
        raise ValueError("ไม่พบตัวเลขของตัวเลือกใน PDF เลย layout อาจไม่ตรงกับที่สคริปต์รองรับ")
    return [sorted(xs)[len(xs) // 2] for _, xs in sorted(by_digit.items())]


def read_scale_header(lines: list[Line], centers: list[float]) -> tuple[str, ...]:
    """รวมป้ายตัวเลือกจากหลายบรรทัดให้เป็นหนึ่งป้ายต่อหนึ่งคอลัมน์

    อ่านระดับตัวอักษรเพราะ span เดียวอาจคร่อมสองคอลัมน์ และใช้ลำดับตัวอักษรเดิมของ PDF
    ภายในบรรทัด (ไม่เรียงตามพิกัด) เพราะวรรณยุกต์และสระบนลอยอยู่สูงกว่าพยัญชนะ ถ้าเรียง
    ตาม y จะได้ "ไมม่ขี้อ" แทน "ไม่มีข้อ"
    """
    bounds = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
    buckets = [""] * len(centers)
    for line in sorted(lines, key=lambda line: line.y):
        for ch in line.chars:
            x = (ch["bbox"][0] + ch["bbox"][2]) / 2
            buckets[sum(x > bound for bound in bounds)] += ch["c"]
    # แก้สระอำก่อน แล้วค่อยลบช่องว่าง ลำดับนี้สลับไม่ได้
    return tuple(re.sub(r"\s+", "", restore_sara_am(bucket)) for bucket in buckets)


def parse(doc: pymupdf.Document) -> list[ParsedItem]:
    lines = read_lines(doc)
    centers = column_centers(lines)

    items: list[ParsedItem] = []
    section_id = "core"
    header_lines: list[Line] = []   # ป้ายตัวเลือกที่กำลังสะสม
    current_labels: tuple[str, ...] = ()
    current: ParsedItem | None = None
    page = -1

    for line in lines:
        text = line.text

        if line.page != page:
            # หัวตารางไม่เคยข้ามหน้า สิ่งที่ค้างจากหน้าก่อนคือข้อความท้ายหน้าที่ไม่ใช่ป้าย
            # (สูตรคิดคะแนนท้ายหน้า 3 พิมพ์อยู่ฝั่งขวาพอดี ถ้าไม่ล้างจะไปปนกับหัวตารางหน้า 4)
            page = line.page
            header_lines = []

        for marker_id, marker in _SECTION_MARKERS:
            if marker in text:
                section_id = marker_id
                current = None
                header_lines = []
                break

        if line.x0 >= _RIGHT_MIN_X:
            if is_digit_line(line):
                # แถวตัวเลขยืนยันว่าข้อนี้มีกี่ตัวเลือก ป้ายที่สะสมมาก่อนหน้าถือว่าจบแล้ว
                if header_lines:
                    current_labels = read_scale_header(header_lines, centers)
                    header_lines = []
                    if current is not None and not current.labels:
                        current.labels = current_labels
                if current is not None:
                    current.option_count += 1
            elif current is None or current.option_count:
                # ข้อความฝั่งขวาที่ไม่ใช่ตัวเลข ระหว่างข้อ = หัวตารางของ scale ถัดไป
                header_lines.append(line)
            continue

        if line.x0 > _LEFT_MAX_X:
            continue

        match = _ITEM_START.match(text)
        if match:
            if header_lines:
                current_labels = read_scale_header(header_lines, centers)
                header_lines = []
            current = ParsedItem(section_id, int(match.group(1)), [match.group(2)], current_labels)
            items.append(current)
        elif current is not None and not current.option_count:
            # บรรทัดต่อของข้อเดิม (ยังไม่ถึงแถวตัวเลข)
            current.lines.append(text)
        elif current is not None and current.option_count and _continues_item(text):
            # บางข้อพิมพ์ตัวเลขไว้บรรทัดแรก แล้วข้อความบรรทัดสองตามหลังมา
            current.lines.append(text)

    return items


def _continues_item(text: str) -> bool:
    """บรรทัดซ้ายที่ตามหลังแถวตัวเลข ยังเป็นส่วนของข้อเดิมไหม

    ตัดบรรทัดคำสั่ง ลิขสิทธิ์ และคำอธิบายการคิดคะแนน ซึ่งไม่ใช่ข้อความคำถาม
    """
    noise = ("©", "คะแนน", "ขณะที่", "โปรด", "DISABILITIES", "ชุดคำถาม", "O ไม่ต้อง",
             "คำถามต่อไปนี้", "ให้บอก", "หากท่าน", "Thai translation", "Thailand", "การคิดคะแนน")
    return bool(text) and not any(text.startswith(word) or word in text for word in noise)


# --- ประกอบเป็นฟอร์ม -----------------------------------------------------------

# รหัสข้อในแต่ละชุด ใช้ d สำหรับ DASH 30 ข้อ เพื่อไม่ให้สับสนกับ q ของ QuickDASH
# (DASH ข้อ 24 คือ QuickDASH ข้อ 9 ถ้าใช้ q ทั้งคู่จะชนกัน) ส่วน w/s ตรงกับ QuickDASH
_ID_PREFIX = {"core": "d", "work": "w", "sports": "s"}

_SECTION_NAMES = {
    "core": "ชุดคำถามหลัก",
    "work": "ชุดคำถามเกี่ยวกับการทำงาน",
    "sports": "ชุดคำถามเกี่ยวกับการเล่นกีฬา/การทำงานด้านศิลปะ",
}

# เกณฑ์คิดคะแนนตามที่พิมพ์ไว้ใน PDF
#   หน้า 3: "คะแนน DASH ไม่สามารถคิดคะแนนได้ หากมีข้อที่ไม่ตอบเกิน 3 ข้อ"
#   หน้า 4: "คะแนนชุดคำถามนี้ไม่สามารถคิดคะแนนได้หากไม่ตอบข้อใดข้อหนึ่ง"
_SCORING = {
    "core": {"method": "dash", "total_items": 30, "min_answered": 27,
             "note_th": "คะแนน DASH ไม่สามารถคิดคะแนนได้ หากมีข้อที่ไม่ตอบเกิน 3 ข้อ"},
    "work": {"method": "dash", "total_items": 4, "min_answered": 4,
             "note_th": "คะแนนชุดคำถามนี้ไม่สามารถคิดคะแนนได้หากไม่ตอบข้อใดข้อหนึ่ง"},
    "sports": {"method": "dash", "total_items": 4, "min_answered": 4,
               "note_th": "คะแนนชุดคำถามนี้ไม่สามารถคิดคะแนนได้หากไม่ตอบข้อใดข้อหนึ่ง"},
}

# ข้อผิดพลาดในตัว PDF เอง ใช้เฉพาะเมื่อสั่ง --apply-errata
# ไม่แก้เงียบ ๆ เพราะเป็นข้อความทางคลินิกที่จะถูกอ่านให้ผู้ป่วยฟัง คนต้องเห็นและตัดสินเอง
ERRATA: dict[str, tuple[str, str, str]] = {
    "d24": ("อาการอาการ", "อาการ", "คำซ้ำในต้นฉบับ QuickDASH ฉบับที่ตรวจแล้วเขียนว่า 'มีอาการปวด'"),
    "d25": ("อาการอาการ", "อาการ", "คำซ้ำในต้นฉบับ แบบเดียวกับข้อ 24"),
}

# scale ที่ไม่มีใน QuickDASH ต้องตั้งชื่อเอง จับคู่จากป้ายตัวเลือกตัวแรกกับตัวสุดท้าย
_NEW_SCALE_IDS = {("ไม่เห็นด้วยอย่างยิ่ง", "เห็นด้วยอย่างยิ่ง"): ("agreement", "ระดับความเห็นด้วย")}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def build_form(items: list[ParsedItem], reference: dict | None, apply_errata: bool,
               pdf_path: Path) -> tuple[dict, list[str]]:
    """ประกอบผลที่ parse ได้ให้เป็นฟอร์ม คืนฟอร์มและรายการสิ่งที่คนต้องตรวจ"""
    review: list[str] = []
    reference = reference or {}
    ref_scales = reference.get("scales", {})
    ref_by_labels = {
        tuple(option["label_th"] for option in scale["options"]): scale_id
        for scale_id, scale in ref_scales.items()
    }
    ref_items = {
        _normalize(item["text_th"]): item
        for section in reference.get("sections", [])
        for item in section.get("items", [])
    }
    ref_sections = {section["id"]: section for section in reference.get("sections", [])}

    scales: dict[str, dict] = {}
    new_scale_ids: dict[tuple[str, ...], str] = {}   # scale ใหม่ กันไม่ให้ข้อที่ใช้ป้ายชุดเดียวกันได้คนละ id
    sections: dict[str, list[dict]] = {}

    for parsed in items:
        if not parsed.labels:
            raise ValueError(f"ข้อ {parsed.section_id} {parsed.number} ไม่มีหัวตารางตัวเลือกนำหน้า")
        if parsed.option_count != len(parsed.labels):
            raise ValueError(
                f"ข้อ {parsed.section_id} {parsed.number}: มีตัวเลข {parsed.option_count} ตัว "
                f"แต่ป้ายตัวเลือก {len(parsed.labels)} อัน {parsed.labels}"
            )

        scale_id = ref_by_labels.get(parsed.labels) or new_scale_ids.get(parsed.labels)
        if scale_id is not None and scale_id in ref_scales:
            scales.setdefault(scale_id, ref_scales[scale_id])
        elif scale_id is None:
            key = (parsed.labels[0], parsed.labels[-1])
            scale_id, name_th = _NEW_SCALE_IDS.get(key, (f"scale_{len(new_scale_ids) + 1}", ""))
            new_scale_ids[parsed.labels] = scale_id
            if scale_id not in scales:
                scales[scale_id] = {
                    "id": scale_id,
                    "name_th": name_th,
                    "options_prompt_th": "",
                    "options": [
                        {"value": value, "label_th": label, "description_th": "", "aliases": []}
                        for value, label in enumerate(parsed.labels, start=1)
                    ],
                    "needs_review": True,
                }
                review.append(
                    f"scale '{scale_id}' {list(parsed.labels)} ไม่มีในฟอร์มอ้างอิง "
                    "ต้องเขียน description_th ของทุกตัวเลือก และ options_prompt_th เพิ่ม "
                    "(extractor ใช้ description_th เทียบกับคำตอบ ถ้าว่างจะเดาไม่ได้)"
                )

        item_id = f"{_ID_PREFIX[parsed.section_id]}{parsed.number}"
        text = parsed.text
        if apply_errata and item_id in ERRATA:
            wrong, right, why = ERRATA[item_id]
            if wrong in text:
                text = text.replace(wrong, right, 1)
                review.append(f"{item_id}: แก้ '{wrong}' -> '{right}' ({why})")
        elif item_id in ERRATA and ERRATA[item_id][0] in text:
            review.append(
                f"{item_id}: พบ '{ERRATA[item_id][0]}' ในต้นฉบับ น่าจะพิมพ์ผิด "
                "(คงไว้ตาม PDF ใช้ --apply-errata ถ้าจะแก้)"
            )

        item = {"id": item_id, "number": parsed.number, "scale": scale_id, "text_th": text}

        # จับคู่กับ QuickDASH ด้วยข้อความที่แก้คำผิดแล้วเสมอ เพราะเป็นการระบุว่าเป็นข้อเดียวกัน
        # ไม่ใช่การเลือกข้อความที่จะใช้ (text_th ข้างบนยังคงตามต้นฉบับถ้าไม่ได้สั่ง --apply-errata)
        corrected = text
        if item_id in ERRATA:
            corrected = corrected.replace(ERRATA[item_id][0], ERRATA[item_id][1], 1)
        matched = ref_items.get(_normalize(corrected))
        if matched is not None and matched["text_th"] != text:
            # จับคู่ได้แต่เขียนไม่เหมือนกันทุกตัวอักษร ฉบับอ้างอิงถูกคนแก้มาจากต้นฉบับ
            # (เช่นเติมวรรค "หนัก ๆ") ไม่ใช่ปัญหาของการดึงข้อความ แต่คนควรเลือกว่าจะใช้ฉบับไหน
            review.append(
                f"{item_id}: ต่างจาก {matched['id']} ในฟอร์มอ้างอิง  "
                f"PDF='{text}'  อ้างอิง='{matched['text_th']}'"
            )
        if matched is not None:
            # ข้อที่ตรงกับ QuickDASH ได้ฟิลด์สำหรับระบบเสียงที่คนเขียนไว้แล้วมาด้วย
            for key in ("prompt_th", "image", "image_caption_th", "keywords_th"):
                if key in matched:
                    item[key] = matched[key]
            if parsed.section_id == "core":
                item["quickdash_id"] = matched["id"]
        else:
            item["prompt_th"] = text

        sections.setdefault(parsed.section_id, []).append(item)

    form = {
        "form_id": "dash_th",
        "version": "1.0.0",
        "title_th": "DASH ฉบับภาษาไทย",
        "title_en": "Disabilities of the Arm, Shoulder and Hand (Thai)",
        "language": "th",
        "source_document": str(pdf_path.relative_to(ROOT)).replace("\\", "/")
        if pdf_path.is_relative_to(ROOT) else str(pdf_path),
        "generated_by": "tools/pdf_to_form.py",
    }
    for key in ("copyright_th", "recall_period_th", "instructions_th", "speech"):
        if key in reference:
            form[key] = reference[key]
    if "speech" in form:
        review.append(
            "speech.greeting_th คัดลอกมาจาก QuickDASH ซึ่งบอกผู้ป่วยว่ามี 'สิบเอ็ดข้อ' "
            "ต้องแก้เป็นจำนวนข้อของ DASH ก่อนใช้งาน"
        )

    if "yes_no" in ref_scales and any(sid in sections for sid in ("work", "sports")):
        scales.setdefault("yes_no", ref_scales["yes_no"])
    form["scales"] = scales

    form["sections"] = []
    for section_id, section_items in sections.items():
        section = {
            "id": section_id,
            "name_th": _SECTION_NAMES[section_id],
            "required": section_id == "core",
        }
        ref_section = ref_sections.get(section_id, {})
        for key in ("gate", "free_text", "intro_th"):
            if key in ref_section:
                section[key] = ref_section[key]
        section["scoring"] = _SCORING[section_id]
        section["items"] = section_items
        form["sections"].append(section)

    return form, review


# --- ตรวจผล -------------------------------------------------------------------


def validate(form: dict, reference: dict | None) -> list[str]:
    """ตรวจว่าผลที่ได้ครบและตรงกับสิ่งที่รู้อยู่แล้วเกี่ยวกับ DASH คืนรายการปัญหา"""
    problems = []
    expected = {"core": 30, "work": 4, "sports": 4}
    by_section = {section["id"]: section["items"] for section in form["sections"]}

    for section_id, count in expected.items():
        numbers = [item["number"] for item in by_section.get(section_id, [])]
        if numbers != list(range(1, count + 1)):
            problems.append(f"ชุด {section_id}: ได้ข้อ {numbers} แต่ควรเป็น 1..{count}")

    for section in form["sections"]:
        for item in section["items"]:
            if item["scale"] not in form["scales"]:
                problems.append(f"{item['id']}: อ้าง scale '{item['scale']}' ที่ไม่มี")
            if re.search(f"[{_THAI_CONSONANT}] [{_THAI_TONE}]?{_SARA_AA}", item["text_th"]):
                problems.append(f"{item['id']}: ยังมีสระอำที่แก้ไม่ครบ '{item['text_th']}'")

    if reference:
        # QuickDASH ทุกข้อในชุดหลักต้องหาเจอใน DASH เป็นตัวพิสูจน์ว่าดึงข้อความถูก
        found = {item.get("quickdash_id") for item in by_section.get("core", [])}
        for ref_section in reference.get("sections", []):
            if ref_section["id"] != "core":
                continue
            for ref_item in ref_section["items"]:
                if ref_item["id"] not in found:
                    problems.append(
                        f"หาข้อ QuickDASH {ref_item['id']} '{ref_item['text_th']}' ใน DASH ไม่เจอ "
                        "ข้อความที่ดึงได้อาจเพี้ยน หรือต้นฉบับสองฉบับเขียนต่างกัน"
                    )
    return problems


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "forms" / "DASH_Thai.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "forms" / "dash_th.json")
    parser.add_argument("--reuse", type=Path, default=ROOT / "forms" / "quickdash_th.json",
                        help="ฟอร์มที่คนเขียนไว้แล้ว ใช้ดึง description_th, prompt_th ฯลฯ "
                             "ที่ PDF ไม่มี และใช้ตรวจผล ใส่ 'none' ถ้าไม่ต้องการ")
    parser.add_argument("--apply-errata", action="store_true",
                        help="แก้คำที่พิมพ์ผิดในต้นฉบับตามตาราง ERRATA (ค่าตั้งต้นคงไว้ตาม PDF)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    reference = None
    if str(args.reuse).lower() != "none":
        reference = json.loads(args.reuse.read_text(encoding="utf-8"))

    with pymupdf.open(args.pdf) as doc:
        items = parse(doc)

    form, review = build_form(items, reference, args.apply_errata, args.pdf.resolve())
    problems = validate(form, reference)

    for section in form["sections"]:
        print(f"\n[{section['id']}] {section['name_th']} — {len(section['items'])} ข้อ")
        for item in section["items"]:
            tag = f"  = QuickDASH {item['quickdash_id']}" if "quickdash_id" in item else ""
            print(f"  {item['id']:<4} [{item['scale']:<16}] {item['text_th']}{tag}")

    print("\nscale ที่ใช้:")
    for scale_id, scale in form["scales"].items():
        labels = [option["label_th"] for option in scale["options"]]
        flag = "  <- ต้องเขียน description เพิ่ม" if scale.get("needs_review") else ""
        print(f"  {scale_id:<17} {labels}{flag}")

    if problems:
        print("\n!! ตรวจไม่ผ่าน ไม่ได้เขียนไฟล์:")
        for problem in problems:
            print(f"   - {problem}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(form, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nเขียนไฟล์แล้ว: {args.out}")

    if review:
        print("\nสิ่งที่คนต้องตรวจ/เขียนเพิ่ม:")
        for note in review:
            print(f"   - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
