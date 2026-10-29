"""Score calculator: แปลงคะแนนรายข้อเป็นคะแนนรวมตามสูตรของแบบประเมินต้นฉบับ

การคิดคะแนนทำด้วยโปรแกรมล้วน ๆ ไม่ผ่านโมเดลภาษา เพื่อให้ตรงกับคู่มือของแบบประเมิน
และได้ผลเท่าเดิมทุกครั้ง

กฎของแต่ละชุดคำถามมาจาก ``sections[].scoring`` ในไฟล์ฟอร์ม ไม่ได้เขียนตายไว้ในโค้ด

- ``method``        ชื่อสูตร (ดู ``METHODS``)
- ``total_items``   จำนวนข้อของชุดนั้น ใช้ตรวจว่าไฟล์ฟอร์มไม่ขาดข้อ
- ``min_answered``  ตอบน้อยกว่านี้คิดคะแนนไม่ได้

## สูตร DASH (forms/Scoring_DASH_Thai.pdf)

    คะแนน = ((ผลรวมคะแนนของข้อที่ตอบ / n) - 1) x 25     n = จำนวนข้อที่ตอบ

ได้ช่วง 0-100 คะแนนสูงหมายถึงความบกพร่องสูง ใช้สูตรเดียวกันทั้งชุดหลักและชุดเสริม
(การทำงาน / กีฬา) ต่างกันแค่จำนวนข้อขั้นต่ำ QuickDASH ใช้สูตรเดียวกันกับ 11 ข้อ

## ข้อที่ไม่มีคำตอบ

ข้อที่ dialogue manager ส่งให้พยาบาลตรวจ (flagged) ยังไม่มีคะแนน จึงนับเป็นข้อที่ไม่ตอบ
จนกว่าพยาบาลจะยืนยัน แล้วค่อยส่งค่าที่ยืนยันเข้ามาคิดใหม่
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from extractor.forms import Form, Section


class ScoreStatus(str, Enum):
    SCORED = "scored"
    INSUFFICIENT = "insufficient"          # ตอบไม่ถึงจำนวนข้อขั้นต่ำ
    NOT_ADMINISTERED = "not_administered"  # ชุดเสริมที่ไม่ได้ถาม (ไม่มีคำตอบเลย)


@dataclass(frozen=True)
class SectionScore:
    section_id: str
    method: str
    status: ScoreStatus
    score: float | None
    answered: int
    total_items: int
    min_answered: int
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "method": self.method,
            "status": self.status.value,
            "score": None if self.score is None else round(self.score, 2),
            "answered": self.answered,
            "total_items": self.total_items,
            "min_answered": self.min_answered,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class FormScore:
    form_id: str
    sections: dict[str, SectionScore] = field(default_factory=dict)

    def score(self, section_id: str) -> float | None:
        return self.sections[section_id].score

    def to_dict(self) -> dict:
        return {
            "form_id": self.form_id,
            "sections": {section_id: score.to_dict() for section_id, score in self.sections.items()},
        }


def dash_score(values: list[int]) -> float:
    """((ผลรวม / n) - 1) x 25"""
    return (sum(values) / len(values) - 1) * 25


# ชื่อสูตรใน ``scoring.method`` ของไฟล์ฟอร์ม -> ฟังก์ชันที่รับคะแนนของข้อที่ตอบ
# TESS จะเพิ่มที่นี่เมื่อมีไฟล์ฟอร์ม
METHODS: dict[str, Callable[[list[int]], float]] = {
    "dash": dash_score,
}


def score_section(form: Form, section: Section, answers: Mapping[str, int | None]) -> SectionScore:
    rules = section.scoring
    method = rules.get("method")
    if method not in METHODS:
        known = ", ".join(METHODS)
        raise ValueError(
            f"ชุดคำถาม {section.id!r} ของฟอร์ม {form.form_id} ใช้สูตร {method!r} ที่ยังไม่รองรับ (มีเฉพาะ: {known})"
        )

    total_items = int(rules.get("total_items", len(section.item_ids)))
    if total_items != len(section.item_ids):
        raise ValueError(
            f"ชุดคำถาม {section.id!r} ของฟอร์ม {form.form_id} ระบุว่ามี {total_items} ข้อ "
            f"แต่ในไฟล์มี {len(section.item_ids)} ข้อ"
        )
    min_answered = int(rules.get("min_answered", total_items))

    values: list[int] = []
    missing: list[str] = []
    for item_id in section.item_ids:
        value = answers.get(item_id)
        if value is None:
            missing.append(item_id)
            continue
        scale = form.scale_for(form.item(item_id))
        if not scale.is_valid(value):
            raise ValueError(f"คะแนน {value!r} ของข้อ {item_id} ไม่อยู่ใน scale {scale.id} ({scale.values})")
        values.append(value)

    if not values and not section.required:
        status, score = ScoreStatus.NOT_ADMINISTERED, None
    elif len(values) < min_answered:
        status, score = ScoreStatus.INSUFFICIENT, None
    else:
        status, score = ScoreStatus.SCORED, METHODS[method](values)

    return SectionScore(
        section_id=section.id,
        method=method,
        status=status,
        score=score,
        answered=len(values),
        total_items=total_items,
        min_answered=min_answered,
        missing=tuple(missing),
    )


def score_form(form: Form, answers: Mapping[str, int | None]) -> FormScore:
    """คิดคะแนนทุกชุดคำถามของฟอร์ม ``answers`` คือ item_id -> คะแนน (None = ไม่มีคำตอบ)"""
    unknown = [item_id for item_id in answers if item_id not in form.items]
    if unknown:
        raise KeyError(f"ไม่มีคำถาม {', '.join(unknown)} ในฟอร์ม {form.form_id}")
    return FormScore(
        form_id=form.form_id,
        sections={section.id: score_section(form, section, answers) for section in form.sections},
    )
