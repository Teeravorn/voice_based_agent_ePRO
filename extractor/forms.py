"""โหลดแบบฟอร์ม PROM จากไฟล์ JSON ให้อยู่ในรูป object ที่ส่วนอื่นใช้ต่อได้

โครงสร้างของไฟล์ฟอร์มแยก "คำถาม" (item) ออกจาก "ชุดตัวเลือก" (scale) เพราะคำถาม
หลายข้อใช้ scale เดียวกัน เช่น QuickDASH ข้อ 1-6 ใช้ scale ``difficulty`` ทั้งหมด
การแยกแบบนี้ทำให้เพิ่มคำถามใหม่ได้โดยไม่ต้องเขียนคำอธิบายคะแนนซ้ำ และทำให้ตอนสร้าง
JSON schema ให้ LLM รู้ได้ว่าคำถามข้อนี้ตอบได้กี่ค่า

``description_th`` ของแต่ละตัวเลือกคือหัวใจ ไม่ใช่ ``label_th`` เพราะผู้ป่วยไม่ได้พูด
ชื่อตัวเลือกออกมาตรง ๆ ตัวที่ LLM ต้องใช้เทียบคือคำอธิบายว่าอาการระดับนั้นหน้าตาเป็นยังไง
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Option:
    """ตัวเลือกหนึ่งค่าใน scale (เช่น 3 = ยากปานกลาง)"""

    value: int
    label_th: str
    description_th: str
    # aliases ยังไม่ถูกใช้ใน POC นี้ เก็บไว้ให้ rule-based extractor ในอนาคต
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scale:
    """ชุดตัวเลือกแบบ ordinal หนึ่งชุด"""

    id: str
    name_th: str
    options_prompt_th: str
    options: tuple[Option, ...]

    @property
    def values(self) -> list[int]:
        return [option.value for option in self.options]

    def is_valid(self, value: int) -> bool:
        return value in self.values

    def label(self, value: int) -> str | None:
        for option in self.options:
            if option.value == value:
                return option.label_th
        return None


@dataclass(frozen=True)
class Item:
    """คำถามหนึ่งข้อ"""

    id: str
    number: int
    scale_id: str
    text_th: str
    prompt_th: str
    section_id: str
    image: str | None = None
    keywords_th: tuple[str, ...] = ()


@dataclass(frozen=True)
class Form:
    form_id: str
    version: str
    title_th: str
    recall_period_th: str
    scales: dict[str, Scale] = field(default_factory=dict)
    items: dict[str, Item] = field(default_factory=dict)

    def item(self, item_id: str) -> Item:
        try:
            return self.items[item_id]
        except KeyError:
            known = ", ".join(self.items)
            raise KeyError(f"ไม่มีคำถาม {item_id!r} ในฟอร์ม {self.form_id} (มีเฉพาะ: {known})") from None

    def scale(self, scale_id: str) -> Scale:
        try:
            return self.scales[scale_id]
        except KeyError:
            known = ", ".join(self.scales)
            raise KeyError(f"ไม่มี scale {scale_id!r} ในฟอร์ม {self.form_id} (มีเฉพาะ: {known})") from None

    def scale_for(self, item: Item) -> Scale:
        return self.scale(item.scale_id)


def load_form(path: str | Path) -> Form:
    """อ่านไฟล์ฟอร์ม JSON แล้วคืน ``Form``

    คำถามไม่ได้อยู่ที่ระดับบนสุดของไฟล์ แต่ซ้อนอยู่ใน ``sections[].items[]``
    เพราะฟอร์มจริงมีส่วนเสริม (module การทำงาน / เล่นกีฬา) ที่ถามเฉพาะผู้ป่วยบางคน
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    scales = {
        scale_id: Scale(
            id=scale_raw.get("id", scale_id),
            name_th=scale_raw.get("name_th", ""),
            options_prompt_th=scale_raw.get("options_prompt_th", ""),
            options=tuple(
                Option(
                    value=int(option["value"]),
                    label_th=option["label_th"],
                    description_th=option.get("description_th", ""),
                    aliases=tuple(option.get("aliases", ())),
                )
                for option in scale_raw["options"]
            ),
        )
        for scale_id, scale_raw in raw["scales"].items()
    }

    items: dict[str, Item] = {}
    for section in raw.get("sections", []):
        section_id = section.get("id", "")
        for item_raw in section.get("items", []):
            item_id = item_raw["id"]
            scale_id = item_raw["scale"]
            if scale_id not in scales:
                raise ValueError(
                    f"คำถาม {item_id!r} ในไฟล์ {path} อ้าง scale {scale_id!r} ที่ไม่มีอยู่จริง"
                )
            if item_id in items:
                raise ValueError(f"คำถาม id ซ้ำกัน: {item_id!r} ในไฟล์ {path}")
            items[item_id] = Item(
                id=item_id,
                number=int(item_raw.get("number", 0)),
                scale_id=scale_id,
                text_th=item_raw["text_th"],
                prompt_th=item_raw.get("prompt_th", item_raw["text_th"]),
                section_id=section_id,
                image=item_raw.get("image"),
                keywords_th=tuple(item_raw.get("keywords_th", ())),
            )

    return Form(
        form_id=raw["form_id"],
        version=raw.get("version", ""),
        title_th=raw.get("title_th", ""),
        recall_period_th=raw.get("recall_period_th", ""),
        scales=scales,
        items=items,
    )
