"""อ่านไฟล์ manifest ที่เก็บสคริปต์คำตอบและเฉลย

หนึ่งแถว = คำตอบหนึ่งอันของผู้ป่วยสมมติ ประกอบด้วย
- ``script_th`` ประโยคที่ใช้อัดเสียง และใช้เป็น reference transcript ตอนรันโหมด text
- ``gold_value`` คะแนนเฉลย รองรับหลายค่าคั่นด้วย ``|`` สำหรับคำตอบที่กำกวมจริง ๆ
  ซึ่งพยาบาลสองคนก็อาจให้คะแนนต่างกัน การบังคับให้มีเฉลยเดียวในเคสแบบนี้
  จะทำให้ตัวเลขความแม่นต่ำกว่าความจริง
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {"audio", "item_id", "answer_type", "script_th", "gold_value"}


@dataclass(frozen=True)
class ManifestRow:
    audio: str
    item_id: str
    answer_type: str
    script_th: str
    gold_values: tuple[int, ...]   # เฉลยที่ยอมรับได้ (0 = ไม่ควรให้คะแนน)
    gold_meta: str | None
    note: str = ""

    @property
    def is_scorable(self) -> bool:
        """แถวนี้ควรได้คะแนนไหม (0 = ไม่ควร เช่น ขอให้ทวนคำถาม)"""
        return self.gold_values != (0,)

    @property
    def gold_display(self) -> str:
        return "|".join(str(v) for v in self.gold_values)


def load_manifest(path: str | Path) -> list[ManifestRow]:
    path = Path(path)
    # utf-8-sig เพราะไฟล์มี BOM ให้ Excel เปิดภาษาไทยได้ถูก
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest {path} ขาดคอลัมน์: {sorted(missing)}")

        rows = []
        for line_no, raw in enumerate(reader, start=2):
            gold_raw = (raw.get("gold_value") or "").strip()
            try:
                gold_values = tuple(int(part) for part in gold_raw.split("|") if part.strip())
            except ValueError:
                raise ValueError(
                    f"manifest {path} บรรทัด {line_no}: gold_value {gold_raw!r} ไม่ใช่ตัวเลข"
                ) from None
            if not gold_values:
                raise ValueError(f"manifest {path} บรรทัด {line_no}: ไม่ได้ใส่ gold_value")

            meta = (raw.get("gold_meta") or "").strip()
            rows.append(
                ManifestRow(
                    audio=(raw.get("audio") or "").strip(),
                    item_id=(raw.get("item_id") or "").strip(),
                    answer_type=(raw.get("answer_type") or "").strip(),
                    script_th=(raw.get("script_th") or "").strip(),
                    gold_values=gold_values,
                    gold_meta=meta or None,
                    note=(raw.get("note") or "").strip(),
                )
            )
    return rows
