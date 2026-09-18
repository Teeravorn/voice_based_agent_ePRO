"""Option extractor: แปลงคำตอบภาษาพูดของผู้ป่วยให้เป็นคะแนนของแบบฟอร์ม

นี่คือชิ้นที่เสี่ยงที่สุดของระบบทั้งหมด เพราะผู้ป่วยไม่ได้ตอบเป็นตัวเลือก แต่เล่าอาการ
ด้วยภาษาพูด และคะแนนที่ได้จะถูกเอาไปเทียบกับคะแนนที่พยาบาลให้

หลักสำคัญ: **ไม่เชื่อผลจากโมเดลโดยไม่ตรวจ** value ต้องอยู่ใน scale จริง ๆ และ label
ต้องเอามาจากไฟล์ฟอร์มเสมอ ไม่ใช้ข้อความที่โมเดลสร้างขึ้นเอง
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from engine import ChatEngine, LLMUnavailable

from .forms import Form, Item, Scale
from .prompt import SYSTEM_PROMPT, UNDECIDED, build_schema, build_user_prompt


@dataclass
class ExtractionResult:
    """ผลการแปลงคำตอบหนึ่งครั้ง"""

    value: int | None = None          # คะแนนที่ map ได้ (None = แปลงไม่ได้)
    label: str | None = None          # label_th จากฟอร์ม ไม่ใช่จากโมเดล
    confidence: float = 0.0
    meta_intent: str | None = None    # "none" จะถูกแปลงเป็น None
    reason_th: str = ""
    latency_ms: float = 0.0
    error: str | None = None          # ตั้งเมื่อเรียกโมเดลไม่ได้ / ผลใช้ไม่ได้
    raw: dict = field(default_factory=dict)  # prompt + คำตอบดิบ ไว้ debug และใส่ appendix

    @property
    def has_value(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "label": self.label,
            "confidence": self.confidence,
            "meta_intent": self.meta_intent,
            "reason_th": self.reason_th,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "raw": self.raw,
        }


class OptionExtractor:
    def __init__(self, engine: ChatEngine, form: Form):
        self.engine = engine
        self.form = form

    def extract(self, transcript: str, item_id: str) -> ExtractionResult:
        item = self.form.item(item_id)
        scale = self.form.scale_for(item)
        return self.extract_for(transcript, item, scale)

    def extract_for(self, transcript: str, item: Item, scale: Scale) -> ExtractionResult:
        started = time.perf_counter()

        if not transcript or not transcript.strip():
            # ไม่มีอะไรให้วิเคราะห์ ไม่ต้องเปลืองการเรียกโมเดล
            return ExtractionResult(
                reason_th="ไม่มีข้อความให้วิเคราะห์",
                error="empty_transcript",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        user_prompt = build_user_prompt(transcript, item, scale)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.engine.chat_json(messages, build_schema(scale))
        except LLMUnavailable as exc:
            # ห้ามโยนต่อ เพราะ runner วนหลายไฟล์ ควรได้ผลบางส่วนเสมอ
            return ExtractionResult(
                reason_th=f"เรียก LLM ไม่ได้: {exc}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw={"prompt": user_prompt},
            )

        return self._to_result(response.data, scale, user_prompt, response, started)

    def _to_result(self, data, scale: Scale, user_prompt, response, started) -> ExtractionResult:
        latency_ms = (time.perf_counter() - started) * 1000
        raw = {
            "prompt": user_prompt,
            "response": data,
            "llm_latency_ms": round(response.latency_ms, 1),
            "model": response.model,
        }

        try:
            value = int(data.get("value", UNDECIDED))
        except (TypeError, ValueError):
            value = UNDECIDED

        meta = data.get("meta_intent") or "none"
        meta_intent = None if meta == "none" else meta

        try:
            confidence = min(max(float(data.get("confidence", 0.0) or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0

        reason_th = data.get("reason", "") or ""

        # เชื่อเฉพาะค่าที่อยู่ใน scale จริง ๆ schema กันไว้ชั้นหนึ่งแล้ว แต่ถ้าวันหนึ่ง
        # เปลี่ยนไปใช้ backend ที่ไม่บังคับ schema ด่านนี้จะยังกันให้อยู่
        if value == UNDECIDED or not scale.is_valid(value):
            return ExtractionResult(
                value=None,
                confidence=confidence if meta_intent else 0.0,
                meta_intent=meta_intent,
                reason_th=reason_th or "โมเดลระบุว่าแปลงเป็นคะแนนไม่ได้",
                latency_ms=latency_ms,
                raw=raw,
            )

        return ExtractionResult(
            value=value,
            label=scale.label(value),
            confidence=confidence,
            meta_intent=meta_intent,
            reason_th=reason_th,
            latency_ms=latency_ms,
            raw=raw,
        )
