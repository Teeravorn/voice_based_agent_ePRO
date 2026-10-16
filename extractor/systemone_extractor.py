"""Option extractor ตัวที่สอง: ใช้ OpenThai-SystemOne แทน LLM แบบ generative

ต่างจาก ``OptionExtractor`` (qwen3 ผ่าน Ollama) ในเชิงโครงสร้าง ไม่ใช่แค่เปลี่ยนโมเดล

- **ไม่ได้ generate ข้อความ** โมเดลนี้เป็น decision head (``OpenThaiSystemOneForDecision``)
  ยิง forward pass เดียวแล้วอ่านค่าจาก head ออกมาตรง ๆ จึงเร็วกว่ามากและไม่มีทางตอบ
  นอกรูปแบบ ไม่ต้องใช้ JSON schema มาบังคับ
- **ได้ probability distribution จริง** ทุกระดับคะแนน ต่างจากการถาม LLM ว่า confidence
  เท่าไหร่ ซึ่งโมเดลจะตอบ 0.85/0.90/0.95 วนไปโดยไม่สัมพันธ์กับความถูกจริง
  ตัวเลขชุดนี้เอาไปใช้ตัดสินได้ว่าควรถามย้ำผู้ป่วยไหม
- **แลกมาด้วยการไม่มีเหตุผลกำกับ** ``reason`` ที่ LLM เขียนให้นั้นเอาไปให้พยาบาลตรวจย้อนหลังได้
  ตัวนี้ให้แค่ตัวเลข

ถาม Score กับ Choice ไปในคำขอเดียว ทั้งสองคำถามใช้ forward pass ร่วมกัน จึงแทบไม่มีต้นทุนเพิ่ม
"""

from __future__ import annotations

import time
from functools import lru_cache

from openthai_systemone import Choice, Score, SystemOneClient

from .forms import Form, Item, Scale
from .option_extractor import ExtractionResult
from .prompt import META_INTENTS

MODEL_ID = "iapp/OpenThai-SystemOne"

# คำอธิบายเจตนาที่ไม่ใช่การตอบคำถาม ใช้เป็น criteria ของคำถาม Choice
META_CRITERIA = {
    "none": "ตอบคำถามเรื่องอาการของตัวเองตามปกติ",
    "repeat": "ขอให้ทวนคำถาม บอกว่าฟังไม่ชัด หรือถามว่าเมื่อกี้ถามอะไร",
    "dont_know": "บอกว่าไม่รู้ ไม่แน่ใจ จำไม่ได้ หรือไม่เคยทำกิจกรรมนั้น",
    "need_help": "บอกว่าไม่เข้าใจคำถาม ขอให้อธิบายเพิ่ม",
    "out_of_scope": "พูดเรื่องอื่นที่ไม่เกี่ยวกับอาการหรือคำถามเลย",
}


class SystemOneExtractor:
    """แปลงคำตอบเป็นคะแนนด้วย OpenThai-SystemOne

    ``decision`` เลือกวิธีแปลง distribution ให้เป็นคะแนนเดียว
    - ``"expected"`` ปัดค่า ``score`` ที่โมเดลคืนมา ซึ่งเป็นค่าคาดหวังถ่วงน้ำหนัก (ค่าตั้งต้น)
    - ``"argmax"``  เอาระดับที่มี probability สูงสุด

    **ค่าตั้งต้นเป็น expected เพราะวัดแล้วต่างกันมาก** บนคำตอบ 15 อันชุดเดียวกัน
    (เสียงจริงผ่าน ASR) expected ได้ exact match 10/13 = 76.9% ขณะที่ argmax ได้
    7/13 = 53.8% โมเดลเดียวกัน เปลี่ยนแค่วิธีอ่านผล เหตุผลคือ distribution มักคร่อม
    สองระดับที่ติดกัน (เช่น 0.52/0.46) การเอา mode จึงทิ้งข้อมูลครึ่งหนึ่งไป
    ส่วนค่าคาดหวังใช้ทั้ง distribution ซึ่งเหมาะกับสเกล ordinal มากกว่า

    ทั้งสองแบบถูกบันทึกไว้ใน ``raw`` ทุกครั้ง จะเปลี่ยนใจภายหลังก็เทียบย้อนหลังได้
    โดยไม่ต้องรันใหม่
    """

    name = "systemone"

    def __init__(
        self,
        form: Form,
        model_path: str = MODEL_ID,
        client: SystemOneClient | None = None,
        decision: str = "expected",
        meta_threshold: float = 0.5,
    ):
        if decision not in ("argmax", "expected"):
            raise ValueError(f"decision ต้องเป็น argmax หรือ expected ไม่ใช่ {decision!r}")
        self.form = form
        self.decision = decision
        # ถือว่าผู้ป่วยไม่ได้ตอบคำถาม เมื่อ probability ของเจตนาอื่นรวมกันเกินค่านี้
        self.meta_threshold = meta_threshold
        self.client = client or SystemOneClient(model_path)
        self.model_path = model_path

    @lru_cache(maxsize=32)
    def _questions(self, scale_id: str) -> dict:
        """สร้างชุดคำถามของ scale นี้

        ไม่มีข้อความคำถามของ item อยู่ในนี้ คำถามถูกส่งไปทาง ``state`` แทน
        (ดู ``_build_state``) ชุดนี้จึงขึ้นกับ scale อย่างเดียวและ cache ร่วมกันได้
        ทุกข้อที่ใช้ scale เดียวกัน — QuickDASH ข้อ 1-6 ใช้ ``difficulty`` ร่วมกันหมด

        ``criteria[i]`` อธิบายระดับ i เรียงจากน้อยไปมาก ตาม API ของ SystemOne
        ซึ่งนับเริ่มจาก 0 ขณะที่ฟอร์มนับคะแนนเริ่มจาก 1 การแปลงกลับอยู่ใน ``_to_result``

        **criteria คือตัวที่พาข้อมูลจริง ไม่ใช่ข้อความคำถาม** ทดสอบด้วยการส่งคำถาม
        ผิดข้อไปโดยตั้งใจ (คำถามเรื่องการนอนไปคู่กับข้อเปิดฝาขวด) ผลเปลี่ยนแค่ 1 ใน 15 ข้อ
        และตัดคำถามทิ้งทั้งหมดก็ได้คะแนนเท่าเดิม เพราะคำอธิบายของแต่ละตัวเลือกบอกมิติ
        ที่กำลังถามอยู่ในตัวแล้ว ถ้าจะแก้อะไรให้ผลดีขึ้น ให้ไปแก้ ``description_th``
        ในไฟล์ฟอร์ม ไม่ใช่มาเกลาข้อความคำถามตรงนี้
        """
        scale = self.form.scale(scale_id)
        criteria = [f"{option.label_th}: {option.description_th}" for option in scale.options]
        return {
            "score": Score(
                instructions="ประเมินจากคำตอบของผู้ป่วยว่าอาการอยู่ในระดับไหน",
                criteria=criteria,
            ),
            "meta": Choice(
                instructions="ผู้ป่วยกำลังทำอะไรในคำตอบนี้",
                criteria=META_CRITERIA,
            ),
        }

    def _build_state(self, transcript: str, item: Item) -> dict:
        """สิ่งที่ให้โมเดลตัดสิน: คำถามที่ถาม คู่กับคำตอบที่ผู้ป่วยให้

        ส่งคำถามมาทางนี้แทนที่จะไว้ใน ``instructions`` ด้วยสองเหตุผล

        1. วัดแล้วไม่เคยพลาดเกินหนึ่งระดับเลย (±1 = 100%) ทั้งสองรอบที่ทดสอบ ขณะที่
           การวางคำถามไว้ใน instructions มีเคสพลาดสองระดับ ความต่างของ exact match
           อยู่ที่ 1 ข้อจาก 13 ซึ่งไม่มีนัยสำคัญ แต่การไม่พลาดหนักสำคัญกว่าในทางคลินิก
        2. โครงสร้างตรงกับ prompt ที่ให้ qwen (คำถามกับคำตอบอยู่ด้วยกัน) ทำให้เทียบสอง
           โมเดลได้โดยไม่ต้องแก้ตัวว่าตัวไหนได้ข้อมูลมากกว่ากัน

        ``recall_period_th`` ยังไม่ถูกใส่ตรงนี้ เพราะจะทำให้ตัวเลข baseline ที่วัดไว้
        ใช้เทียบไม่ได้ ถ้าจะเพิ่มต้องรันวัดใหม่ทั้งชุดทั้งสอง extractor
        """
        return {
            "คำถามที่พยาบาลถาม": item.text_th,
            "คำตอบของผู้ป่วย": transcript.strip(),
        }

    def extract(self, transcript: str, item_id: str) -> ExtractionResult:
        item = self.form.item(item_id)
        return self.extract_for(transcript, item, self.form.scale_for(item))

    def extract_for(self, transcript: str, item: Item, scale: Scale) -> ExtractionResult:
        started = time.perf_counter()

        if not transcript or not transcript.strip():
            return ExtractionResult(
                reason_th="ไม่มีข้อความให้วิเคราะห์",
                error="empty_transcript",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        questions = self._questions(scale.id)
        try:
            response = self.client.system_one(
                state=self._build_state(transcript, item), questions=questions
            )
        except Exception as exc:
            # เหมือน OptionExtractor: ห้ามโยนต่อ runner ต้องได้ผลบางส่วนเสมอ
            return ExtractionResult(
                reason_th=f"เรียก SystemOne ไม่ได้: {exc}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return self._to_result(response, scale, started)

    def _to_result(self, response, scale: Scale, started) -> ExtractionResult:
        score_answer = response.answers["score"]
        meta_answer = response.answers["meta"]

        # probabilities มี key เป็นสตริงของ index ที่เริ่มจาก 0
        probs = {int(level): float(prob) for level, prob in score_answer.probabilities.items()}
        argmax_level = max(probs, key=probs.get)
        expected_level = round(float(score_answer.score))

        level = argmax_level if self.decision == "argmax" else expected_level
        # index 0-based ของ SystemOne -> ค่าคะแนนจริงในฟอร์ม
        value = scale.values[level] if 0 <= level < len(scale.values) else None

        meta_probs = {key: float(prob) for key, prob in meta_answer.probabilities.items()}
        meta_intent = meta_answer.choice if meta_answer.choice in META_INTENTS else None
        # เชื่อ meta_intent เฉพาะตอนที่มั่นใจพอ ไม่งั้นคำตอบปกติจะถูกตีเป็น meta ทิ้งคะแนนไปเปล่า ๆ
        if meta_intent is not None and meta_probs.get(meta_answer.choice, 0.0) < self.meta_threshold:
            meta_intent = None

        raw = {
            "score_expected": float(score_answer.score),
            "level_argmax": argmax_level,
            "level_expected": expected_level,
            "value_argmax": scale.values[argmax_level] if 0 <= argmax_level < len(scale.values) else None,
            "value_expected": scale.values[expected_level] if 0 <= expected_level < len(scale.values) else None,
            "decision": self.decision,
            "probabilities": {str(k): round(v, 4) for k, v in probs.items()},
            "head_confidence": float(score_answer.confidence),
            "meta_choice": meta_answer.choice,
            "meta_probabilities": {k: round(v, 4) for k, v in meta_probs.items()},
            "meta_abstain": meta_answer.abstain,
            "model": self.model_path,
        }

        latency_ms = (time.perf_counter() - started) * 1000

        # ผู้ป่วยไม่ได้ตอบคำถาม -> ไม่ให้คะแนน เหมือนที่ LLM extractor ตอบ value 0
        if meta_intent is not None:
            return ExtractionResult(
                value=None,
                confidence=meta_probs.get(meta_answer.choice, 0.0),
                meta_intent=meta_intent,
                reason_th=f"ตีความว่าผู้ป่วยไม่ได้ตอบคำถาม ({meta_answer.choice})",
                latency_ms=latency_ms,
                raw=raw,
            )

        if value is None or not scale.is_valid(value):
            return ExtractionResult(
                value=None,
                confidence=0.0,
                reason_th=f"แปลงระดับ {level} เป็นคะแนนของ scale {scale.id} ไม่ได้",
                latency_ms=latency_ms,
                raw=raw,
            )

        return ExtractionResult(
            value=value,
            label=scale.label(value),
            # ใช้ probability ของระดับที่เลือกเป็น confidence เพราะเป็นค่าที่มีความหมายจริง
            confidence=probs.get(level, 0.0),
            meta_intent=None,
            reason_th=f"p({scale.label(value)})={probs.get(level, 0.0):.2f}",
            latency_ms=latency_ms,
            raw=raw,
        )
