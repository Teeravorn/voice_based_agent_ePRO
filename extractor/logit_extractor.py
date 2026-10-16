"""Option extractor ตัวที่สาม: ใช้ qwen เป็น decision model อ่าน logit ตรง ๆ

แนวคิดคือยืม**วิธีทำงาน**ของ OpenThai-SystemOne มาใช้กับ qwen ซึ่งเป็น generative model
เพื่อแยกให้ออกว่าผลที่ต่างกันระหว่างสองตัวมาจาก *โมเดล* หรือมาจาก *วิธีอ่านคำตอบ*

- ``OptionExtractor`` ให้โมเดลเขียน JSON ออกมาแล้วอ่านค่า ``value`` ที่มันเขียน
- ตัวนี้บังคับให้โมเดลตอบเป็นตัวเลขตัวเดียว แล้ว**ไม่สนใจ** token ที่มันเลือก
  แต่ไปอ่าน logprob ของตัวเลือกทุกตัวที่ตำแหน่งนั้น ทำ softmax เฉพาะในกลุ่มตัวเลือก
  แล้วเอาตัวที่ probability สูงสุด

ข้อดีที่ได้มาคือ **probability distribution ของทุกระดับคะแนน** เหมือนที่ SystemOne ให้
ซึ่งเอาไปตัดสินได้ว่าควรถามผู้ป่วยย้ำไหม ต่างจาก confidence ที่ LLM รายงานเองซึ่งวนอยู่
แค่ 0.85/0.90/0.95

ข้อจำกัด: Ollama ให้ ``top_logprobs`` ได้มากสุด 20 อันดับ ถ้าตัวเลือกไหนหลุดออกไป
จะไม่มีค่ากลับมา ตัวนั้นถือว่า probability เป็น 0 (บันทึกไว้ใน ``raw`` ว่าหลุดตัวไหน)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from engine import ChatEngine, LLMUnavailable

from .forms import Form, Item, Scale
from .option_extractor import ExtractionResult
from .prompt import META_INTENTS, UNDECIDED, build_user_prompt

SYSTEM_PROMPT = """คุณเป็นแบบจำลองตัดสินใจ (decision model) ไม่ใช่ผู้ช่วยสนทนา

หน้าที่ของคุณคือเลือกคะแนนที่ตรงกับคำตอบของผู้ป่วยมากที่สุด โดยเทียบสิ่งที่ผู้ป่วยเล่า
กับคำอธิบายของแต่ละตัวเลือก ผู้ป่วยส่วนใหญ่ไม่ได้พูดชื่อตัวเลือกออกมาตรง ๆ

หลักการตัดสิน
- ถ้าคำตอบบอกเป็นนัยว่าอาการอยู่ระดับไหน ให้เลือกระดับนั้น
- ถ้าลังเลระหว่างสองระดับ ให้เลือกระดับที่ใกล้เคียงกว่า
- เลือก 0 เฉพาะเมื่อผู้ป่วยไม่ได้ตอบคำถามเลย เช่น ขอให้ทวนคำถาม หรือพูดเรื่องอื่น
- ระวังประโยคปฏิเสธซ้อน ให้ตัดสินจากสภาพตอนนี้ ไม่ใช่การเปรียบเทียบกับอดีต

ตอบด้วยตัวเลขของตัวเลือกเพียงตัวเดียว ห้ามมีคำอธิบาย ห้ามมีเครื่องหมายวรรคตอน
ห้ามมีข้อความอื่นใดนอกจากตัวเลขนั้น"""

ANSWER_CUE = "\n\nตอบด้วยตัวเลขตัวเดียว:"

META_PROMPT = """คุณเป็นแบบจำลองตัดสินใจ (decision model) ไม่ใช่ผู้ช่วยสนทนา

ผู้ป่วยตอบคำถามของพยาบาลมาแบบที่แปลงเป็นคะแนนไม่ได้ หน้าที่ของคุณคือระบุว่าผู้ป่วย
กำลังทำอะไร โดยเลือกจากตัวเลือกที่ให้

ตอบด้วยตัวเลขของตัวเลือกเพียงตัวเดียว ห้ามมีข้อความอื่นใด"""

# ตัวเลือกของคำถาม meta เรียงตามหมายเลขที่ใช้ใน prompt
META_OPTIONS: tuple[tuple[str, str], ...] = (
    ("repeat", "ขอให้ทวนคำถาม บอกว่าฟังไม่ชัด หรือถามว่าเมื่อกี้ถามอะไร"),
    ("dont_know", "บอกว่าไม่รู้ ไม่แน่ใจ จำไม่ได้ หรือไม่เคยทำกิจกรรมนั้น"),
    ("need_help", "บอกว่าไม่เข้าใจคำถาม ขอให้อธิบายเพิ่ม"),
    ("out_of_scope", "พูดเรื่องอื่นที่ไม่เกี่ยวกับอาการหรือคำถามเลย"),
)


@dataclass
class _Decision:
    """ผลการอ่าน logit หนึ่งครั้ง"""

    choice: int | None
    probabilities: dict[int, float]
    missing: list[int]
    chosen_token: str
    position: int | None


def softmax_over(logprobs: dict[int, float]) -> dict[int, float]:
    """ทำ softmax เฉพาะในกลุ่มตัวเลือก

    ค่าที่ Ollama คืนมาเป็น log probability เทียบกับ vocabulary ทั้งก้อนแล้ว การ exp
    ตรง ๆ จะได้ probability ที่รวมกันไม่ถึง 1 เพราะ token อื่นกินไปส่วนหนึ่ง จึงต้อง
    normalize ใหม่ในกลุ่ม ซึ่งเทียบเท่ากับการทำ softmax บน logit ของเฉพาะตัวเลือก
    (ลบ max ก่อนเพื่อกัน overflow)
    """
    if not logprobs:
        return {}
    top = max(logprobs.values())
    weights = {key: math.exp(value - top) for key, value in logprobs.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


class LogitExtractor:
    """อ่าน logit ของตัวเลือกจาก qwen แล้วตัดสินด้วย softmax"""

    name = "logit"

    def __init__(self, engine: ChatEngine, form: Form, top_logprobs: int = 20):
        self.engine = engine
        self.form = form
        self.top_logprobs = top_logprobs

    # --- การอ่าน logprob ---------------------------------------------------

    def _decide(self, messages: list[dict[str, str]], options: list[int]) -> _Decision:
        """ยิงหนึ่งครั้งแล้วอ่าน logprob ของ ``options`` ที่ตำแหน่งคำตอบ

        ตอนปิด thinking คำตอบคือ token แรกที่สร้าง จึงจำกัด ``num_predict=1`` ได้
        ตอนเปิด thinking โมเดลต้องเขียน reasoning ก่อน จำกัดไม่ได้ ต้องไล่หาตำแหน่ง
        ของ token คำตอบจากท้าย array แทน
        """
        thinking_on = self.engine.think is not False
        response = self.engine.chat_logprobs(
            messages,
            num_predict=None if thinking_on else 1,
            top_logprobs=self.top_logprobs,
        )

        wanted = {str(option): option for option in options}
        position = None
        entries = response.entries
        # ไล่จากท้ายเพื่อให้ได้ token คำตอบ ไม่ใช่เลขที่โผล่อยู่ใน reasoning
        for index in range(len(entries) - 1, -1, -1):
            if entries[index].get("token", "").strip() in wanted:
                position = index
                break

        if position is None:
            return _Decision(None, {}, list(options), response.text.strip(), None)

        entry = entries[position]
        logprobs: dict[int, float] = {}
        for candidate in entry.get("top_logprobs") or []:
            token = (candidate.get("token") or "").strip()
            if token in wanted:
                option = wanted[token]
                # token เดียวกันอาจโผล่หลายรูปแบบ เก็บค่าที่สูงสุดไว้
                logprobs[option] = max(logprobs.get(option, -math.inf), candidate["logprob"])

        probabilities = softmax_over(logprobs)
        missing = [option for option in options if option not in logprobs]
        choice = max(probabilities, key=probabilities.get) if probabilities else None

        decision = _Decision(choice, probabilities, missing,
                             entry.get("token", "").strip(), position)
        decision.latency_ms = response.latency_ms          # type: ignore[attr-defined]
        decision.eval_count = response.eval_count          # type: ignore[attr-defined]
        decision.text = response.text                      # type: ignore[attr-defined]
        return decision

    # --- ส่วนที่ runner เรียก ----------------------------------------------

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

        user_prompt = build_user_prompt(transcript, item, scale) + ANSWER_CUE
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        options = [UNDECIDED, *scale.values]

        try:
            decision = self._decide(messages, options)
        except LLMUnavailable as exc:
            return ExtractionResult(
                reason_th=f"เรียก LLM ไม่ได้: {exc}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        raw = {
            "probabilities": {str(k): round(v, 4) for k, v in decision.probabilities.items()},
            "missing_from_top_logprobs": decision.missing,
            "token_ที่โมเดลเลือกเอง": decision.chosen_token,
            "answer_position": decision.position,
            "eval_count": getattr(decision, "eval_count", None),
            "think": self.engine.think,
            "model": self.engine.model,
        }

        if decision.choice is None:
            return ExtractionResult(
                reason_th="อ่าน logprob ของตัวเลือกไม่ได้เลย",
                error="no_option_logprobs",
                latency_ms=(time.perf_counter() - started) * 1000,
                raw=raw,
            )

        probability = decision.probabilities.get(decision.choice, 0.0)

        if decision.choice == UNDECIDED:
            meta_intent, meta_raw = self._classify_meta(transcript, item)
            raw["meta"] = meta_raw
            return ExtractionResult(
                value=None,
                confidence=probability,
                meta_intent=meta_intent,
                reason_th=f"p(แปลงเป็นคะแนนไม่ได้)={probability:.2f}",
                latency_ms=(time.perf_counter() - started) * 1000,
                raw=raw,
            )

        if not scale.is_valid(decision.choice):
            return ExtractionResult(
                reason_th=f"ค่า {decision.choice} ไม่อยู่ใน scale {scale.id}",
                error="value_out_of_scale",
                latency_ms=(time.perf_counter() - started) * 1000,
                raw=raw,
            )

        return ExtractionResult(
            value=decision.choice,
            label=scale.label(decision.choice),
            confidence=probability,
            meta_intent=None,
            reason_th=f"p({scale.label(decision.choice)})={probability:.2f}",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=raw,
        )

    def _classify_meta(self, transcript: str, item: Item) -> tuple[str | None, dict]:
        """ยิงครั้งที่สองเพื่อระบุว่าผู้ป่วยกำลังทำอะไร

        ยิงเฉพาะเมื่อคำถามแรกตอบว่าให้คะแนนไม่ได้ ซึ่งเกิดไม่บ่อย จึงแทบไม่กระทบเวลารวม
        ทำแยกเพราะ logit หนึ่งตำแหน่งตอบได้แค่คำถามเดียว ต่างจาก SystemOne ที่ถามหลาย
        คำถามใน forward pass เดียวได้
        """
        lines = [f"คำถามที่พยาบาลถาม: {item.text_th}", "", "ตัวเลือก:"]
        for number, (_, description) in enumerate(META_OPTIONS, start=1):
            lines.append(f"  {number} = {description}")
        lines += ["", f'คำตอบของผู้ป่วย: "{transcript.strip()}"', ANSWER_CUE.strip()]

        messages = [
            {"role": "system", "content": META_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]

        try:
            decision = self._decide(messages, list(range(1, len(META_OPTIONS) + 1)))
        except LLMUnavailable as exc:
            return None, {"error": str(exc)}

        raw = {
            "probabilities": {
                META_OPTIONS[number - 1][0]: round(probability, 4)
                for number, probability in decision.probabilities.items()
                if 1 <= number <= len(META_OPTIONS)
            }
        }
        if decision.choice is None:
            return None, raw

        intent = META_OPTIONS[decision.choice - 1][0]
        return (intent if intent in META_INTENTS else None), raw
