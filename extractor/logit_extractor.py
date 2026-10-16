"""Option extractor ตัวที่สาม: ใช้ qwen เป็น decision model อ่าน logit ตรง ๆ

แนวคิดคือยืม**วิธีทำงาน**ของ OpenThai-SystemOne มาใช้กับ qwen ซึ่งเป็น generative model
เพื่อแยกให้ออกว่าผลที่ต่างกันระหว่างสองตัวมาจาก *โมเดล* หรือมาจาก *วิธีอ่านคำตอบ*

- ``OptionExtractor`` ให้โมเดลเขียน JSON ออกมาแล้วอ่านค่า ``value`` ที่มันเขียน
- ตัวนี้บังคับให้โมเดลตอบเป็นตัวเลขตัวเดียว แล้ว**ไม่สนใจ** token ที่มันเลือก
  แต่ไปอ่าน logprob ของตัวเลือกทุกตัวที่ตำแหน่งนั้น ทำ softmax เฉพาะในกลุ่มตัวเลือก
  แล้วเอาตัวที่ probability สูงสุด

ใช้กฎและ few-shot ชุดเดียวกับ ``OptionExtractor`` (``LOGIT_SYSTEM_PROMPT`` ใน prompt.py)
เพื่อให้เทียบกันได้ว่าต่างกันแค่วิธีอ่านคำตอบ ไม่ใช่ prompt

ข้อดีที่ได้มาคือ **probability distribution ของทุกระดับคะแนน** เหมือนที่ SystemOne ให้
ซึ่งเอาไปตัดสินได้ว่าควรถามผู้ป่วยย้ำไหม ต่างจาก confidence ที่ LLM รายงานเองซึ่งวนอยู่
แค่ 0.85/0.90/0.95

ข้อจำกัด: Ollama ให้ ``top_logprobs`` ได้มากสุด 20 อันดับ ถ้าตัวเลือกไหนหลุดออกไป
จะไม่มีค่ากลับมา ตัวนั้นถือว่า probability เป็น 0 (บันทึกไว้ใน ``raw`` ว่าหลุดตัวไหน)
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass

from engine import ChatEngine, LLMUnavailable

from .forms import Form, Item, Scale
from .option_extractor import ExtractionResult, OptionExtractor
from .prompt import (
    LOGIT_SYSTEM_PROMPT,
    META_INTENTS,
    SYSTEM_PROMPT,
    UNDECIDED,
    build_schema,
    build_user_prompt,
)

ANSWER_CUE = "\n\nตอบด้วยตัวเลขตัวเดียว:"

META_PROMPT = """คุณเป็นแบบจำลองตัดสินใจ (decision model) ไม่ใช่ผู้ช่วยสนทนา

ผู้ป่วยตอบคำถามของพยาบาลมาแบบที่แปลงเป็นคะแนนไม่ได้ หน้าที่ของคุณคือระบุว่าผู้ป่วย
กำลังทำอะไร โดยเลือกจากตัวเลือกที่ให้

ตอบด้วยตัวเลขของตัวเลือกเพียงตัวเดียว ห้ามมีข้อความอื่นใด"""

# ตัวเลือกของคำถาม meta เรียงตามหมายเลขที่ใช้ใน prompt
META_OPTIONS: tuple[tuple[str, str], ...] = (
    ("repeat", "ขอให้ทวนคำถาม บอกว่าฟังไม่ชัด หรือขอให้พูดอีกรอบ"),
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


def option_logprobs(entry: dict, options: list[int]) -> dict[int, float]:
    """ดึง logprob ของ ``options`` จาก ``top_logprobs`` ของ token หนึ่งตำแหน่ง"""
    wanted = {str(option): option for option in options}
    logprobs: dict[int, float] = {}
    for candidate in entry.get("top_logprobs") or []:
        token = (candidate.get("token") or "").strip()
        if token in wanted:
            option = wanted[token]
            # token เดียวกันอาจโผล่หลายรูปแบบ เก็บค่าที่สูงสุดไว้
            logprobs[option] = max(logprobs.get(option, -math.inf), candidate["logprob"])
    return logprobs


# key ``"value"`` ของ JSON ที่ไม่ได้อยู่ในสตริง (ใน reason จะถูก escape เป็น \"value\")
_VALUE_KEY = re.compile(r'(?<!\\)"value"\s*:')


def find_value_position(entries: list[dict]) -> int | None:
    """หาตำแหน่ง token ของค่า ``value`` ใน JSON ที่โมเดลเขียน

    ต่อ token ทั้งหมดกลับเป็นข้อความ หา key ``"value":`` แล้วเอา token แรกที่ไม่ใช่
    ช่องว่างหลังจากนั้น ถ้า token นั้นไม่ใช่ตัวเลขล้วน (เช่น tokenizer รวมกับเครื่องหมาย
    อื่น) คืน None เพราะอ่าน logprob ของตัวเลือกจากตำแหน่งนั้นไม่ได้
    """
    text = "".join(entry.get("token", "") for entry in entries)
    match = _VALUE_KEY.search(text)
    if match is None:
        return None

    offset = 0
    for index, entry in enumerate(entries):
        token = entry.get("token", "")
        if offset >= match.end() and token.strip():
            return index if token.strip().isdigit() else None
        offset += len(token)
    return None


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
        logprobs = option_logprobs(entry, options)

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
            {"role": "system", "content": LOGIT_SYSTEM_PROMPT},
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


class ReasonLogitExtractor(OptionExtractor):
    """ให้โมเดลเขียน reason ก่อนตาม schema เดิม แล้วอ่าน logit ของตัวเลือกที่ตำแหน่ง value

    รวมข้อดีของสองตัวก่อนหน้า
    - จาก ``OptionExtractor``: prompt, few-shot และ schema ชุดเดียวกันทุกอย่าง โมเดลได้
      เขียนเหตุผลก่อนตัดสิน (``LogitExtractor`` ต้องฟันธงที่ token แรก ซึ่งแม่นน้อยกว่า)
    - จาก ``LogitExtractor``: probability distribution ของทุกระดับคะแนน แทน confidence
      ที่โมเดลเขียนเองซึ่งวนอยู่แค่ 0.85/0.90/0.95

    ข้อควรรู้: ที่ temperature 0 token ที่โมเดลเขียนคือตัวที่ logprob สูงสุดอยู่แล้ว
    argmax จึงแทบเท่ากับ value ที่เขียนเสมอ (ต่างกันได้เฉพาะเมื่อ grammar ตัด token
    อันดับหนึ่งทิ้ง) exact match จึงไม่ควรต่างจาก ``OptionExtractor`` สิ่งที่ได้เพิ่มคือ
    distribution ซึ่งใช้ตัดสินว่าควรถามย้ำไหม และใช้คิด ``value_expected`` ที่เก็บไว้ใน raw

    meta_intent ยังอ่านจาก JSON ที่โมเดลเขียน ไม่ต้องยิงครั้งที่สองแบบ ``LogitExtractor``
    """

    name = "reason-logit"

    def __init__(self, engine: ChatEngine, form: Form, top_logprobs: int = 20):
        super().__init__(engine=engine, form=form)
        self.top_logprobs = top_logprobs

    def extract_for(self, transcript: str, item: Item, scale: Scale) -> ExtractionResult:
        started = time.perf_counter()

        if not transcript or not transcript.strip():
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
            response = self.engine.chat_logprobs(
                messages, format=build_schema(scale), top_logprobs=self.top_logprobs
            )
        except LLMUnavailable as exc:
            return ExtractionResult(
                reason_th=f"เรียก LLM ไม่ได้: {exc}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw={"prompt": user_prompt},
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            return ExtractionResult(
                reason_th="โมเดลตอบมาไม่ใช่ JSON object",
                error="invalid_json",
                latency_ms=(time.perf_counter() - started) * 1000,
                raw={"prompt": user_prompt, "text": response.text[:500]},
            )

        # อ่าน reason / meta_intent / ค่าที่เขียนด้วยกติกาเดียวกับ OptionExtractor ก่อน
        result = self._to_result(data, scale, user_prompt, response, started)

        options = [UNDECIDED, *scale.values]
        position = find_value_position(response.entries)
        logprobs = option_logprobs(response.entries[position], options) if position is not None else {}
        probabilities = softmax_over(logprobs)

        result.raw.update({
            "probabilities": {str(k): round(v, 4) for k, v in probabilities.items()},
            "missing_from_top_logprobs": [option for option in options if option not in logprobs],
            "value_ที่โมเดลเขียน": data.get("value"),
            "confidence_ที่โมเดลเขียน": data.get("confidence"),
            "value_expected": expected_value(probabilities, scale),
            "answer_position": position,
            "eval_count": response.eval_count,
            "think": self.engine.think,
        })

        if not probabilities:
            # อ่าน logprob ไม่ได้ ใช้ value ที่โมเดลเขียนต่อไป แต่ไม่มี confidence ให้เชื่อ
            result.confidence = 0.0
            result.raw["fallback"] = "value_ที่โมเดลเขียน"
            return result

        choice = max(probabilities, key=probabilities.get)
        result.confidence = probabilities[choice]
        if choice == UNDECIDED or not scale.is_valid(choice):
            result.value = None
            result.label = None
        else:
            result.value = choice
            result.label = scale.label(choice)
        return result


def expected_value(probabilities: dict[int, float], scale: Scale) -> int | None:
    """ค่าคาดหวังของคะแนน ปัดเป็นจำนวนเต็ม คิดเฉพาะค่าที่อยู่ใน scale

    ไม่รวม 0 (แปลงไม่ได้) เพราะไม่ใช่ระดับอาการ ถ้ารวมเข้าไปจะดึงค่าลงผิดทิศ
    และคืน None เมื่อ 0 มี probability สูงสุด เพราะคำตอบนั้นไม่ควรได้คะแนนตั้งแต่แรก
    เก็บไว้เทียบกับ argmax เหมือนที่ SystemOne พบว่า expected แม่นกว่า
    """
    if probabilities and max(probabilities, key=probabilities.get) == UNDECIDED:
        return None
    weights = {value: p for value, p in probabilities.items() if scale.is_valid(value)}
    total = sum(weights.values())
    if total <= 0:
        return None
    return round(sum(value * p for value, p in weights.items()) / total)
