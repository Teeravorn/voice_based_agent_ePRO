"""Dialogue manager: ควบคุมบทสนทนาแบบ finite-state machine ตามแบบประเมิน

หลักคือ **แยกการควบคุมบทสนทนาออกจากการตีความคำตอบ** ลำดับคำถามและถ้อยคำทุกคำ
มาจากไฟล์ฟอร์มเท่านั้น ผู้ป่วยทุกคนจึงได้ยินคำถามเหมือนกันตรงตามต้นฉบับ ส่วนโมเดลภาษา
(option extractor) ทำหน้าที่แค่บอกว่าคำตอบหนึ่งครั้งตรงกับระดับคะแนนไหน มั่นใจแค่ไหน
หรือผู้ป่วยกำลังขออย่างอื่น แล้วตัวนี้เป็นคนตัดสินว่าจะทำอะไรต่อ

## ไม่ผูกกับเสียง

manager ไม่ได้เล่นเสียงหรืออัดเสียงเอง แต่คืน "action" ให้ผู้เรียกไปทำ

- ``Say``       ให้เล่นเสียงที่อัดไว้ล่วงหน้า ``key`` คือชื่อไฟล์เสียง ส่วน ``text`` คือถ้อยคำ
- ``ShowImage`` ให้แสดงภาพประกอบของข้อนั้น
- ``Listen``    ให้อัดเสียงคำตอบหนึ่งครั้งแล้วส่งข้อความที่ถอดได้กลับมาที่ ``handle``
- ``End``       จบการสนทนา

ผลคือทดสอบ logic ทั้งหมดได้ด้วยการพิมพ์ข้อความ (``python -m dialogue``) และเทสต์ offline
ได้โดยไม่ต้องมีไมโครโฟนหรือ GPU ส่วน question presenter / speech recognizer มาเสียบทีหลัง

ทุกข้อความที่ manager พูดเป็นชุดจำกัดที่รู้ล่วงหน้า (คำถาม, การอ่านตัวเลือกของ scale,
ประโยคจาก ``speech`` ในไฟล์ฟอร์ม) จึงสังเคราะห์เสียงและให้คนตรวจไว้ก่อนได้ทั้งหมด
ไม่มีการสร้างประโยคใหม่ระหว่างคุยกับผู้ป่วย

## การตัดสินคำตอบหนึ่งครั้ง

1. ข้อความว่าง -> ไม่ได้ยินเสียง ขอให้พูดใหม่
2. คำสั่งสั้น ๆ "ย้อนกลับ" / "ขอหยุด" -> ตรวจด้วยกฎก่อนเรียกโมเดล (ดู ``detect_command``)
3. ส่งให้ extractor ถ้าได้ระดับคะแนนและ confidence ถึงเกณฑ์ -> บันทึกแล้วไปข้อถัดไป
4. ไม่ถึงเกณฑ์ หรือผู้ป่วยขอให้ทวน / ไม่เข้าใจ / ไม่รู้ -> ตอบสนองตามเจตนา แล้วฟังใหม่
5. ไม่สำเร็จครบ ``max_attempts`` ครั้ง -> บันทึกข้อนั้นให้พยาบาลตรวจ (flagged) แล้วไปต่อ

ข้อที่ flag จะเก็บคะแนนที่โมเดลเดาได้ดีที่สุดไว้ใน ``suggested_value`` ให้พยาบาลยืนยันได้เร็ว
แต่ไม่นับเป็นคำตอบจนกว่าพยาบาลจะยืนยัน
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from extractor.forms import Form, Item, Scale, Section
from extractor.option_extractor import ExtractionResult
from scoring import FormScore, score_form

# --- action ที่คืนให้ผู้เรียก --------------------------------------------------


@dataclass(frozen=True)
class Say:
    """เล่นเสียงที่อัดไว้ล่วงหน้า ``key`` ใช้หาไฟล์เสียง เช่น ``item/q1``, ``options/difficulty``"""

    key: str
    text: str


@dataclass(frozen=True)
class ShowImage:
    item_id: str
    path: str | None
    caption_th: str = ""


@dataclass(frozen=True)
class Listen:
    """อัดเสียงคำตอบหนึ่งครั้ง ``step_id`` คือข้อที่กำลังรอคำตอบ"""

    step_id: str


@dataclass(frozen=True)
class End:
    status: str  # "completed" หรือ "stopped"


Action = Say | ShowImage | Listen | End


# --- สถานะ -------------------------------------------------------------------


class DialogueState(str, Enum):
    NOT_STARTED = "not_started"
    AWAITING_ANSWER = "awaiting_answer"
    COMPLETED = "completed"
    STOPPED = "stopped"


class Outcome(str, Enum):
    """ผลของคำตอบหนึ่งครั้ง เก็บไว้ทุกครั้งเพื่อวิเคราะห์ว่าบทสนทนาติดตรงไหน"""

    ACCEPTED = "accepted"
    LOW_CONFIDENCE = "low_confidence"
    UNDECIDED = "undecided"
    NO_SPEECH = "no_speech"
    REPEAT = "repeat"
    NEED_HELP = "need_help"
    DONT_KNOW = "dont_know"
    OUT_OF_SCOPE = "out_of_scope"
    ERROR = "error"
    BACK = "back"
    STOP = "stop"


class StepKind(str, Enum):
    ITEM = "item"
    GATE = "gate"
    FREE_TEXT = "free_text"


class Status(str, Enum):
    PENDING = "pending"
    ANSWERED = "answered"
    FLAGGED = "flagged"   # ต้องให้พยาบาลตรวจ
    SKIPPED = "skipped"   # ชุดเสริมที่ผู้ป่วยไม่ได้ทำ (gate ตอบว่าไม่)


_META_OUTCOMES = {
    "repeat": Outcome.REPEAT,
    "need_help": Outcome.NEED_HELP,
    "dont_know": Outcome.DONT_KNOW,
    "out_of_scope": Outcome.OUT_OF_SCOPE,
}


# --- คำสั่งที่ตรวจด้วยกฎ -------------------------------------------------------

# extractor ไม่ได้ตัดสินเรื่องย้อนกลับ/หยุด (ดู META_INTENTS ใน prompt.py) จึงตรวจด้วยกฎที่นี่
# ต้องเป็นวลีที่ไม่ใช่คำบรรยายอาการ เช่นห้ามใช้ "หยุด" เฉย ๆ เพราะผู้ป่วยพูด "ต้องหยุดพัก" บ่อย
_BACK_PHRASES = ("ย้อนกลับ", "ข้อที่แล้ว", "ข้อก่อนหน้า", "กลับไปข้อ", "ขอแก้คำตอบ", "แก้ข้อที่แล้ว")
_STOP_PHRASES = ("ขอหยุด", "หยุดก่อน", "หยุดทำ", "ไม่ทำต่อ", "ไม่อยากทำต่อ", "เลิกทำ", "พอแค่นี้", "จบเลย")
# "ต้องขอหยุดพักบ่อย" เป็นคำบรรยายอาการ ไม่ใช่คำสั่ง แม้จะสั้นและมี "ขอหยุด"
_NOT_STOP_PHRASES = ("หยุดพัก",)

# คำสั่งจริงเป็นประโยคสั้น ประโยคยาวกว่านี้ถือว่าเป็นคำตอบ แม้จะมีวลีข้างบนปนอยู่
# เช่น "ทำได้แต่ต้องขอหยุดพักเป็นช่วง ๆ" ต้องไปถึง extractor ไม่ใช่หยุดการสอบถาม
COMMAND_MAX_CHARS = 25

_SPACES = re.compile(r"\s+")


def detect_command(transcript: str) -> str | None:
    """คืน ``"back"`` / ``"stop"`` ถ้าข้อความเป็นคำสั่งสั้น ๆ ไม่อย่างนั้นคืน None"""
    compact = _SPACES.sub("", transcript)
    if not compact or len(compact) > COMMAND_MAX_CHARS:
        return None
    if any(phrase in compact for phrase in _STOP_PHRASES) and not any(
        phrase in compact for phrase in _NOT_STOP_PHRASES
    ):
        return "stop"
    if any(phrase in compact for phrase in _BACK_PHRASES):
        return "back"
    return None


# --- บันทึกผล ----------------------------------------------------------------


class Extractor(Protocol):
    """extractor ทุกตัวใน ``extractor/`` มี method นี้"""

    def extract_for(self, transcript: str, item: Item, scale: Scale) -> ExtractionResult: ...


@dataclass
class Turn:
    """คำตอบหนึ่งครั้งของผู้ป่วย"""

    transcript: str
    outcome: Outcome
    at_s: float                       # วินาทีนับจากเริ่ม session
    value: int | None = None
    confidence: float = 0.0
    meta_intent: str | None = None
    reason_th: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "outcome": self.outcome.value,
            "at_s": round(self.at_s, 3),
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "meta_intent": self.meta_intent,
            "reason_th": self.reason_th,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class StepRecord:
    """ผลของหนึ่งข้อ (คำถาม, gate หรือคำถามปลายเปิด)"""

    step_id: str
    kind: StepKind
    section_id: str
    status: Status = Status.PENDING
    value: int | None = None
    label: str | None = None
    text: str = ""                     # คำตอบของคำถามปลายเปิด
    suggested_value: int | None = None  # สำหรับข้อที่ flag: ค่าที่โมเดลเดาได้ดีที่สุด
    flag_reason: str = ""
    turns: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "kind": self.kind.value,
            "section_id": self.section_id,
            "status": self.status.value,
            "value": self.value,
            "label": self.label,
            "text": self.text,
            "suggested_value": self.suggested_value,
            "flag_reason": self.flag_reason,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class _Step:
    id: str
    kind: StepKind
    section: Section


# --- ตัวควบคุม ---------------------------------------------------------------


class DialogueManager:
    """ควบคุมการสอบถามหนึ่ง session

        manager = DialogueManager(form, extractor)
        actions = manager.start()
        while not manager.finished:
            transcript = ...  # ทำตาม actions แล้วถอดเสียงคำตอบเมื่อเจอ Listen
            actions = manager.handle(transcript)

    ``confidence_threshold`` ขึ้นกับ extractor ที่ใช้ เพราะ confidence ที่ LLM เขียนเอง
    (OptionExtractor) กับ probability จาก logit (ReasonLogitExtractor) อยู่คนละสเกล
    ต้องเลือกค่าจากผลการทดลอง ไม่ใช่ใช้ค่าตั้งต้นไปตลอด
    """

    def __init__(
        self,
        form: Form,
        extractor: Extractor,
        confidence_threshold: float = 0.6,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts ต้องอย่างน้อย 1")
        self.form = form
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts
        self._clock = clock

        self.steps = self._build_steps(form)
        self.records: dict[str, StepRecord] = {
            step.id: StepRecord(step_id=step.id, kind=step.kind, section_id=step.section.id)
            for step in self.steps
        }
        self.state = DialogueState.NOT_STARTED

        self._index = -1            # ข้อที่กำลังรอคำตอบ
        self._history: list[int] = []  # ข้อที่ผ่านมาแล้ว ไว้ใช้ตอนย้อนกลับ
        self._failures = 0          # จำนวนครั้งที่ไม่สำเร็จของข้อปัจจุบัน
        self._intro_said: set[str] = set()

        self._t0: float | None = None
        self.first_question_s: float | None = None
        self.last_answer_s: float | None = None
        self.ended_s: float | None = None

    # --- API ---------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.state in (DialogueState.COMPLETED, DialogueState.STOPPED)

    @property
    def current_step_id(self) -> str | None:
        if self.state is not DialogueState.AWAITING_ANSWER:
            return None
        return self.steps[self._index].id

    def start(self) -> list[Action]:
        if self.state is not DialogueState.NOT_STARTED:
            raise RuntimeError("session นี้เริ่มไปแล้ว")
        self._t0 = self._clock()
        actions: list[Action] = [
            self._speech("greeting_th"),
            self._speech("ready_th"),
        ]
        return actions + self._advance()

    def handle(self, transcript: str) -> list[Action]:
        """รับข้อความที่ถอดจากเสียงคำตอบหนึ่งครั้ง แล้วคืน action ถัดไป"""
        if self.state is not DialogueState.AWAITING_ANSWER:
            raise RuntimeError(f"ไม่ได้รอคำตอบอยู่ (state = {self.state.value})")
        transcript = (transcript or "").strip()

        if not transcript:
            self._log(Turn(transcript="", outcome=Outcome.NO_SPEECH, at_s=self._now()))
            return self._retry(Outcome.NO_SPEECH)

        command = detect_command(transcript)
        if command is not None:
            return self.command(command, transcript)

        step = self.steps[self._index]
        if step.kind is StepKind.FREE_TEXT:
            return self._handle_free_text(transcript)
        return self._handle_scaled(step, transcript)

    def command(self, name: str, transcript: str = "") -> list[Action]:
        """คำสั่งจากผู้ป่วย มาจากเสียง (ผ่าน ``detect_command``) หรือจากปุ่มบนหน้าจอก็ได้"""
        if self.state is not DialogueState.AWAITING_ANSWER:
            raise RuntimeError(f"ไม่ได้รอคำตอบอยู่ (state = {self.state.value})")
        if name == "stop":
            self._log(Turn(transcript=transcript, outcome=Outcome.STOP, at_s=self._now()))
            self.state = DialogueState.STOPPED
            self.ended_s = self._now()
            return [self._speech("stopped_th"), End(status=DialogueState.STOPPED.value)]
        if name == "back":
            self._log(Turn(transcript=transcript, outcome=Outcome.BACK, at_s=self._now()))
            return self._go_back()
        if name == "repeat":
            self._log(Turn(transcript=transcript, outcome=Outcome.REPEAT, at_s=self._now()))
            return self._retry(Outcome.REPEAT)
        raise ValueError(f"ไม่รู้จักคำสั่ง {name!r}")

    def review_items(self) -> list[StepRecord]:
        """ข้อที่พยาบาลต้องตรวจ"""
        return [record for record in self.records.values() if record.status is Status.FLAGGED]

    def answers(self) -> dict[str, int | None]:
        """คะแนนรายข้อของคำถามในฟอร์ม (ไม่รวม gate) ข้อที่ยังไม่มีคำตอบเป็น None"""
        return {
            record.step_id: record.value if record.status is Status.ANSWERED else None
            for record in self.records.values()
            if record.kind is StepKind.ITEM
        }

    def scores(self, confirmed: Mapping[str, int] | None = None) -> FormScore:
        """คะแนนรวมตามสูตรของฟอร์ม ``confirmed`` คือคะแนนที่พยาบาลยืนยันให้ข้อที่ flag ไว้"""
        answers = self.answers()
        answers.update(confirmed or {})
        return score_form(self.form, answers)

    def to_dict(self) -> dict:
        return {
            "form": {"form_id": self.form.form_id, "version": self.form.version},
            "state": self.state.value,
            "confidence_threshold": self.confidence_threshold,
            "max_attempts": self.max_attempts,
            # เวลาที่ผู้ป่วยใช้กับระบบ: เริ่มถามข้อแรก -> ได้คำตอบข้อสุดท้าย (การทดลองที่ 3)
            "patient_time_s": (
                round(self.last_answer_s - self.first_question_s, 3)
                if self.first_question_s is not None and self.last_answer_s is not None
                else None
            ),
            "first_question_s": self.first_question_s,
            "last_answer_s": self.last_answer_s,
            "ended_s": self.ended_s,
            "review_items": [record.step_id for record in self.review_items()],
            # คะแนนก่อนพยาบาลตรวจ ข้อที่ flag นับเป็นข้อที่ไม่ตอบ
            "scores": self.scores().to_dict()["sections"],
            "records": [self.records[step.id].to_dict() for step in self.steps],
        }

    # --- การเดินไปข้างหน้า / ย้อนกลับ ----------------------------------------

    @staticmethod
    def _build_steps(form: Form) -> list[_Step]:
        steps: list[_Step] = []
        for section in form.sections:
            if section.gate is not None:
                steps.append(_Step(section.gate.id, StepKind.GATE, section))
            if section.free_text is not None:
                steps.append(_Step(section.free_text.id, StepKind.FREE_TEXT, section))
            steps.extend(_Step(item_id, StepKind.ITEM, section) for item_id in section.item_ids)
        if not steps:
            raise ValueError(f"ฟอร์ม {form.form_id} ไม่มีคำถาม")
        return steps

    def _section_skipped(self, section: Section) -> bool:
        """ชุดนี้ถูกข้ามเพราะ gate ตอบว่าไม่ หรือ gate ถูก flag (ให้พยาบาลถามเอง)"""
        if section.gate is None:
            return False
        gate_record = self.records[section.gate.id]
        if gate_record.status is Status.FLAGGED:
            return True
        return gate_record.status is Status.ANSWERED and gate_record.value == section.gate.skip_on_value

    def _advance(self, preamble: list[Action] | None = None) -> list[Action]:
        """ไปข้อถัดไปที่ต้องถาม ข้ามชุดที่ gate บอกให้ข้าม ถ้าหมดแล้วจบ session"""
        actions = list(preamble or [])
        if self._index >= 0:
            self._history.append(self._index)

        index = self._index + 1
        while index < len(self.steps):
            step = self.steps[index]
            if step.kind is StepKind.GATE or not self._section_skipped(step.section):
                break
            record = self.records[step.id]
            record.status = Status.SKIPPED
            record.value = record.label = None
            index += 1

        if index >= len(self.steps):
            self._index = len(self.steps)
            self.state = DialogueState.COMPLETED
            self.ended_s = self._now()
            return actions + [self._speech("closing_th"), End(status=DialogueState.COMPLETED.value)]

        return actions + self._enter(index)

    def _enter(self, index: int) -> list[Action]:
        self._index = index
        self._failures = 0
        self.state = DialogueState.AWAITING_ANSWER
        step = self.steps[index]
        record = self.records[step.id]
        record.status = Status.PENDING
        record.value = record.label = record.suggested_value = None
        record.text = record.flag_reason = ""

        actions: list[Action] = []
        if step.kind is StepKind.GATE:
            # ถาม gate ใหม่ (ย้อนกลับมา) ผลการข้ามชุดเดิมใช้ไม่ได้แล้ว จะตัดสินใหม่ตอนเดินหน้า
            for other in self.steps:
                other_record = self.records[other.id]
                if other.section.id == step.section.id and other_record.status is Status.SKIPPED:
                    other_record.status = Status.PENDING
        elif step.kind is StepKind.ITEM:
            section = step.section
            if section.intro_th and section.id not in self._intro_said:
                self._intro_said.add(section.id)
                actions.append(Say(f"intro/{section.id}", section.intro_th))
            if self.first_question_s is None:
                self.first_question_s = self._now()
        return actions + self._ask(step)

    def _go_back(self) -> list[Action]:
        # ย้อนไปข้อก่อนหน้าที่ผู้ป่วยตอบเองได้ ข้ามข้อที่ระบบข้ามให้ (skipped) เพราะไม่ได้ถาม
        while self._history:
            previous = self._history.pop()
            if self.records[self.steps[previous].id].status is not Status.SKIPPED:
                return [self._speech("back_th")] + self._enter(previous)
        return [self._speech("no_previous_th"), Listen(self.steps[self._index].id)]

    # --- การถาม --------------------------------------------------------------

    def _ask(self, step: _Step, with_options: bool = False) -> list[Action]:
        actions: list[Action] = []
        if step.kind is StepKind.ITEM:
            item = self.form.item(step.id)
            actions.append(ShowImage(item.id, item.image, item.image_caption_th))
            actions.append(Say(f"item/{item.id}", item.prompt_th))
        elif step.kind is StepKind.GATE:
            actions.append(Say(f"gate/{step.id}", step.section.gate.prompt_th))
        else:
            actions.append(Say(f"free_text/{step.id}", step.section.free_text.prompt_th))
        if with_options:
            actions.extend(self._options(step))
        actions.append(Listen(step.id))
        return actions

    def _options(self, step: _Step) -> list[Action]:
        if step.kind is StepKind.FREE_TEXT:
            return []
        scale = self._scale_of(step)
        if not scale.options_prompt_th:
            return []
        return [Say(f"options/{scale.id}", scale.options_prompt_th)]

    def _retry(self, outcome: Outcome) -> list[Action]:
        """คำตอบนี้ใช้ไม่ได้ ถ้ายังไม่ครบจำนวนครั้งให้ถามใหม่ตามเจตนาของผู้ป่วย ไม่อย่างนั้น flag"""
        step = self.steps[self._index]
        self._failures += 1
        if self._failures >= self.max_attempts:
            return self._flag(step, f"ไม่ได้คำตอบที่ชัดเจนใน {self.max_attempts} ครั้ง")

        if outcome is Outcome.NO_SPEECH:
            return [self._speech("no_speech_th")] + self._ask(step)
        if outcome is Outcome.REPEAT:
            return self._ask(step)
        if outcome is Outcome.NEED_HELP:
            # อ่านตัวเลือกให้ฟังช่วยให้เข้าใจว่าคำถามต้องการคำตอบแบบไหน
            return self._options(step) + [Listen(step.id)]
        if outcome is Outcome.DONT_KNOW:
            return [self._speech("dont_know_hint_th")] + self._options(step) + [Listen(step.id)]

        # ตีความไม่ได้ / ไม่มั่นใจ: ครั้งแรกอ่านตัวเลือกให้ฟัง ครั้งต่อไปถามใหม่ทั้งข้อ
        if self._failures == 1:
            return [self._speech("not_understood_th")] + self._options(step) + [Listen(step.id)]
        return [self._speech("retry_prefix_th")] + self._ask(step, with_options=True)

    def _flag(self, step: _Step, reason: str) -> list[Action]:
        record = self.records[step.id]
        record.status = Status.FLAGGED
        record.flag_reason = reason
        # เก็บคะแนนที่เดาได้มั่นใจที่สุดไว้ให้พยาบาลดู ไม่นับเป็นคำตอบ
        candidates = [turn for turn in record.turns if turn.value is not None]
        if candidates:
            record.suggested_value = max(candidates, key=lambda turn: turn.confidence).value
        self.last_answer_s = self._now()
        return self._advance([self._speech("flagged_th")])

    # --- การรับคำตอบ ----------------------------------------------------------

    def _handle_free_text(self, transcript: str) -> list[Action]:
        step = self.steps[self._index]
        self._log(Turn(transcript=transcript, outcome=Outcome.ACCEPTED, at_s=self._now()))
        record = self.records[step.id]
        record.status = Status.ANSWERED
        record.text = transcript
        return self._advance([self._speech("confirm_accepted_th")])

    def _handle_scaled(self, step: _Step, transcript: str) -> list[Action]:
        scale = self._scale_of(step)
        result = self.extractor.extract_for(transcript, self._item_of(step), scale)
        outcome = self._classify(result)
        self._log(
            Turn(
                transcript=transcript,
                outcome=outcome,
                at_s=self._now(),
                value=result.value,
                confidence=result.confidence,
                meta_intent=result.meta_intent,
                reason_th=result.reason_th,
                latency_ms=result.latency_ms,
            )
        )

        if outcome is Outcome.ERROR:
            # เรียกโมเดลไม่ได้ ถามซ้ำก็ไม่ช่วย ส่งให้พยาบาลทันที
            return self._flag(step, f"extractor ใช้งานไม่ได้: {result.error}")
        if outcome is not Outcome.ACCEPTED:
            return self._retry(outcome)

        record = self.records[step.id]
        record.status = Status.ANSWERED
        record.value = result.value
        record.label = scale.label(result.value)
        self.last_answer_s = self._now()

        preamble: list[Action] = []
        if step.kind is StepKind.GATE and result.value == step.section.gate.skip_on_value:
            gate = step.section.gate
            if gate.skip_message_th:
                preamble.append(Say(f"skip/{gate.id}", gate.skip_message_th))
        return self._advance(preamble)

    def _classify(self, result: ExtractionResult) -> Outcome:
        if result.error and result.error != "empty_transcript":
            return Outcome.ERROR
        if result.has_value and result.confidence >= self.confidence_threshold:
            return Outcome.ACCEPTED
        if not result.has_value and result.meta_intent in _META_OUTCOMES:
            return _META_OUTCOMES[result.meta_intent]
        if result.has_value:
            return Outcome.LOW_CONFIDENCE
        return Outcome.UNDECIDED

    # --- helpers ------------------------------------------------------------

    def _scale_of(self, step: _Step) -> Scale:
        if step.kind is StepKind.GATE:
            return self.form.scale(step.section.gate.scale_id)
        return self.form.scale_for(self.form.item(step.id))

    def _item_of(self, step: _Step) -> Item:
        if step.kind is StepKind.ITEM:
            return self.form.item(step.id)
        # gate ไม่ใช่ item ในฟอร์ม แต่ extractor ต้องการ Item จึงประกอบขึ้นจาก gate
        gate = step.section.gate
        return Item(
            id=gate.id,
            number=0,
            scale_id=gate.scale_id,
            text_th=gate.prompt_th,
            prompt_th=gate.prompt_th,
            section_id=step.section.id,
        )

    def _speech(self, key: str) -> Say:
        return Say(f"speech/{key}", self.form.speech.get(key, ""))

    def _log(self, turn: Turn) -> None:
        self.records[self.steps[self._index].id].turns.append(turn)

    def _now(self) -> float:
        if self._t0 is None:
            return 0.0
        return self._clock() - self._t0
