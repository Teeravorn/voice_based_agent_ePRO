"""เทสต์ที่รันได้โดยไม่ต้องมี Ollama หรือ GPU

ทั้งหมดเป็น offline เพื่อให้รันได้เร็วและไม่ผูกกับผลลัพธ์ของโมเดล ซึ่งเป็นคนละเรื่อง
กับความแม่นที่วัดด้วย ``main.py``
"""

from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import LLMResponse, LLMUnavailable  # noqa: E402
from evaluation.manifest import load_manifest  # noqa: E402
from extractor import OptionExtractor, load_form  # noqa: E402
from extractor.prompt import (  # noqa: E402
    LOGIT_SYSTEM_PROMPT,
    META_INTENTS,
    SYSTEM_PROMPT,
    REASON_MIN_CHARS,
    UNDECIDED,
    build_schema,
    build_user_prompt,
)

FORM_PATH = ROOT / "forms" / "quickdash_th.json"
MANIFEST_PATH = ROOT / "evaluation" / "manifest.csv"

POC_ITEMS = {"q1": "difficulty", "q9": "severity", "q11": "sleep_difficulty"}


@pytest.fixture(scope="module")
def form():
    return load_form(FORM_PATH)


class FakeEngine:
    """ChatEngine ปลอม คืนค่าที่กำหนดไว้ หรือโยน exception ตามที่สั่ง"""

    def __init__(self, data=None, raises=None):
        self.data = data or {}
        self.raises = raises
        self.calls = []

    def chat_json(self, messages, schema):
        self.calls.append((messages, schema))
        if self.raises:
            raise self.raises
        return LLMResponse(data=self.data, text="", latency_ms=12.0, model="fake")


# --- ฟอร์ม ---------------------------------------------------------------


def test_form_loads_all_items(form):
    assert form.form_id == "quickdash_th"
    # 11 ข้อหลัก + 4 ข้อ module งาน + 4 ข้อ module กีฬา/ดนตรี
    assert len(form.items) == 19


@pytest.mark.parametrize("item_id,scale_id", POC_ITEMS.items())
def test_poc_items_use_expected_scale(form, item_id, scale_id):
    item = form.item(item_id)
    assert item.scale_id == scale_id
    assert form.scale_for(item).values == [1, 2, 3, 4, 5]


def test_every_option_has_description(form):
    """description_th คือสิ่งที่ LLM ใช้เทียบ ถ้าว่างแปลว่า prompt จะขาดข้อมูล"""
    for scale in form.scales.values():
        for option in scale.options:
            assert option.description_th.strip(), f"{scale.id}/{option.value} ไม่มีคำอธิบาย"


def test_unknown_item_raises_readable_error(form):
    with pytest.raises(KeyError, match="q99"):
        form.item("q99")


# --- schema / prompt -----------------------------------------------------


def test_schema_puts_reason_before_value(form):
    """ลำดับ field มีผลต่อความแม่นโดยตรง ดูคำอธิบายใน extractor/prompt.py"""
    schema = build_schema(form.scale("severity"))
    assert list(schema["properties"]) == ["reason", "value", "meta_intent", "confidence"]


def test_schema_forces_non_empty_reason(form):
    schema = build_schema(form.scale("severity"))
    assert schema["properties"]["reason"]["minLength"] == REASON_MIN_CHARS


def test_schema_enum_matches_scale(form):
    schema = build_schema(form.scale("severity"))
    assert schema["properties"]["value"]["enum"] == [UNDECIDED, 1, 2, 3, 4, 5]
    assert schema["properties"]["meta_intent"]["enum"] == ["none", *META_INTENTS]


def test_user_prompt_contains_every_option_description(form):
    item = form.item("q9")
    scale = form.scale_for(item)
    prompt = build_user_prompt("ปวดนิดหน่อยค่ะ", item, scale)

    assert item.text_th in prompt
    assert "ปวดนิดหน่อยค่ะ" in prompt
    for option in scale.options:
        assert option.description_th in prompt
        assert f"{option.value} = {option.label_th}" in prompt


# --- extractor -----------------------------------------------------------


def test_extract_maps_value_and_takes_label_from_form(form):
    engine = FakeEngine({"reason": "ผู้ป่วยบอกว่าปวดเล็กน้อยเท่านั้น", "value": 2,
                         "meta_intent": "none", "confidence": 0.9})
    result = OptionExtractor(engine, form).extract("ปวดนิดหน่อย", "q9")

    assert result.value == 2
    # label ต้องมาจากไฟล์ฟอร์ม ไม่ใช่ข้อความที่โมเดลสร้าง
    assert result.label == form.scale("severity").label(2)
    assert result.meta_intent is None      # "none" ถูกแปลงเป็น None
    assert result.confidence == 0.9


def test_extract_rejects_value_outside_scale(form):
    engine = FakeEngine({"reason": "เหตุผลที่ยาวพอสำหรับการทดสอบนี้", "value": 9,
                         "meta_intent": "none", "confidence": 0.9})
    result = OptionExtractor(engine, form).extract("อะไรสักอย่าง", "q9")

    assert result.value is None
    assert result.label is None


def test_extract_keeps_meta_intent(form):
    engine = FakeEngine({"reason": "ผู้ป่วยขอให้ทวนคำถามอีกครั้ง", "value": 0,
                         "meta_intent": "repeat", "confidence": 0.8})
    result = OptionExtractor(engine, form).extract("เมื่อกี้ถามว่าอะไรนะ", "q1")

    assert result.value is None
    assert result.meta_intent == "repeat"
    assert result.confidence == 0.8


def test_extract_does_not_propagate_llm_failure(form):
    """runner วนหลายไฟล์ ถ้า ollama ล่มกลางทางต้องได้ผลบางส่วน ไม่ใช่ตายทั้ง run"""
    engine = FakeEngine(raises=LLMUnavailable("connection refused"))
    result = OptionExtractor(engine, form).extract("ปวดมาก", "q9")

    assert result.value is None
    assert result.error and "connection refused" in result.error


def test_extract_skips_llm_on_empty_transcript(form):
    engine = FakeEngine({"reason": "ไม่ควรถูกเรียก", "value": 1, "meta_intent": "none", "confidence": 1.0})
    result = OptionExtractor(engine, form).extract("   ", "q9")

    assert result.value is None
    assert result.error == "empty_transcript"
    assert engine.calls == []


def test_extract_clamps_bad_confidence(form):
    engine = FakeEngine({"reason": "เหตุผลที่ยาวพอสำหรับการทดสอบนี้", "value": 3,
                         "meta_intent": "none", "confidence": 7.5})
    result = OptionExtractor(engine, form).extract("ก็กลาง ๆ", "q9")

    assert result.confidence == 1.0


# --- manifest ------------------------------------------------------------


def test_manifest_rows_are_consistent_with_form(form):
    rows = load_manifest(MANIFEST_PATH)
    assert len(rows) == 15

    for row in rows:
        item = form.item(row.item_id)
        scale = form.scale_for(item)
        assert row.script_th, f"{row.audio} ไม่มีสคริปต์คำตอบ"
        assert row.audio.endswith(".wav")
        for gold in row.gold_values:
            assert gold == UNDECIDED or scale.is_valid(gold), f"{row.audio} เฉลย {gold} ไม่อยู่ใน {scale.id}"


def test_manifest_meta_rows_are_not_scorable():
    rows = {row.audio: row for row in load_manifest(MANIFEST_PATH)}
    meta_rows = [row for row in rows.values() if row.gold_meta]

    assert meta_rows, "ควรมีแถวที่เป็น meta-intent ไว้ทดสอบ"
    for row in meta_rows:
        assert not row.is_scorable
        assert row.gold_meta in META_INTENTS


def test_manifest_covers_all_answer_types():
    types = {row.answer_type for row in load_manifest(MANIFEST_PATH)}
    assert types == {"direct", "natural", "ambiguous", "meta"}


# --- การนับคะแนนในรายงาน --------------------------------------------------


def test_meta_row_counts_as_correct_when_model_gives_no_score():
    """เฉลย 0 แปลว่า 'ไม่ควรให้คะแนน' การที่โมเดลไม่ให้คะแนนจึงถูกต้อง"""
    from evaluation.report import RunRecord, summarize
    from extractor import ExtractionResult

    rows = {row.audio: row for row in load_manifest(MANIFEST_PATH)}
    meta_row = rows["q1_meta_1.wav"]
    record = RunRecord(
        row=meta_row,
        mode="text",
        transcript=meta_row.script_th,
        result=ExtractionResult(value=None, meta_intent="repeat", confidence=0.9, latency_ms=1.0),
    )

    assert record.is_correct
    assert record.meta_correct
    # แถวแบบนี้ไม่ถูกนับรวมใน exact match เพราะไม่ได้วัดเรื่องคะแนน
    summary = summarize([record], "text")
    assert summary.scorable == 0
    assert summary.meta_total == 1 and summary.meta_correct == 1


def test_scorable_row_with_no_score_counts_as_undecided():
    from evaluation.report import RunRecord, summarize
    from extractor import ExtractionResult

    rows = {row.audio: row for row in load_manifest(MANIFEST_PATH)}
    record = RunRecord(
        row=rows["q9_natural_1.wav"],
        mode="text",
        transcript="...",
        result=ExtractionResult(value=None, latency_ms=1.0),
    )

    summary = summarize([record], "text")
    assert not record.is_correct
    assert summary.scorable == 1 and summary.undecided == 1


# --- การส่งพารามิเตอร์ think ให้ Ollama ------------------------------------


def _capture_ollama_call(monkeypatch):
    """ดักการเรียก ollama.chat แล้วคืน kwargs ที่ถูกส่งไปจริง"""
    captured = {}

    class FakeMessage:
        content = '{"reason": "x", "value": 1, "meta_intent": "none", "confidence": 0.5}'
        thinking = "เหตุผลที่โมเดลคิดไว้"

    class FakeResponse:
        message = FakeMessage()
        eval_count = 10

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    import engine.chat as chat_module

    monkeypatch.setattr(chat_module, "chat", fake_chat)
    return captured


def test_chat_json_passes_think_false(monkeypatch):
    """โหมด JSON ส่ง think=False ได้ เพราะ format บังคับ token แรกเป็น { อยู่แล้ว
    และประหยัดเวลาไปราว 9 เท่า ดู docstring ของ chat_json()"""
    from engine import ChatEngine

    captured = _capture_ollama_call(monkeypatch)
    ChatEngine(model="qwen3:4b", think=False).chat_json([{"role": "user", "content": "x"}], {})

    assert captured["think"] is False
    assert "format" in captured


def test_chat_omits_think_false_in_free_text_mode(monkeypatch):
    """โหมดข้อความธรรมดาห้ามส่ง think=False ไม่งั้น reasoning จะไหลลงไปปนใน content

    วัดกับ qwen3:4b: think=False ได้ content 2167 ตัวอักษรที่เป็น reasoning ล้วน
    ส่วนการไม่ส่ง think ได้ content 6 ตัวอักษร และ reasoning 2151 ตัวอักษรแยกออกมา
    """
    from engine import ChatEngine

    captured = _capture_ollama_call(monkeypatch)
    engine = ChatEngine(model="qwen3:4b", think=False)
    engine.chat("สวัสดี")

    assert "think" not in captured
    assert "format" not in captured
    # reasoning ถูกแยกเก็บไว้ ไม่ปนกับคำตอบ
    assert engine.last_thinking == "เหตุผลที่โมเดลคิดไว้"


def test_chat_never_sends_think_in_free_text_mode(monkeypatch):
    """โหมดข้อความธรรมดาไม่ส่ง think เลย ปล่อยให้ Ollama แยก reasoning ตามค่าตั้งต้น

    เพราะ think=False ที่นี่ไม่ได้ปิดการคิด แค่ปิดการแยก reasoning ออกจาก content
    (ดู test ด้านบน) จึงไม่มีค่าไหนของ think ที่ควรส่งในโหมดนี้
    """
    from engine import ChatEngine

    for think in (False, True, "low", None):
        captured = _capture_ollama_call(monkeypatch)
        ChatEngine(model="qwen3:4b", think=think).chat("สวัสดี")
        assert "think" not in captured, f"think={think!r} ไม่ควรถูกส่ง"


# --- ข้อสอบต้องไม่หลุดเข้าไปใน prompt -------------------------------------

# ข้อความยาวเท่านี้ที่ตรงกันเป๊ะถือว่าหลุด วัดจาก prompt เดิมที่หลุดจริง: ประโยคที่ลอกมา
# ซ้ำกันยาว 11-25 ตัวอักษร ส่วนคำทั่วไปที่บังเอิญซ้ำ เช่น "เกือบทุก" "เมื่อก่อน" ยาวไม่เกิน 9
LEAK_MIN_CHARS = 10


@pytest.mark.parametrize("prompt_name", ["SYSTEM_PROMPT", "LOGIT_SYSTEM_PROMPT"])
def test_prompt_does_not_leak_manifest(prompt_name):
    """ตัวอย่างใน prompt ต้องไม่ใช่ประโยคเดียวกับข้อสอบ ไม่อย่างนั้นความแม่นที่วัดได้จะสูงเกินจริง"""
    prompt = {"SYSTEM_PROMPT": SYSTEM_PROMPT, "LOGIT_SYSTEM_PROMPT": LOGIT_SYSTEM_PROMPT}[prompt_name]
    for row in load_manifest(MANIFEST_PATH):
        script = row.script_th
        match = SequenceMatcher(None, script, prompt, autojunk=False).find_longest_match(
            0, len(script), 0, len(prompt)
        )
        overlap = script[match.a:match.a + match.size]
        assert match.size < LEAK_MIN_CHARS, f"{row.audio} ซ้ำกับ {prompt_name}: {overlap!r}"
