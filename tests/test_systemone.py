"""เทสต์ SystemOneExtractor แบบ offline ไม่โหลดโมเดลจริง

จุดที่ต้องกันไว้ที่สุดคือ **การนับ index** SystemOne นับระดับเริ่มจาก 0 แต่ฟอร์มนับ
คะแนนเริ่มจาก 1 ถ้าพลาดตรงนี้ คะแนนจะเพี้ยนไปหนึ่งระดับทุกข้อโดยไม่มีอะไรฟ้อง
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor import load_form  # noqa: E402
from extractor.systemone_extractor import SystemOneExtractor  # noqa: E402

FORM_PATH = ROOT / "forms" / "quickdash_th.json"


@pytest.fixture(scope="module")
def form():
    return load_form(FORM_PATH)


class FakeScoreAnswer:
    def __init__(self, score, probabilities, confidence=0.5):
        self.score = score
        self.probabilities = probabilities
        self.confidence = confidence
        self.legend = {}


class FakeChoiceAnswer:
    def __init__(self, choice, probabilities, abstain=0.001):
        self.choice = choice
        self.probabilities = probabilities
        self.abstain = abstain
        self.confidence = max(probabilities.values())


class FakeResponse:
    def __init__(self, score_answer, meta_answer):
        self.answers = {"score": score_answer, "meta": meta_answer}


class FakeClient:
    """SystemOneClient ปลอม คืนคำตอบที่กำหนดไว้ล่วงหน้า"""

    def __init__(self, score_answer, meta_answer=None, raises=None):
        self.score_answer = score_answer
        self.meta_answer = meta_answer or FakeChoiceAnswer("none", {"none": 0.95})
        self.raises = raises
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.raises:
            raise self.raises
        return FakeResponse(self.score_answer, self.meta_answer)


def make(form, score, probs, meta=None, decision="expected", **kwargs):
    client = FakeClient(FakeScoreAnswer(score, probs), meta)
    extractor = SystemOneExtractor(form=form, client=client, decision=decision, **kwargs)
    return extractor, client


def test_level_zero_maps_to_form_value_one(form):
    """ระดับ 0 ของ SystemOne = คะแนน 1 ของฟอร์ม ไม่ใช่ 0"""
    extractor, _ = make(form, 0.1, {"0": 0.9, "1": 0.1})
    result = extractor.extract("ไม่ปวดเลย", "q9")

    assert result.value == 1
    assert result.label == form.scale("severity").label(1)


def test_top_level_maps_to_form_value_five(form):
    extractor, _ = make(form, 4.0, {"4": 0.9, "3": 0.1})
    result = extractor.extract("ปวดทนไม่ไหว", "q9")

    assert result.value == 5


def test_expected_and_argmax_can_disagree(form):
    """distribution คร่อมสองระดับ: argmax เอา mode, expected ใช้ทั้ง distribution"""
    probs = {"0": 0.05, "1": 0.46, "2": 0.44, "3": 0.05}
    # ค่าคาดหวังราว 1.49 -> ปัดเป็น 1 -> คะแนน 2 ; mode คือ 1 -> คะแนน 2 เช่นกัน
    expected_extractor, _ = make(form, 1.62, probs, decision="expected")
    argmax_extractor, _ = make(form, 1.62, probs, decision="argmax")

    # score 1.62 ปัดเป็น 2 -> คะแนน 3 ขณะที่ mode ยังเป็นระดับ 1 -> คะแนน 2
    assert expected_extractor.extract("ก็กลาง ๆ", "q9").value == 3
    assert argmax_extractor.extract("ก็กลาง ๆ", "q9").value == 2


def test_confidence_is_real_probability_not_self_report(form):
    extractor, _ = make(form, 2.0, {"0": 0.01, "1": 0.04, "2": 0.9, "3": 0.05})
    result = extractor.extract("ปวดปานกลาง", "q9")

    assert result.value == 3
    assert result.confidence == pytest.approx(0.9)
    # เก็บ distribution ทั้งชุดไว้ ให้ dialogue manager เอาไปตัดสินว่าควรถามย้ำไหม
    assert result.raw["probabilities"] == {"0": 0.01, "1": 0.04, "2": 0.9, "3": 0.05}


def test_records_both_decision_rules_for_later_reanalysis(form):
    extractor, _ = make(form, 1.62, {"0": 0.05, "1": 0.46, "2": 0.44, "3": 0.05})
    raw = extractor.extract("ก็กลาง ๆ", "q9").raw

    assert raw["value_expected"] == 3
    assert raw["value_argmax"] == 2


def test_confident_meta_intent_blocks_scoring(form):
    extractor, _ = make(form, 0.3, {"0": 0.8, "1": 0.2},
                        meta=FakeChoiceAnswer("repeat", {"none": 0.1, "repeat": 0.85}))
    result = extractor.extract("เมื่อกี้ถามว่าอะไรนะ", "q9")

    assert result.value is None
    assert result.meta_intent == "repeat"


def test_weak_meta_intent_is_ignored(form):
    """ไม่งั้นคำตอบปกติจะถูกตีเป็น meta แล้วทิ้งคะแนนไปเปล่า ๆ"""
    extractor, _ = make(form, 1.0, {"1": 0.9, "0": 0.1},
                        meta=FakeChoiceAnswer("dont_know", {"none": 0.45, "dont_know": 0.3}))
    result = extractor.extract("ปวดนิดหน่อย", "q9")

    assert result.meta_intent is None
    assert result.value == 2


def test_failure_does_not_propagate(form):
    client = FakeClient(FakeScoreAnswer(1.0, {"1": 1.0}), raises=RuntimeError("cuda oom"))
    extractor = SystemOneExtractor(form=form, client=client)
    result = extractor.extract("ปวดมาก", "q9")

    assert result.value is None
    assert result.error and "cuda oom" in result.error


def test_empty_transcript_skips_model(form):
    extractor, client = make(form, 1.0, {"1": 1.0})
    result = extractor.extract("   ", "q9")

    assert result.error == "empty_transcript"
    assert client.calls == []


def test_questions_include_every_option_description(form):
    extractor, client = make(form, 1.0, {"1": 1.0})
    extractor.extract("ปวดนิดหน่อย", "q9")

    _, questions = client.calls[0]
    criteria = questions["score"].criteria
    scale = form.scale("severity")
    assert len(criteria) == len(scale.options)
    for option, line in zip(scale.options, criteria):
        assert option.description_th in line


def test_rejects_unknown_decision_rule(form):
    with pytest.raises(ValueError, match="decision"):
        SystemOneExtractor(form=form, client=object(), decision="median")


# --- คำถามถูกส่งไปทาง state ไม่ใช่ instructions -----------------------------


def test_state_carries_question_and_answer(form):
    extractor, client = make(form, 1.0, {"1": 1.0})
    extractor.extract("ปวดนิดหน่อย", "q9")

    state, _ = client.calls[0]
    assert state == {
        "คำถามที่พยาบาลถาม": form.item("q9").text_th,
        "คำตอบของผู้ป่วย": "ปวดนิดหน่อย",
    }


def test_instructions_do_not_repeat_the_question(form):
    """คำถามอยู่ใน state ที่เดียว ไม่ซ้ำใน instructions"""
    extractor, client = make(form, 1.0, {"1": 1.0})
    extractor.extract("ปวดนิดหน่อย", "q9")

    _, questions = client.calls[0]
    assert form.item("q9").text_th not in questions["score"].instructions


def test_questions_are_shared_across_items_with_same_scale(form):
    """q1-q6 ใช้ scale difficulty ร่วมกัน จึงใช้ชุดคำถามเดียวกันได้ทั้งหมด"""
    extractor, client = make(form, 1.0, {"1": 1.0})
    extractor.extract("ทำได้ปกติ", "q1")
    extractor.extract("ทำได้ปกติ", "q2")

    (_, q1_questions), (_, q2_questions) = client.calls
    assert q1_questions["score"] is q2_questions["score"]
    # แต่ state ต้องต่างกัน เพราะคนละคำถาม
    assert client.calls[0][0] != client.calls[1][0]
