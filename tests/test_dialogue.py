"""เทสต์ dialogue manager แบบ offline ด้วย extractor ปลอม

extractor ปลอมคืนผลตามที่กำหนดไว้ต่อข้อความ จึงทดสอบได้เฉพาะ logic การควบคุมบทสนทนา
ไม่ผูกกับความแม่นของโมเดล
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dialogue import (  # noqa: E402
    DialogueManager,
    DialogueState,
    End,
    Listen,
    Outcome,
    Say,
    ShowImage,
    Status,
    detect_command,
)
from extractor import ExtractionResult, load_form  # noqa: E402

QUICKDASH = ROOT / "forms" / "quickdash_th.json"
DASH = ROOT / "forms" / "dash_th.json"

CORE = [f"q{n}" for n in range(1, 12)]
WORK = ["work_gate", "work_occupation", "w1", "w2", "w3", "w4"]
SPORTS = ["sports_gate", "sports_activity", "s1", "s2", "s3", "s4"]


@pytest.fixture(scope="module")
def form():
    return load_form(QUICKDASH)


class FakeExtractor:
    """แปลงข้อความเป็นผลตามตาราง ข้อความที่ไม่อยู่ในตารางให้คะแนน 2 มั่นใจเต็ม"""

    RESULTS = {
        "ใช่": ExtractionResult(value=1, label="ใช่", confidence=0.95),
        "ไม่ใช่": ExtractionResult(value=2, label="ไม่ใช่", confidence=0.95),
        "ไม่แน่ใจ": ExtractionResult(value=3, confidence=0.3),
        "ไม่ค่อยแน่ใจ": ExtractionResult(value=4, confidence=0.5),
        "ทวนอีกที": ExtractionResult(meta_intent="repeat", confidence=0.9),
        "ไม่เข้าใจ": ExtractionResult(meta_intent="need_help", confidence=0.9),
        "ไม่รู้": ExtractionResult(meta_intent="dont_know", confidence=0.9),
        "ฝนตก": ExtractionResult(meta_intent="out_of_scope", confidence=0.9),
        "พัง": ExtractionResult(error="ollama down"),
    }

    def __init__(self):
        self.calls = []

    def extract_for(self, transcript, item, scale):
        self.calls.append((transcript, item.id, scale.id))
        return self.RESULTS.get(transcript, ExtractionResult(value=2, confidence=1.0))


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += 1.0
        return self.now


def make(form, **kwargs):
    extractor = FakeExtractor()
    manager = DialogueManager(form, extractor, clock=FakeClock(), **kwargs)
    return manager, extractor


def listening(actions):
    listens = [action for action in actions if isinstance(action, Listen)]
    assert len(listens) <= 1
    return listens[0].step_id if listens else None


def keys(actions):
    return [action.key for action in actions if isinstance(action, Say)]


def answer_core(manager, answer="ยากนิดหน่อย"):
    actions = []
    for _ in CORE:
        actions = manager.handle(answer)
    return actions


# --- การโหลดฟอร์ม --------------------------------------------------------


def test_form_sections_and_speech(form):
    assert [section.id for section in form.sections] == ["core", "work", "sports"]
    assert list(form.sections[0].item_ids) == CORE
    work = form.sections[1]
    assert work.gate.id == "work_gate" and work.gate.scale_id == "yes_no"
    assert work.free_text.id == "work_occupation"
    assert form.speech["greeting_th"]


@pytest.mark.parametrize("path", [QUICKDASH, DASH])
def test_gate_skip_value_is_a_real_option(path):
    # value 0 ชนกับ UNDECIDED ของ extractor ถ้า "ไม่ใช่" = 0 ผู้ป่วยจะตอบว่าไม่ที่ gate ไม่ได้เลย
    form = load_form(path)
    for section in form.sections:
        if section.gate:
            scale = form.scale(section.gate.scale_id)
            assert section.gate.skip_on_value != 0
            assert scale.is_valid(section.gate.skip_on_value)


def test_every_spoken_text_exists():
    # ทุก key ที่ manager พูดต้องมีข้อความในฟอร์ม ไม่อย่างนั้นจะได้เสียงเงียบ
    needed = [
        "greeting_th", "ready_th", "no_speech_th", "not_understood_th", "retry_prefix_th",
        "dont_know_hint_th", "flagged_th", "back_th", "no_previous_th", "closing_th",
        "stopped_th", "confirm_accepted_th",
    ]
    for path in (QUICKDASH, DASH):
        speech = load_form(path).speech
        assert [key for key in needed if not speech.get(key)] == []


def test_dash_greeting_says_thirty_items():
    assert "สามสิบข้อ" in load_form(DASH).speech["greeting_th"]


# --- flow ปกติ -----------------------------------------------------------


def test_start_greets_and_asks_first_item(form):
    manager, _ = make(form)
    actions = manager.start()
    assert keys(actions)[:3] == ["speech/greeting_th", "speech/ready_th", "item/q1"]
    assert isinstance(actions[2], ShowImage) and actions[2].item_id == "q1"
    assert listening(actions) == "q1"
    assert manager.state is DialogueState.AWAITING_ANSWER


def test_full_session_in_order(form):
    manager, extractor = make(form)
    asked = [listening(manager.start())]
    answers = {"work_gate": "ใช่", "sports_gate": "ใช่", "work_occupation": "ครู", "sports_activity": "ฟุตบอล"}
    while not manager.finished:
        asked.append(listening(manager.handle(answers.get(asked[-1], "ยากนิดหน่อย"))))
    assert asked[:-1] == CORE + WORK + SPORTS
    assert manager.state is DialogueState.COMPLETED
    assert all(record.status is Status.ANSWERED for record in manager.records.values())
    assert manager.records["work_occupation"].text == "ครู"
    # คำถามปลายเปิดไม่ต้องเรียกโมเดล
    assert "work_occupation" not in [item_id for _, item_id, _ in extractor.calls]


def test_scores_and_labels_come_from_form(form):
    manager, _ = make(form)
    manager.start()
    manager.handle("ยากนิดหน่อย")
    record = manager.records["q1"]
    assert record.value == 2
    assert record.label == form.scale("difficulty").label(2)


def test_section_intro_said_once_before_first_item(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    manager.handle("ใช่")
    actions = manager.handle("ครู")
    assert keys(actions)[:3] == ["speech/confirm_accepted_th", "intro/work", "item/w1"]
    assert "intro/work" not in keys(manager.handle("ยากนิดหน่อย"))


# --- gate ----------------------------------------------------------------


def test_gate_no_skips_section(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    actions = manager.handle("ไม่ใช่")
    assert "skip/work_gate" in keys(actions)
    assert listening(actions) == "sports_gate"
    for step_id in WORK[1:]:
        assert manager.records[step_id].status is Status.SKIPPED


def test_both_gates_no_completes(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    manager.handle("ไม่ใช่")
    actions = manager.handle("ไม่ใช่")
    assert isinstance(actions[-1], End) and actions[-1].status == "completed"
    assert keys(actions)[-1] == "speech/closing_th"


def test_flagged_gate_skips_section_for_nurse(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    for _ in range(3):
        actions = manager.handle("ฝนตก")
    assert manager.records["work_gate"].status is Status.FLAGGED
    assert listening(actions) == "sports_gate"
    assert manager.records["w1"].status is Status.SKIPPED


# --- ถามซ้ำ / flag -------------------------------------------------------


def test_low_confidence_reads_options_then_reasks(form):
    manager, _ = make(form)
    manager.start()
    first = manager.handle("ไม่แน่ใจ")
    assert keys(first) == ["speech/not_understood_th", "options/difficulty"]
    assert listening(first) == "q1"
    second = manager.handle("ไม่แน่ใจ")
    assert keys(second) == ["speech/retry_prefix_th", "item/q1", "options/difficulty"]
    assert manager.records["q1"].status is Status.PENDING


def test_flag_after_max_attempts_keeps_best_guess(form):
    manager, _ = make(form, max_attempts=3)
    manager.start()
    manager.handle("ไม่แน่ใจ")       # 3 @ 0.3
    manager.handle("ไม่ค่อยแน่ใจ")   # 4 @ 0.5
    actions = manager.handle("ฝนตก")
    record = manager.records["q1"]
    assert record.status is Status.FLAGGED
    assert record.value is None
    assert record.suggested_value == 4
    assert keys(actions)[0] == "speech/flagged_th"
    assert listening(actions) == "q2"
    assert [r.step_id for r in manager.review_items()] == ["q1"]


def test_threshold_is_configurable(form):
    manager, _ = make(form, confidence_threshold=0.25)
    manager.start()
    manager.handle("ไม่แน่ใจ")
    assert manager.records["q1"].value == 3


@pytest.mark.parametrize(
    "answer, expected_keys",
    [
        ("ทวนอีกที", ["item/q1"]),
        ("ไม่เข้าใจ", ["options/difficulty"]),
        ("ไม่รู้", ["speech/dont_know_hint_th", "options/difficulty"]),
        ("", ["speech/no_speech_th", "item/q1"]),
        ("ฝนตก", ["speech/not_understood_th", "options/difficulty"]),
    ],
)
def test_response_follows_intent(form, answer, expected_keys):
    manager, _ = make(form)
    manager.start()
    actions = manager.handle(answer)
    assert keys(actions) == expected_keys
    assert listening(actions) == "q1"


def test_empty_answer_does_not_call_extractor(form):
    manager, extractor = make(form)
    manager.start()
    manager.handle("   ")
    assert extractor.calls == []
    assert manager.records["q1"].turns[0].outcome is Outcome.NO_SPEECH


def test_extractor_error_flags_immediately(form):
    manager, _ = make(form)
    manager.start()
    actions = manager.handle("พัง")
    assert manager.records["q1"].status is Status.FLAGGED
    assert "ollama down" in manager.records["q1"].flag_reason
    assert listening(actions) == "q2"


def test_single_attempt_flags_on_first_failure(form):
    manager, _ = make(form, max_attempts=1)
    manager.start()
    manager.handle("ไม่แน่ใจ")
    assert manager.records["q1"].status is Status.FLAGGED


# --- คำสั่ง ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ย้อนกลับค่ะ", "back"),
        ("ขอย้อนกลับไปข้อที่แล้ว", "back"),
        ("ขอหยุดก่อนนะ", "stop"),
        ("พอแค่นี้ครับ", "stop"),
        ("ทำได้แต่ต้องขอหยุดพักเป็นช่วง ๆ ตลอดเลยค่ะ", None),  # ยาว = คำตอบ
        ("ต้องหยุดพักบ่อย", None),
        ("ต้องขอหยุดพักบ่อย", None),  # สั้นแต่เป็นคำบรรยายอาการ
        ("ยากปานกลาง", None),
        ("", None),
    ],
)
def test_detect_command(text, expected):
    assert detect_command(text) == expected


def test_back_reasks_previous_and_clears_answer(form):
    manager, extractor = make(form)
    manager.start()
    manager.handle("ยากนิดหน่อย")
    actions = manager.handle("ย้อนกลับ")
    assert keys(actions)[:2] == ["speech/back_th", "item/q1"]
    assert manager.records["q1"].status is Status.PENDING
    assert manager.records["q1"].value is None
    assert len(extractor.calls) == 1  # คำสั่งไม่ผ่านโมเดล
    assert listening(manager.handle("ไม่แน่ใจ")) == "q1"
    manager.handle("ยากนิดหน่อย")
    assert manager.current_step_id == "q2"


def test_back_on_first_item(form):
    manager, _ = make(form)
    manager.start()
    actions = manager.command("back")
    assert keys(actions) == ["speech/no_previous_th"]
    assert listening(actions) == "q1"


def test_back_skips_over_skipped_section(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    manager.handle("ไม่ใช่")            # ข้ามชุดการทำงาน
    actions = manager.command("back")   # จาก sports_gate
    assert listening(actions) == "work_gate"


def test_back_to_gate_and_change_answer_reopens_section(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    manager.handle("ไม่ใช่")
    manager.command("back")
    actions = manager.handle("ใช่")
    assert listening(actions) == "work_occupation"
    assert manager.records["w1"].status is Status.PENDING


def test_stop_ends_session(form):
    manager, _ = make(form)
    manager.start()
    manager.handle("ยากนิดหน่อย")
    actions = manager.handle("ขอหยุดก่อน")
    assert keys(actions) == ["speech/stopped_th"]
    assert isinstance(actions[-1], End) and actions[-1].status == "stopped"
    assert manager.finished
    with pytest.raises(RuntimeError):
        manager.handle("ยากนิดหน่อย")


def test_unknown_command(form):
    manager, _ = make(form)
    manager.start()
    with pytest.raises(ValueError):
        manager.command("jump")


# --- เวลาและผลลัพธ์ -------------------------------------------------------


def test_patient_time_and_export(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)
    manager.handle("ไม่ใช่")
    manager.handle("ไม่ใช่")
    data = manager.to_dict()
    assert data["state"] == "completed"
    assert data["patient_time_s"] is not None and data["patient_time_s"] > 0
    assert data["first_question_s"] < data["last_answer_s"]
    assert [record["step_id"] for record in data["records"]] == CORE + WORK + SPORTS
    assert data["records"][0]["turns"][0]["outcome"] == "accepted"


def test_scores_from_session(form):
    manager, _ = make(form)
    manager.start()
    answer_core(manager)            # ทุกข้อได้ 2
    manager.handle("ไม่ใช่")
    manager.handle("ไม่ใช่")
    scores = manager.scores()
    assert scores.sections["core"].score == pytest.approx(25.0)
    assert scores.sections["work"].status.value == "not_administered"
    assert manager.to_dict()["scores"]["core"]["score"] == 25.0


def test_flagged_items_count_as_missing_until_confirmed(form):
    manager, _ = make(form, max_attempts=1)
    manager.start()
    for _ in range(4):                # q1-q4 flag ทั้งหมด
        manager.handle("ไม่แน่ใจ")
    while manager.current_step_id in CORE:
        manager.handle("ยากนิดหน่อย")
    assert manager.answers()["q1"] is None
    assert manager.scores().sections["core"].status.value == "insufficient"  # QuickDASH ต้องตอบอย่างน้อย 8 ข้อ ตอบได้ 7
    confirmed = manager.scores(confirmed={"q1": 2, "q2": 2, "q3": 2, "q4": 2})
    assert confirmed.sections["core"].score == pytest.approx(25.0)


def test_start_twice_raises(form):
    manager, _ = make(form)
    manager.start()
    with pytest.raises(RuntimeError):
        manager.start()


def test_dash_runs_end_to_end():
    manager, _ = make(load_form(DASH))
    manager.start()
    while not manager.finished:
        step = manager.current_step_id
        manager.handle("ไม่ใช่" if step.endswith("_gate") else "ยากนิดหน่อย")
    answered = [r for r in manager.records.values() if r.status is Status.ANSWERED]
    assert len(answered) == 30 + 2
