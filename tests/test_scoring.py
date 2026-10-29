"""เทสต์ score calculator เทียบกับสูตรใน forms/Scoring_DASH_Thai.pdf"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor import load_form  # noqa: E402
from scoring import ScoreStatus, dash_score, score_form, score_section  # noqa: E402

QUICKDASH = ROOT / "forms" / "quickdash_th.json"
DASH = ROOT / "forms" / "dash_th.json"


@pytest.fixture(scope="module")
def quickdash():
    return load_form(QUICKDASH)


@pytest.fixture(scope="module")
def dash():
    return load_form(DASH)


def section_answers(form, section_id, value):
    section = next(s for s in form.sections if s.id == section_id)
    return {item_id: value for item_id in section.item_ids}


# --- สูตร ----------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1] * 30, 0.0),     # ไม่มีความบกพร่อง
        ([5] * 30, 100.0),   # บกพร่องมากที่สุด
        ([3] * 30, 50.0),
        ([1, 2, 3, 4], 37.5),  # (10/4 - 1) x 25
        ([2] * 11, 25.0),
    ],
)
def test_dash_formula(values, expected):
    assert dash_score(values) == pytest.approx(expected)


def test_missing_items_divide_by_answered_count(dash):
    # n = จำนวนข้อที่ตอบ ไม่ใช่ 30 ข้อที่ไม่ตอบไม่ถูกนับเป็นศูนย์
    answers = section_answers(dash, "core", 3)
    answers["d1"] = answers["d2"] = answers["d3"] = None
    result = score_form(dash, answers).sections["core"]
    assert result.status is ScoreStatus.SCORED
    assert result.answered == 27
    assert result.score == pytest.approx(50.0)
    assert result.missing == ("d1", "d2", "d3")


# --- ข้อขั้นต่ำ -----------------------------------------------------------


def test_dash_core_needs_27_of_30(dash):
    answers = section_answers(dash, "core", 2)
    for item_id in ("d1", "d2", "d3", "d4"):
        answers[item_id] = None
    result = score_form(dash, answers).sections["core"]
    assert result.status is ScoreStatus.INSUFFICIENT
    assert result.score is None
    assert result.min_answered == 27


@pytest.mark.parametrize("path", [QUICKDASH, DASH])
def test_optional_module_needs_all_four(path):
    form = load_form(path)
    answers = section_answers(form, "work", 4)
    answers["w4"] = None
    result = score_form(form, answers).sections["work"]
    assert result.status is ScoreStatus.INSUFFICIENT
    answers["w4"] = 4
    assert score_form(form, answers).sections["work"].score == pytest.approx(75.0)


def test_optional_module_not_administered(quickdash):
    result = score_form(quickdash, section_answers(quickdash, "core", 1))
    assert result.sections["core"].score == 0.0
    assert result.sections["work"].status is ScoreStatus.NOT_ADMINISTERED
    assert result.sections["sports"].status is ScoreStatus.NOT_ADMINISTERED


def test_required_section_with_no_answers_is_insufficient(quickdash):
    assert score_form(quickdash, {}).sections["core"].status is ScoreStatus.INSUFFICIENT


def test_rules_come_from_form_file(quickdash, dash):
    core = {s.id: s for s in quickdash.sections}["core"]
    assert core.scoring["total_items"] == 11
    assert {s.id: s for s in dash.sections}["core"].scoring["min_answered"] == 27


# --- ตรวจข้อมูลเข้า ---------------------------------------------------------


def test_out_of_scale_value_rejected(quickdash):
    with pytest.raises(ValueError, match="ไม่อยู่ใน scale"):
        score_form(quickdash, {"q1": 6})


def test_unknown_item_rejected(quickdash):
    with pytest.raises(KeyError):
        score_form(quickdash, {"x99": 1})


def test_gate_is_not_scored(quickdash):
    # gate ไม่ใช่คำถามในฟอร์ม ส่งเข้ามาถือว่าผิด
    with pytest.raises(KeyError):
        score_form(quickdash, {"work_gate": 1})


def test_unknown_method_rejected(quickdash):
    core = quickdash.sections[0]
    with pytest.raises(ValueError, match="ยังไม่รองรับ"):
        score_section(quickdash, replace(core, scoring={**core.scoring, "method": "tess"}), {})


def test_total_items_mismatch_rejected(quickdash):
    core = quickdash.sections[0]
    with pytest.raises(ValueError, match="ระบุว่ามี"):
        score_section(quickdash, replace(core, scoring={**core.scoring, "total_items": 30}), {})


def test_to_dict_rounds(quickdash):
    answers = section_answers(quickdash, "core", 1)
    answers["q1"] = 2  # (12/11 - 1) x 25 = 2.2727...
    data = score_form(quickdash, answers).to_dict()
    assert data["sections"]["core"]["score"] == 2.27
    assert data["sections"]["work"]["status"] == "not_administered"
