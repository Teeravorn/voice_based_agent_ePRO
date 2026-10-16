"""เทสต์ offline ของ ReasonLogitExtractor (ไม่ต้องมี Ollama)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import LogprobResponse  # noqa: E402
from extractor import load_form  # noqa: E402
from extractor.logit_extractor import (  # noqa: E402
    ReasonLogitExtractor,
    expected_value,
    find_value_position,
)

FORM_PATH = ROOT / "forms" / "quickdash_th.json"


@pytest.fixture(scope="module")
def form():
    return load_form(FORM_PATH)


def _entries(tokens: list[str], value_top: dict[str, float] | None = None) -> list[dict]:
    """สร้าง entries แบบที่ Ollama คืน ใส่ top_logprobs ให้เฉพาะ token ที่เป็นตัวเลข"""
    entries = []
    for token in tokens:
        top = []
        if token.strip().isdigit() and value_top is not None:
            top = [{"token": t, "logprob": lp} for t, lp in value_top.items()]
        entries.append({"token": token, "logprob": 0.0, "top_logprobs": top})
    return entries


# token ชุดนี้แบ่งแบบเดียวกับที่ qwen3.5:4b แบ่งจริง (วัดจาก Ollama 0.34)
REASON_WITH_DIGIT = ['{"', 'reason', '":', ' "', 'ระดับ ', '5', ' ใน \\"value\\": 9', '",',
                     ' "', 'value', '":', ' ', '4', ',', ' "', 'meta_intent', '":', ' "none"',
                     ',', ' "', 'confidence', '":', ' ', '0', '.9', '}']


class FakeLogprobEngine:
    think = False

    def __init__(self, tokens, value_top):
        self.response = LogprobResponse(
            text="".join(tokens), entries=_entries(tokens, value_top),
            latency_ms=5.0, model="fake", eval_count=len(tokens),
        )
        self.calls = []

    def chat_logprobs(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


def test_find_value_position_skips_digits_inside_reason():
    entries = _entries(REASON_WITH_DIGIT)
    position = find_value_position(entries)
    assert entries[position]["token"] == "4"
    assert position == REASON_WITH_DIGIT.index("4")


def test_find_value_position_none_without_value_key():
    assert find_value_position(_entries(['{"', 'reason', '":', ' "x"', '}'])) is None


def test_extractor_uses_logprobs_at_value_position(form):
    engine = FakeLogprobEngine(REASON_WITH_DIGIT, {"4": -0.1, "5": -2.5, "6": -9.0, "3": -6.0})
    result = ReasonLogitExtractor(engine=engine, form=form).extract("ปวดมาก", "q9")

    assert result.value == 4
    assert result.label == form.scale_for(form.item("q9")).label(4)
    # 6 อยู่นอก scale จึงต้องไม่ถูกนับใน softmax
    assert set(result.raw["probabilities"]) == {"3", "4", "5"}
    assert result.confidence == pytest.approx(result.raw["probabilities"]["4"], abs=1e-4)
    assert result.raw["value_ที่โมเดลเขียน"] == 4
    # ส่ง schema ไปด้วย โมเดลจึงได้เขียน reason ก่อน value
    assert "format" in engine.calls[0][1]


def test_extractor_falls_back_to_written_value(form):
    engine = FakeLogprobEngine(REASON_WITH_DIGIT, value_top={})
    result = ReasonLogitExtractor(engine=engine, form=form).extract("ปวดมาก", "q9")
    assert result.value == 4
    assert result.confidence == 0.0
    assert result.raw["fallback"]


def test_undecided_when_zero_wins(form):
    tokens = ['{"', 'reason', '":', ' "ขอให้ทวน"', ',', ' "', 'value', '":', ' ', '0', ',',
              ' "', 'meta_intent', '":', ' "repeat"', ',', ' "', 'confidence', '":', ' 0.9', '}']
    engine = FakeLogprobEngine(tokens, {"0": -0.01, "1": -5.0})
    result = ReasonLogitExtractor(engine=engine, form=form).extract("ถามว่าอะไรนะ", "q9")
    assert result.value is None
    assert result.meta_intent == "repeat"


def test_expected_value_ignores_undecided(form):
    scale = form.scale_for(form.item("q9"))
    assert expected_value({4: 0.54, 5: 0.39, 3: 0.07}, scale) == 4
    assert expected_value({4: 0.3, 5: 0.7, 0: 0.2}, scale) == 5
    assert expected_value({0: 0.9, 4: 0.05, 5: 0.05}, scale) is None
    assert expected_value({0: 1.0}, scale) is None


def test_invalid_json_is_reported(form):
    engine = FakeLogprobEngine(["ไม่ใช่ json"], {})
    result = ReasonLogitExtractor(engine=engine, form=form).extract("ปวด", "q9")
    assert result.error == "invalid_json"
    json.dumps(result.to_dict(), ensure_ascii=False)
