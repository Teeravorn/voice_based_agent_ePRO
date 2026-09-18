"""รวมผลการรันเป็นตัวเลขสรุป และพิมพ์ตารางออกหน้าจอ

ตัวชี้วัดที่เลือกใช้ และเหตุผล

- **exact match** ตัวเลขหลักที่จะเอาไปเทียบกับคะแนนที่พยาบาลให้
- **±1 agreement** คะแนนเป็น ordinal การพลาดจาก 3 ไปเป็น 4 ไม่เท่ากับพลาดจาก 3 ไปเป็น 5
  ตัวเลขนี้บอกว่าโมเดลเข้าใจ "ทิศทาง" ของอาการถูกไหม แม้จะไม่ตรงเป๊ะ
- **undecided rate** ตอบ 0 ทั้งที่ควรให้คะแนนได้ คือ failure mode ที่เจ็บที่สุดในทางคลินิก
  เพราะแปลว่าต้องให้พยาบาลมาถามซ้ำ ระบบไม่ได้ช่วยลดงาน
- **meta-intent accuracy** วัดว่าแยกออกไหมว่าผู้ป่วยไม่ได้กำลังตอบคำถาม

ไม่คำนวณ quadratic weighted kappa ในเฟสนี้ เพราะ n = 15 ยังไม่มีความหมายทางสถิติ
ไฟล์ผลลัพธ์เก็บข้อมูลดิบครบพอให้คิดย้อนหลังได้เมื่อมีข้อมูลจากผู้ป่วยจริง
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from extractor import ExtractionResult

from .manifest import ManifestRow


@dataclass
class RunRecord:
    """ผลของหนึ่งแถว manifest ในหนึ่งโหมด"""

    row: ManifestRow
    mode: str                      # "text" หรือ "asr"
    transcript: str
    result: ExtractionResult
    asr_latency_ms: float | None = None
    skipped_reason: str | None = None

    @property
    def predicted(self) -> int | None:
        return self.result.value

    @property
    def is_correct(self) -> bool:
        if not self.row.is_scorable:
            # เฉลย 0 แปลว่า "ไม่ควรให้คะแนน" การที่โมเดลไม่ให้คะแนนจึงถูกต้องแล้ว
            return self.predicted is None
        return self.predicted in self.row.gold_values

    @property
    def distance(self) -> int | None:
        """ระยะห่างจากเฉลยที่ใกล้ที่สุด (None เมื่อโมเดลไม่ได้ให้คะแนน)"""
        if self.predicted is None:
            return None
        return min(abs(self.predicted - gold) for gold in self.row.gold_values)

    @property
    def meta_correct(self) -> bool:
        return (self.result.meta_intent or None) == (self.row.gold_meta or None)

    def to_dict(self) -> dict:
        return {
            "audio": self.row.audio,
            "item_id": self.row.item_id,
            "answer_type": self.row.answer_type,
            "mode": self.mode,
            "script_th": self.row.script_th,
            "transcript": self.transcript,
            "gold_value": self.row.gold_display,
            "gold_meta": self.row.gold_meta,
            "predicted": self.predicted,
            "correct": self.is_correct,
            "distance": self.distance,
            "meta_correct": self.meta_correct,
            "asr_latency_ms": None if self.asr_latency_ms is None else round(self.asr_latency_ms, 1),
            "skipped_reason": self.skipped_reason,
            "extraction": self.result.to_dict(),
        }


@dataclass
class Summary:
    mode: str
    n: int = 0
    scorable: int = 0
    exact: int = 0
    within_one: int = 0
    undecided: int = 0        # ควรให้คะแนนได้ แต่โมเดลตอบว่าให้ไม่ได้
    meta_total: int = 0
    meta_correct: int = 0
    errors: int = 0
    llm_latencies: list[float] = field(default_factory=list)
    asr_latencies: list[float] = field(default_factory=list)
    by_type: dict[str, list[int]] = field(default_factory=dict)  # type -> [ถูก, ทั้งหมด]

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("llm_latencies")
        data.pop("asr_latencies")
        data["exact_match"] = ratio(self.exact, self.scorable)
        data["within_one"] = ratio(self.within_one, self.scorable)
        data["undecided_rate"] = ratio(self.undecided, self.scorable)
        data["meta_accuracy"] = ratio(self.meta_correct, self.meta_total)
        data["llm_latency_ms"] = latency_stats(self.llm_latencies)
        data["asr_latency_ms"] = latency_stats(self.asr_latencies)
        return data


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def latency_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    # p95 แบบเลือก index ตรง ๆ พอสำหรับ n เล็ก ไม่ต้อง interpolate
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "median": round(statistics.median(ordered), 1),
        "p95": round(ordered[p95_index], 1),
        "max": round(ordered[-1], 1),
    }


def summarize(records: list[RunRecord], mode: str) -> Summary:
    summary = Summary(mode=mode)
    for record in records:
        if record.skipped_reason:
            continue
        summary.n += 1
        if record.result.error:
            summary.errors += 1
        if record.result.latency_ms:
            summary.llm_latencies.append(record.result.latency_ms)
        if record.asr_latency_ms:
            summary.asr_latencies.append(record.asr_latency_ms)

        if record.row.gold_meta:
            summary.meta_total += 1
            if record.meta_correct:
                summary.meta_correct += 1

        if not record.row.is_scorable:
            continue

        summary.scorable += 1
        bucket = summary.by_type.setdefault(record.row.answer_type, [0, 0])
        bucket[1] += 1
        if record.is_correct:
            summary.exact += 1
            summary.within_one += 1
            bucket[0] += 1
        elif record.predicted is None:
            summary.undecided += 1
        elif record.distance is not None and record.distance <= 1:
            summary.within_one += 1

    return summary


def print_table(records: list[RunRecord], mode: str) -> None:
    print()
    print(f"=== โหมด {mode} ===")
    header = "{:<22} {:<5} {:<10} {:>5} {:>5} {:>3} {:>5} {:>8}".format(
        "audio", "item", "type", "gold", "pred", "ok", "conf", "llm ms"
    )
    if mode == "asr":
        header += "{:>8}".format(" asr ms")
    print(header)
    print("-" * len(header))

    for record in records:
        if record.skipped_reason:
            print(f"{record.row.audio:<22} {record.row.item_id:<5} ข้าม: {record.skipped_reason}")
            continue
        mark = "OK" if record.is_correct else "X"
        pred = "-" if record.predicted is None else str(record.predicted)
        line = "{:<22} {:<5} {:<10} {:>5} {:>5} {:>3} {:>5.2f} {:>8.0f}".format(
            record.row.audio,
            record.row.item_id,
            record.row.answer_type,
            record.row.gold_display,
            pred,
            mark,
            record.result.confidence,
            record.result.latency_ms,
        )
        if mode == "asr":
            line += "{:>8.0f}".format(record.asr_latency_ms or 0)
        print(line)


def print_failures(records: list[RunRecord]) -> None:
    """พิมพ์เคสที่พลาดแบบเต็ม ๆ นี่คือส่วนที่เอาไปวิเคราะห์ต่อได้จริง"""
    failures = [
        record
        for record in records
        if not record.skipped_reason
        and (not record.is_correct or (record.row.gold_meta and not record.meta_correct))
    ]
    if not failures:
        print("\nไม่มีเคสที่พลาด")
        return

    print(f"\n--- เคสที่พลาด {len(failures)} เคส ---")
    for record in failures:
        print(f"\n[{record.mode}] {record.row.audio}  ({record.row.item_id} / {record.row.answer_type})")
        print(f"  สคริปต์  : {record.row.script_th}")
        if record.transcript != record.row.script_th:
            print(f"  ถอดได้   : {record.transcript}")
        gold_line = record.row.gold_display
        if record.row.gold_meta:
            gold_line += f" / meta {record.row.gold_meta}"
        print(f"  เฉลย     : {gold_line}")
        pred_line = str(record.predicted)
        if record.result.meta_intent:
            pred_line += f" / meta {record.result.meta_intent}"
        print(f"  โมเดลตอบ : {pred_line}")
        print(f"  เหตุผล   : {record.result.reason_th}")
        if record.result.error:
            print(f"  error    : {record.result.error}")


def print_summary(summary: Summary) -> None:
    data = summary.as_dict()
    print(f"\n--- สรุปโหมด {summary.mode} ---")
    print(f"  จำนวนที่รัน         : {summary.n} (ให้คะแนนได้ {summary.scorable})")
    print(f"  exact match         : {summary.exact}/{summary.scorable} = {pct(data['exact_match'])}")
    print(f"  +-1 agreement       : {summary.within_one}/{summary.scorable} = {pct(data['within_one'])}")
    print(f"  undecided (ตอบ 0)   : {summary.undecided}/{summary.scorable} = {pct(data['undecided_rate'])}")
    if summary.meta_total:
        print(f"  meta-intent         : {summary.meta_correct}/{summary.meta_total} = {pct(data['meta_accuracy'])}")
    if summary.errors:
        print(f"  เรียกโมเดลไม่สำเร็จ  : {summary.errors}")

    if summary.by_type:
        print("  แยกตามประเภทคำตอบ:")
        for answer_type, (correct, total) in sorted(summary.by_type.items()):
            print(f"    {answer_type:<12} {correct}/{total} = {pct(ratio(correct, total))}")

    if data["llm_latency_ms"]:
        stats = data["llm_latency_ms"]
        print(f"  เวลา LLM (ms)       : mean {stats['mean']} / median {stats['median']} / p95 {stats['p95']}")
    if data["asr_latency_ms"]:
        stats = data["asr_latency_ms"]
        print(f"  เวลา ASR (ms)       : mean {stats['mean']} / median {stats['median']} / p95 {stats['p95']}")


def print_mode_comparison(text_summary: Summary | None, asr_summary: Summary | None) -> None:
    """ช่องว่างระหว่างสองโหมด = ต้นทุนความแม่นที่ ASR เพิ่มเข้ามา"""
    if text_summary is None or asr_summary is None:
        return
    text_exact = ratio(text_summary.exact, text_summary.scorable)
    asr_exact = ratio(asr_summary.exact, asr_summary.scorable)
    if text_exact is None or asr_exact is None:
        return

    print("\n--- เทียบสองโหมด ---")
    print(f"  extractor อย่างเดียว (text) : {pct(text_exact)}")
    print(f"  ทั้ง pipeline (asr)          : {pct(asr_exact)}")
    print(f"  ส่วนต่างที่มาจาก ASR         : {pct(round(text_exact - asr_exact, 4))}")


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"
