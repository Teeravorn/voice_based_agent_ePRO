"""POC runner: ไฟล์เสียงคำตอบผู้ป่วย -> ASR -> option extractor -> คะแนน QuickDASH

รันได้สองโหมด และความต่างของสองโหมดนี้คือสิ่งที่ POC ต้องการวัด

- ``text`` เอาประโยคต้นฉบับจาก manifest ป้อนเข้า extractor ตรง ๆ ไม่ผ่าน ASR
  ผลที่ได้คือ **เพดานความแม่นของ extractor** และรันได้โดยไม่ต้องมีไฟล์เสียงหรือ GPU
- ``asr`` ถอดเสียงจริงก่อนแล้วค่อยเข้า extractor ผลที่ได้คือ **ความแม่นของทั้ง pipeline**

ตัวอย่างการใช้งาน
    python main.py --mode text                 # ยังไม่ต้องมีไฟล์เสียง
    python main.py --mode both                 # หลังอัดเสียงครบแล้ว
    python main.py --mode text --model qwen3:8b
    python main.py --only q9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from engine import ChatEngine
from evaluation.manifest import ManifestRow, load_manifest
from evaluation.report import (
    RunRecord,
    print_failures,
    print_mode_comparison,
    print_summary,
    print_table,
    summarize,
)
from extractor import ExtractionResult, OptionExtractor, load_form

ROOT = Path(__file__).resolve().parent
DEFAULT_FORM = ROOT / "forms" / "quickdash_th.json"
DEFAULT_MANIFEST = ROOT / "evaluation" / "manifest.csv"
DEFAULT_AUDIO_DIR = ROOT / "evaluation" / "answers"
DEFAULT_RESULTS_DIR = ROOT / "evaluation" / "results"

ASR_MODEL_ID = "typhoon-ai/typhoon-whisper-turbo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["text", "asr", "both"], default="text",
                        help="text = ข้าม ASR, asr = ถอดเสียงจริง, both = รันทั้งสองแล้วเทียบกัน")
    parser.add_argument("--engine", choices=["ollama", "logit", "reason-logit", "systemone"], default="ollama",
                        help="ollama = qwen แบบ generative เขียน JSON, "
                             "logit = qwen แบบ decision model อ่าน logit ของตัวเลือกมาทำ softmax, "
                             "reason-logit = qwen เขียน JSON (reason ก่อน) แล้วอ่าน logit ที่ตำแหน่ง value, "
                             "systemone = OpenThai-SystemOne decision head")
    parser.add_argument("--model", default=None,
                        help="ชื่อโมเดล (ollama: qwen3:4b, systemone: iapp/OpenThai-SystemOne)")
    parser.add_argument("--think", choices=["off", "on", "low", "medium", "high"], default=None,
                        help="เฉพาะ ollama: off = ปิด reasoning ซึ่งเร็วกว่ามากในโหมด JSON "
                             "โดยความแม่นเท่ากัน (ดู docstring ของ ChatEngine.chat_json) "
                             "ไม่ระบุ = ใช้ค่าตั้งต้นของ ChatEngine")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="เฉพาะ ollama/logit/reason-logit: context window ค่าตั้งต้น 4096 ซึ่งไม่พอเมื่อเปิด "
                             "thinking (reasoning กิน context หมดก่อนเขียนคำตอบ) ใช้ 16384 ขึ้นไป")
    parser.add_argument("--decision", choices=["expected", "argmax"], default="expected",
                        help="เฉพาะ systemone: แปลง distribution เป็นคะแนนเดียวด้วยวิธีไหน "
                             "(expected วัดแล้วดีกว่า argmax ชัดเจน ดู docstring ของ SystemOneExtractor)")
    parser.add_argument("--form", type=Path, default=DEFAULT_FORM)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--only", action="append", default=None,
                        help="รันเฉพาะคำถามที่ระบุ ใส่ซ้ำได้ เช่น --only q9 --only q11")
    parser.add_argument("--limit", type=int, default=None, help="รันแค่ N แถวแรก")
    parser.add_argument("--out", type=Path, default=None, help="ที่เก็บผลดิบ (ค่าตั้งต้นอยู่ใน evaluation/results/)")
    parser.add_argument("--asr-model", default=ASR_MODEL_ID)
    return parser.parse_args(argv)


def build_engine_kwargs(args) -> dict:
    """แปลง flag ของ CLI เป็นพารามิเตอร์ของ ChatEngine"""
    kwargs = {}
    if args.think is not None:
        # "off"/"on" -> bool ส่วน low/medium/high ส่งเป็นสตริงตามที่ Ollama รับ
        kwargs["think"] = {"off": False, "on": True}.get(args.think, args.think)
    if args.num_ctx is not None:
        kwargs["num_ctx"] = args.num_ctx
    return kwargs


def select_rows(rows: list[ManifestRow], only: list[str] | None, limit: int | None) -> list[ManifestRow]:
    if only:
        wanted = {item_id.strip() for item_id in only}
        rows = [row for row in rows if row.item_id in wanted]
    if limit is not None:
        rows = rows[:limit]
    return rows


def warm_up_extractor(rows: list[ManifestRow], extractor) -> float:
    """ยิง extractor ทิ้งหนึ่งครั้งก่อนเริ่มจับเวลา

    การเรียกครั้งแรกสุดแพงกว่าครั้งต่อ ๆ ไปหลายเท่า (วัดกับ qwen3.5:4b ได้ 8.9 วินาที
    เทียบกับค่ากลาง 1.6 วินาที) เพราะโมเดลต้องถูกโหลดเข้า VRAM และ compile kernel
    ถ้าไม่ตัดออก ค่า mean จะถูกดึงขึ้นจนรายงานผิด และยังพบว่าการเรียกครั้งแรกบางทีให้
    ผลลัพธ์ต่างจากครั้งถัด ๆ ไปแม้ตั้ง temperature=0 กับ seed คงที่

    คืนเวลาที่ใช้ warm up เพื่อรายงานแยก ไม่เอาไปรวมในสถิติ
    """
    if not rows:
        return 0.0
    print("warm up extractor ...", flush=True)
    result = extractor.extract(rows[0].script_th, rows[0].item_id)
    return result.latency_ms


def run_text_mode(rows: list[ManifestRow], extractor: OptionExtractor) -> list[RunRecord]:
    records = []
    for index, row in enumerate(rows, start=1):
        print(f"  [{index}/{len(rows)}] {row.audio} (text)", flush=True)
        result = extractor.extract(row.script_th, row.item_id)

        print("result", result)

        records.append(RunRecord(row=row, mode="text", transcript=row.script_th, result=result))
    return records


def run_asr_mode(rows: list[ManifestRow], extractor: OptionExtractor, audio_dir: Path, asr_model: str) -> list[RunRecord]:
    existing = [(row, audio_dir / row.audio) for row in rows]
    missing = [row.audio for row, path in existing if not path.exists()]
    if len(missing) == len(rows):
        print(f"\nไม่พบไฟล์เสียงเลยใน {audio_dir} ข้ามโหมด asr")
        return [
            RunRecord(row=row, mode="asr", transcript="", result=ExtractionResult(),
                      skipped_reason="ไม่พบไฟล์เสียง")
            for row in rows
        ]
    if missing:
        print(f"\nไม่พบไฟล์เสียง {len(missing)} ไฟล์ จะข้ามไป: {', '.join(missing)}")

    # import ตรงนี้ ไม่ใช่หัวไฟล์ เพราะโหมด text และกรณีที่ยังไม่มีไฟล์เสียง
    # ไม่ควรต้องรอ torch โหลด
    from asr import ASRWhisper

    print(f"\nกำลังโหลด ASR ({asr_model}) ...")
    asr = ASRWhisper()
    asr.load_model(asr_model)

    # warm up ครั้งแรกด้วยไฟล์จริงไฟล์แรก เพื่อไม่ให้เวลา compile/allocate ไปปนกับ
    # เวลาที่จะรายงาน การวัด latency ครั้งแรกโดยไม่ warm up จะสูงกว่าความจริงหลายเท่า
    first_audio = next((path for _, path in existing if path.exists()), None)
    if first_audio is not None:
        print("warm up ASR ...")
        asr.transcribe_file(str(first_audio))

    records = []
    for index, (row, path) in enumerate(existing, start=1):
        if not path.exists():
            records.append(
                RunRecord(row=row, mode="asr", transcript="", result=ExtractionResult(),
                          skipped_reason="ไม่พบไฟล์เสียง")
            )
            continue

        print(f"  [{index}/{len(existing)}] {row.audio} (asr)", flush=True)
        started = time.perf_counter()
        transcript = asr.transcribe_file(str(path))
        asr_latency_ms = (time.perf_counter() - started) * 1000

        result = extractor.extract(transcript, row.item_id)
        records.append(
            RunRecord(row=row, mode="asr", transcript=transcript, result=result,
                      asr_latency_ms=asr_latency_ms)
        )
    return records


def save_results(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nเก็บผลดิบไว้ที่ {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    form = load_form(args.form)
    rows = select_rows(load_manifest(args.manifest), args.only, args.limit)
    if not rows:
        print("ไม่มีแถวให้รัน (ลองตรวจ --only / --limit)")
        return 1

    # ตรวจ manifest กับฟอร์มก่อนเสียเวลาโหลดโมเดล
    for row in rows:
        item = form.item(row.item_id)
        scale = form.scale_for(item)
        for gold in row.gold_values:
            if gold != 0 and not scale.is_valid(gold):
                print(f"manifest ผิด: {row.audio} เฉลย {gold} ไม่อยู่ใน scale {scale.id} ของ {row.item_id}")
                return 1

    if args.engine in ("logit", "reason-logit"):
        from extractor.logit_extractor import LogitExtractor, ReasonLogitExtractor

        model_name = args.model or "qwen3.5:4b"
        engine = ChatEngine(model=model_name, **build_engine_kwargs(args))
        if not engine.health():
            print("เรียก Ollama ไม่ได้ ตรวจว่า server รันอยู่หรือยัง (ollama serve) แล้วลองใหม่")
            return 1
        extractor_class = LogitExtractor if args.engine == "logit" else ReasonLogitExtractor
        extractor = extractor_class(engine=engine, form=form)
        model_label = f"{model_name} ({args.engine}, think={engine.think})"
    elif args.engine == "systemone":
        from extractor.systemone_extractor import MODEL_ID, SystemOneExtractor

        model_name = args.model or MODEL_ID
        print(f"กำลังโหลด {model_name} ...")
        extractor = SystemOneExtractor(form=form, model_path=model_name, decision=args.decision)
        model_label = f"{model_name} (decision={args.decision})"
    else:
        model_name = args.model or "qwen3:4b"
        engine = ChatEngine(model=model_name, **build_engine_kwargs(args))
        if not engine.health():
            print("เรียก Ollama ไม่ได้ ตรวจว่า server รันอยู่หรือยัง (ollama serve) แล้วลองใหม่")
            return 1
        extractor = OptionExtractor(engine=engine, form=form)
        model_label = f"{model_name} (think={engine.think})"

    print(f"ฟอร์ม   : {form.title_th} ({form.form_id} v{form.version})")
    print(f"โมเดล   : {model_label}")
    print(f"จำนวน   : {len(rows)} คำตอบ จาก {len({row.item_id for row in rows})} คำถาม")

    warm_up_ms = warm_up_extractor(rows, extractor)
    print(f"warm up : {warm_up_ms:.0f} ms (ไม่นับรวมในสถิติ)")

    text_records = asr_records = None
    text_summary = asr_summary = None

    if args.mode in ("text", "both"):
        print("\nรันโหมด text (ข้าม ASR) ...")
        text_records = run_text_mode(rows, extractor)
        text_summary = summarize(text_records, "text")
        print_table(text_records, "text")
        print_failures(text_records)
        print_summary(text_summary)

    if args.mode in ("asr", "both"):
        asr_records = run_asr_mode(rows, extractor, args.audio_dir, args.asr_model)
        asr_summary = summarize(asr_records, "asr")
        if asr_summary.n:
            print_table(asr_records, "asr")
            print_failures(asr_records)
            print_summary(asr_summary)

    print_mode_comparison(text_summary, asr_summary if asr_summary and asr_summary.n else None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or DEFAULT_RESULTS_DIR / f"run_{timestamp}.json"
    save_results(
        out_path,
        {
            "timestamp": timestamp,
            "form": {"form_id": form.form_id, "version": form.version},
            "engine": args.engine,
            "llm_model": model_name,
            "decision": args.decision if args.engine == "systemone" else None,
            "asr_model": args.asr_model if args.mode in ("asr", "both") else None,
            "mode": args.mode,
            "warm_up_ms": round(warm_up_ms, 1),
            "num_ctx": args.num_ctx,
            "summaries": {
                name: summary.as_dict()
                for name, summary in (("text", text_summary), ("asr", asr_summary))
                if summary is not None
            },
            "records": [
                record.to_dict()
                for records in (text_records, asr_records)
                if records
                for record in records
            ],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
