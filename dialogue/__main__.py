"""จำลองบทสนทนาด้วยการพิมพ์ แทนการพูดและฟัง

ใช้ตรวจ flow ของ dialogue manager กับ extractor ตัวจริง ก่อนจะมี question presenter
และการอัดเสียงสด ทุกอย่างที่ระบบจะ "พูด" ถูกพิมพ์ออกมาพร้อม key ของไฟล์เสียง

    python -m dialogue
    python -m dialogue --form forms/dash_th.json --engine reason-logit --threshold 0.7
    python -m dialogue --script answers.txt     # อ่านคำตอบทีละบรรทัดจากไฟล์

พิมพ์บรรทัดว่างแทนการไม่พูดอะไร พิมพ์ ``/back`` ``/stop`` ``/repeat`` แทนการกดปุ่มบนหน้าจอ
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from engine import ChatEngine
from extractor import OptionExtractor, load_form

from .manager import DialogueManager, End, Listen, Say, ShowImage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORM = ROOT / "forms" / "quickdash_th.json"
DEFAULT_RESULTS_DIR = ROOT / "evaluation" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--form", type=Path, default=DEFAULT_FORM)
    parser.add_argument("--engine", choices=["ollama", "reason-logit"], default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--threshold", type=float, default=0.6, help="confidence ขั้นต่ำที่ยอมรับคำตอบ")
    parser.add_argument("--max-attempts", type=int, default=3, help="ถามซ้ำได้กี่ครั้งก่อนส่งให้พยาบาลตรวจ")
    parser.add_argument("--script", type=Path, default=None, help="ไฟล์คำตอบ บรรทัดละหนึ่งคำตอบ")
    parser.add_argument("--out", type=Path, default=None, help="ที่เก็บผล session (ค่าตั้งต้นอยู่ใน evaluation/results/)")
    return parser.parse_args(argv)


def show(actions) -> None:
    for action in actions:
        if isinstance(action, Say):
            print(f"  ระบบ  [{action.key}] {action.text}")
        elif isinstance(action, ShowImage):
            print(f"  ภาพ   [{action.path}] {action.caption_th}")
        elif isinstance(action, End):
            print(f"  -- จบ ({action.status}) --")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    form = load_form(args.form)

    # think=False เพราะความแม่นเท่ากันแต่เร็วกว่ามาก (ดู README)
    engine = ChatEngine(model=args.model, think=False)
    if not engine.health():
        print("เรียก Ollama ไม่ได้ ตรวจว่า server รันอยู่หรือยัง (ollama serve) แล้วลองใหม่")
        return 1
    if args.engine == "reason-logit":
        from extractor.logit_extractor import ReasonLogitExtractor

        extractor = ReasonLogitExtractor(engine=engine, form=form)
    else:
        extractor = OptionExtractor(engine=engine, form=form)

    # การเรียกครั้งแรกต้องโหลดโมเดลเข้า VRAM (วัดได้ ~58 วินาที) ถ้าไม่ warm up ก่อน
    # เวลานี้จะไปตกที่คำตอบข้อแรกของผู้ป่วย และทำให้ patient_time_s สูงเกินจริง
    print("warm up extractor ...", flush=True)
    first_item = form.item(form.sections[0].item_ids[0])
    extractor.extract_for("ไม่ยากค่ะ", first_item, form.scale_for(first_item))

    manager = DialogueManager(
        form, extractor, confidence_threshold=args.threshold, max_attempts=args.max_attempts
    )
    scripted = iter(args.script.read_text(encoding="utf-8").splitlines()) if args.script else None

    actions = manager.start()
    while True:
        show(actions)
        if manager.finished:
            break
        listen = next(action for action in actions if isinstance(action, Listen))
        if scripted is not None:
            answer = next(scripted, None)
            if answer is None:
                print("คำตอบในไฟล์หมดแล้ว หยุดก่อนจบ")
                break
            print(f"  ผู้ป่วย [{listen.step_id}] {answer}")
        else:
            try:
                answer = input(f"  ผู้ป่วย [{listen.step_id}] > ")
            except EOFError:
                break

        if answer.strip() in ("/back", "/stop", "/repeat"):
            actions = manager.command(answer.strip()[1:])
        else:
            actions = manager.handle(answer)

    review = manager.review_items()
    print(f"\nข้อที่ต้องให้พยาบาลตรวจ: {', '.join(r.step_id for r in review) or '-'}")
    for section_score in manager.scores().sections.values():
        score = "-" if section_score.score is None else f"{section_score.score:.1f}"
        print(
            f"คะแนน {section_score.section_id:<7}: {score:>5}  ({section_score.status.value}, "
            f"ตอบ {section_score.answered}/{section_score.total_items})"
        )

    out_path = args.out or DEFAULT_RESULTS_DIR / f"dialogue_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"engine": args.engine, "llm_model": args.model, **manager.to_dict()}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"เก็บผล session ไว้ที่ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
