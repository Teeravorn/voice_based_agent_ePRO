"""ตัวห่อ Ollama สำหรับเรียก LLM ที่รันบนเครื่อง

เหตุผลที่ต้องรันบนเครื่อง: ข้อมูลอาการผู้ป่วยห้ามออกนอกโรงพยาบาล

``chat_json`` คือตัวที่ option extractor ใช้ ต่างจาก ``chat`` ตรงที่บังคับรูปแบบผลลัพธ์
ด้วย JSON schema (constrained decoding) แทนที่จะหวังว่าโมเดลจะตอบ JSON มาเอง
"""

import json
import time
from dataclasses import dataclass, field
from urllib import response

import ollama
from ollama import ChatResponse, chat
from typing import Literal

# seed คงที่เพื่อให้รันซ้ำแล้วได้ผลเดิม ถ้าผลไม่นิ่ง การเทียบคะแนนกับพยาบาล
# จะไม่มีความหมาย เพราะแยกไม่ออกว่าคะแนนต่างเพราะโมเดลหรือเพราะ sampling
DEFAULT_SEED = 20260101


@dataclass
class LLMResponse:
    """ผลลัพธ์จากการเรียก LLM หนึ่งครั้ง"""

    data: dict            # JSON ที่ parse แล้ว
    text: str             # ข้อความดิบที่โมเดลตอบมา
    latency_ms: float
    model: str
    raw: dict = field(default_factory=dict)


class LLMUnavailable(RuntimeError):
    """เรียก Ollama ไม่ได้ หรือโมเดลตอบกลับมาในรูปแบบที่ใช้ไม่ได้"""


class ChatEngine:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        seed: int = DEFAULT_SEED,
        num_ctx: int = 4096,
        think: bool | Literal['low', 'medium', 'high'] | None = None
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        # think=False มีผลจริงเฉพาะตอนส่ง format=schema ไปด้วย
        # ดูคำอธิบายใน chat_json() และ _free_text_think
        self.think = think
        self.last_thinking = ""

    @property
    def options(self) -> dict:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }

    def health(self) -> bool:
        """เช็กว่า Ollama server ตอบอยู่ไหม ก่อนจะยิงงานยาว ๆ"""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def chat_json(self, messages: list[dict[str, str]], schema: dict) -> LLMResponse:
        """เรียก LLM โดยบังคับให้ตอบตาม JSON schema ที่ให้ไป

        Ollama รับ schema ผ่านพารามิเตอร์ ``format`` แล้วใช้ constrained decoding
        ทำให้ผลลัพธ์ parse ได้เสมอ ไม่ต้องมานั่งแกะ markdown fence หรือข้อความนำ

        ตรงนี้ส่ง ``think`` ไปตามที่ตั้งไว้ได้ เพราะ ``format`` บังคับให้ token แรก
        ต้องเป็น ``{`` โมเดลจึงเปิดบล็อก ``<think>`` ไม่ได้ตั้งแต่แรก grammar เป็นคนปิด
        reasoning ไม่ใช่ตัว flag เอง

        วัดกับ qwen3:4b คำถามเดียวกัน (q9 คำตอบกำกวม) ได้ value เท่ากันทั้งสามแบบ
            think=False       2.8 วินาที   156 tokens
            ไม่ใส่ / think=True  23-27 วินาที  ~1300 tokens
        เร็วกว่าราว 9 เท่าโดยไม่เสียความแม่น เพราะ reason ใน schema ทำหน้าที่แทน
        การคิดอยู่แล้ว **อย่าเอา think=False ออกโดยไม่วัดใหม่**
        """
        started = time.perf_counter()
        try:
            response: ChatResponse = chat(
                model=self.model,
                messages=messages,
                format=schema,
                options=self.options,
                think=self.think,
                stream=False,
            )
        except Exception as exc:
            raise LLMUnavailable(f"เรียก Ollama ไม่สำเร็จ: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        text = response.message.content or ""

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"โมเดลตอบมาไม่ใช่ JSON: {text[:200]!r}") from exc

        if not isinstance(data, dict):
            raise LLMUnavailable(f"โมเดลตอบ JSON ที่ไม่ใช่ object: {text[:200]!r}")

        return LLMResponse(
            data=data,
            text=text,
            latency_ms=latency_ms,
            model=self.model,
            raw={"eval_count": getattr(response, "eval_count", None)},
        )

    def chat(self, message: str, stream: bool = False) -> str:
        """คุยแบบข้อความธรรมดา (ใช้ตอน debug / ลองโมเดล ไม่ได้ใช้ใน pipeline)

        คืนเฉพาะคำตอบ ส่วน reasoning ดูได้จาก ``self.last_thinking``
        """
        if stream:
            raise ValueError("ถ้าต้องการ stream ให้เรียก stream_chat() แทน")
        print("Thinking...")
        response: ChatResponse = chat(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            options=self.options,
            stream=False,
        )

        self.last_thinking = getattr(response.message, "thinking", None) or ""
        return response.message.content

    def stream_chat(self, message: str) -> str:
        """คุยแบบ stream พิมพ์ออกหน้าจอทีละ chunk แล้วคืนข้อความเต็ม"""
        pieces = []
        for chunk in chat(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            options=self.options,
            stream=True,
            **self._free_text_think,
        ):
            piece = chunk.message.content or ""
            pieces.append(piece)
            print(piece, end="", flush=True)
        print()
        return "".join(pieces)


if __name__ == "__main__":
    engine = ChatEngine(model="qwen3:4b")
    print("ollama พร้อมใช้งาน:", engine.health())

    answer = engine.chat("ทักทายสั้น ๆ เป็นภาษาไทยหนึ่งประโยค")
    print(f"คำตอบ    : {answer}")
    # reasoning ถูกแยกออกมา ไม่ปนกับคำตอบ ถ้าบรรทัดนี้ยาวแต่คำตอบสะอาด แปลว่าทำงานถูก
    print(f"reasoning: {len(engine.last_thinking)} ตัวอักษร (แยกไว้ใน last_thinking)")
