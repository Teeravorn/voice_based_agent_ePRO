import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

class ASR_Whisper:
    def __init__(self):

        self.task_name = "automatic-speech-recognition"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.mode = None
        self.processor = None

    def load_model(self, model_id: str):

        print(f"Loading model ....")
        print(f"   Device: {self.device.upper()}")
        print(f"   Model: {model_id}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True)

        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
                self.task_name,
                model= self.model,
                tokenizer= self.processor.tokenizer,
                feature_extractor= self.processor.feature_extractor,
                max_new_tokens=444,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                dtype= self.dtype,
                device= self.device,
                )

    def transcribe(self, data: np.ndarray):
        pass

    def transcrive_file(self, file_path: str) -> dict:
        result = self.pipe(file_path, generate_kwargs={"language": "thai"})
        print(result["text"])

        return result