import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

class ASR_Whisper:
    def __init__(self):

        self.task_name = "automatic-speech-recognition"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self, model_id: str, language: str = "thai", task: str = "transcribe",
                   return_timestamps: bool = False):
        """Load the Whisper model and build the transcription pipeline.

        Decoding options (language, task, timestamps) are set on the model's
        generation config instead of being passed to `pipeline()` or to
        `generate()` as separate keyword arguments: the pipeline always hands
        `generate()` a generation config, and transformers deprecates mixing a
        generation config with generation-related arguments.

        `return_timestamps=True` adds a "chunks" entry (text + timestamp per
        segment) to the result, but the pipeline then passes `return_timestamps`
        to `generate()` itself, which brings the deprecation warning back.
        """

        print(f"Loading model ....")
        print(f"   Device: {self.device.upper()}")
        print(f"   Model: {model_id}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True)

        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model.generation_config.language = language
        self.model.generation_config.task = task
        self.model.generation_config.return_timestamps = return_timestamps
        # self.model.generation_config.max_new_tokens = 448

        # The pipeline copies the generation config from the model, so the
        # settings above have to be in place before this call.
        self.pipe = pipeline(
                self.task_name,
                model= self.model,
                tokenizer= self.processor.tokenizer,
                feature_extractor= self.processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                dtype= self.dtype,
                device= self.device,
                )

    def transcribe(self, data: np.ndarray):
        pass

    def transcrive_file(self, file_path: str) -> dict:
        result = self.pipe(file_path)
        print(result["text"])

        return result
