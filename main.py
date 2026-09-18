from asr import ASR_Whisper
import time
def main():
    pass

if __name__ == "__main__":
    # main()

    file_path = "ASR/Test/test_1.wav"
    model_id = "typhoon-ai/typhoon-whisper-turbo"
    
    asr = ASR_Whisper()
    asr.load_model(model_id)

    while(1):
        file_path = input("path: ")

        start_time = time.time()
        result = asr.transcrive_file(file_path)
        end_time = time.time()
        print(f"elapsed time {end_time-start_time} seconds")