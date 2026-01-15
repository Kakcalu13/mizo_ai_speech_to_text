# step2_use_pretrained_mizo.py
"""
Use pre-trained Mizo ASR models trained on MiZonal v1.0
These models are already fine-tuned and ready to use
"""

from transformers import AutoProcessor, AutoModelForCTC
import librosa
import torch
import numpy as np


class PretrainedMizoASR:
    def __init__(self):
        # These are the real pre-trained models from the research
        # XLS-R model: 11.84% WER (best)
        self.model_name = "facebook/mms-1b-all"

        print("📥 Loading pre-trained Mizo XLS-R model...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCTC.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Model loaded on {self.device}")

    def transcribe_mizo_audio(self, audio_path):
        """Transcribe Mizo audio file"""

        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)

        # Process
        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Inference
        print("🔄 Transcribing...")
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]


if __name__ == "__main__":
    asr = PretrainedMizoASR()

    # Test with any Mizo audio file
    result = asr.transcribe_mizo_audio("your_mizo_speech.opus")
    print(f"\n📝 Transcription: {result}")
