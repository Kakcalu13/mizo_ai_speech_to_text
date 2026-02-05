"""
Step 4: Fine-tune MMS-1B model on your Mizo speech data
This trains the model to recognize YOUR community's Mizo speech better
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
import librosa
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ============================================
# DATA PREPARATION
# ============================================

def load_mizo_dataset(json_file, split_ratio=0.9):
    """Load training data from JSON and split into train/eval"""

    print(f"📂 Loading data from {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Verify all audio files exist
    valid_indices = []
    for i, audio_path in enumerate(data['audio']):
        if Path(audio_path).exists():
            valid_indices.append(i)
        else:
            print(f"⚠️  Warning: {audio_path} not found, skipping...")

    # Filter to valid files only
    filtered_data = {
        'audio': [data['audio'][i] for i in valid_indices],
        'text': [data['text'][i] for i in valid_indices],
    }

    print(f"✅ Loaded {len(filtered_data['audio'])} valid audio files")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(filtered_data)

    # Cast audio column to Audio feature (handles loading automatically)
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split into train/validation
    if len(dataset) < 10:
        print(f"⚠️  Warning: Only {len(dataset)} samples - using 80/20 split")
        split_ratio = 0.8

    split = dataset.train_test_split(test_size=1 - split_ratio, seed=42)

    print(f"   Training samples: {len(split['train'])}")
    print(f"   Validation samples: {len(split['test'])}")

    return split['train'], split['test']


# ============================================
# DATA COLLATOR
# ============================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = []
        label_features = []

        for feature in features:
            audio, sr = librosa.load(feature["audio"], sr=16000)
            input_features.append({"input_values": audio})
            label_features.append({"input_ids": feature["labels"]})

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


# ============================================
# METRICS
# ============================================

def compute_metrics(pred, processor):
    """Calculate WER (Word Error Rate)"""
    from jiwer import wer

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad token
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Calculate WER
    try:
        wer_score = wer(label_str, pred_str)
    except:
        # If WER calculation fails (e.g., empty strings), return 1.0
        wer_score = 1.0

    return {"wer": wer_score}


# ============================================
# TRAINING
# ============================================

class MizoASRTrainer:
    """Fine-tune MMS model on Mizo data"""

    def __init__(self, output_dir="./mizo_asr_finetuned"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("📥 Loading pre-trained MMS-1B model...")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained('facebook/mms-1b-all')
        self.model = AutoModelForCTC.from_pretrained('facebook/mms-1b-all')

        # # Set to Mizo
        # self.processor.tokenizer.set_target_lang("miz")
        # self.model.load_adapter("miz")

        # Freeze feature encoder (only train adapter + LM head)
        self.model.freeze_feature_encoder()

        self.device = torch.device("cpu")
        self.model.to(self.device)

        print(f"✅ Model loaded on {self.device}")
        print(f"   Architecture: {self.model.config.model_type}")
        print(f"   Total parameters: {self.model.num_parameters() / 1e6:.1f}M")
        print(
            f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.1f}M")

    def prepare_dataset(self, dataset):
        """Prepare dataset by tokenizing text"""

        def prepare_example(batch):
            labels = self.processor(
                text=batch["text"],
                return_attention_mask=False
            ).input_ids
            return {"labels": labels}

        # Map adds "labels", then we manually remove only "text"
        dataset = dataset.map(
            prepare_example,
            desc="Tokenizing text"
        )

        # Now remove text column explicitly
        dataset = dataset.remove_columns(["text"])

        print(f"DEBUG: Dataset columns after map: {dataset.column_names}")
        print("DEBUG columns:", dataset.column_names)
        return dataset

    def train(self, train_dataset, eval_dataset, num_epochs=30, batch_size=4):
        """Fine-tune the model"""

        print("\n" + "=" * 70)
        print("🚀 STARTING FINE-TUNING")
        print("=" * 70)

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)
        eval_dataset = self.prepare_dataset(eval_dataset)

        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            group_by_length=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            eval_strategy="epoch",  # Changed from evaluation_strategy
            num_train_epochs=num_epochs,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=1e-4,
            warmup_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=lambda pred: compute_metrics(pred, self.processor),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train
        print("\n⏳ Training started...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: 1e-4")
        print(f"   Total steps: ~{len(train_dataset) // batch_size * num_epochs}")
        print(f"   Device: {self.device}")

        trainer.train()

        # Save final model
        print("\n💾 Saving final model...")
        final_dir = self.output_dir / "final_model"
        self.model.save_pretrained(str(final_dir))
        self.processor.save_pretrained(str(final_dir))

        print(f"✅ Model saved to: {final_dir}")
        print(f"\nTo use your fine-tuned model:")
        print(f"   processor = AutoProcessor.from_pretrained('{final_dir}')")
        print(f"   model = AutoModelForCTC.from_pretrained('{final_dir}')")

        return trainer


# ============================================
# MAIN WORKFLOW
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎓 MIZO ASR FINE-TUNING")
    print("=" * 70)

    # Check for training data
    data_file = Path("mizo_training_data/training_data.json")

    if not data_file.exists():
        print(f"❌ Error: {data_file} not found!")
        print("   Please run step3_data_collection.py first to collect data")
        exit(1)

    # Load data
    train_dataset, eval_dataset = load_mizo_dataset(data_file)

    num_samples = len(train_dataset) + len(eval_dataset)
    if num_samples < 10:
        print(f"⚠️  Warning: Only {num_samples} samples found")
        print("   Recommended minimum: 50-100 samples")
        print("   Training will proceed but accuracy may be limited")

    # Initialize trainer
    trainer = MizoASRTrainer(output_dir="mizo_asr_finetuned")

    # Adjust epochs based on data size
    if num_samples < 50:
        num_epochs = 50  # More epochs for small datasets
    elif num_samples < 100:
        num_epochs = 30
    else:
        num_epochs = 20  # Fewer epochs for larger datasets

    print(f"\n📊 Training configuration:")
    print(f"   Dataset size: {num_samples} samples")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 4 (adjust if out of memory)")

    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        batch_size=1  # Reduce to 2 if out of memory
    )

    print("\n🎉 FINE-TUNING COMPLETE!")
    print("\nNext steps:")
    print("1. Test your model: python3 test_finetuned.py")
    print("2. Compare before/after accuracy")
    print("3. Collect more data if needed")
    print("4. Deploy to Android/Godot")
