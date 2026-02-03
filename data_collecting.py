import sounddevice as sd
import soundfile as sf
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import os


class MizoDataCollector:
    def __init__(self, project_name="mizo_training_data"):
        self.project_dir = Path(project_name)
        self.project_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.audio_dir = self.project_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.metadata_file = self.project_dir / "metadata.csv"
        self.validation_file = self.project_dir / "validation.json"

        # Initialize CSV headers if new
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'filename', 'speaker_id', 'speaker_name',
                    'transcription_mizo', 'transcription_en',
                    'duration_sec', 'date', 'quality_ok', 'notes'
                ])
                writer.writeheader()
                print(f"✅ Created metadata file: {self.metadata_file}")

        print(f"📁 Data collection directory: {self.project_dir}/")
        print(f"📁 Audio files will be saved to: {self.audio_dir}/")

    def record_utterance(self, speaker_id, speaker_name, mizo_text,
                         english_text, quality_check=True, max_retries=2):
        """
        Record ONE Mizo sentence with quality control
        
        Args:
            speaker_id: unique speaker code (e.g., SPK001)
            speaker_name: speaker's name
            mizo_text: the Mizo sentence to record
            english_text: English translation
            quality_check: whether to ask user if quality is good
            max_retries: how many times user can re-record
        """

        print("\n" + "=" * 70)
        print(f"📝 Recording #{self._get_next_recording_number()}")
        print(f"   Mizo:   '{mizo_text}'")
        print(f"   English: '{english_text}'")
        print(f"   Speaker: {speaker_name}")
        print("=" * 70)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"\n♻️  Re-recording (attempt {attempt + 1}/{max_retries + 1})")

            print("\n⏱️  Instructions:")
            print("   1. Read the sentence CLEARLY")
            print("   2. Speak at NORMAL pace (not too fast, not too slow)")
            print("   3. Avoid background noise")
            print("\nPress ENTER when ready to record...")
            input()

            # Record 8 seconds (gives flexibility for longer sentences)
            SAMPLE_RATE = 16000
            DURATION = 8

            print("🔴 Recording... (speak now)")
            audio = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("⏹️  Recording complete!")

            # Check if audio has content (not silent)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.01:
                print("⚠️  WARNING: Recording is too quiet!")
                print("   (RMS level: {:.4f}, needs > 0.01)".format(rms))
                if attempt < max_retries:
                    print("Try again with louder voice...")
                    continue
                else:
                    print("Using quiet recording anyway...")

            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{speaker_id}_{timestamp}.wav"
            filepath = self.audio_dir / filename

            sf.write(str(filepath), audio, SAMPLE_RATE)
            actual_duration = len(audio) / SAMPLE_RATE

            print(f"✅ Saved: {filename}")
            print(f"   Duration: {actual_duration:.2f}s, RMS: {rms:.4f}")

            # Quality check
            quality_ok = True
            if quality_check:
                print("\n❓ Is the recording quality acceptable? (y/n)")
                quality_ok = input().lower().strip() == 'y'

                if not quality_ok and attempt < max_retries:
                    print("Re-recording...")
                    os.remove(filepath)  # Delete bad recording
                    continue

            # Save metadata
            with open(self.metadata_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'filename', 'speaker_id', 'speaker_name',
                    'transcription_mizo', 'transcription_en',
                    'duration_sec', 'date', 'quality_ok', 'notes'
                ])
                writer.writerow({
                    'filename': filename,
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'transcription_mizo': mizo_text,
                    'transcription_en': english_text,
                    'duration_sec': actual_duration,
                    'date': datetime.now().isoformat(),
                    'quality_ok': 'yes' if quality_ok else 'no',
                    'notes': ''
                })

            return filepath

        return None

    def _get_next_recording_number(self):
        """Count existing recordings"""
        if not self.metadata_file.exists():
            return 1
        with open(self.metadata_file, 'r') as f:
            lines = f.readlines()
        return len(lines)  # Header + data lines

    def get_statistics(self):
        """Show collection progress"""
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            print("No data collected yet")
            return None

        # Calculate stats
        total_duration = sum(float(row['duration_sec']) for row in rows)
        speakers = set(row['speaker_id'] for row in rows)
        quality_ok = sum(1 for row in rows if row['quality_ok'].lower() == 'yes')

        print("\n" + "=" * 60)
        print("📊 COLLECTION STATISTICS")
        print("=" * 60)
        print(f"Total recordings:     {len(rows)}")
        print(f"Quality recordings:   {quality_ok}/{len(rows)}")
        print(f"Unique speakers:      {len(speakers)}")
        print(f"Total duration:       {total_duration / 3600:.2f} hours")
        if len(speakers) > 0:
            print(f"Average per speaker:  {total_duration / len(speakers):.1f} seconds")
        print("=" * 60)

        return {
            'total_recordings': len(rows),
            'quality_recordings': quality_ok,
            'unique_speakers': len(speakers),
            'total_hours': total_duration / 3600
        }

    def export_for_training(self):
        """Export data in format ready for HuggingFace training"""
        if not self.metadata_file.exists():
            print("No data to export")
            return None

        with open(self.metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Create training-ready format
        train_data = {
            'audio': [],
            'text': [],
            'speaker_id': [],
        }

        for row in rows:
            if row['quality_ok'].lower() == 'yes':
                audio_path = str(self.audio_dir / row['filename'])
                train_data['audio'].append(audio_path)
                train_data['text'].append(row['transcription_mizo'])
                train_data['speaker_id'].append(row['speaker_id'])

        export_file = self.project_dir / "training_data.json"  # Save as JSON for training
        with open(export_file, 'w') as f:
            json.dump(train_data, f, indent=2)

        print(f"\n✅ Exported {len(train_data['audio'])} quality recordings")
        print(f"   Location: {export_file}")

        return export_file


# ============================================
# MIZO SENTENCES FOR COLLECTION
# ============================================

MIZO_SENTENCES = [
    ("Ka hre lo", "I don't know"),
    ("Enge i tih?", "What are you doing?")
]

# MIZO_SENTENCES = [
#     # Basic greetings
#     ("Hlo, i dam ang em?", "Hello, how are you?"),
#     ("I lum", "I am fine"),
#     ("Engvei mai lo", "Good morning"),
#     ("Engzahlai mai lo", "Good afternoon"),
#     ("Engduhchhuah mai lo", "Good evening"),
#
#     # Common phrases
#     ("Khum inngaihva che", "Excuse me"),
#     ("Zai lo leh", "Thank you"),
#     ("Hrechiang tawk", "Welcome"),
#     ("I hla mi", "I am Mizo"),
#     ("Mizoram hi pian ang che", "Mizoram is beautiful"),
#
#     # Questions
#     ("I min a lo", "What is your name?"),
#     ("Nang chu eng hma a lo?", "Where are you from?"),
#     ("Enge i tih?", "What are you doing?"),
#     ("I phone a kal em?", "Do you have a phone?"),
#     ("Nang chu teacher em?", "Are you a teacher?"),
#
#     # Answers
#     ("I min chu John a ni", "My name is John"),
#     ("I chu Aizawl a kung", "I am from Aizawl"),
#     ("I chu ofisah i tih", "I am working in an office"),
#     ("I phone a kal", "I have a phone"),
#     ("Ai, i teacher a ni", "Yes, I am a teacher"),
#
#     # Statements
#     ("Tuna kal an dah a", "They arrived yesterday"),
#     ("I zawhnak chu mahni a che", "I like studying"),
#     ("Nang chu engine pan man em?", "Are you coming tomorrow?"),
#     ("I bu entir nia an lo", "They don't understand"),
#     ("Mizo zing hi thiamthat tak a ni", "The Mizo culture is very unique"),
#
#     # More complex sentences
#     ("I pawl chu school a kal te a ni", "My group is going to school"),
#     ("Aizawl ah ziarah rawh", "Come visit Aizawl"),
#     ("Hmang zawhna chu a pawimawh tak", "Knowledge is very important"),
#     ("Mizo mi te chu hriselna zawk a nih", "Mizo people are very hospitable"),
#     ("I family chu a awm tak", "I have a big family"),
# ]

# ============================================
# MAIN COLLECTION WORKFLOW
# ============================================

if __name__ == "__main__":
    print("Default input device index:", sd.default.device[0])
    print("Default output device index:", sd.default.device[1])

    print("\nAll audio devices:")
    print(sd.query_devices())

    print("\nCurrent input device info:")
    print(sd.query_devices(sd.default.device[0], "input"))
    print("----" * 70)

    print("\n" + "=" * 70)
    print("🎯 MIZO SPEECH TRAINING DATA COLLECTION")
    print("=" * 70)
    print("\nThis tool will help you collect Mizo speech data for fine-tuning")
    print("the ASR model. High-quality data = better accuracy!\n")

    # Initialize collector
    collector = MizoDataCollector(project_name="mizo_training_data")

    print("\n📋 SPEAKER REGISTRATION")
    print("=" * 70)
    speaker_id = input("Speaker ID (e.g., SPK001, SPK002): ").strip()
    speaker_name = input("Speaker name (e.g., John, Sima): ").strip()

    print(f"\n✅ Registered: {speaker_name} ({speaker_id})")
    print(f"\n📚 Ready to collect {len(MIZO_SENTENCES)} sentences")
    print("\nTips for best results:")
    print("  • Speak clearly and naturally")
    print("  • Use normal speed (not too fast)")
    print("  • Minimize background noise")
    print("  • Re-record if you make mistakes")
    print("  • Review and approve quality\n")

    # Collect all sentences from this speaker
    recorded = 0
    skipped = 0

    for i, (sentence_mizo, sentence_en) in enumerate(MIZO_SENTENCES, 1):
        try:
            print(f"\n[{i}/{len(MIZO_SENTENCES)}]", end=" ")

            filepath = collector.record_utterance(
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                mizo_text=sentence_mizo,
                english_text=sentence_en,
                quality_check=True,
                max_retries=2
            )

            if filepath:
                recorded += 1
            else:
                skipped += 1

            # Ask if continuing
            if i < len(MIZO_SENTENCES):
                continue_recording = input(f"\nContinue to next sentence? (y/n): ").strip().lower()
                if continue_recording != 'y':
                    print("\n⏹️  Recording session ended by user")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped by user (Ctrl+C)")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            skipped += 1
            continue

    # Show final statistics
    print("\n" + "=" * 70)
    print("🎉 SESSION COMPLETE")
    print("=" * 70)
    print(f"Recorded: {recorded} sentences")
    print(f"Skipped:  {skipped} sentences")

    stats = collector.get_statistics()

    # Export for training
    if recorded > 0:
        print("\n💾 Exporting data for training...")
        collector.export_for_training()
        print("\n✅ Data ready for Step 4: Fine-tuning!")

    print(f"\n📁 Your data is saved in: {collector.project_dir}/")
