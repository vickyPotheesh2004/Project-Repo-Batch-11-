import os
import librosa
import soundfile as sf

RAW_AUDIO_DIR = "data/raw_audio"
PROCESSED_AUDIO_DIR = "data/processed_audio"
TARGET_SAMPLE_RATE = 16000


def preprocess_audio(filename):
    input_path = os.path.join(RAW_AUDIO_DIR, filename)
    output_path = os.path.join(
        PROCESSED_AUDIO_DIR,
        os.path.splitext(filename)[0] + ".wav"
    )

    print(f"🔊 Loading audio: {filename}")

    # Load audio (auto converts to mono)
    audio, sr = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Save processed audio
    sf.write(output_path, audio, TARGET_SAMPLE_RATE)

    print(f"✅ Saved cleaned audio to: {output_path}")


if __name__ == "__main__":
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

    for file in os.listdir(RAW_AUDIO_DIR):
        if file.endswith((".mp3", ".wav")):
            preprocess_audio(file)
