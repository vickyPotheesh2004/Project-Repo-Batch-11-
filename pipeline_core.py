import whisper
import json
import sys
from pathlib import Path

from language_adaptation.translator import translate_auto
from language_adaptation.romanizer import romanize_text


# ==============================
# CONFIG
# ==============================
WHISPER_MODEL = "base"


# ==============================
# CORE PIPELINE
# ==============================
def process_audio(audio_path: str) -> dict:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1️⃣ Load Whisper ASR
    model = whisper.load_model(WHISPER_MODEL)

    result = model.transcribe(
        str(audio_path),
        fp16=False,
        verbose=False
    )

    # Global fallback language (Whisper-level)
    detected_lang = result.get("language", "en")

    segments_out = []

    # 2️⃣ Process segments (PER-SEGMENT LANGUAGE SAFE)
    for idx, seg in enumerate(result.get("segments", [])):
        text = seg["text"].strip()

        # 🔥 IMPORTANT: per-segment language if available
        segment_lang = seg.get("language", detected_lang)

        # Translation logic (safe)
        if segment_lang != "en":
            translation = translate_auto(text, segment_lang, "en")
        else:
            translation = text

        # Romanization (language-aware)
        romanized = romanize_text(text, segment_lang)

        segments_out.append({
            "segment_id": idx,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
            "language": segment_lang,
            "translation": translation,
            "romanized": romanized
        })

    # 3️⃣ Final output (LOCKED CONTRACT)
    output = {
        "audio_file": audio_path.name,
        "language_detected": detected_lang,
        "segments": segments_out
    }

    return output


# ==============================
# CLI ENTRY
# ==============================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline_core.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    data = process_audio(audio_file)

    # Preserve existing behavior
    with open("pipeline_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Pipeline output saved to pipeline_output.json")
