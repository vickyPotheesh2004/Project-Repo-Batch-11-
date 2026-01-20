import whisper
import os

model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio and returns plain text (NOT JSON)
    """
    result = model.transcribe(audio_path)

    # Whisper always returns a dict
    if isinstance(result, dict) and "text" in result:
        return result["text"].strip()

    return ""
