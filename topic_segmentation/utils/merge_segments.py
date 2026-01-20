from typing import List, Dict


def merge_short_segments(
    segments: List[Dict],
    min_duration: float = 8.0,
    min_chars: int = 200
) -> List[Dict]:
    """
    Merge adjacent Whisper segments into larger semantic chunks
    before topic segmentation.

    Rules:
    - Merge while BOTH duration < min_duration AND text length < min_chars
    - Preserve timing, text, translation, romanization
    - Language is inherited from the first segment (for now)
    """

    if not segments:
        return []

    merged: List[Dict] = []
    seg_id = 0

    buffer = {
        "segment_id": seg_id,
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "language": segments[0]["language"],
        "translation": segments[0]["translation"],
        "romanized": segments[0]["romanized"],
    }

    for seg in segments[1:]:
        duration = buffer["end"] - buffer["start"]
        char_len = len(buffer["text"])

        if duration < min_duration and char_len < min_chars:
            # Merge into buffer
            buffer["end"] = seg["end"]
            buffer["text"] += " " + seg["text"]
            buffer["translation"] += " " + seg["translation"]
            buffer["romanized"] += " " + seg["romanized"]
        else:
            merged.append(buffer)
            seg_id += 1
            buffer = {
                "segment_id": seg_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "language": seg["language"],
                "translation": seg["translation"],
                "romanized": seg["romanized"],
            }

    merged.append(buffer)

    return merged
