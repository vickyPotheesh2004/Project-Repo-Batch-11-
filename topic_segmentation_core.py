import json
import sys
from pathlib import Path

# 🔹 Algorithm (Hybrid is default & preferred)
from topic_segmentation.algorithms.segment_hybrid_engine import segment

# 🔹 Pre-merge utility (CRITICAL FIX)
from topic_segmentation.utils.merge_segments import merge_short_segments


def process_segmentation(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if "segments" not in data:
        raise ValueError("Input JSON does not contain 'segments'")

    # 🔒 STEP 1 — Pre-merge short Whisper segments
    merged_segments = merge_short_segments(
        data["segments"],
        min_duration=8.0,
        min_chars=200
    )

    # 🧠 STEP 2 — Topic segmentation
    topics = segment(merged_segments)

    # 📦 Output contract
    output = {
        "audio_file": data["audio_file"],
        "language_detected": data["language_detected"],
        "topics": topics
    }

    return output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python topic_segmentation_core.py <pipeline_output.json>")
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()
    result = process_segmentation(input_file)

    output_path = Path("segmented_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Segmented output saved to {output_path}")
