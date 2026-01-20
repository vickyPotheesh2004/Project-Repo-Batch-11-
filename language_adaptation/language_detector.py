from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"
