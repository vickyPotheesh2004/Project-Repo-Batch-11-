# language_adaptation/translator.py

from transformers import MarianMTModel, MarianTokenizer
import torch

# -------------------------------
# Supported translation models
# -------------------------------

MODEL_MAP = {
    ("te", "en"): "Helsinki-NLP/opus-mt-te-en",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    ("kn", "en"): "Helsinki-NLP/opus-mt-kn-en",
    ("ml", "en"): "Helsinki-NLP/opus-mt-ml-en",
    ("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("ur", "en"): "Helsinki-NLP/opus-mt-ur-en",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
}

# Cache loaded models
_LOADED_MODELS = {}

# -------------------------------
# Language normalization
# -------------------------------

def normalize_lang(lang: str) -> str:
    """Normalize language codes coming from Whisper / detectors"""
    if not lang:
        return "en"

    lang = lang.lower()

    if lang.startswith("zh"):
        return "zh"

    return lang


# -------------------------------
# Translation API (STABLE)
# -------------------------------

def translate_auto(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source_lang -> target_lang.
    Safe fallback: returns original text if unsupported.
    """

    if not text or source_lang == target_lang:
        return text

    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)

    key = (source_lang, target_lang)

    if key not in MODEL_MAP:
        # Unsupported language pair → no translation
        return text

    # Lazy load model
    if key not in _LOADED_MODELS:
        tokenizer = MarianTokenizer.from_pretrained(MODEL_MAP[key])
        model = MarianMTModel.from_pretrained(MODEL_MAP[key])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        _LOADED_MODELS[key] = (tokenizer, model, device)

    tokenizer, model, device = _LOADED_MODELS[key]

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_length=512
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception:
        # Absolute safety: never crash pipeline
        return text
