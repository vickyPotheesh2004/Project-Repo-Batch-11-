import streamlit as st
import tempfile
import os

from asr.transcribe import transcribe_audio
from language_adaptation.translator import translate_auto
from language_adaptation.romanized_tone_adapter import romanized_tone

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Language Adaptation System",
    page_icon="🌍",
    layout="centered",
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
st.session_state.setdefault("transcript", None)
st.session_state.setdefault("translated", None)
st.session_state.setdefault("romanized", None)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌍 AI Language Adaptation System")
st.caption("Podcast Translation + Local Romanized Tone")

st.divider()

# --------------------------------------------------
# Upload Audio
# --------------------------------------------------
st.subheader("🎧 Upload Podcast Audio")

audio_file = st.file_uploader(
    "Upload podcast (mp3 / wav)",
    type=["mp3", "wav"],
)

LANGUAGES = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Marathi": "mr",
    "Urdu": "ur",
    "Arabic": "ar",
    "Russian": "ru",
    "Chinese": "zh-cn",
}

target_lang_name = st.selectbox(
    "Select Target Language",
    list(LANGUAGES.keys()),
)

target_lang = LANGUAGES[target_lang_name]

st.divider()

# --------------------------------------------------
# STEP 1 — TRANSCRIBE
# --------------------------------------------------
if audio_file and st.button("🎙️ Transcribe Podcast"):
    with st.spinner("Transcribing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        st.session_state.transcript = transcribe_audio(tmp_path)
        os.remove(tmp_path)

        st.session_state.translated = None
        st.session_state.romanized = None

# --------------------------------------------------
# Show Transcription
# --------------------------------------------------
if st.session_state.transcript:
    st.subheader("🎙️ Transcribed Text")

    st.text_area(
        "Transcribed Text",
        value=st.session_state.transcript,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    st.divider()

    # --------------------------------------------------
    # STEP 2 — TRANSLATE & ADAPT
    # --------------------------------------------------
    if st.button("🌐 Translate & Adapt"):
        with st.spinner("Translating and adapting..."):
            translated_text = translate_auto(
                st.session_state.transcript,
                target_lang,
            )

            romanized_text = romanized_tone(
                translated_text,
                target_lang,   # ✅ FIXED
            )

            st.session_state.translated = translated_text
            st.session_state.romanized = romanized_text

# --------------------------------------------------
# Show Results
# --------------------------------------------------
if st.session_state.translated:
    st.subheader("✅ Translated (Native Script)")
    st.text_area(
        "Translated Output",
        value=st.session_state.translated,
        height=150,
        disabled=True,
        label_visibility="collapsed",
    )

if st.session_state.romanized:
    st.subheader("🗣️ Romanized Local Tone")
    st.text_area(
        "Romanized Output",
        value=st.session_state.romanized,
        height=120,
        disabled=True,
        label_visibility="collapsed",
    )
