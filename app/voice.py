"""
Voice transcription (requirement #7, optional).

Uses faster-whisper, a local/offline Whisper implementation:
  - No external API key needed for transcription itself.
  - Supports Arabic and English out of the box (Whisper is multilingual).
  - Works fine in Google Colab (CPU is enough for the 'small' model on short clips;
    a Colab GPU runtime makes it faster).

The model is loaded lazily on first use so importing this module has no cost
if voice input isn't used.
"""
from typing import Optional

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # "small" is a good accuracy/speed tradeoff for AR+EN on CPU.
        # Use "base" for faster/lower-accuracy, "medium" for higher accuracy if you have a GPU.
        _model = WhisperModel("small", device="auto", compute_type="int8")
    return _model


def transcribe_audio(file_path: str, language_hint: Optional[str] = None) -> str:
    """
    Transcribes an audio file to text.
    language_hint: 'ar', 'en', or None to let Whisper auto-detect.
    """
    model = _get_model()
    segments, info = model.transcribe(
        file_path,
        language=language_hint,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()
