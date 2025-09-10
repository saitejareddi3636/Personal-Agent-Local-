# faster_whisper_stt.py

import numpy as np
import asyncio
from collections import namedtuple
from faster_whisper import WhisperModel
from livekit.agents.stt import SpeechEvent

Alternative = namedtuple("Alternative", ["text", "language", "speaker_id", "confidence"])

class FasterWhisperSTT:
    def __init__(self):
        print("FasterWhisperSTT instance created:", self)
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("[STT] FasterWhisper model loaded")

    @property
    def label(self):
        return "FasterWhisper"

    def __call__(self, *args, **kwargs):
        return self

    def on(self, *args, **kwargs):
        def dummy(*a, **k): pass
        return dummy

    @property
    def capabilities(self):
        class Capabilities:
            streaming = False
            aligned_transcript = False
        return Capabilities()

    async def recognize(self, audio: np.ndarray = None, sample_rate: int = None, **kwargs):
        # Handle LiveKit buffer
        if audio is None and "buffer" in kwargs:
            frame = kwargs["buffer"]
            try:
                if hasattr(frame, "data"):
                    audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                    sample_rate = getattr(frame, "sample_rate", 16000)
                else:
                    print("[STT] Buffer missing data attribute")
            except Exception as e:
                print("[STT] ERROR processing buffer:", str(e))

        # Handle raw bytes
        elif isinstance(audio, bytes):
            try:
                audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                print("[STT] ERROR converting bytes:", str(e))
                audio = None

        # Handle file-like object
        elif hasattr(audio, "read") and callable(audio.read):
            try:
                audio_bytes = audio.read()
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                print("[STT] ERROR reading file-like object:", str(e))
                audio = None

        if sample_rate is None:
            sample_rate = 16000

        if audio is None:
            print("[STT] ERROR: No valid audio data to transcribe.")
            return SpeechEvent(
                type="final",
                alternatives=[Alternative(text="", language="en", speaker_id=None, confidence=0.0)]
            )

        try:
            loop = asyncio.get_running_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe((audio, sample_rate), language="en", beam_size=5)
            )
            full_text = " ".join([seg.text for seg in segments])
            print("[STT] Transcription complete:", full_text)
            return SpeechEvent(
                type="final",
                alternatives=[Alternative(text=full_text, language="en", speaker_id=None, confidence=1.0)]
            )
        except Exception as e:
            print("[STT] ERROR during transcription", str(e))
            return SpeechEvent(
                type="final",
                alternatives=[Alternative(text="", language="en", speaker_id=None, confidence=0.0)]
            )
