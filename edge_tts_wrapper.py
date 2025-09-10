# edge_tts_wrapper.py

import io
from pydub import AudioSegment
import edge_tts

class EdgeTTSWrapper:
    def on(self, *args, **kwargs):
        # Dummy event handler registration for compatibility
        return lambda *a, **k: None
    @property
    def num_channels(self):
        return 1
    def __init__(self, voice="en-US-JennyNeural"):
        self.voice = voice
        print("[TTS] EdgeTTS loaded with voice:", voice)

    @property
    def capabilities(self):
        class Capabilities:
            streaming = False
            aligned_transcript = False
        return Capabilities()

    @property
    def sample_rate(self):
        return 24000

    class _SynthContext:
        def __init__(self, audio_bytes):
            self.audio_bytes = audio_bytes
            self.frame = audio_bytes  # For compatibility with expected interface
        async def __aenter__(self, *args, **kwargs):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass

    async def synthesize(self, text: str, language: str = "en", sample_rate: int = 24000, conn_options=None, **kwargs):
        try:
            print(f"[TTS] Synthesizing: {text}")
            communicate = edge_tts.Communicate(text=text, voice=self.voice)

            mp3_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.write(chunk["data"])
            mp3_data.seek(0)

            mp3_audio = AudioSegment.from_file(mp3_data, format="mp3")
            wav_data = io.BytesIO()
            mp3_audio.set_frame_rate(sample_rate).export(wav_data, format="wav")
            print("[TTS] Synthesis complete.")
            audio_bytes = wav_data.getvalue()
            return self._SynthContext(audio_bytes)

        except Exception as e:
            print("[TTS] ERROR:", e)
            return self._SynthContext(b"")
