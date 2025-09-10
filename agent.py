# ---------- Imports ----------
import logging
import numpy as np
import asyncio
import tempfile
import io
from dotenv import load_dotenv
from collections import namedtuple
from faster_whisper import WhisperModel
import edge_tts

from livekit.agents import (
    Agent, AgentSession, JobContext, JobProcess,
    WorkerOptions, cli, metrics, RoomInputOptions
)
from livekit.agents.stt import SpeechEvent
from livekit.plugins import openai, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
#from livekit.agents.utils.audio import encode_wav

# ---------- Load .env ----------
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# ---------- STT: FasterWhisper ----------
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
        return Capabilities()

    async def recognize(self, audio: np.ndarray = None, sample_rate: int = None, **kwargs):
        if audio is None and "buffer" in kwargs:
            frame = kwargs["buffer"]
            audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = getattr(frame, "sample_rate", 16000)

        elif isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        if sample_rate is None:
            sample_rate = 16000

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
            print("[STT] ERROR during transcription:", str(e))
            return SpeechEvent(
                type="final",
                alternatives=[Alternative(text="", language="en", speaker_id=None, confidence=0.0)]
            )

# ---------- TTS: EdgeTTS ----------
# ---------- TTS: EdgeTTS ----------
class EdgeTTSWrapper:
    def __init__(self, voice="en-US-JennyNeural"):
        self.voice = voice
        print("[TTS] EdgeTTS loaded with voice:", voice)

    @property
    def capabilities(self):
        class Capabilities:
            streaming = False
        return Capabilities()

    async def synthesize(self, text: str, language: str = "en-US", sample_rate: int = 24000) -> bytes:
        try:
            print(f"[TTS] Synthesizing: {text}")
            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            mp3_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])
            mp3_buffer.seek(0)

            # Encode mp3 to wav using ffmpeg-compatible encoder
            from livekit.agents.utils.audio import encode_wav
            wav_bytes = await encode_wav(mp3_buffer.read(), input_format="mp3", sample_rate=sample_rate)
            print("[TTS] Synthesis complete.")
            return wav_bytes
        except Exception as e:
            print("[TTS] ERROR:", e)
            return b""


# ---------- Prompt ----------
EllePrompt = (
    "You are a voice assistant created by Elle. Your name is Elle. "
    "You speak clearly, with warmth and curiosity. "
    "Always keep your answers short and conversational."
)

# ---------- Agent ----------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=EllePrompt,
            stt=FasterWhisperSTT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=EdgeTTSWrapper(),
            turn_detection=MultilingualModel(),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Hi, what's up?", allow_interruptions=False
        )

# ---------- Prewarm ----------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ---------- Entrypoint ----------
async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

# ---------- Main ----------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
