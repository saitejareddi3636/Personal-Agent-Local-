<<<<<<< HEAD
# Personal-Agent-Local-
Will be updated soon
=======
# LiveKit Voice Agent (Local)

## Overview
This project is a Python-based voice agent that integrates LiveKit for real-time audio streaming, speech-to-text (STT), and text-to-speech (TTS) capabilities. The agent is designed to run locally, enabling natural language interaction over voice using custom STT and TTS wrappers.

## Features
- **LiveKit Integration:** Real-time audio streaming and session management.
- **Speech-to-Text (STT):** Uses FasterWhisper for fast, accurate transcription.
- **Text-to-Speech (TTS):** Uses EdgeTTS for natural-sounding voice responses. Planned support for F5 TTS (for enhanced quality and flexibility).
- **Async & Streaming:** Fully async context management and streaming audio frames.
- **Local Execution:** No cloud dependencies; runs entirely on your machine.

## Project Structure
- `agent.py`: Main agent logic, session management, STT/TTS integration.
- `faster_whisper_stt.py`: Custom wrapper for FasterWhisper STT.
- `edge_tts_wrapper.py`: Custom wrapper for EdgeTTS TTS.

## Getting Started
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd my_agent
   ```
2. **Set up Python environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Add your LiveKit credentials and settings to `.env.local`.
4. **Run the agent:**
   ```sh
   python agent.py start
   ```

## Usage
- Start the agent and connect via LiveKit for voice interaction.
- The agent will transcribe incoming audio and respond using TTS.

## Contributing
Pull requests and issues are welcome! Please ensure code is well-documented and tested.

## License
MIT License
>>>>>>> 4081dff (Initial commit without secrets)
