# --- START OF FILE main.py ---

import io
import logging
import os
import wave
import asyncio
import struct
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

if DEBUG_MODE:
    log.info("--- Debug ---")
# ----------------------------------------------------------------

# --- Config ---
WYOMING_HOST = os.getenv("WYOMING_HOST", "127.0.0.1")
WYOMING_PORT = int(os.getenv("WYOMING_PORT", "10205"))

app = FastAPI(
    title="wyoming-openai-tts-gateway",
    description="A gateway that exposes local Wyoming TTS services via an OpenAI-compatible API.",
    version="2.2.0 (production)",
)

def create_wav_header(sample_rate: int, bits_per_sample: int, channels: int) -> bytes:
    chunk_size = 0xFFFFFFFF
    final_data_size = 0xFFFFFFFF
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sL4s4sLHHLLHH4sL",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", final_data_size,
    )

class OpenAISpeechRequest(BaseModel):
    input: str
    model: str
    voice: str
    response_format: Optional[str] = Field("wav", alias="response_format")
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False

# --- Endpoints ---

@app.get("/v1/voices")
async def list_voices():
    log.info("Request for available voices")
    try:
        async with AsyncTcpClient(WYOMING_HOST, WYOMING_PORT) as client:
            await client.write_event(Describe().event())
            event = await client.read_event()
            if event is None or not Info.is_type(event.type): raise HTTPException(status_code=500, detail="Server did not return info")
            info = Info.from_event(event)
            all_voices = [
                {"id": voice.name, "name": voice.description or voice.name, "languages": voice.languages}
                for tts_program in info.tts if tts_program.installed and tts_program.voices
                for voice in tts_program.voices
            ]
            if not all_voices: raise HTTPException(status_code=404, detail="No voices found")
            return {"voices": all_voices}
    except Exception as e:
        log.error(f"Error getting voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech", response_class=Response)
async def openai_text_to_speech(request: OpenAISpeechRequest):
    is_streaming = request.stream or False
    if is_streaming:
        log.info(f"Serving STREAMING request for voice '{request.voice}'")
        return StreamingResponse(generate_streaming_audio(request), media_type="audio/wav", headers={"Connection": "close"})
    else:
        log.info(f"Serving NON-STREAMING request for voice '{request.voice}'")
        return await generate_complete_audio(request)

async def generate_complete_audio(request: OpenAISpeechRequest) -> Response:
    audio_params = None
    audio_chunks = []
    try:
        async with AsyncTcpClient(WYOMING_HOST, WYOMING_PORT) as client:
            voice_opts = SynthesizeVoice(name=request.voice.strip())
            synthesize_event = Synthesize(text=request.input, voice=voice_opts)
            if request.speed is not None: synthesize_event.speech_rate = request.speed
            await client.write_event(synthesize_event.event())
            while event := await client.read_event():
                if AudioStart.is_type(event.type): audio_params = AudioStart.from_event(event)
                elif AudioChunk.is_type(event.type): audio_chunks.append(AudioChunk.from_event(event).audio)
                elif AudioStop.is_type(event.type):
                    await asyncio.sleep(0.01)
                    break
        if not audio_params or not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio received")
        complete_audio_data = b"".join(audio_chunks)
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_writer:
                wav_writer.setframerate(audio_params.rate)
                wav_writer.setsampwidth(audio_params.width)
                wav_writer.setnchannels(audio_params.channels)
                wav_writer.writeframes(complete_audio_data)
            wav_bytes = wav_io.getvalue()
    except Exception as e:
        log.error(f"Error during complete audio generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    log.info(f"Generated {len(wav_bytes) / 1024:.1f} KB of WAV audio.")
    return Response(content=wav_bytes, media_type="audio/wav")

async def generate_streaming_audio(request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
    try:
        async with AsyncTcpClient(WYOMING_HOST, WYOMING_PORT) as client:
            voice_opts = SynthesizeVoice(name=request.voice.strip())
            synthesize_event = Synthesize(text=request.input, voice=voice_opts)
            if request.speed is not None: synthesize_event.speech_rate = request.speed
            await client.write_event(synthesize_event.event())
            wav_header_sent = False
            while event := await client.read_event():
                if not wav_header_sent and AudioStart.is_type(event.type):
                    audio_start = AudioStart.from_event(event)
                    bits_per_sample = audio_start.width * 8
                    header = create_wav_header(
                        sample_rate=audio_start.rate,
                        bits_per_sample=bits_per_sample,
                        channels=audio_start.channels,
                    )
                    yield header
                    wav_header_sent = True
                    log.debug("Streaming: Robust WAV header sent.")
                elif wav_header_sent and AudioChunk.is_type(event.type):
                    yield AudioChunk.from_event(event).audio
                elif AudioStop.is_type(event.type):
                    log.debug("Streaming: 'AudioStop' received. Finishing.")
                    await asyncio.sleep(0.01)
                    break
    except Exception as e:
        log.error(f"Error in streaming generator: {e}", exc_info=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8555, log_config=None)
