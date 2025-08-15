# server.py
import os
import asyncio
import threading
import json
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Optional ngrok support
try:
    from pyngrok import ngrok, conf as ngrok_conf
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False

# ----- Pedalboard audio processing -----
from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)
import soundfile as sf

# --- Fixed voice prompt ---
VOICE_PROMPT_PATH = os.environ.get(
    "VOICE_PROMPT_PATH",
    os.path.join(os.path.dirname(__file__), "voice.wav")
)
if not os.path.isfile(VOICE_PROMPT_PATH):
    raise FileNotFoundError(f"Voice prompt not found: {VOICE_PROMPT_PATH}")

# --- Effects chain ---
board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=100),
    LowShelfFilter(gain_db=-2.0, cutoff_frequency_hz=200),
    HighShelfFilter(gain_db=3.0, cutoff_frequency_hz=8000),
    Compressor(threshold_db=-18, ratio=3, attack_ms=15, release_ms=100),
    PitchShift(semitones=-1),
    Chorus(rate_hz=1.5, depth=0.2, centre_delay_ms=15, feedback=0.1),
    Delay(delay_seconds=0.035, mix=0.15),
    Reverb(room_size=0.15, damping=0.5, wet_level=0.15, dry_level=0.85),
    Distortion(drive_db=5.0)
])

def add_background_hum(audio, sr, hum_file="server_hum.wav", mix=0.08):
    hum, hum_sr = sf.read(hum_file)
    if hum_sr != sr:
        raise ValueError("Hum sample rate must match voice sample rate")
    if hum.ndim == 1:
        hum = np.expand_dims(hum, axis=0)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    if hum.shape[0] != audio.shape[0]:
        if hum.shape[0] == 1:
            hum = np.repeat(hum, audio.shape[0], axis=0)
        else:
            raise ValueError("Hum and audio must have same number of channels")
    hum_len = hum.shape[1]
    audio_len = audio.shape[1]
    if hum_len < audio_len:
        repeats = int(np.ceil(audio_len / hum_len))
        hum = np.tile(hum, (1, repeats))
    hum = hum[:, :audio_len]
    return audio * (1 - mix) + hum * mix

def add_monster_layer(audio, sr, semitones=-9, mix=0.25):
    monster = PitchShift(semitones=semitones)(audio, sr)
    monster = HighpassFilter(cutoff_frequency_hz=50)(monster, sr)
    monster = LowShelfFilter(gain_db=6.0, cutoff_frequency_hz=150)(monster, sr)
    return (audio * (1 - mix)) + (monster * mix)

def process_audio_chunk(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.expand_dims(audio, 0)
    effected = board(audio, sr)
    # Uncomment to add hum or monster undertone
    # effected = add_background_hum(effected, sr, hum_file="server_hum.wav", mix=0.05)
    # effected = add_monster_layer(effected, sr, semitones=-3, mix=0.3)
    # Bitcrush for edge
    bit_depth = 12
    max_val = np.max(np.abs(effected)) or 1.0
    effected = np.round(effected / max_val * (2**(bit_depth - 1))) / (2**(bit_depth - 1)) * max_val
    return effected.astype(np.float32)

def interleave_channels(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    return arr.T.reshape(-1)

# ----- Load TTS model -----
import torch
from chatterbox.tts import ChatterboxTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
try:
    model.t3 = torch.compile(model.t3, fullgraph=True, mode="reduce-overhead")
except Exception:
    pass
sr = getattr(model, "sr", 48000)

# ----- FastAPI app -----
app = FastAPI()

@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    try:
        init_msg = await ws.receive_text()
        try:
            req = json.loads(init_msg)
        except Exception:
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            await ws.close()
            return

        text = req.get("text", "")
        chunk_size = int(req.get("chunk_size", 100))

        if not text:
            await ws.send_json({"type": "error", "message": "No text provided"})
            await ws.close()
            return

        # Send meta
        await ws.send_json({"type": "meta", "sr": sr, "channels": 1, "voice_prompt": os.path.basename(VOICE_PROMPT_PATH)})

        # Produce and send chunks
        def produce_chunks():
            try:
                for chunk, metrics in model.generate_stream(
                    text, audio_prompt_path=VOICE_PROMPT_PATH,
                    chunk_size=chunk_size, exaggeration=0.7, cfg_weight=0.3
                ):
                    effected = process_audio_chunk(chunk, sr)
                    frames = effected.shape[1]
                    interleaved = interleave_channels(effected)
                    # send chunk info
                    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "chunk", "frames": frames}), loop).result()
                    asyncio.run_coroutine_threadsafe(ws.send_bytes(interleaved.tobytes()), loop).result()
                asyncio.run_coroutine_threadsafe(ws.send_json({"type": "end"}), loop).result()
            except Exception as e:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "error", "message": str(e)}), loop).result()
                except Exception:
                    pass

        thr = threading.Thread(target=produce_chunks, daemon=True)
        thr.start()

        # Keep socket alive
        while thr.is_alive():
            try:
                msg = await ws.receive_text()
                try:
                    js = json.loads(msg)
                    if js.get("cmd") == "stop":
                        await ws.send_json({"type": "info", "message": "Stop requested"})
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            except WebSocketDisconnect:
                break
        await ws.close()
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WebSocket error:", e)
        try:
            await ws.close()
        except Exception:
            pass

# ----- Main -----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    token = "2aoU5n18n0HAlIEzBmb7OzVArLv_5X9adLmyQydLV4wvgPXvh"
    if token:
        ngrok_conf.get_default().auth_token = token
    try:
        public_url = ngrok.connect(port, "http").public_url
        print("ngrok:", public_url)
        print("WebSocket:", public_url.replace("http", "ws") + "/ws/tts")
    except Exception as e:
        print("ngrok failed:", e)

uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
