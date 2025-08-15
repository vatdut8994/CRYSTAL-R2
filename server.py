# server.py
import asyncio
import threading
import json
import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# If you want ngrok automatic tunnel
try:
    from pyngrok import ngrok, conf as ngrok_conf
    PYNGROK_AVAILABLE = True
except Exception:
    PYNGROK_AVAILABLE = False

# Your original imports / audio processing (kept mostly intact)
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)

# ----- Your audio processing definitions (copied / minimally adapted) -----

def add_monster_layer(audio, sr, semitones=-9, mix=0.25):
    monster = PitchShift(semitones=semitones)(audio, sr)
    monster = HighpassFilter(cutoff_frequency_hz=50)(monster, sr)
    monster = LowShelfFilter(gain_db=6.0, cutoff_frequency_hz=150)(monster, sr)
    return (audio * (1 - mix)) + (monster * mix)

board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=100),
    LowShelfFilter(gain_db=-2.0, cutoff_frequency_hz=200),
    HighShelfFilter(gain_db=3.0, cutoff_frequency_hz=8000),
    Compressor(threshold_db=-18, ratio=3, attack_ms=15, release_ms=100),
    PitchShift(semitones=0.03),
    Chorus(rate_hz=1.5, depth=0.2, centre_delay_ms=15, feedback=0.1),
    Delay(delay_seconds=0.035, mix=0.15),
    Reverb(room_size=0.15, damping=0.5, wet_level=0.15, dry_level=0.85),
    Distortion(drive_db=5.0)
])

def process_chunk_ai_voice(audio_tensor, sr):
    # Accepts either torch tensor or numpy array like your original code
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = np.array(audio_tensor)

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)  # (channels, samples)

    effected = board(audio_np, sr)

    # One monster layer
    effected = add_monster_layer(effected, sr, semitones=-3, mix=0.2)

    # Ensure float32, shape (channels, samples)
    if effected.dtype != np.float32:
        effected = effected.astype(np.float32)
    return effected

# -------------------------------------------------------------------------

app = FastAPI()

# Initialize/load model at startup (blocking)
print("Loading TTS model... (this could take a while)")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
# Optional optimizations from your original:
try:
    model.t3 = torch.compile(model.t3, fullgraph=True, mode="reduce-overhead")
except Exception as e:
    print("Warning: torch.compile failed or not beneficial on this platform:", e)
print("Model loaded. Sample rate:", getattr(model, "sr", None))

# Helper: interleave (channels, samples) -> interleaved samples per frame (float32)
def interleave_channels(arr: np.ndarray) -> np.ndarray:
    # arr shape: (channels, frames)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    channels, frames = arr.shape
    # transpose to (frames, channels), then flatten -> interleaved
    return arr.T.flatten()

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        init_msg = await websocket.receive_text()
        # Expecting a JSON: {"text": "...", "audio_prompt_path": "...", "chunk_size": 100}
        try:
            req = json.loads(init_msg)
        except Exception:
            await websocket.send_json({"type": "error", "message": "Invalid init JSON"})
            await websocket.close()
            return

        text = req.get("text", "")
        audio_prompt_path = req.get("audio_prompt_path", None)
        chunk_size = req.get("chunk_size", 100)

        if not text:
            await websocket.send_json({"type": "error", "message": "No text provided"})
            await websocket.close()
            return

        sr = getattr(model, "sr", 48000)
        channels = 1  # assume your model returns mono; adapt if needed

        # Inform client about audio meta
        await websocket.send_json({"type": "meta", "sr": sr, "channels": channels, "dtype": "float32"})

        # Use a thread to run the blocking generator (model.generate_stream)
        def produce_and_send():
            try:
                for audio_chunk, metrics in model.generate_stream(
                        text,
                        audio_prompt_path=audio_prompt_path,
                        chunk_size=chunk_size,
                        exaggeration=req.get("exaggeration", 0.7),
                        cfg_weight=req.get("cfg_weight", 0.3),
                ):
                    effected = process_chunk_ai_voice(audio_chunk, sr)  # (channels, frames) float32

                    # Ensure shape (channels, frames)
                    if effected.ndim == 1:
                        effected = np.expand_dims(effected, 0)

                    channels_local, frames = effected.shape
                    interleaved = interleave_channels(effected)  # float32 1-D

                    # Tell client a chunk is coming
                    coro_send_json = websocket.send_json({"type": "chunk", "frames": frames})
                    fut = asyncio.run_coroutine_threadsafe(coro_send_json, loop)
                    fut.result()

                    # Send raw bytes (Float32 little-endian)
                    coro_send_bytes = websocket.send_bytes(interleaved.tobytes())
                    fut2 = asyncio.run_coroutine_threadsafe(coro_send_bytes, loop)
                    fut2.result()

                    # Optionally send metrics for debugging
                    try:
                        coro_metrics = websocket.send_json({"type": "metrics", "rtf": getattr(metrics, "rtf", None), "chunk_count": getattr(metrics, "chunk_count", None)})
                        asyncio.run_coroutine_threadsafe(coro_metrics, loop).result()
                    except Exception:
                        pass

                # Finished sending all chunks
                asyncio.run_coroutine_threadsafe(websocket.send_json({"type": "end"}), loop).result()
            except Exception as e:
                try:
                    asyncio.run_coroutine_threadsafe(websocket.send_json({"type": "error", "message": str(e)}), loop).result()
                except Exception:
                    pass

        thr = threading.Thread(target=produce_and_send, daemon=True)
        thr.start()

        # keep websocket open until thread finishes or client disconnects
        while thr.is_alive():
            try:
                msg = await websocket.receive_text()
                # Client can send commands; if they send {"cmd":"stop"} we break
                try:
                    js = json.loads(msg)
                    if js.get("cmd") == "stop":
                        # No standard way to stop model.generate_stream; we rely on it checking external flag in a more advanced setup
                        await websocket.send_json({"type": "info", "message": "Stop requested (not implemented)."})
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            except WebSocketDisconnect:
                break
        # Ensure finished
        await websocket.close()
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Websocket error:", e)
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    use_ngrok = os.environ.get("USE_NGROK", "1") == "1" and PYNGROK_AVAILABLE

    if use_ngrok:
        # Attempt to open ngrok HTTP tunnel to the server port
        ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
        if ngrok_token:
            ngrok_conf.get_default().auth_token = ngrok_token
        try:
            public_url = ngrok.connect(port, "http").public_url
            print("ngrok tunnel created:", public_url)
            print("WebSocket endpoint (ws):", public_url.replace("http", "ws") + "/ws/tts")
        except Exception as e:
            print("ngrok failed:", e)
            print("Start server without ngrok or install/configure pyngrok.")
    else:
        print("ngrok disabled or not available. Running locally on port", port)

    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
