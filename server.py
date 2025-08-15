# server.py
import asyncio
import threading
import json
import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Optional ngrok
try:
    from pyngrok import ngrok, conf as ngrok_conf
    PYNGROK_AVAILABLE = True
except Exception:
    PYNGROK_AVAILABLE = False

# --- Your original stack ---
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)

# ----------------- FIXED VOICE CLONE PROMPT -----------------
# Always use voice.wav (can override via env VOICE_PROMPT_PATH if you really want)
AUDIO_PROMPT_PATH = os.environ.get(
    "VOICE_PROMPT_PATH",
    os.path.join(os.path.dirname(__file__), "voice.wav")
)

if not os.path.isfile(AUDIO_PROMPT_PATH):
    print(f"[FATAL] voice clone prompt not found: {AUDIO_PROMPT_PATH}")
    print("        Put voice.wav next to server.py or set VOICE_PROMPT_PATH.")
    # Don't exit; let the websocket respond with a clean error instead.
# ------------------------------------------------------------

# ----- Audio FX pipeline (unchanged) -----
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
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = np.array(audio_tensor)

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)  # (channels, frames)

    effected = board(audio_np, sr)
    effected = add_monster_layer(effected, sr, semitones=-3, mix=0.2)

    if effected.dtype != np.float32:
        effected = effected.astype(np.float32)
    return effected  # (channels, frames) float32
# -----------------------------------------

app = FastAPI()

print("Loading TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
try:
    model.t3 = torch.compile(model.t3, fullgraph=True, mode="reduce-overhead")
except Exception as e:
    print("torch.compile not used:", e)
print("Model ready. SR:", getattr(model, "sr", None))

def interleave_channels(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    # arr: (channels, frames) -> interleaved 1D
    return arr.T.reshape(-1)

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # Hard fail early if voice prompt missing
    if not os.path.isfile(AUDIO_PROMPT_PATH):
        await websocket.send_json({
            "type": "error",
            "message": f"voice prompt not found at {AUDIO_PROMPT_PATH}"
        })
        await websocket.close()
        return

    try:
        # Accept init JSON but ignore any audio_prompt_path from client
        init_msg = await websocket.receive_text()
        try:
            req = json.loads(init_msg)
        except Exception:
            await websocket.send_json({"type": "error", "message": "Invalid init JSON"})
            await websocket.close()
            return

        text = req.get("text", "")
        if not text:
            await websocket.send_json({"type": "error", "message": "No text provided"})
            await websocket.close()
            return

        chunk_size = int(req.get("chunk_size", 100))
        exaggeration = float(req.get("exaggeration", 0.7))
        cfg_weight = float(req.get("cfg_weight", 0.3))

        sr = getattr(model, "sr", 48000)
        channels = 1  # most TTS is mono; client deinterleaves if needed

        # Tell client meta (and explicitly state which prompt file we're using)
        await websocket.send_json({
            "type": "meta",
            "sr": sr,
            "channels": channels,
            "dtype": "float32",
            "voice_prompt": os.path.basename(AUDIO_PROMPT_PATH)
        })

        def produce_and_send():
            try:
                for audio_chunk, metrics in model.generate_stream(
                    text,
                    audio_prompt_path=AUDIO_PROMPT_PATH,  # <-- FORCE clone prompt here
                    chunk_size=chunk_size,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                ):
                    effected = process_chunk_ai_voice(audio_chunk, sr)  # (C, F)
                    if effected.ndim == 1:
                        effected = np.expand_dims(effected, 0)

                    frames = effected.shape[1]
                    interleaved = interleave_channels(effected)  # float32 1-D

                    # announce + send bytes
                    fut1 = asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"type": "chunk", "frames": frames}),
                        loop
                    )
                    fut1.result()

                    fut2 = asyncio.run_coroutine_threadsafe(
                        websocket.send_bytes(interleaved.tobytes()),
                        loop
                    )
                    fut2.result()

                    # optional metrics
                    try:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_json({
                                "type": "metrics",
                                "rtf": getattr(metrics, "rtf", None),
                                "chunk_count": getattr(metrics, "chunk_count", None)
                            }),
                            loop
                        ).result()
                    except Exception:
                        pass

                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": "end"}),
                    loop
                ).result()
            except Exception as e:
                try:
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"type": "error", "message": str(e)}),
                        loop
                    ).result()
                except Exception:
                    pass

        thr = threading.Thread(target=produce_and_send, daemon=True)
        thr.start()

        # Keep socket alive until producer finishes or client bails
        while thr.is_alive():
            try:
                msg = await websocket.receive_text()
                # Accept stop command; no hard interruption hook unless model supports it
                try:
                    js = json.loads(msg)
                    if js.get("cmd") == "stop":
                        await websocket.send_json({"type": "info", "message": "Stop requested (generator will finish current step)."})
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.01)
            except WebSocketDisconnect:
                break

        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WebSocket error:", e)
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    use_ngrok = os.environ.get("USE_NGROK", "1") == "1" and PYNGROK_AVAILABLE

    if use_ngrok:
        token = os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok_conf.get_default().auth_token = token
        try:
            public_url = ngrok.connect(port, "http").public_url
            print("ngrok:", public_url)
            print("WebSocket:", public_url.replace("http", "ws") + "/ws/tts")
        except Exception as e:
            print("ngrok failed:", e)

    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
