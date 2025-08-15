import torchaudio as ta
import torch
import numpy as np
from chatterbox.tts import ChatterboxTTS
from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)

# Simplified AI Voice Filter for speed
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
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    effected = board(audio_np, sr)

    # Simplified: One monster layer
    effected = add_monster_layer(effected, sr, semitones=-3, mix=0.2)

    return effected

# Streaming TTS with CUDA optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16  # bfloat16 for CUDA to reduce memory and boost speed

model = ChatterboxTTS.from_pretrained(device=device)

# Set dtype for key components
# model.t3.to(dtype=dtype)
# model.conds.t3.to(dtype=dtype)

# Compile the t3 module for 2-3x speedup (uses CUDA graphs via reduce-overhead mode)
model.t3 = torch.compile(model.t3, fullgraph=True, mode="reduce-overhead")

text = "Greetings! I am Crystal, what can I do for you today? Please let me know if you need any assistance."
AUDIO_PROMPT_PATH = "voice.wav"

audio_chunks = []

for audio_chunk, metrics in model.generate_stream(
    text, audio_prompt_path=AUDIO_PROMPT_PATH, chunk_size=100,  # Larger for efficiency/lower RTF
    exaggeration=0.7, cfg_weight=0.3,  # Faster speech/pacing
):
    effected_chunk = process_chunk_ai_voice(audio_chunk, model.sr)

    audio_chunks.append(torch.from_numpy(effected_chunk))
    print(
        f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}"
        if metrics.rtf else f"Chunk {metrics.chunk_count}"
    )

# Combine and save
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("test-2.wav", final_audio, model.sr)
