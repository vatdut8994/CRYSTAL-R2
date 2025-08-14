import torchaudio as ta
import torch
import numpy as np
import sounddevice as sd
from chatterbox.tts import ChatterboxTTS
from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)

# ===== AI Voice Filter =====
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
    # Convert torch Tensor -> numpy [channels, samples]
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    # Apply effect chain
    effected = board(audio_np, sr)

    # Bitcrush
    bit_depth = 12
    max_val = np.max(np.abs(effected))
    if max_val > 0:
        effected = np.round(effected / max_val * (2**(bit_depth - 1))) / (2**(bit_depth - 1)) * max_val

    # Monster undertones
    effected = add_monster_layer(effected, sr, semitones=1, mix=0.2)
    effected = add_monster_layer(effected, sr, semitones=-3, mix=0.3)

    return effected

# ===== Streaming TTS with AI filter and live playback =====
model = ChatterboxTTS.from_pretrained(device="mps")

text = "Greetings! I am Crystal, what can I do for you today? Please let me know if you need any assistance."
AUDIO_PROMPT_PATH = "voice.wav"

audio_chunks = []
for audio_chunk, metrics in model.generate_stream(text, audio_prompt_path=AUDIO_PROMPT_PATH):
    # Apply AI effect chain to chunk
    effected_chunk = process_chunk_ai_voice(audio_chunk, model.sr)

    # Play immediately
    sd.play(effected_chunk.T, model.sr, blocking=True)

    audio_chunks.append(torch.from_numpy(effected_chunk))
    print(
        f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}"
        if metrics.rtf else f"Chunk {metrics.chunk_count}"
    )

# Combine all chunks for final save
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("test-2.wav", final_audio, model.sr)
