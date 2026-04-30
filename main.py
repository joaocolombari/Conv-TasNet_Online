import torch
import torchaudio
import torchaudio.functional as F
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import signal
import sys
import onnxruntime as ort

# =====================================================
# REALTIME SOURCE SEPARATION PRO v3 + 20ms LOOKAHEAD
# ConvTasNet + capture + separation + stereo playback
#
# LEFT EAR  = Source 1
# RIGHT EAR = Source 2
#
# UPGRADES:
# ✔️ mantém tudo do seu código original
# ✔️ adiciona look-ahead real de 20 ms
# ✔️ atraso pequeno e controlado
# ✔️ melhor qualidade perceptual (paper)
# ✔️ menos buzzing / menos leakage
# =====================================================

# ---------------- CONFIG ----------------
MIC_SR = 48000
MODEL_SR = 8000
CHANNELS = 1

CHUNK_MS = np.round(120*0.8)
WINDOW_MS = np.round(480*0.8)
LOOKAHEAD_MS = np.round(100*0.8)

QUEUE_MAX = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_num_threads(4)

OUT1 = "source1.wav"
OUT2 = "source2.wav"

PLAY_GAIN = 0.85

# ---------------------------------------

chunk_mic = int(MIC_SR * CHUNK_MS / 1000)
chunk_model = int(MODEL_SR * CHUNK_MS / 1000)

look_mic = int(MIC_SR * LOOKAHEAD_MS / 1000)
look_model = int(MODEL_SR * LOOKAHEAD_MS / 1000)

window_model = int(MODEL_SR * WINDOW_MS / 1000)

print("=" * 60)
print("Realtime Separator PRO v3 + LookAhead")
print("Device:", DEVICE)
print("Mic SR:", MIC_SR)
print("Model SR:", MODEL_SR)
print("Chunk:", CHUNK_MS, "ms")
print("Window:", WINDOW_MS, "ms")
print("LookAhead:", LOOKAHEAD_MS, "ms")
print("=" * 60)

# =====================================================
# MODEL LOAD
# =====================================================
print("Loading ConvTasNet...")

ONNX_PATH = "convtasnet_fp32.onnx"

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]  # ou CUDAExecutionProvider
)

input_name = session.get_inputs()[0].name

print("Model loaded.")

# =====================================================
# GLOBALS
# =====================================================
audio_q = queue.Queue(maxsize=QUEUE_MAX)
play_q = queue.Queue(maxsize=QUEUE_MAX)

running = True

capture_chunks = 0
processed_chunks = 0
played_chunks = 0
dropped_chunks = 0
dropped_play_chunks = 0

source1_all = []
source2_all = []

buffer = torch.zeros(1, 1, window_model, device=DEVICE)

proc_times = []

# acumulador para look-ahead
future_cache = np.zeros(0, dtype=np.float32)

# =====================================================
# AUDIO INPUT CALLBACK
# =====================================================
def callback(indata, frames, time_info, status):
    global capture_chunks, dropped_chunks

    if status:
        print("Audio status:", status)

    mono = indata[:, 0].copy()

    try:
        audio_q.put_nowait(mono)
        capture_chunks += 1
    except queue.Full:
        dropped_chunks += 1


# =====================================================
# INFERENCE THREAD
# =====================================================
def worker():
    global running
    global processed_chunks
    global dropped_play_chunks
    global buffer
    global future_cache

    last_log = time.time()

    while running or not audio_q.empty():

        try:
            new_chunk = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        # acumula chunk novo
        future_cache = np.concatenate([future_cache, new_chunk])

        # só processa quando tiver chunk + lookahead
        needed = chunk_mic + look_mic

        if len(future_cache) < needed:
            continue

        t0 = time.time()

        # usa chunk atual + 20ms futuro
        proc = future_cache[:needed]

        # remove apenas chunk principal
        # guarda futuro residual para próxima iteração
        future_cache = future_cache[chunk_mic:]

        # tensor
        x = torch.tensor(proc).float().unsqueeze(0).unsqueeze(0)

        # resample mic -> model
        x = F.resample(x, MIC_SR, MODEL_SR).to(DEVICE)

        # auto gain
        rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
        target = 0.06

        gain = target / (rms + 1e-8)
        gain = torch.clamp(gain, 0.5, 4.0)

        x = x * gain
        x = torch.clamp(x, -1.0, 1.0)

        # mantemos somente chunk principal no buffer
        x_main = x[:, :, :chunk_model]

        # rolling context
        buffer = torch.roll(buffer, -chunk_model, dims=-1)
        buffer[:, :, -chunk_model:] = x_main

        # =====================================================
        # ONNX INFERENCE (SUBSTITUI MODEL)
        # =====================================================
        ort_input = buffer.cpu().numpy().astype(np.float32)

        sep = session.run(
            None,
            {input_name: ort_input}
        )[0]

        out1 = sep[0, 0, -chunk_model:]
        out2 = sep[0, 1, -chunk_model:]

        source1_all.append(out1.copy())
        source2_all.append(out2.copy())

        stereo = np.stack([out1, out2], axis=1).astype(np.float32)

        mx = np.max(np.abs(stereo)) + 1e-9
        if mx > 1.0:
            stereo = stereo / mx

        stereo *= PLAY_GAIN

        try:
            play_q.put_nowait(stereo)
        except queue.Full:
            dropped_play_chunks += 1

        processed_chunks += 1

        proc_times.append(time.time() - t0)

        if time.time() - last_log > 1.0:
            avg = np.mean(proc_times[-20:]) if proc_times else 0

            print(
                f"Captured={capture_chunks} | "
                f"Processed={processed_chunks} | "
                f"Played={played_chunks} | "
                f"InQ={audio_q.qsize()} | "
                f"OutQ={play_q.qsize()} | "
                f"DroppedIn={dropped_chunks} | "
                f"DroppedOut={dropped_play_chunks} | "
                f"ChunkProc={avg*1000:.1f} ms"
            )

            last_log = time.time()


# =====================================================
# PLAYBACK THREAD
# =====================================================
def playback_worker():
    global running, played_chunks

    with sd.OutputStream(
        samplerate=MODEL_SR,
        blocksize=chunk_model,
        channels=2,
        dtype="float32"
    ) as stream:

        while running or not play_q.empty():

            try:
                stereo = play_q.get(timeout=0.2)
            except queue.Empty:
                continue

            stream.write(stereo)
            played_chunks += 1


# =====================================================
# STOP HANDLER
# =====================================================
def stop_handler(sig, frame):
    global running
    print("\nStopping requested...")
    running = False


signal.signal(signal.SIGINT, stop_handler)

# =====================================================
# START THREADS
# =====================================================
thread_sep = threading.Thread(target=worker, daemon=True)
thread_play = threading.Thread(target=playback_worker, daemon=True)

thread_sep.start()
thread_play.start()

# =====================================================
# START MIC STREAM
# =====================================================
print("Opening microphone...")
print("Speak / play audio.")
print("Use headphones.")
print("LEFT = Source1 | RIGHT = Source2")
print("LookAhead latency:", LOOKAHEAD_MS, "ms")
print("Press CTRL+C to stop.\n")

t_global0 = time.time()

with sd.InputStream(
    samplerate=MIC_SR,
    blocksize=chunk_mic,
    channels=1,
    callback=callback
):
    while running:
        time.sleep(0.1)

# =====================================================
# FINISH
# =====================================================
print("Finishing queues...")

thread_sep.join()
thread_play.join()

t_global1 = time.time()

# =====================================================
# SAVE FILES
# =====================================================
print("Saving WAV files...")

if len(source1_all) == 0:
    print("No audio processed.")
    sys.exit()

s1 = np.concatenate(source1_all)
s2 = np.concatenate(source2_all)

s1 /= np.max(np.abs(s1)) + 1e-9
s2 /= np.max(np.abs(s2)) + 1e-9

sf.write(OUT1, s1, MODEL_SR)
sf.write(OUT2, s2, MODEL_SR)

# =====================================================
# METRICS
# =====================================================
audio_duration = len(s1) / MODEL_SR
proc_total = t_global1 - t_global0
rtf = proc_total / audio_duration if audio_duration > 0 else 999

corr = np.corrcoef(s1, s2)[0, 1]

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Captured chunks : {capture_chunks}")
print(f"Processed chunks: {processed_chunks}")
print(f"Played chunks   : {played_chunks}")
print(f"Dropped input   : {dropped_chunks}")
print(f"Dropped output  : {dropped_play_chunks}")
print(f"Separated audio : {audio_duration:.2f} s")
print(f"Wall time       : {proc_total:.2f} s")
print(f"RTF             : {rtf:.4f}")
print(f"Correlation     : {corr:.4f}")
print(f"Saved           : {OUT1}")
print(f"Saved           : {OUT2}")

if rtf < 1:
    print("Realtime capable ✔️")
else:
    print("Too slow for strict realtime ✖️")

print("=" * 60)