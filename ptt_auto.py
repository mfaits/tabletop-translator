import json, time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

TARGET_SR = 16000
CHUNK_MS = 50
import json
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

# =========================
# Tunables
# =========================
TARGET_SR = 16000          # STT sample rate
CHUNK_MS = 50              # detection/recording chunk size
START_THRESH = 1200        # raise if it triggers too easily; lower if it misses speech
END_THRESH_RATIO = 0.6     # end threshold = START_THRESH * ratio
END_SILENCE_MS = 500       # how long of silence means "utterance ended"
MAX_UTTERANCE_S = 12       # safety cap in case of constant noise
COOLDOWN_MS = 200          # small pause after an utterance to avoid double-triggers

# =========================
# Helpers
# =========================
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def rms_int16(x: np.ndarray) -> float:
    # x is int16 array (N,)
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))

def resample_to_16k(audio_int16: np.ndarray, src_sr: int) -> bytes:
    """audio_int16: mono int16 (N,) -> bytes at 16k int16"""
    if src_sr == TARGET_SR:
        return audio_int16.tobytes()

    audio_f = audio_int16.astype(np.float32)
    audio_rs = resample_poly(audio_f, TARGET_SR, src_sr)
    audio_rs = np.clip(audio_rs, -32768, 32767).astype(np.int16)
    return audio_rs.tobytes()

# =========================
# Audio capture
# =========================
def record_utterance(device_index: int, src_sr: int) -> bytes:
    """
    Wait for sound above START_THRESH, then keep recording until
    END_SILENCE_MS of silence (below START_THRESH*END_THRESH_RATIO).
    """
    chunk = int(src_sr * CHUNK_MS / 1000)
    end_sil_chunks = int(END_SILENCE_MS / CHUNK_MS)
    max_chunks = int(MAX_UTTERANCE_S * 1000 / CHUNK_MS)

    frames = []
    silent_run = 0
    started = False
    chunks_recorded = 0

    end_thresh = START_THRESH * END_THRESH_RATIO

    with sd.InputStream(device=device_index, samplerate=src_sr, channels=1, dtype="int16") as stream:
        # Wait for speech start
        while True:
            data, _ = stream.read(chunk)  # (chunk, 1)
            level = rms_int16(data[:, 0])
            if level > START_THRESH:
                started = True
                frames.append(data.copy())
                break

        # Record until silence or max duration
        while started:
            data, _ = stream.read(chunk)
            frames.append(data.copy())
            chunks_recorded += 1

            level = rms_int16(data[:, 0])
            if level < end_thresh:
                silent_run += 1
            else:
                silent_run = 0

            if silent_run >= end_sil_chunks:
                break

            if chunks_recorded >= max_chunks:
                break

    audio = np.concatenate(frames, axis=0)[:, 0]  # int16 mono
    return resample_to_16k(audio, src_sr)

def detect_pressed_mic(en_dev: int, en_sr: int, ru_dev: int, ru_sr: int) -> str:
    """
    Opens both streams briefly, checks RMS level, and returns "en" or "ru"
    when one crosses START_THRESH.
    Closes both streams before returning.
    """
    en_chunk = int(en_sr * CHUNK_MS / 1000)
    ru_chunk = int(ru_sr * CHUNK_MS / 1000)

    with sd.InputStream(device=en_dev, samplerate=en_sr, channels=1, dtype="int16") as en_stream, \
         sd.InputStream(device=ru_dev, samplerate=ru_sr, channels=1, dtype="int16") as ru_stream:

        while True:
            en_data, _ = en_stream.read(en_chunk)
            ru_data, _ = ru_stream.read(ru_chunk)

            en_level = rms_int16(en_data[:, 0])
            ru_level = rms_int16(ru_data[:, 0])

            if en_level > START_THRESH:
                return "en"
            if ru_level > START_THRESH:
                return "ru"

            time.sleep(0.01)

# =========================
# Cloud calls
# =========================
def stt_and_translate(pcm16k: bytes, source_lang: str, target_lang: str):
    stt_client = speech.SpeechClient()
    tr_client = translate.Client()

    audio = speech.RecognitionAudio(content=pcm16k)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SR,
        language_code=source_lang,
        enable_automatic_punctuation=True,
    )

    t0 = time.perf_counter()
    resp = stt_client.recognize(config=config, audio=audio)
    stt_t = time.perf_counter() - t0

    if not resp.results:
        return "", "", stt_t, 0.0

    text = resp.results[0].alternatives[0].transcript.strip()

    t1 = time.perf_counter()
    translated = tr_client.translate(text, target_language=target_lang)["translatedText"]
    tr_t = time.perf_counter() - t1

    return text, translated, stt_t, tr_t

# =========================
# Main loop
# =========================
def main():
    cfg = load_config()
    en_dev = cfg["english_mic_device"]   # you said 2
    ru_dev = cfg["russian_mic_device"]   # should be 3

    en_sr = int(sd.query_devices(en_dev)["default_samplerate"])
    ru_sr = int(sd.query_devices(ru_dev)["default_samplerate"])

    print("Auto-PTT engine running.")
    print("Hold a mic button and speak; release to end.")
    print("Ctrl+C to quit.\n")
    print(f"Config: EN dev={en_dev} sr={en_sr}, RU dev={ru_dev} sr={ru_sr}")
    print(f"Threshold: START_THRESH={START_THRESH}, END_SILENCE_MS={END_SILENCE_MS}\n")

    while True:
        winner = detect_pressed_mic(en_dev, en_sr, ru_dev, ru_sr)

        if winner == "en":
            print("[EN mic] detected… recording utterance")
            pcm = record_utterance(en_dev, en_sr)
            text, tr, stt_t, tr_t = stt_and_translate(pcm, "en-US", "ru")
            if text:
                print(f"\nEN: {text}\nRU: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")
            else:
                print("(no speech recognized)\n")

        elif winner == "ru":
            print("[RU mic] detected… recording utterance")
            pcm = record_utterance(ru_dev, ru_sr)
            text, tr, stt_t, tr_t = stt_and_translate(pcm, "ru-RU", "en")
            if text:
                print(f"\nRU: {text}\nEN: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")
            else:
                print("(no speech recognized)\n")

        # brief cooldown so the detection loop doesn't immediately retrigger
        time.sleep(COOLDOWN_MS / 1000)

if __name__ == "__main__":
    main()
START_THRESH = 800      # tune if needed
END_SILENCE_MS = 500    # how long of silence = "done"
MAX_UTTERANCE_S = 12    # safety cap

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def rms_int16(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))

def record_utterance(device_index: int, src_sr: int) -> bytes:
    """Wait for sound, then record until END_SILENCE_MS of silence."""
    chunk = int(src_sr * CHUNK_MS / 1000)
    end_sil_chunks = int(END_SILENCE_MS / CHUNK_MS)
    max_chunks = int(MAX_UTTERANCE_S * 1000 / CHUNK_MS)

    frames = []
    silent_run = 0
    started = False
    started_at = None

    with sd.InputStream(device=device_index, samplerate=src_sr, channels=1, dtype="int16") as stream:
        # Wait for speech start
        while True:
            data, _ = stream.read(chunk)   # shape (chunk, 1)
            level = rms_int16(data[:, 0])
            if level > START_THRESH:
                started = True
                started_at = time.perf_counter()
                frames.append(data.copy())
                break

        # Record until silence
        while True:
            data, _ = stream.read(chunk)
            frames.append(data.copy())

            level = rms_int16(data[:, 0])
            if level < START_THRESH * 0.6:
                silent_run += 1
            else:
                silent_run = 0

            if silent_run >= end_sil_chunks:
                break

            if (time.perf_counter() - started_at) > MAX_UTTERANCE_S:
                break

    audio = np.concatenate(frames, axis=0)[:, 0]  # int16 mono
    # Resample to 16k for STT consistency
    if src_sr != TARGET_SR:
        audio_f = audio.astype(np.float32)
        audio_rs = resample_poly(audio_f, TARGET_SR, src_sr)
        audio_rs = np.clip(audio_rs, -32768, 32767).astype(np.int16)
    else:
        audio_rs = audio

    return audio_rs.tobytes()

def stt_and_translate(pcm16k: bytes, source_lang: str, target_lang: str):
    stt_client = speech.SpeechClient()
    tr_client = translate.Client()

    audio = speech.RecognitionAudio(content=pcm16k)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SR,
        language_code=source_lang,
        enable_automatic_punctuation=True,
    )

    t0 = time.perf_counter()
    resp = stt_client.recognize(config=config, audio=audio)
    stt_t = time.perf_counter() - t0

    if not resp.results:
        return "", "", stt_t, 0.0

    text = resp.results[0].alternatives[0].transcript.strip()

    t1 = time.perf_counter()
    translated = tr_client.translate(text, target_language=target_lang)["translatedText"]
    tr_t = time.perf_counter() - t1

    return text, translated, stt_t, tr_t

def main():
    cfg = load_config()
    en_dev = cfg["english_mic_device"]  # you said 2
    ru_dev = cfg["russian_mic_device"]  # 3

    en_sr = int(sd.query_devices(en_dev)["default_samplerate"])
    ru_sr = int(sd.query_devices(ru_dev)["default_samplerate"])

    print("Auto-PTT engine running.")
    print("Hold a mic button and speak; release to end.")
    print("Ctrl+C to quit.\n")

    while True:
        # Poll both mics quickly by reading a tiny chunk from each until one crosses threshold.
        # Simpler approach: alternate checking levels.
        en_chunk = int(en_sr * CHUNK_MS / 1000)
        ru_chunk = int(ru_sr * CHUNK_MS / 1000)

        with sd.InputStream(device=en_dev, samplerate=en_sr, channels=1, dtype="int16") as en_stream, \
             sd.InputStream(device=ru_dev, samplerate=ru_sr, channels=1, dtype="int16") as ru_stream:

            while True:
                en_data, _ = en_stream.read(en_chunk)
                ru_data, _ = ru_stream.read(ru_chunk)
                en_level = rms_int16(en_data[:, 0])
                ru_level = rms_int16(ru_data[:, 0])

                if en_level > START_THRESH:
                    print("[EN mic] detected… recording utterance")
                    pcm = record_utterance(en_dev, en_sr)
                    text, tr, stt_t, tr_t = stt_and_translate(pcm, "en-US", "ru")
                    if text:
                        print(f"EN: {text}\nRU: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")
                    else:
                        print("(no speech recognized)\n")
                    break

                if ru_level > START_THRESH:
                    print("[RU mic] detected… recording utterance")
                    pcm = record_utterance(ru_dev, ru_sr)
                    text, tr, stt_t, tr_t = stt_and_translate(pcm, "ru-RU", "en")
                    if text:
                        print(f"RU: {text}\nEN: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")
                    else:
                        print("(no speech recognized)\n")
                    break

                # tiny sleep to reduce CPU
                time.sleep(0.01)

if __name__ == "__main__":
    main()
