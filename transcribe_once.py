import argparse, time, sys
import sounddevice as sd
import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

SAMPLE_RATE = 16000

def record(device, seconds):
    print(f"[REC] device={device}, {seconds}s")
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="int16",
                   device=device,
                   blocking=True)
    sd.wait()
    return audio.tobytes()

def speech_to_text(pcm_bytes, lang):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=pcm_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=lang,
        enable_automatic_punctuation=True,
    )
    t0 = time.perf_counter()
    resp = client.recognize(config=config, audio=audio)
    dt = time.perf_counter() - t0
    if not resp.results:
        return "", dt
    return resp.results[0].alternatives[0].transcript, dt

def translate_text(text, target):
    client = translate.Client()
    t0 = time.perf_counter()
    out = client.translate(text, target_language=target)
    dt = time.perf_counter() - t0
    return out["translatedText"], dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--seconds", type=float, default=4)
    args = ap.parse_args()

    pcm = record(args.device, args.seconds)
    text, stt_t = speech_to_text(pcm, args.lang)

    if not text:
        print("❌ No speech recognized")
        sys.exit(1)

    translated, tr_t = translate_text(text, args.to)

    print("\n=== RESULT ===")
    print(f"Original [{args.lang}]: {text}")
    print(f"Translated → [{args.to}]: {translated}")
    print(f"\nTiming: STT={stt_t:.2f}s  Translate={tr_t:.2f}s  Total≈{stt_t+tr_t:.2f}s")

if __name__ == "__main__":
    main()
