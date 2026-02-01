import json, time
import sounddevice as sd
import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

SAMPLE_RATE = 16000

def record(device, seconds=4):
    print(f"[REC] dev={device} {seconds}s")
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="int16",
                   device=device,
                   blocking=True)
    sd.wait()
    return audio.tobytes()

def speech_to_text(pcm, lang):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=pcm)
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
    return resp.results[0].alternatives[0].transcript.strip(), dt

def translate_text(text, target):
    client = translate.Client()
    t0 = time.perf_counter()
    out = client.translate(text, target_language=target)
    dt = time.perf_counter() - t0
    return out["translatedText"], dt

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def main():
    cfg = load_config()
    en_dev = cfg["english_mic_device"]   # 2
    ru_dev = cfg["russian_mic_device"]   # 3

    print("PTT CLI ready.")
    print("Type e + Enter: English mic (EN→RU)")
    print("Type r + Enter: Russian mic (RU→EN)")
    print("Type q + Enter: quit\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break

        if cmd == "e":
            pcm = record(en_dev, 4)
            text, stt_t = speech_to_text(pcm, "en-US")
            if not text:
                print("No speech recognized.\n")
                continue
            tr, tr_t = translate_text(text, "ru")
            print(f"\nEN: {text}\nRU: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")

        elif cmd == "r":
            pcm = record(ru_dev, 4)
            text, stt_t = speech_to_text(pcm, "ru-RU")
            if not text:
                print("No speech recognized.\n")
                continue
            tr, tr_t = translate_text(text, "en")
            print(f"\nRU: {text}\nEN: {tr}\nTiming: {stt_t+tr_t:.2f}s\n")

        else:
            print("Use e / r / q\n")

if __name__ == "__main__":
    main()
import json, time
import sounddevice as sd
import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

SAMPLE_RATE = 16000

def record(device, seconds=4):
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="int16",
                   device=device,
                   blocking=True)
    sd.wait()
    return audio.tobytes()

def speech_to_text(pcm, lang):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=pcm)
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
    return resp.results[0].alternatives[0].transcript.strip(), dt

def translate_text(text, target):
    client = translate.Client()
    t0 = time.perf_counter()
    out = client.translate(text, target_language=target)
    dt = time.perf_counter() - t0
    return out["translatedText"], dt

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def main():
    cfg = load_config()
    en_dev = cfg["english_mic_device"]
    ru_dev = cfg["russian_mic_device"]

    print("PTT CLI ready.")
    print("Press 'e' then Enter for EN mic (EN→RU)")
    print("Press 'r' then Enter for RU mic (RU→EN)")
    print("Press 'q' then Enter to quit.\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "e":
            pcm = record(en_dev, 4)
            text, stt_t = speech_to_text(pcm, "en-US")
            if not text:
                print("No speech recognized.")
                continue
            tr, tr_t = translate_text(text, "ru")
            print(f"\nEN: {text}\nRU: {tr}\n(Timing {stt_t+tr_t:.2f}s)\n")
        elif cmd == "r":
            pcm = record(ru_dev, 4)
            text, stt_t = speech_to_text(pcm, "ru-RU")
            if not text:
                print("No speech recognized.")
                continue
            tr, tr_t = translate_text(text, "en")
            print(f"\nRU: {text}\nEN: {tr}\n(Timing {stt_t+tr_t:.2f}s)\n")
        else:
            print("Use e / r / q")

if __name__ == "__main__":
    main()
