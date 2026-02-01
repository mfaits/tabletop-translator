import json
import time
import queue
import threading
import html

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import tkinter as tk
from tkinter import ttk

import google.cloud.speech as speech
from google.cloud import translate_v2 as translate

# =========================
# Tunables
# =========================
TARGET_SR = 16000

CHUNK_MS = 50
START_THRESH = 1200        # raise if it triggers too easily; lower if it misses
END_THRESH_RATIO = 0.6
END_SILENCE_MS = 500
MAX_UTTERANCE_S = 12
COOLDOWN_MS = 200

# Some USB audio devices refuse mono streams via PortAudio/ALSA.
# Open in stereo and downmix to mono.
STREAM_CHANNELS = 2

# UI
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 16              # compact; bump to 18/20 if needed
PADX = 14
PADY = 10
LINE_SPACING = 10

# Subtle fade-in animation for new utterances
FADE_STEPS = 5
FADE_STEP_MS = 40           # total fade ~200ms

# Colors
COLOR_EN = "#1f77b4"         # blue
COLOR_RU = "#2ca02c"         # green
COLOR_BG = "#ffffff"
COLOR_HEADER = "#333333"

# Instruction banner (two columns)
INSTRUCTIONS_TEXT_EN = "Runs on Google Translate. Say “system clear” to clear."
INSTRUCTIONS_TEXT_RU = "Работает на Google Translate. Скажите «система очистить», чтобы очистить."
INSTRUCTIONS_COLOR = "#555555"
INSTRUCTIONS_FONT_SIZE = 12

# Clear commands (exact match after normalization)
CLEAR_CMD_EN = "system clear"
CLEAR_CMD_RU = "система очистить"


# =========================
# Helpers
# =========================
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def rms_int16(x: np.ndarray) -> float:
    """RMS for mono int16 vector (N,)"""
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))

def downmix_to_mono(frames_2d: np.ndarray) -> np.ndarray:
    """
    frames_2d: (N, C) int16. Returns mono int16 (N,).
    If C==1, uses that channel. If C>=2, averages channels.
    """
    if frames_2d.ndim != 2:
        raise ValueError("Expected 2D audio array (N, C)")
    if frames_2d.shape[1] == 1:
        return frames_2d[:, 0]
    return frames_2d.mean(axis=1).astype(np.int16)

def resample_to_16k(audio_int16_mono: np.ndarray, src_sr: int) -> bytes:
    """mono int16 (N,) -> bytes at 16k int16"""
    if src_sr == TARGET_SR:
        return audio_int16_mono.tobytes()
    audio_f = audio_int16_mono.astype(np.float32)
    audio_rs = resample_poly(audio_f, TARGET_SR, src_sr)
    audio_rs = np.clip(audio_rs, -32768, 32767).astype(np.int16)
    return audio_rs.tobytes()

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def lerp(a, b, t: float):
    return int(round(a + (b - a) * t))

def blend_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex((lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t)))

def normalize_cmd(text: str) -> str:
    """
    Normalize STT output for command matching:
    - lower
    - strip
    - remove punctuation
    - collapse whitespace
    """
    if not text:
        return ""
    t = text.strip().lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace())
    t = " ".join(t.split())
    return t

def is_clear_command(text: str) -> bool:
    """
    Triggers only for exact commands:
      - 'system clear'
      - 'система очистить'
    """
    t = normalize_cmd(text)
    return t == CLEAR_CMD_EN or t == CLEAR_CMD_RU

def list_input_devices():
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            out.append((i, dev.get("name", "")))
    return out

def list_gn_usb_pt_indices():
    idx = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0 and "GN-USB-PT" in dev.get("name", ""):
            idx.append(i)
    return idx

def validate_or_autofix_devices(en_dev, ru_dev):
    """If configured devices are not input devices, fall back to GN-USB-PT devices."""
    def is_input(d):
        try:
            return sd.query_devices(d).get("max_input_channels", 0) > 0
        except Exception:
            return False

    if is_input(en_dev) and is_input(ru_dev) and en_dev != ru_dev:
        return en_dev, ru_dev

    candidates = list_gn_usb_pt_indices()
    if len(candidates) >= 2:
        return candidates[0], candidates[1]

    raise RuntimeError(
        "Could not find two GN-USB-PT microphone devices. "
        "Check USB connections. Input devices seen: "
        + ", ".join(f"{i}:{name}" for i, name in list_input_devices())
    )


# =========================
# Audio capture
# =========================
def record_utterance(device_index: int, src_sr: int) -> bytes:
    """
    Wait for sound above START_THRESH, then record until END_SILENCE_MS of silence.
    Uses STREAM_CHANNELS and downmixes to mono; resamples to TARGET_SR for STT.
    """
    chunk = int(src_sr * CHUNK_MS / 1000)
    end_sil_chunks = int(END_SILENCE_MS / CHUNK_MS)
    max_chunks = int(MAX_UTTERANCE_S * 1000 / CHUNK_MS)

    frames = []
    silent_run = 0
    end_thresh = START_THRESH * END_THRESH_RATIO
    chunks_recorded = 0

    with sd.InputStream(
        device=device_index,
        samplerate=src_sr,
        channels=STREAM_CHANNELS,
        dtype="int16"
    ) as stream:
        # Wait for start
        while True:
            data, _ = stream.read(chunk)  # (chunk, C)
            mono = downmix_to_mono(data)
            if rms_int16(mono) > START_THRESH:
                frames.append(data.copy())
                break

        # Record until silence
        while True:
            data, _ = stream.read(chunk)
            frames.append(data.copy())
            chunks_recorded += 1

            mono = downmix_to_mono(data)
            level = rms_int16(mono)

            if level < end_thresh:
                silent_run += 1
            else:
                silent_run = 0

            if silent_run >= end_sil_chunks:
                break
            if chunks_recorded >= max_chunks:
                break

    audio = np.concatenate(frames, axis=0)  # (N, C)
    mono = downmix_to_mono(audio)           # (N,)
    return resample_to_16k(mono, src_sr)

def detect_pressed_mic(en_dev: int, en_sr: int, ru_dev: int, ru_sr: int) -> str:
    """
    Returns "en" if English mic crosses threshold, else "ru" if Russian crosses.
    """
    en_chunk = int(en_sr * CHUNK_MS / 1000)
    ru_chunk = int(ru_sr * CHUNK_MS / 1000)

    with sd.InputStream(device=en_dev, samplerate=en_sr, channels=STREAM_CHANNELS, dtype="int16") as en_stream, \
         sd.InputStream(device=ru_dev, samplerate=ru_sr, channels=STREAM_CHANNELS, dtype="int16") as ru_stream:

        while True:
            en_data, _ = en_stream.read(en_chunk)
            ru_data, _ = ru_stream.read(ru_chunk)

            en_mono = downmix_to_mono(en_data)
            ru_mono = downmix_to_mono(ru_data)

            if rms_int16(en_mono) > START_THRESH:
                return "en"
            if rms_int16(ru_mono) > START_THRESH:
                return "ru"

            time.sleep(0.01)


# =========================
# Cloud calls
# =========================
class CloudClients:
    def __init__(self):
        self.stt = speech.SpeechClient()
        self.tr = translate.Client()

def stt_and_translate(clients: CloudClients, pcm16k: bytes, source_lang: str, target_lang: str):
    audio = speech.RecognitionAudio(content=pcm16k)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SR,
        language_code=source_lang,
        enable_automatic_punctuation=True,
    )

    t0 = time.perf_counter()
    resp = clients.stt.recognize(config=config, audio=audio)
    stt_t = time.perf_counter() - t0

    if not resp.results:
        return "", "", stt_t, 0.0

    text = resp.results[0].alternatives[0].transcript.strip()

    t1 = time.perf_counter()
    translated = clients.tr.translate(text, target_language=target_lang)["translatedText"]
    translated = html.unescape(translated)

    tr_t = time.perf_counter() - t1

    return text, translated, stt_t, tr_t


# =========================
# UI
# =========================
class TranslatorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Translator")
        self.configure(bg=COLOR_BG)
        self.attributes("-fullscreen", True)

        # Escape to quit
        self.bind("<Escape>", lambda e: self.on_close())

        # Optional backup clear shortcut while keyboard attached
        self.bind("<Control-l>", lambda e: self.clear_transcript())
        self.bind("c", lambda e: self.clear_transcript())

        # Header labels (language names)
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill="x", padx=PADX, pady=(PADY, 0))

        tk.Label(
            header, text="English", bg=COLOR_BG, fg=COLOR_HEADER,
            font=(FONT_FAMILY, FONT_SIZE, "bold")
        ).pack(side="left", expand=True, anchor="w")

        tk.Label(
            header, text="Русский", bg=COLOR_BG, fg=COLOR_HEADER,
            font=(FONT_FAMILY, FONT_SIZE, "bold")
        ).pack(side="right", expand=True, anchor="e")

        # Scrollable transcript area
        container = tk.Frame(self, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=0, pady=(8, 0))

        self.canvas = tk.Canvas(container, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=COLOR_BG)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # rows (utterance widgets) live below the instruction banner
        self.max_rows = 200
        self.rows = []

        # instruction banner
        self.banner = None
        self.show_instructions()

        # Status bar
        self.status = tk.Label(
            self, text="Ready. Hold mic button and speak.",
            bg=COLOR_BG, fg="#666666", font=(FONT_FAMILY, 12)
        )
        self.status.pack(fill="x", padx=PADX, pady=(6, PADY))

    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width)

    def show_instructions(self):
        if self.banner is not None:
            try:
                self.banner.destroy()
            except Exception:
                pass

        self.banner = tk.Frame(self.inner, bg=COLOR_BG)
        self.banner.pack(fill="x", padx=PADX, pady=(0, 10))

        wrap = self.winfo_screenwidth() // 2 - (PADX * 4)

        row = tk.Frame(self.banner, bg=COLOR_BG)
        row.pack(fill="x")

        left = tk.Label(
            row,
            text=INSTRUCTIONS_TEXT_EN,
            bg=COLOR_BG,
            fg=INSTRUCTIONS_COLOR,
            font=(FONT_FAMILY, INSTRUCTIONS_FONT_SIZE),
            anchor="w",
            justify="left",
            wraplength=max(200, wrap),
        )
        right = tk.Label(
            row,
            text=INSTRUCTIONS_TEXT_RU,
            bg=COLOR_BG,
            fg=INSTRUCTIONS_COLOR,
            font=(FONT_FAMILY, INSTRUCTIONS_FONT_SIZE),
            anchor="e",
            justify="right",
            wraplength=max(200, wrap),
        )

        left.pack(side="left", fill="x", expand=True, padx=(0, PADX))
        right.pack(side="right", fill="x", expand=True, padx=(PADX, 0))

    def clear_transcript(self):
        for row in self.rows:
            row.destroy()
        self.rows = []
        self.show_instructions()
        self.canvas.yview_moveto(0.0)

    def _fade_in_labels(self, labels, final_color: str):
        start_color = blend_hex(final_color, COLOR_BG, 0.80)

        def step(k):
            t = k / (FADE_STEPS - 1) if FADE_STEPS > 1 else 1.0
            c = blend_hex(start_color, final_color, t)
            for w in labels:
                try:
                    w.configure(fg=c)
                except Exception:
                    pass
            if k < FADE_STEPS - 1:
                self.after(FADE_STEP_MS, lambda: step(k + 1))

        step(0)

    def add_utterance(self, speaker: str, en_text: str, ru_text: str):
        final_color = COLOR_EN if speaker == "en" else COLOR_RU

        row = tk.Frame(self.inner, bg=COLOR_BG)
        row.pack(fill="x", padx=PADX, pady=(0, LINE_SPACING))

        wrap = self.winfo_screenwidth() // 2 - (PADX * 4)
        initial_color = blend_hex(final_color, COLOR_BG, 0.80)

        content = tk.Frame(row, bg=COLOR_BG)
        content.pack(fill="x")

        left = tk.Label(
            content, text=en_text, bg=COLOR_BG, fg=initial_color,
            font=(FONT_FAMILY, FONT_SIZE), justify="left", anchor="w",
            wraplength=wrap
        )
        right = tk.Label(
            content, text=ru_text, bg=COLOR_BG, fg=initial_color,
            font=(FONT_FAMILY, FONT_SIZE), justify="right", anchor="e",
            wraplength=wrap
        )

        left.pack(side="left", fill="x", expand=True, padx=(0, PADX))
        right.pack(side="right", fill="x", expand=True, padx=(PADX, 0))

        self.rows.append(row)
        if len(self.rows) > self.max_rows:
            old = self.rows.pop(0)
            old.destroy()

        self._fade_in_labels([left, right], final_color)
        self.canvas.yview_moveto(1.0)

    def set_status(self, text: str):
        self.status.config(text=text)

    def on_close(self):
        self.destroy()


# =========================
# Worker thread
# =========================
def worker_loop(out_q: queue.Queue, stop_event: threading.Event):
    cfg = load_config()
    en_dev = cfg.get("english_mic_device", None)
    ru_dev = cfg.get("russian_mic_device", None)

    if en_dev is None or ru_dev is None:
        raise RuntimeError("config.json must contain english_mic_device and russian_mic_device.")

    en_dev, ru_dev = validate_or_autofix_devices(en_dev, ru_dev)

    en_sr = int(sd.query_devices(en_dev)["default_samplerate"])
    ru_sr = int(sd.query_devices(ru_dev)["default_samplerate"])

    clients = CloudClients()
    out_q.put(("status", f"Ready. EN dev={en_dev}, RU dev={ru_dev}. Hold mic button and speak."))

    while not stop_event.is_set():
        try:
            winner = detect_pressed_mic(en_dev, en_sr, ru_dev, ru_sr)
            if stop_event.is_set():
                break

            if winner == "en":
                out_q.put(("status", "Recording English…"))
                pcm = record_utterance(en_dev, en_sr)

                out_q.put(("status", "Transcribing + translating…"))
                en_text, ru_text, stt_t, tr_t = stt_and_translate(clients, pcm, "en-US", "ru")
                total = stt_t + tr_t

                if en_text:
                    if is_clear_command(en_text):
                        out_q.put(("clear",))
                        out_q.put(("status", "Cleared. Ready."))
                    else:
                        out_q.put(("utterance", "en", en_text, ru_text, total))
                        out_q.put(("status", f"Ready. (Last: {total:.2f}s)"))
                else:
                    out_q.put(("status", "No speech recognized. Ready."))

            else:  # "ru"
                out_q.put(("status", "Recording Russian…"))
                pcm = record_utterance(ru_dev, ru_sr)

                out_q.put(("status", "Transcribing + translating…"))
                ru_text, en_text, stt_t, tr_t = stt_and_translate(clients, pcm, "ru-RU", "en")
                total = stt_t + tr_t

                if ru_text:
                    if is_clear_command(ru_text):
                        out_q.put(("clear",))
                        out_q.put(("status", "Cleared. Ready."))
                    else:
                        out_q.put(("utterance", "ru", en_text, ru_text, total))
                        out_q.put(("status", f"Ready. (Last: {total:.2f}s)"))
                else:
                    out_q.put(("status", "No speech recognized. Ready."))

            time.sleep(COOLDOWN_MS / 1000)

        except Exception as e:
            out_q.put(("status", f"Error: {e!r}"))
            time.sleep(0.5)


# =========================
# Main
# =========================
def main():
    ui = TranslatorUI()

    q_out = queue.Queue()
    stop_event = threading.Event()

    t = threading.Thread(target=worker_loop, args=(q_out, stop_event), daemon=True)
    t.start()

    def pump():
        try:
            while True:
                msg = q_out.get_nowait()
                kind = msg[0]

                if kind == "status":
                    ui.set_status(msg[1])

                elif kind == "utterance":
                    _, speaker, en_text, ru_text, _total = msg
                    ui.add_utterance(speaker, en_text, ru_text)

                elif kind == "clear":
                    ui.clear_transcript()

        except queue.Empty:
            pass

        ui.after(50, pump)

    def on_close():
        stop_event.set()
        ui.on_close()

    ui.protocol("WM_DELETE_WINDOW", on_close)
    ui.after(50, pump)
    ui.mainloop()
    stop_event.set()

if __name__ == "__main__":
    main()
