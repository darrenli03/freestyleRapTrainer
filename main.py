import json
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf
from vosk import KaldiRecognizer, Model

from rhymedict import (
    find_rhymes,
    find_rhymes_by_phonemes,
    get_line_rhyme_tail,
    get_word_phonemes,
    prewarm_caches,
    random_diverse_rhymes,
    random_diverse_rhymes_by_phonemes,
)


MODEL_PATHS = [
    "vosk-model-small-en-us-0.15",
    "vosk-model-en-us-0.22-lgraph",
]


class FreestyleRapTrainerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Freestyle Rap Trainer")
        self.root.geometry("900x500")

        self.model: Model | None = None
        self.recognizer: KaldiRecognizer | None = None
        self.audio_stream: sd.RawInputStream | None = None
        self.audio_queue: queue.Queue[bytes] = queue.Queue()

        self.current_words: list[str] = []
        self.partial_words: list[str] = []
        self.exclude_words: set[str] = set()

        self.beat_data: np.ndarray | None = None
        self.beat_samplerate: int = 44100
        self.beat_filename: str | None = None
        self.beat_playing: bool = False
        self.beat_speed: float = 1.0

        self.completed_line_var = tk.StringVar(value="Completed line will appear here")
        self.live_line_var = tk.StringVar(value="")
        self.rhymes_var = tk.StringVar(
            value="Rhymes will appear after you commit a line"
        )
        self.status_var = tk.StringVar(value="Status: Loading...")

        self._build_ui()
        self.root.bind("<Return>", self.commit_line)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._prewarm_caches()

    def _prewarm_caches(self) -> None:
        """Pre-warm rhyme caches in a background thread."""

        def update_status(msg: str) -> None:
            self.status_var.set(msg)
            self.root.update_idletasks()

        def warm_caches() -> None:
            try:
                prewarm_caches(progress_callback=update_status)
                self.root.after(
                    0,
                    lambda: (
                        self.status_var.set("Status: Ready"),
                        self.start_button.config(state="normal"),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0, lambda: self.status_var.set(f"Status: Cache error - {exc}")
                )

        thread = threading.Thread(target=warm_caches, daemon=True)
        thread.start()

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, padx=16, pady=16)
        container.pack(fill="both", expand=True)

        title = tk.Label(
            container,
            text="Freestyle Rap Trainer",
            font=("Helvetica", 20, "bold"),
        )
        title.pack(anchor="w", pady=(0, 12))

        controls = tk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))

        self.start_button = tk.Button(
            controls,
            text="Start Recording",
            command=self.start_recording,
            width=18,
            state="disabled",
        )
        self.start_button.pack(side="left")

        self.stop_button = tk.Button(
            controls,
            text="Stop Recording",
            command=self.stop_recording,
            width=18,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        self.quit_button = tk.Button(
            controls,
            text="Quit",
            command=self.on_close,
            width=8,
        )
        self.quit_button.pack(side="right")

        instructions = tk.Label(
            controls,
            text="Speak your line, then press Enter to lock it in and refresh rhymes.",
        )
        instructions.pack(side="left", padx=(12, 0))

        status = tk.Label(container, textvariable=self.status_var, fg="#444")
        status.pack(anchor="w", pady=(0, 8))

        beat_controls = tk.Frame(container)
        beat_controls.pack(fill="x", pady=(0, 16))

        beat_label = tk.Label(
            beat_controls, text="Beat Controls:", font=("Helvetica", 10, "bold")
        )
        beat_label.pack(anchor="w")

        beat_row = tk.Frame(beat_controls)
        beat_row.pack(fill="x", pady=(4, 0))

        self.load_beat_button = tk.Button(
            beat_row, text="Load Beat", command=self.load_beat, width=10
        )
        self.load_beat_button.pack(side="left")

        self.play_beat_button = tk.Button(
            beat_row, text="Play", command=self.toggle_beat, width=10, state="disabled"
        )
        self.play_beat_button.pack(side="left", padx=(8, 0))

        tk.Label(beat_row, text="Speed:").pack(side="left", padx=(16, 4))

        self.speed_slider = tk.Scale(
            beat_row,
            from_=0.5,
            to=1.5,
            resolution=0.05,
            orient="horizontal",
            showvalue=False,
            width=12,
            command=self.on_speed_change,
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side="left")

        self.speed_label = tk.Label(beat_row, text="1.00x", width=6)
        self.speed_label.pack(side="left", padx=(4, 0))

        self.beat_filename_var = tk.StringVar(value="No beat loaded")
        self.beat_filename_label = tk.Label(
            beat_row, textvariable=self.beat_filename_var, fg="#666"
        )
        self.beat_filename_label.pack(side="left", padx=(16, 0))

        completed_title = tk.Label(
            container, text="Completed Line", font=("Helvetica", 14, "bold")
        )
        completed_title.pack(anchor="w")

        completed_box = tk.Label(
            container,
            textvariable=self.completed_line_var,
            anchor="w",
            justify="left",
            wraplength=850,
            bg="#f3f3f3",
            padx=10,
            pady=10,
        )
        completed_box.pack(fill="x", pady=(4, 16))

        live_title = tk.Label(
            container, text="Current Spoken Line", font=("Helvetica", 14, "bold")
        )
        live_title.pack(anchor="w")

        live_box = tk.Label(
            container,
            textvariable=self.live_line_var,
            anchor="w",
            justify="left",
            wraplength=850,
            bg="#e9f4ff",
            padx=10,
            pady=10,
        )
        live_box.pack(fill="x", pady=(4, 16))

        rhyme_title = tk.Label(
            container, text="Suggested Rhymes", font=("Helvetica", 14, "bold")
        )
        rhyme_title.pack(anchor="w")

        rhyme_box = tk.Label(
            container,
            textvariable=self.rhymes_var,
            anchor="w",
            justify="left",
            wraplength=850,
            bg="#fff5e8",
            padx=10,
            pady=10,
        )
        rhyme_box.pack(fill="x", pady=(4, 0))

    def _find_model_path(self) -> str | None:
        for path in MODEL_PATHS:
            if os.path.exists(path):
                return path
        return None

    def start_recording(self) -> None:
        if self.audio_stream is not None:
            return

        model_path = self._find_model_path()
        if model_path is None:
            messagebox.showerror(
                "Model Missing",
                "No Vosk model found. Expected one of: " + ", ".join(MODEL_PATHS),
            )
            return

        try:
            if self.model is None:
                self.status_var.set(f"Status: Loading model from {model_path}...")
                self.root.update_idletasks()
                self.model = Model(model_path)

            self.recognizer = KaldiRecognizer(self.model, 16000)
            print("Model loaded, starting audio stream...")

            self.audio_stream = sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
            )
            self.audio_stream.start()
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set(
                "Status: Recording (press Enter to commit current line)"
            )
            self.root.after(40, self.process_audio_queue)
        except Exception as exc:
            self.audio_stream = None
            messagebox.showerror("Audio Error", f"Could not start recording: {exc}")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_var.set("Status: Ready")

    def stop_recording(self) -> None:
        if self.audio_stream is None:
            return

        try:
            self.audio_stream.stop()
            self.audio_stream.close()
        except Exception:
            pass

        self.audio_stream = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Status: Recording stopped")
        self.exclude_words = set()

    def audio_callback(self, indata, frames, time, status) -> None:
        if status:
            # Avoid touching Tk state from callback thread.
            print(f"Audio callback status: {status}")
        self.audio_queue.put(bytes(indata))

    def _safe_json(self, payload: str) -> dict:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def process_audio_queue(self) -> None:
        if self.audio_stream is None or self.recognizer is None:
            return

        try:
            while True:
                data = self.audio_queue.get_nowait()
                if self.recognizer.AcceptWaveform(data):
                    result = self._safe_json(self.recognizer.Result())
                    words = result.get("text", "").split()
                    if words:
                        self.current_words.extend(words)
                    self.partial_words = []
                else:
                    partial = self._safe_json(self.recognizer.PartialResult())
                    self.partial_words = partial.get("partial", "").split()
        except queue.Empty:
            pass

        self.live_line_var.set(" ".join(self.current_words + self.partial_words))
        self.root.after(40, self.process_audio_queue)

    def _extract_last_word(self, text: str) -> str:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        return words[-1] if words else ""

    def _drain_audio_queue(self) -> None:
        # Drop any buffered audio so old speech does not leak into the next line.
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

    def _find_rhyme_source(
        self, line_words: list[str], min_rhymes: int = 3
    ) -> tuple[str, tuple[str, ...]] | None:
        """
        Find the best phoneme sequence for rhyme lookup.

        Tries progressively more syllables from the end until we find
        enough rhymes, falling back to the last word alone.

        Args:
            line_words: List of words in the line.
            min_rhymes: Minimum number of rhymes needed to accept a source.

        Returns:
            Tuple of (last_word, phones) where phones is the phoneme sequence
            to query, or None if no suitable source found.
        """
        if not line_words:
            return None

        last_word = line_words[-1]
        rhymes = find_rhymes(last_word, min_score=0.5, limit=10)
        if len(rhymes) >= min_rhymes:
            phones = get_word_phonemes(last_word)
            if phones:
                return (last_word, tuple(phones))

        for n in [2, 3]:
            tail = get_line_rhyme_tail(line_words, n)
            if tail:
                rhymes = find_rhymes_by_phonemes(tail, min_score=0.5, limit=10)
                if len(rhymes) >= min_rhymes:
                    return (last_word, tail)

        phones = get_word_phonemes(last_word)
        return (last_word, tuple(phones)) if phones else None

    def commit_line(self, _event=None) -> None:
        line_words = self.current_words + self.partial_words
        if not line_words:
            return

        completed_line = " ".join(line_words)
        self.completed_line_var.set(completed_line)

        rhymes_display = "No rhymes found."
        rhyme_source = self._find_rhyme_source(line_words)
        if rhyme_source:
            last_word, phones = rhyme_source
            self.exclude_words.add(last_word)
            rhymes = random_diverse_rhymes_by_phonemes(
                phones,
                exclude=self.exclude_words,
                n=5,
                freq_weight=0.5,
                exclude_word=last_word,
            )
            if rhymes:
                words_only = [entry[0] for entry in rhymes]
                rhymes_display = ", ".join(words_only)

        self.rhymes_var.set(rhymes_display)

        # Clear recognizer/queue state so the next line starts clean after Enter.
        self._drain_audio_queue()
        if self.recognizer is not None and hasattr(self.recognizer, "Reset"):
            self.recognizer.Reset()

        self.current_words = []
        self.partial_words = []
        self.live_line_var.set("")

    def load_beat(self) -> None:
        """Open file dialog to select audio file."""
        filename = filedialog.askopenfilename(
            title="Select Beat",
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")],
        )
        if not filename:
            return

        try:
            self.beat_data, self.beat_samplerate = sf.read(filename, dtype="float32")
            self.beat_filename = filename
            basename = os.path.basename(filename)
            self.beat_filename_var.set(basename)
            self.play_beat_button.config(state="normal")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load audio: {exc}")

    def toggle_beat(self) -> None:
        """Toggle play/pause."""
        if self.beat_playing:
            self._pause_beat()
        else:
            self._play_beat()

    def _play_beat(self) -> None:
        """Start or resume beat playback."""
        if self.beat_data is None:
            return

        sd.play(
            self.beat_data,
            samplerate=int(self.beat_samplerate * self.beat_speed),
            loop=True,
        )
        self.beat_playing = True
        self.play_beat_button.config(text="Pause")

        basename = (
            os.path.basename(self.beat_filename) if self.beat_filename else "beat"
        )
        self.beat_filename_var.set(f"{basename} (looping)")

    def _pause_beat(self) -> None:
        """Pause beat playback."""
        sd.stop()
        self.beat_playing = False
        self.play_beat_button.config(text="Play")

        basename = (
            os.path.basename(self.beat_filename) if self.beat_filename else "beat"
        )
        self.beat_filename_var.set(f"{basename} (paused)")

    def on_speed_change(self, value: str) -> None:
        """Handle speed slider change."""
        self.beat_speed = float(value)
        self.speed_label.config(text=f"{self.beat_speed:.2f}x")

        if self.beat_playing:
            sd.stop()
            sd.play(
                self.beat_data,
                samplerate=int(self.beat_samplerate * self.beat_speed),
                loop=True,
            )

    def _cleanup_beat(self) -> None:
        """Clean up beat resources on close."""
        if self.beat_playing:
            sd.stop()
        self.beat_data = None

    def on_close(self) -> None:
        self.stop_recording()
        self._cleanup_beat()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = FreestyleRapTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
