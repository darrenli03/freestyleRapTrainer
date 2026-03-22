import json
import os
import queue
import re
import tkinter as tk
from tkinter import messagebox

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from rhymedict import diverse_rhymes


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

		self.completed_line_var = tk.StringVar(value="Completed line will appear here")
		self.live_line_var = tk.StringVar(value="")
		self.rhymes_var = tk.StringVar(value="Rhymes will appear after you commit a line")
		self.status_var = tk.StringVar(value="Status: Ready")

		self._build_ui()
		self.root.bind("<Return>", self.commit_line)
		self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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

		instructions = tk.Label(
			controls,
			text="Speak your line, then press Enter to lock it in and refresh rhymes.",
		)
		instructions.pack(side="left", padx=(12, 0))

		status = tk.Label(container, textvariable=self.status_var, fg="#444")
		status.pack(anchor="w", pady=(0, 16))

		completed_title = tk.Label(container, text="Completed Line", font=("Helvetica", 14, "bold"))
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

		live_title = tk.Label(container, text="Current Spoken Line", font=("Helvetica", 14, "bold"))
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

		rhyme_title = tk.Label(container, text="Suggested Rhymes", font=("Helvetica", 14, "bold"))
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
			self.status_var.set("Status: Recording (press Enter to commit current line)")
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

	def commit_line(self, _event=None) -> None:
		line_words = self.current_words + self.partial_words
		if not line_words:
			return

		completed_line = " ".join(line_words)
		self.completed_line_var.set(completed_line)

		last_word = self._extract_last_word(completed_line)
		rhymes_display = "No rhymes found."
		if last_word:
			rhymes = diverse_rhymes(last_word, n=5)
			if rhymes:
				words_only = [entry[0] for entry in rhymes]
				rhymes_display = ", ".join(words_only)

		self.rhymes_var.set(rhymes_display)

		self.current_words = []
		self.partial_words = []
		self.live_line_var.set("")

	def on_close(self) -> None:
		self.stop_recording()
		self.root.destroy()


def main() -> None:
	root = tk.Tk()
	app = FreestyleRapTrainerApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
