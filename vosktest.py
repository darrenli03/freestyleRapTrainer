import os
import sys
import queue
import sounddevice as sd
import select  # Import select for non-blocking input
from vosk import Model, KaldiRecognizer

# Path to the Vosk model directory
MODEL_PATH = "vosk-model-en-us-0.22-lgraph"

# Ensure the model directory exists
if not os.path.exists(MODEL_PATH):
    print(f"Please download the model from https://alphacephei.com/vosk/models and unpack it as '{MODEL_PATH}' in the current folder.")
    sys.exit(1)

# Load the Vosk model
model = Model(MODEL_PATH)

# Initialize the recognizer
recognizer = KaldiRecognizer(model, 16000)

# Create a queue to store audio data
audio_queue = queue.Queue()

# Variable to store the final line of text
final_line = ""

# Array to store all lines (2D array)
all_lines = []

# Array to store words for the current line
current_line_words = []

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def main():
    global final_line, current_line_words
    print("Listening... Speak into the microphone. Type anything and press ENTER to append the line.")

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=audio_callback):
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    print("Result:", result)

                    # Extract words from the result and add to current_line_words
                    words = eval(result).get("text", "").split()
                    current_line_words.extend(words)

                else:
                    partial_result = recognizer.PartialResult()
                    print("Partial result:", partial_result)

                # Check if there is input from the terminal
                if select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline()
                    if user_input:  # If the user typed something
                        all_lines.append(current_line_words)  # Append the current line to all_lines
                        print("Appended line. Current lines:", all_lines)
                        current_line_words = []  # Reset the current line words
                        print("Listening... Speak into the microphone. Type anything and press ENTER to append the line.")
    except KeyboardInterrupt:
        print("\nExiting...")
        print("All lines:", all_lines)  # Print all lines when exiting
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()