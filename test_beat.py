import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestBeatPlayback:
    @patch("main.sf")
    @patch("main.sd")
    def test_load_beat_success(self, mock_sd, mock_sf):
        mock_sf.read.return_value = (np.array([1.0, 2.0]), 44100)

        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)

        with patch(
            "tkinter.filedialog.askopenfilename", return_value="/path/to/beat.wav"
        ):
            app.load_beat()

        assert app.beat_data is not None
        assert app.beat_filename == "/path/to/beat.wav"
        assert app.play_beat_button.cget("state") == "normal"

        root.destroy()

    @patch("main.sf")
    def test_load_beat_cancelled(self, mock_sf):
        mock_sf.read.return_value = (np.array([1.0, 2.0]), 44100)

        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)

        with patch("tkinter.filedialog.askopenfilename", return_value=""):
            app.load_beat()

        assert app.beat_data is None

        root.destroy()

    @patch("main.sf")
    @patch("main.sd")
    def test_load_beat_error(self, mock_sd, mock_sf):
        mock_sf.read.side_effect = Exception("Invalid file")

        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)

        with patch(
            "tkinter.filedialog.askopenfilename", return_value="/path/to/invalid.wav"
        ):
            with patch("tkinter.messagebox.showerror"):
                app.load_beat()

        assert app.beat_data is None
        assert app.play_beat_button.cget("state") == "disabled"

        root.destroy()

    @patch("main.sf")
    @patch("main.sd")
    def test_toggle_beat_play(self, mock_sd, mock_sf):
        mock_sf.read.return_value = (np.array([1.0, 2.0]), 44100)

        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)
        app.beat_data = np.array([1.0, 2.0])
        app.beat_samplerate = 44100
        app.beat_filename = "beat.wav"
        app.beat_playing = False

        app.toggle_beat()

        assert app.beat_playing is True
        assert app.play_beat_button.cget("text") == "Pause"
        mock_sd.play.assert_called_once()

    @patch("main.sf")
    @patch("main.sd")
    def test_toggle_beat_pause(self, mock_sd, mock_sf):
        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)
        app.beat_playing = True
        app.beat_filename = "beat.wav"

        app.toggle_beat()

        assert app.beat_playing is False
        assert app.play_beat_button.cget("text") == "Play"
        mock_sd.stop.assert_called_once()

    @patch("main.sf")
    @patch("main.sd")
    def test_speed_change_while_playing(self, mock_sd, mock_sf):
        mock_sf.read.return_value = (np.array([1.0, 2.0]), 44100)

        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)
        app.beat_data = np.array([1.0, 2.0])
        app.beat_samplerate = 44100
        app.beat_playing = True
        app.beat_speed = 1.0

        app.on_speed_change("0.75")

        assert app.beat_speed == 0.75
        assert app.speed_label.cget("text") == "0.75x"
        assert mock_sd.stop.call_count == 1
        assert mock_sd.play.call_count == 1

    @patch("main.sf")
    @patch("main.sd")
    def test_cleanup_beat(self, mock_sd, mock_sf):
        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)
        app.beat_data = np.array([1.0, 2.0])
        app.beat_playing = True

        app._cleanup_beat()

        mock_sd.stop.assert_called_once()
        assert app.beat_data is None

    @patch("main.sf")
    @patch("main.sd")
    def test_on_close_cleans_up_beat(self, mock_sd, mock_sf):
        from main import FreestyleRapTrainerApp
        import tkinter as tk

        root = tk.Tk()
        app = FreestyleRapTrainerApp(root)
        app.beat_data = np.array([1.0, 2.0])
        app.beat_playing = True
        app.audio_stream = None

        app.on_close()

        mock_sd.stop.assert_called_once()


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not installed. Run: pip install pytest")
