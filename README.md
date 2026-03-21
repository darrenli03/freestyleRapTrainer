create virtual environment: 
python3 -m venv venv

activate venv:
source venv/bin/activate

use natural language toolkit cmudict corpus
pip install nltk
python -m nltk.downloader cmudict

download the vosk model
https://alphacephei.com/vosk/models