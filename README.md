create virtual environment: 
```bash
python3 -m venv venv
```

activate venv:
```bash
source venv/bin/activate
```

use natural language toolkit cmudict corpus
```bash
pip install nltk
python -m nltk.downloader cmudict brown
```

download the vosk model
https://alphacephei.com/vosk/models
model name: vosk-model-en-us-0.22-lgraph
