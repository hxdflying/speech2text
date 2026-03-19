# Medical Speech2Text Web Demo (Alibaba fun-asr-realtime)

This demo provides a simple web page for:

- Uploading a medical conversation audio file
- Choosing a hotword set: `pediatrics` or `surgery`
- Calling Alibaba DashScope realtime ASR model `fun-asr-realtime`
- Returning transcript + raw API response

## 1) Install dependencies

```bash
cd /root/test/xudong/project/speech2text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Set environment variables

Required:

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

Optional:

```bash
# default: cn
export DASHSCOPE_REGION="cn"

# vocabulary prefix when auto-creating hotword lists
export HOTWORD_PREFIX="medasr"

# reuse existing vocabulary IDs if you already created them
export PEDIATRICS_VOCABULARY_ID="vocab-xxxx"
export SURGERY_VOCABULARY_ID="vocab-yyyy"
```

If `PEDIATRICS_VOCABULARY_ID` or `SURGERY_VOCABULARY_ID` is not set, the app will create that hotword list automatically and cache it into `.hotword_cache.json`.

## 3) Run

```bash
python test.py
```

Open `http://127.0.0.1:8000`.

You can also customize service port:

```bash
export PORT=8000
python test.py
```

If your selected port is occupied, the app will automatically try the next available port.

## Notes

- Supported upload extensions: `.wav`, `.pcm`, `.mp3`, `.amr`, `.aac`, `.m4a`, `.ogg`, `.opus`, `.flac`
- For best accuracy, use clear mono speech and set sample rate correctly (commonly 16000).
- `.m4a` uploads are auto-converted to `wav(16k, mono)` before ASR (requires `ffmpeg` installed).
- If needed, manual conversion command:
  `ffmpeg -i input.m4a -ac 1 -ar 16000 output.wav`
