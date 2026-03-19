import json
import os
import shutil
import socket
import subprocess
import tempfile
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

import dashscope
from dashscope.audio.asr import Recognition
from dashscope.audio.asr.vocabulary import VocabularyService

try:
    from dashscope.audio.asr import RecognitionCallback
except Exception:  # pragma: no cover
    class RecognitionCallback:  # type: ignore[no-redef]
        pass

MODEL_NAME = "fun-asr-realtime"
DEFAULT_SAMPLE_RATE = 16000
CACHE_FILE = Path(__file__).with_name(".hotword_cache.json")

SUPPORTED_AUDIO_FORMATS = {
    ".wav": "wav",
    ".pcm": "pcm",
    ".mp3": "mp3",
    ".amr": "amr",
    ".aac": "aac",
    # m4a is a container; DashScope expects the underlying codec format (commonly AAC).
    ".m4a": "aac",
    ".ogg": "ogg",
    ".opus": "opus",
    ".flac": "flac",
}

HOTWORD_SETS = {
    "pediatrics": [
        "儿科",
        "发热",
        "高热",
        "咳嗽",
        "喘息",
        "气促",
        "腹泻",
        "呕吐",
        "脱水",
        "支气管炎",
        "肺炎",
        "毛细支气管炎",
        "哮喘",
        "中耳炎",
        "手足口病",
        "疱疹性咽峡炎",
        "轮状病毒",
        "腺病毒",
        "甲流",
        "乙流",
        "新生儿黄疸",
        "川崎病",
        "热性惊厥",
        "过敏性鼻炎",
        "湿疹",
        "生长发育",
        "体格检查",
        "头围",
        "母乳喂养",
        "辅食添加",
        "维生素D",
        "雾化吸入",
        "布地奈德",
        "孟鲁司特",
        "口服补液盐",
        "阿莫西林",
        "头孢克肟",
        "对乙酰氨基酚",
        "布洛芬",
        "退热",
        "疫苗接种",
        "麻腮风",
        "水痘疫苗",
    ],
    "surgery": [
        "外科",
        "急腹症",
        "阑尾炎",
        "胆囊炎",
        "胆结石",
        "胰腺炎",
        "肠梗阻",
        "腹股沟疝",
        "切口疝",
        "疝修补",
        "甲状腺结节",
        "乳腺结节",
        "乳腺癌",
        "胃癌",
        "结直肠癌",
        "肝癌",
        "静脉曲张",
        "痔疮",
        "肛瘘",
        "肛裂",
        "包皮环切",
        "清创缝合",
        "术前评估",
        "术后复查",
        "全麻",
        "腰麻",
        "局麻",
        "腹腔镜",
        "开腹",
        "微创",
        "吻合口",
        "引流管",
        "负压引流",
        "换药",
        "拆线",
        "病理活检",
        "冰冻切片",
        "抗凝",
        "止血",
        "脓肿",
        "败血症",
        "深静脉血栓",
        "肺栓塞",
    ],
}

HOTWORD_ENV = {
    "pediatrics": "PEDIATRICS_VOCABULARY_ID",
    "surgery": "SURGERY_VOCABULARY_ID",
}


def configure_dashscope() -> None:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    region = os.getenv("DASHSCOPE_REGION", "cn").strip().lower()
    if region in {"cn", "china", "cn-beijing"}:
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        dashscope.base_websocket_api_url = (
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
        )
    else:
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        dashscope.base_websocket_api_url = (
            "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/"
        )


def _extract_vocabulary_id(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if not isinstance(response, dict):
        return ""

    if isinstance(response.get("vocabulary_id"), str):
        return response["vocabulary_id"]

    for key in ("output", "data", "result"):
        nested = response.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("vocabulary_id"), str):
            return nested["vocabulary_id"]
    return ""


def _extract_text_from_result(payload: Any) -> str:
    texts: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                texts.append(node.strip())
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return

        for key in ("text", "transcript"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        sentence_value = node.get("sentence")
        if isinstance(sentence_value, str) and sentence_value.strip():
            texts.append(sentence_value.strip())

        for key in ("sentence", "sentences", "segments", "sentence_list"):
            value = node.get(key)
            if isinstance(value, list):
                _walk(value)
        for key in ("output", "result", "data"):
            value = node.get(key)
            if isinstance(value, (dict, list)):
                _walk(value)

    _walk(payload)
    seen: set[str] = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return "\n".join(unique_texts)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return str(value)
    return str(value)


def _extract_api_error(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    code = str(payload.get("code") or "").strip()
    message = str(payload.get("message") or "").strip()
    status_code = payload.get("status_code")

    has_error_code = bool(code) and code not in {"0", "200", "OK"}
    has_error_status = isinstance(status_code, int) and status_code != 200

    if not has_error_code and not has_error_status:
        return ""

    if code and message and code != message:
        return f"{code}: {message}"
    if code:
        return code
    if message:
        return message
    return f"ASR request failed with status_code={status_code}"


class CollectingRecognitionCallback(RecognitionCallback):
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.sentences: list[Any] = []
        self.error: Any = None

    def on_open(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_close(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_event(self, result: Any, *args: Any, **kwargs: Any) -> None:
        self.events.append(_to_jsonable(result))
        getter = getattr(result, "get_sentence", None)
        if callable(getter):
            try:
                sentence = getter()
                if sentence:
                    self.sentences.append(sentence)
            except Exception:
                pass

    def on_complete(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_error(self, result: Any, *args: Any, **kwargs: Any) -> None:
        self.error = _to_jsonable(result)
        self.events.append({"error": self.error})

    def export_payload(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "sentences": self.sentences,
            "error": self.error,
        }


def _recognition_requires_callback() -> bool:
    try:
        callback_param = signature(Recognition.__init__).parameters.get("callback")
        if callback_param is None:
            return False
        return callback_param.default is Parameter.empty
    except Exception:
        return False


def _create_recognition(
    *,
    model: str,
    audio_format: str,
    sample_rate: int,
    vocabulary_id: str,
) -> tuple[Recognition, CollectingRecognitionCallback]:
    callback = CollectingRecognitionCallback()
    kwargs = {
        "model": model,
        "format": audio_format,
        "sample_rate": sample_rate,
        "vocabulary_id": vocabulary_id,
    }
    if _recognition_requires_callback():
        return Recognition(callback=callback, **kwargs), callback

    try:
        return Recognition(**kwargs), callback
    except TypeError as exc:
        if "callback" in str(exc):
            return Recognition(callback=callback, **kwargs), callback
        raise


class HotwordManager:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.service = VocabularyService()
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_file.exists():
            return {}
        try:
            raw = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {
                    str(k): str(v)
                    for k, v in raw.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
        except Exception:
            return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_file.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_vocabulary_id(self, domain: str) -> str:
        env_key = HOTWORD_ENV[domain]
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            return env_value

        cached = self.cache.get(domain, "").strip()
        if cached:
            return cached

        return self._create_vocabulary(domain)

    def _create_vocabulary(self, domain: str) -> str:
        vocabulary_data = [{"text": text, "weight": 4} for text in HOTWORD_SETS[domain]]
        prefix = os.getenv("HOTWORD_PREFIX", "medasr").strip() or "medasr"
        response = self.service.create_vocabulary(
            prefix=prefix,
            target_model=MODEL_NAME,
            vocabulary=vocabulary_data,
        )
        vocabulary_id = _extract_vocabulary_id(response)
        if not vocabulary_id:
            raise RuntimeError(f"Unexpected create_vocabulary response: {response}")

        self.cache[domain] = vocabulary_id
        self._save_cache()
        return vocabulary_id

    def inspect_vocabulary_sources(self) -> dict[str, dict[str, str]]:
        summary: dict[str, dict[str, str]] = {}
        for domain, env_name in HOTWORD_ENV.items():
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                summary[domain] = {"vocabulary_id": env_value, "source": "env"}
                continue
            cached = self.cache.get(domain, "").strip()
            if cached:
                summary[domain] = {"vocabulary_id": cached, "source": "cache"}
            else:
                summary[domain] = {"vocabulary_id": "", "source": "not_created"}
        return summary


configure_dashscope()
hotword_manager = HotwordManager(CACHE_FILE)
app = Flask(__name__)


@app.after_request
def add_cors_headers(response: Any) -> Any:
    # Keep local demo flexible when UI and API are opened from different local origins.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index() -> str:
    return render_template("index.html", model_name=MODEL_NAME)


@app.get("/health")
def health() -> tuple[dict[str, Any], int]:
    if not dashscope.api_key:
        return {"ok": False, "message": "DASHSCOPE_API_KEY is not configured"}, 500
    return {"ok": True}, 200


@app.get("/api/hotwords")
def hotwords() -> tuple[dict[str, Any], int]:
    return (
        {
            "model": MODEL_NAME,
            "sets": HOTWORD_SETS,
            "vocabulary": hotword_manager.inspect_vocabulary_sources(),
        },
        200,
    )


@app.route("/api/transcribe", methods=["POST", "OPTIONS"])
def transcribe() -> Any:
    if request.method == "OPTIONS":
        return "", 204

    if not dashscope.api_key:
        return jsonify({"error": "DASHSCOPE_API_KEY is not configured."}), 400

    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"error": "Please upload an audio file."}), 400

    specialty = request.form.get("specialty", "").strip().lower()
    if specialty not in HOTWORD_SETS:
        return jsonify({"error": "specialty must be pediatrics or surgery."}), 400

    try:
        sample_rate = int(request.form.get("sample_rate", DEFAULT_SAMPLE_RATE))
    except (TypeError, ValueError):
        return jsonify({"error": "sample_rate must be an integer."}), 400

    if sample_rate <= 0:
        return jsonify({"error": "sample_rate must be > 0."}), 400

    suffix = Path(audio.filename).suffix.lower()
    audio_format = SUPPORTED_AUDIO_FORMATS.get(suffix)
    if not audio_format:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        return jsonify({"error": f"Unsupported file extension. Supported: {supported}"}), 400

    temp_path = ""
    converted_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            audio.save(tmp.name)

        processing_path = temp_path
        processing_format = audio_format
        processing_sample_rate = sample_rate

        if suffix == ".m4a":
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                return (
                    jsonify(
                        {
                            "error": (
                                "上传的是 .m4a 文件，当前编码可能不被实时模型支持，"
                                "且系统未安装 ffmpeg 无法自动转码。"
                                "请安装 ffmpeg，或先手动转成 wav(单声道,16kHz) 再上传。"
                            )
                        }
                    ),
                    400,
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                converted_path = tmp_wav.name

            transcode_cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                temp_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                converted_path,
            ]
            transcode = subprocess.run(
                transcode_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if transcode.returncode != 0:
                return (
                    jsonify(
                        {
                            "error": (
                                "m4a 自动转码失败，请先手动转码为 wav 后重试。"
                                f" ffmpeg: {transcode.stderr[-300:]}"
                            )
                        }
                    ),
                    400,
                )

            processing_path = converted_path
            processing_format = "wav"
            processing_sample_rate = 16000

        vocabulary_id = hotword_manager.get_vocabulary_id(specialty)
        recognition, callback = _create_recognition(
            model=MODEL_NAME,
            audio_format=processing_format,
            sample_rate=processing_sample_rate,
            vocabulary_id=vocabulary_id,
        )
        result = recognition.call(processing_path)
        jsonable_result = _to_jsonable(result)
        if jsonable_result in (None, "", {}):
            jsonable_result = callback.export_payload()

        api_error = _extract_api_error(jsonable_result)
        if api_error:
            if suffix == ".m4a" and "UNSUPPORTED_FORMAT" in api_error:
                api_error = (
                    "UNSUPPORTED_FORMAT: 当前 m4a 编码不被实时模型支持。"
                    "请先转成 wav(单声道, 16kHz) 或标准 aac 后重试。"
                    "例如: ffmpeg -i ars_medical.m4a -ac 1 -ar 16000 ars_medical.wav"
                )
            return jsonify({"error": api_error, "raw_result": jsonable_result}), 400

        transcript = _extract_text_from_result(jsonable_result)
        if not transcript and callback.sentences:
            transcript = "\n".join(str(s).strip() for s in callback.sentences if str(s).strip())
        if not transcript:
            transcript = "No transcript text was extracted. Check raw_result."

        return jsonify(
            {
                "model": MODEL_NAME,
                "specialty": specialty,
                "vocabulary_id": vocabulary_id,
                "transcript": transcript,
                "raw_result": jsonable_result,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)


if __name__ == "__main__":
    preferred_port = int(os.getenv("PORT", "8000"))

    def find_available_port(start_port: int, max_tries: int = 50) -> int:
        for port in range(start_port, start_port + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(
            f"No free port found in range {start_port} - {start_port + max_tries - 1}"
        )

    run_port = find_available_port(preferred_port)
    if run_port != preferred_port:
        print(f"Port {preferred_port} is busy, fallback to port {run_port}.")

    app.run(host="0.0.0.0", port=run_port, debug=True)
