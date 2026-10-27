"""Thin clients for Meta Model API and Baseten. Every call is written to the cost ledger."""
import base64
from pathlib import Path

from openai import OpenAI

from . import ledger
from .keys import key

META_BASE = "https://api.meta.ai/v1"
BASETEN_BASE = "https://inference.baseten.co/v1"
WONDER = "muse-wonder-1.0"
EARS = "muse-spark-1.2"  # docs: 1.3 audio is "not fully supported"
SELF_CHECK = "meta-models/muse-spark-open-1026"

_clients: dict[str, OpenAI] = {}


def _client(provider: str) -> OpenAI:
    if provider not in _clients:
        if provider == "meta":
            _clients[provider] = OpenAI(base_url=META_BASE, api_key=key("META_API_KEY"), timeout=600)
        else:
            _clients[provider] = OpenAI(base_url=BASETEN_BASE, api_key=key("BASETEN_API_KEY"), timeout=600)
    return _clients[provider]


def chat(provider: str, model: str, messages: list, purpose: str, estimate_usd: float = 0.05, **kw):
    ledger.guard(estimate_usd, provider)
    resp = _client(provider).chat.completions.create(model=model, messages=messages, **kw)
    u = resp.usage
    usd = ledger.token_cost(model, u.prompt_tokens, u.completion_tokens)
    ledger.record(provider, model, purpose, usd, prompt_tokens=u.prompt_tokens,
                  completion_tokens=u.completion_tokens)
    return resp


def meta_audio_part(path: Path) -> dict:
    """Meta Chat Completions takes base64 MP3/WAV as an input_audio part."""
    fmt = "mp3" if path.suffix.lower() == ".mp3" else "wav"
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(path.read_bytes()).decode(), "format": fmt}}


def baseten_audio_part(path: Path) -> dict:
    """Baseten takes audio_url, either https or a base64 data: URL (25 MB, 5 min max)."""
    mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}[path.suffix.lower()]
    url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return {"type": "audio_url", "audio_url": {"url": url}}


def text(resp) -> str:
    return resp.choices[0].message.content or ""


def transcribe(wav: bytes, seconds: float, purpose: str = "producer direction") -> str:
    """Muse Voice Transcribe: the producer's spoken direction. wav = mono 16-bit PCM at 16 or 24 kHz."""
    import httpx

    usd = ledger.TRANSCRIBE_USD_PER_HOUR * max(seconds, 1) / 3600
    ledger.guard(usd, "meta")
    r = httpx.post(f"{META_BASE}/asr/transcribe", headers={"Authorization": f"Bearer {key('META_API_KEY')}"},
                   files={"request": (None, '{"mode":"PUSH_TO_TALK","model":"muse-voice-transcribe-1.0","audioEncoding":"WAV"}', "application/json"),
                          "audio": ("say.wav", wav, "audio/wav")}, timeout=120)
    r.raise_for_status()
    ledger.record("meta", "muse-voice-transcribe-1.0", purpose, usd, seconds=round(seconds, 1))
    return r.json().get("transcript", "")


def image(prompt: str, purpose: str, size: str = "1024x1024") -> bytes:
    """Muse Image: one image, returned as PNG bytes."""
    ledger.guard(ledger.IMAGE_USD, "meta")
    resp = _client("meta").images.generate(model="muse-image-1.0", prompt=prompt, n=1, size=size,
                                           response_format="b64_json", extra_body={"output_format": "png"})
    ledger.record("meta", "muse-image-1.0", purpose, ledger.IMAGE_USD)
    return base64.b64decode(resp.data[0].b64_json)
