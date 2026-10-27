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
