"""Load keys from ~/.secrets/meta-hackathon.env. Keys never live in this repo."""
import os
from pathlib import Path

SECRETS = Path.home() / ".secrets" / "meta-hackathon.env"


def key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name and v.strip():
                return v.strip()
    raise RuntimeError(f"{name} is not set in {SECRETS}")
