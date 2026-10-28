"""Cost ledger. Every paid call is appended to ledger/calls.jsonl before results are used.

Prices per 1M tokens (USD), from dev.meta.ai/docs/pricing-rate-limits (read 2026-10-27).
Baseten Muse Spark Open: $1.25 / $4.25 per 1M, billed against the $100 Baseten credit.
"""
import json
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALLS = ROOT / "ledger" / "calls.jsonl"

PRICES = {
    "muse-wonder-1.0": (4.00, 20.00),
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
    "muse-spark-1.1": (1.25, 4.25),
}
BASETEN_PRICE = {"meta-models/muse-spark-open-1026": (1.25, 4.25)}  # baseten.co/pricing, read 2026-10-27
IMAGE_USD = 0.01
TRANSCRIBE_USD_PER_HOUR = 0.18
DAILY_STOP_USD = 40.0  # Paul raised the daily stop from $20 to $40 on 2026-10-27; ask before going past it
# Hard stops below the prepaid credits ($150 Meta, $100 Baseten) so no card charge is possible.
TOTAL_STOP_USD = {"meta": 140.0, "baseten": 95.0}


def token_cost(model: str, prompt: int, completion: int) -> float:
    pin, pout = PRICES.get(model) or BASETEN_PRICE.get(model) or (0.0, 0.0)
    return (prompt * pin + completion * pout) / 1e6


def record(provider: str, model: str, purpose: str, usd: float, **detail) -> dict:
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": provider, "model": model,
           "purpose": purpose, "usd": round(usd, 6), **detail}
    CALLS.parent.mkdir(exist_ok=True)
    with CALLS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def spent(day: str | None = None, provider: str | None = None) -> float:
    if not CALLS.exists():
        return 0.0
    total = 0.0
    for line in CALLS.read_text().splitlines():
        r = json.loads(line)
        if day and not r["ts"].startswith(day):
            continue
        if provider and r["provider"] != provider:
            continue
        total += r["usd"]
    return total


def guard(estimate_usd: float, provider: str = "meta") -> None:
    """Refuse a call that would pass the daily stop or the provider's credit stop."""
    total = spent(provider=provider)
    if total + estimate_usd > TOTAL_STOP_USD.get(provider, 0.0):
        raise RuntimeError(f"credit stop: {provider} {total:.2f} + {estimate_usd:.2f} > {TOTAL_STOP_USD[provider]}")
    today = date.today().isoformat()
    if spent(today) + estimate_usd > DAILY_STOP_USD:
        raise RuntimeError(f"daily stop: {spent(today):.2f} + {estimate_usd:.2f} > {DAILY_STOP_USD}; ask Paul")
