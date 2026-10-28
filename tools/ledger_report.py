"""Write LEDGER.md from ledger/calls.jsonl: spend per day, provider and purpose."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "ledger" / "calls.jsonl").read_text().splitlines() if l.strip()]
by_day = defaultdict(lambda: defaultdict(float))
by_purpose = defaultdict(lambda: [0, 0.0])
for r in rows:
    by_day[r["ts"][:10]][r["provider"]] += r["usd"]
    by_purpose[(r["provider"], r["model"], r["purpose"])][0] += 1
    by_purpose[(r["provider"], r["model"], r["purpose"])][1] += r["usd"]
total = defaultdict(float)
for d in by_day.values():
    for p, v in d.items():
        total[p] += v
out = ["# Cost ledger", "", "Generated from `ledger/calls.jsonl` by `tools/ledger_report.py`. Every paid call is",
       "logged before its result is used. Credits: $150 Meta Model API, $100 Baseten, $100 Modal.",
       "Code-enforced stops: $140 Meta, $95 Baseten, $40 a day across both.", "",
       "| Provider | Spent | Credit |", "|---|---:|---:|"]
credit = {"meta": 150, "baseten": 100, "modal": 100}
for p in ("meta", "baseten", "modal"):
    out.append(f"| {p} | ${total.get(p, 0):.4f} | ${credit[p]} |")
out += ["", "## By day", "", "| Day | Meta | Baseten | Modal |", "|---|---:|---:|---:|"]
for day in sorted(by_day):
    d = by_day[day]
    out.append(f"| {day} | ${d.get('meta', 0):.4f} | ${d.get('baseten', 0):.4f} | ${d.get('modal', 0):.4f} |")
out += ["", "## By purpose", "", "| Provider | Model | Purpose | Calls | Spent |", "|---|---|---|---:|---:|"]
for (p, m, purpose), (n, usd) in sorted(by_purpose.items(), key=lambda kv: -kv[1][1]):
    out.append(f"| {p} | {m} | {purpose} | {n} | ${usd:.4f} |")
(ROOT / "LEDGER.md").write_text("\n".join(out) + "\n")
print("\n".join(out[6:11]))
