# Producer: parked

Paused on **Oct 28, 2026**. The Meta Global AI Developer Hackathon entry moved to a social story
creator for autistic kids (a separate, new repo). Producer is not the submission. Nothing here was
deleted; it can be picked up as is.

## State at pause

- **Code:** everything through the live session is committed on `main` (last feature commit
  `7b08d71`, "Live session: palettes of versions..."). See the README's "What was built when"
  table for the full build record.
- **Pipeline stages 1 to 5 are working:** ears (Spark 1.2 + Wonder song map, SAM Audio stems),
  hands (beat-grid renderer), musicians (Wonder tool-calling takes, cited influence research),
  self-check (Spark Open on Baseten with measuring tools), and the session loop (taste to
  instruction, keep / send back, bandleader agent, finish + Muse Image cover).
- **Live Rick Rubin session** (`producer/live.py`, `web/live.html`): each player pre-writes a
  palette of versions, the song loops, Spark Open routes spoken reactions to a version in about
  2.5 s and switches at the next bar; liked versions seed new ones in the background. Working,
  but the end-to-end user experience is not finished, which is why it was paused.
- **Web app:** still running as PM2 service `producer` (fleet registry, port 8064), left on per
  the parking brief. It only spends money when someone uploads or runs a session.
- **SAM Audio on Modal:** app `producer-sam-audio` stays deployed with 0 containers. It scales to
  zero (`scaledown_window=60`, no warm containers), so it costs nothing while idle.
- **Baseten:** Spark Open is used through the shared model API (pay per token); there are no
  dedicated deployments to stop.
- **Local data not in git:** `audio/` and `sessions/` (watchtower, wt-band, wt-live) stay on the
  Mini.

## Spend at pause

From [LEDGER.md](../LEDGER.md) (generated from `ledger/calls.jsonl`; last paid call Oct 27):

| Provider | Spent | Credit |
|---|---:|---:|
| Meta Model API | $35.47 | $150 |
| Baseten | $2.80 | $100 |
| Modal | $0.78 | ~$42 |
| **Total** | **$39.05** | |

The social story project shares these credits, so what's left belongs to it now.

## Open ideas

- Finish the live session UX: one continuous loop where the producer only talks and reacts, no
  waiting between takes; make version switching feel instant and musical.
- Tighten the "changes must be audible on the first listen" rule across every instrument.
- Album finish: stems export, a shareable page per song.
- Hum / beatbox steering into the live loop, not just the take loop.

## How to resume

1. `cd ~/projects/producer && git pull --ff-only`
2. Keys come from `~/.secrets/meta-hackathon.env` (never commit). The spend stops in
   `producer/models.py` ($40 a day, $140 Meta, $95 Baseten) still apply; recheck them against
   whatever credit is left.
3. Python env: `.venv` (pinned in `requirements.lock.txt`).
4. Web app: start or restart only through the fleet registry (`producer`, port 8064), never bare
   PM2.
5. SAM Audio: `.venv/bin/modal deploy modal_app/separate.py` if the Modal app was removed.
6. Regenerate the ledger with `.venv/bin/python tools/ledger_report.py`.
7. Start with `web/live.html` and `producer/live.py`; that's where the work stopped.
