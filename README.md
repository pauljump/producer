# Producer

> "I know nothing about music. I know what I like and what I don't like, and I'm decisive about it."
> Rick Rubin, to Anderson Cooper (CBS, *60 Minutes*)

You bring an unfinished recording. You hire musicians. They listen to the whole song, tell you
what they heard and where they will play, and play a part that sits in time and in key under
your recording. You say what you feel in plain words ("too busy", "I don't believe it", "the
chorus should feel like relief", "that, but less") and they play it again. Two buttons: keep, or
send it back. No instrument, no theory, no soundboard.

Entry for the Meta Global AI Developer Hackathon (Cerebral Valley), Indie track, theme
**Build for Your World**. Everything in this repository was written during the building period
(Oct 27 to Nov 4, 2026). See [What was built when](#what-was-built-when).

## The rule: Meta models are the musicians' ears, brains and self-check

Our code is only their hands: a renderer that plays the notes a musician wrote, with sampled
instruments, on the song's timeline. No musical decision comes from our code.

| Role | Meta product | Where it runs | What it does |
|---|---|---|---|
| Separation | **SAM Audio** (open weights) | Self-hosted on Modal GPU | Splits the recording into stems by text prompt ("acoustic guitar", "male vocal") |
| Ears | **Muse Spark 1.2** | Meta Model API | Listens to the song and each stem; describes sections, feel, room. Reconciles with measured tempo, key, beats and chords |
| Bandleader | **Muse Wonder** | Meta Model API | Plans the arrangement, runs one sub-agent per musician, translates the producer's taste into musical instruction, calls tools to analyze and render |
| Self-check | **Muse Spark Open** (`meta-models/muse-spark-open-1026`) | Baseten | Hears each take mixed with the song and reports in time / in key / leaving room; the musician revises before the producer hears it |
| Direction | **Muse Voice Transcribe** | Meta Model API | The producer's spoken direction |
| Art | **Muse Image** | Meta Model API | Album art and musician cards |
| Build | **Muse Code** | Meta Model API | Used for real build tasks; the files it wrote are marked below |

Measured truth (`producer/analysis.py`) is ordinary signal processing (beat_this for beats,
key profiles, chord templates with Viterbi smoothing). It gives the ears facts to check against.

## Open-weight eligibility (Best Build on Meta Open Weight Models)

Every take is checked by **Muse Spark Open** served by Baseten (partner-provided inference),
model id `meta-models/muse-spark-open-1026`. Calls are made from `producer/models.py`
(`SELF_CHECK`) and every call is logged in [LEDGER.md](LEDGER.md) with provider `baseten`.
SAM Audio also runs as open weights on our own Modal GPU.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock.txt
```

Keys are read from `~/.secrets/meta-hackathon.env` (never from this repo):
`META_API_KEY`, `BASETEN_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`.

```bash
.venv/bin/python -m producer.analysis song.wav     # measured truth -> song.analysis.json
.venv/bin/uvicorn producer.server:app --port 7731   # the producer's page
```

## Spend

Every paid call goes through `producer/models.py`, which refuses calls past $20 a day, $140 on
Meta, or $95 on Baseten (below the $150 and $100 credits), and logs each call to
`ledger/calls.jsonl`. `tools/ledger_report.py` writes [LEDGER.md](LEDGER.md).

## What was built when

| Date (ET) | Built | Files |
|---|---|---|
| Oct 26 night / Oct 27 | Repo, key loading, cost ledger with hard stops, Meta + Baseten clients, measured analysis (beats, key, chords, sections, bar energy) | `producer/keys.py`, `producer/ledger.py`, `producer/models.py`, `producer/analysis.py`, `tools/ledger_report.py` |
| Oct 27 | Stage 1 listening test; ears = Spark 1.2 phrase-by-phrase + Wonder song map with stem evidence | `producer/ears.py`, `docs/LISTENING-TEST.md` |
| Oct 27 | Hands: beat-grid renderer (parts follow a drifting live tempo) and loudness-relative mix; chart and measured fit; Wonder casts a musician and the musician (Wonder, tool calling `check_fit` / `submit_take`) writes the first bass part | `producer/hands.py`, `producer/score.py`, `producer/band.py`, `tools/take.py` |
| Oct 27 | Self-check: Spark Open (Baseten) as session engineer, with measuring tools; planted-error tests by ear vs with meters | `producer/selfcheck.py`, `tools/selfcheck_test.py`, `tools/pairwise_test.py`, `docs/SELF-CHECK-TEST.md` |
| Oct 27 | Session loop: invite, say what you feel (Wonder translates taste into instruction), engineer sends takes back, keep / send back; drummer | `producer/session.py`, `tools/session.py` |
| Oct 27 | Web app: upload, band stage, Muse Image musician cards, characters for the original performers, typed or spoken direction (Muse Voice Transcribe), synced solo / mute | `producer/server.py`, `web/index.html` |
| Oct 27 | SAM Audio separation on Modal (written, awaiting first run) | `modal_app/separate.py` |
| Oct 27 | Hum / beatbox steering: pYIN + onset tracking, Spark 1.2 hears the feel, Wonder places the idea, the musician plays it in character | `producer/steer.py` |
| Oct 27 | Musicians research their influences (Wonder + Meta `web_search`, cited); Finish the song: final mix + Wonder-briefed Muse Image cover | `producer/models.py` (`research`), `producer/band.py`, `producer/session.py` |
| Oct 27 | Bandleader agent: Wonder hires, briefs musician sub-agents in parallel, and reports back through its own tools | `producer/bandleader.py` |

Prior exploration (a separate, earlier Slopify app) is not part of this entry and none of its
code is used here.
