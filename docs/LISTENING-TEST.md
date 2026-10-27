# Stage 1 listening test: can Meta's models hear a recording?

Run 2026-10-27 on two of Paul's unreleased live / demo recordings (audio not in this repo).
Truth comes from `producer/analysis.py` (beat_this beats, key profiles) and from what is known
about the recordings (both are one voice and one acoustic guitar, no band).

## Findings

1. **Audio reaches all three models.** Muse Spark 1.2, Muse Wonder and Muse Spark Open (Baseten)
   all transcribe a spoken test clip word for word. Audio costs about 14 input tokens a second.
2. **Leading questions produce an imaginary band.** Asked to "describe the instruments", every
   model added bass, drums or keys to a solo acoustic recording. Spark Open sometimes refused
   ("I can't hear audio") until given a system prompt stating it has audio input.
3. **Narrow questions on short phrases work.** Asked about one 4-bar phrase at a time
   ("is someone singing, what words, which instruments can you clearly hear, intensity 1 to 5"),
   Spark 1.2 got singing on/off and the lyrics right, and intensity tracked measured loudness.
   It still occasionally named drums in loud strummed passages.
4. **The bandleader reconciles.** Muse Wonder, given the whole recording plus the phrase notes
   plus the measurements, rejected the phrase-level drum hallucinations, kept the measured
   tempo when its own guess differed, and produced a correct song map: performers, sections
   with lyrics, energy, and where new parts have room.

| | Truth | Spark 1.2 (phrases) | Wonder (song map) | Spark Open |
|---|---|---|---|---|
| Hold On: performers | voice + acoustic guitar | guitar; drums in 2 of 9 phrases | voice + acoustic guitar | full band (leading Q); solo acoustic (narrow Q) |
| Hold On: tempo | ~88-91 BPM | n/a | 88 (took measured) | ~88 (narrow Q) |
| Hold On: words | spoken intro + verse/chorus | correct, partial | correct, partial | n/a |
| Free: performers | voice + acoustic guitar | correct | voice + acoustic guitar | not run |
| Free: structure | intro, verse, chorus, refrain, outro | lyrics per phrase | intro 0-14, verse 14-33, chorus 33-51, refrain 51-62, outro | not run |

## Design consequence

The ears are: measured facts for timing and pitch; Spark 1.2 for what happens, phrase by
phrase (words, entries, intensity); Wonder for the whole-song map that the musicians and the
producer's directions refer to ("the second chorus", "where she says the truth won't set you
free"). The self-check on Spark Open must ask narrow, falsifiable questions and will be tested
with deliberately wrong parts (off the beat, wrong key) before we trust it.

Cost of this test: about $0.55 on Meta, $0.02 on Baseten (see LEDGER.md).
