# Stage 3 self-check test: does the check catch a bad take?

Run 2026-10-27 on the first bass take over the Watchtower passage. We rendered the good take
and planted four mistakes, then asked for a verdict on three 15-second passages of each.

## 1. By ear alone: no

Forced choice: two clips, one good and one broken; which one has the problem? Chance is 3/6.

| Model | Semitone out | 150 ms late | Sloppy timing |
|---|---|---|---|
| Muse Spark Open (Baseten) | 1/6 | 3/6 | 2/6 |
| Muse Spark 1.2 | 3/6 | 4/6 | 3/6 |
| Muse Wonder | 3/6 | 3/6 | 6/6 |

Spark Open also often says it "cannot hear audio" even though the audio tokens arrive; asking
again usually gets an answer (`selfcheck.refused`, retried).

## 2. Engineer with meters: yes

A session engineer does not judge timing by ear alone. Spark Open now hears the passage and
calls measuring tools on the rendered audio: `measure_timing` (part onsets against the
recording's onsets), `measure_tuning` (part pitch energy the guitar is not playing),
`measure_room` (part energy in the singer's band, overall level). It decides and writes the fix.

| Take | Passage 1 | Passage 2 | Passage 3 |
|---|---|---|---|
| good | PASS | PASS | PASS |
| 150 ms late | pass (missed) | FAIL sloppy | FAIL sloppy |
| semitone out | FAIL | FAIL clashes | FAIL clashes |
| random timing | FAIL sloppy | FAIL sloppy | FAIL sloppy |
| 10 LU too loud | FAIL too loud | "too loud" but passed | FAIL too loud |

No false alarms on the good take; 12 of 12 broken passages flagged as wrong in some way, 10
with the right reason and a hard fail. Raw verdicts with the measurements each one used are
kept next to each take (`*.selfcheck-test.json`, local).

Cost: about $0.60 on Baseten for these tests.
