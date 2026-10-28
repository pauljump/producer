# SAM Audio vs Demucs on a live duo recording

Run 2026-10-27 on the Watchtower passage (one voice, one guitar, a room). SAM Audio
(`facebook/sam-audio-large`, Meta open weights) runs on a Modal L40S, in 15 s chunks with a 1 s
crossfade (the whole song at once ran a 48 GB GPU out of memory). Demucs `htdemucs_6s` runs on
the Mini's CPU.

How well each guitar stem reads the song's known chord loop (Am Am G G | F F F G), beat by beat:

| Source | Beats matching the loop |
|---|---|
| Demucs guitar + other | 63% |
| SAM Audio "guitar" | 46% |
| full mix, no separation | 37% |
| SAM Audio "acoustic guitar" | 27% |
| SAM Audio "strummed guitar" | 14% |

Voice: SAM Audio "male singing voice" and Demucs vocals agree closely (spectral correlation 0.85),
and neither leaks much guitar (0.09 vs 0.11).

SAM Audio's strength is prompts Demucs has no stem for: "crowd noise" measured -16.6 dB here
(a real room). "people talking" came out at the level of the singing voice, so it can't separate
speech from singing on this recording.

Decision: Demucs keeps the instruments (chords, the engineer's guitar reference); SAM Audio
separates the voice and the room, and its levels go to the ears as evidence.

Cost: $0.63 of Modal credit for the setup runs and tests; about $0.15 per song after that.
