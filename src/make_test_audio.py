#!/usr/bin/env python3
"""Generate a short, clearly tonal solo-piano clip for pipeline self-test.

Writes a MIDI and (via fluidsynth) a 44.1 kHz WAV so we can exercise
audio -> Transkun -> MIDI -> MusicXML -> PDF without any network/YouTube.
"""
import subprocess
import sys
from pathlib import Path

import pretty_midi

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIDI = OUT_DIR / "selftest_piano.mid"
WAV = OUT_DIR / "selftest_piano.wav"
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

BPM = 92.0
BEAT = 60.0 / BPM

# I-vi-IV-V in C major, two passes.
PROG = [
    ([60, 64, 67], [60]),   # C
    ([57, 60, 64], [57]),   # Am
    ([53, 57, 60], [53]),   # F
    ([55, 59, 62], [55]),   # G
] * 2

# Simple right-hand melody (scale-ish), one note per beat.
MELODY = [72, 71, 67, 69, 72, 76, 74, 72,
          71, 69, 67, 69, 71, 74, 72, 67,
          72, 71, 67, 69, 72, 76, 79, 76,
          74, 72, 71, 69, 67, 69, 71, 72]


def main():
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")

    t = 0.0
    for chord_notes, bass in PROG:
        # One bar = 4 beats; left-hand chord held + bass note.
        for p in chord_notes:
            piano.notes.append(pretty_midi.Note(velocity=64, pitch=p, start=t, end=t + 4 * BEAT))
        piano.notes.append(pretty_midi.Note(velocity=72, pitch=bass[0] - 12, start=t, end=t + 2 * BEAT))
        t += 4 * BEAT

    # Right-hand melody over the same span, one note per beat.
    t = 0.0
    for i, p in enumerate(MELODY):
        dur = BEAT * (0.9 if i % 2 == 0 else 0.45)
        piano.notes.append(pretty_midi.Note(velocity=88, pitch=p, start=t, end=t + dur))
        t += BEAT

    pm.instruments.append(piano)
    pm.write(str(MIDI))
    print("MIDI:", MIDI, "end_time:", round(pm.get_end_time(), 2), "s")

    subprocess.run(
        ["fluidsynth", "-ni", "-g", "1.0", "-r", "44100", "-F", str(WAV), SF2, str(MIDI)],
        check=True,
    )
    print("WAV:", WAV)


if __name__ == "__main__":
    main()
