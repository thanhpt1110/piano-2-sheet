#!/usr/bin/env python3
"""Hand-split a single-track piano MIDI into RH/LH tracks for a grand staff.

Works at the MIDI-event level (mido) so tempo/time/key meta are preserved exactly;
only the staff a note lands on changes, never its pitch or timing.
Usage: notation.py <in.mid> <out.mid> [split_point]   (default 60 = C4)
"""
import sys

import mido


def split_hands(in_path, out_path, split_point=60):
    src = mido.MidiFile(in_path)
    tpb = src.ticks_per_beat

    meta = []   # (abs_tick, msg): tempo / time sig / key sig / pedal / program
    notes = []  # (abs_tick, msg): note_on / note_off
    for track in src.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type in ("note_on", "note_off"):
                notes.append((t, msg))
            elif msg.type == "end_of_track":
                continue
            else:
                meta.append((t, msg))

    def is_note_off(msg):
        return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)

    def build(events):
        # Stable sort by tick; at the same tick release notes before striking new
        # ones so simultaneous re-articulations don't get swallowed.
        events = sorted(events, key=lambda am: (am[0], 0 if is_note_off(am[1]) else 1))
        track = mido.MidiTrack()
        last = 0
        for abs_tick, msg in events:
            track.append(msg.copy(time=abs_tick - last))
            last = abs_tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track

    right = [(t, m) for (t, m) in notes if m.note >= split_point]
    left = [(t, m) for (t, m) in notes if m.note < split_point]

    out = mido.MidiFile(ticks_per_beat=tpb, type=1)
    out.tracks.append(build(meta))   # shared tempo / time / key / pedal
    out.tracks.append(build(right))  # right hand -> treble staff
    out.tracks.append(build(left))   # left hand -> bass staff
    out.save(out_path)
    return len(right), len(left)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    split_point = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    r, l = split_hands(in_path, out_path, split_point)
    print(f"split at MIDI {split_point}: right_hand_events={r} left_hand_events={l} -> {out_path}")


if __name__ == "__main__":
    main()
