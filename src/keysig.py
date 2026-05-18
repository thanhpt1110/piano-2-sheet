#!/usr/bin/env python3
"""Data-driven key / key-change detection for transcribed piano.

No hard-coded key or modulation count: slide a duration-weighted pitch-class
histogram, score each window against the 24 Krumhansl-Kessler profiles (Pearson r),
then a Viterbi pass (switch-penalised) tracks the key so it only changes on strong,
sustained evidence. Short runs are merged. Segments are reported by their MAJOR-key
equivalent (relative major/minor share a key signature and note spelling).
"""
import argparse
import sys

import mido
import numpy as np

# Krumhansl-Kessler tonal profiles; index 0 = tonic scale degree.
KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# pitch-class -> conventional MAJOR key spelling (minimal accidentals, mido-valid).
MAJOR_NAME = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}
MINOR_NAME = {
    0: "Cm", 1: "C#m", 2: "Dm", 3: "Ebm", 4: "Em", 5: "Fm",
    6: "F#m", 7: "Gm", 8: "G#m", 9: "Am", 10: "Bbm", 11: "Bm",
}
PC_LABEL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_notes(path):
    """Return (ticks_per_beat, first_tempo, [(onset_tick, pitch, dur_tick), ...])."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo, seen = 500000, False
    notes, on = [], {}
    t = 0
    for msg in mido.merge_tracks(mid.tracks):
        t += msg.time
        if msg.type == "set_tempo" and not seen:
            tempo, seen = msg.tempo, True
        if msg.type == "note_on" and msg.velocity > 0:
            on.setdefault(msg.note, []).append(t)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            st = on.get(msg.note)
            if st:
                t0 = st.pop(0)
                notes.append((t0, msg.note, max(1, t - t0)))
    return tpb, tempo, notes


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _emissions(notes, tpb, step_beats, context_beats):
    """Per-window Pearson correlation against all 24 keys (0-11 maj, 12-23 min)."""
    step = max(1, int(round(step_beats * tpb)))
    ctx = max(step, int(round(context_beats * tpb)))
    end = max((o + d for o, _, d in notes), default=0)
    n = max(1, -(-end // step))

    onsets = np.array([o for o, _, _ in notes])
    pcs = np.array([p % 12 for _, p, _ in notes])
    durs = np.array([float(d) for _, _, d in notes])

    H = np.zeros((n, 12))
    for w in range(n):
        c = (w + 0.5) * step
        m = (onsets >= c - ctx / 2) & (onsets < c + ctx / 2)
        if m.any():
            np.add.at(H[w], pcs[m], durs[m])

    E = np.zeros((n, 24))
    for ki in range(24):
        prof = KK_MAJOR if ki < 12 else KK_MINOR
        rp = np.roll(prof, ki % 12)
        s = H.sum(axis=1)
        for w in range(n):
            E[w, ki] = _pearson(H[w], rp) if s[w] > 0 else 0.0
    return E, H, step


def detect_segments(notes, tpb, step_beats=1.0, context_beats=16.0, penalty=1.5,
                    min_segment_beats=24.0):
    E, H, step = _emissions(notes, tpb, step_beats, context_beats)
    n = E.shape[0]

    # Viterbi (explicit DP arrays; clear over clever).
    k = 24
    dp = E[0].copy()
    back = np.zeros((n, k), dtype=int)
    for w in range(1, n):
        arg = int(np.argmax(dp))
        best = dp[arg]
        masked = dp.copy()
        masked[arg] = -np.inf
        arg2 = int(np.argmax(masked))
        second = dp[arg2]
        nxt = np.empty(k)
        for s in range(k):
            if s == arg:
                stay, sw_val, sw_from = dp[s], second - penalty, arg2
            else:
                stay, sw_val, sw_from = dp[s], best - penalty, arg
            if sw_val > stay:
                back[w, s], base = sw_from, sw_val
            else:
                back[w, s], base = s, stay
            nxt[s] = base + E[w, s]
        dp = nxt
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(dp))
    for w in range(n - 1, 0, -1):
        path[w - 1] = back[w, path[w]]

    # Collapse to KEY SIGNATURE (relative major/minor share one), so a flip between
    # e.g. Gb major and Eb minor is not treated as a key change.
    def sig_of(ki):
        return ki % 12 if ki < 12 else (ki % 12 + 3) % 12

    sigpath = [sig_of(int(ki)) for ki in path]

    def build_runs(seq):
        runs, s0 = [], 0
        for w in range(1, len(seq) + 1):
            if w == len(seq) or seq[w] != seq[s0]:
                runs.append([s0, w])
                s0 = w
        return runs

    runs = build_runs(sigpath)

    # Merge runs shorter than the musical minimum, then coalesce neighbours that
    # ended up on the same signature.
    min_w = max(1, int(round(min_segment_beats / step_beats)))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (a, b) in enumerate(runs):
            if b - a < min_w:
                if i > 0:
                    runs[i - 1][1] = b
                else:
                    runs[i + 1][0] = a
                runs.pop(i)
                changed = True
                break
    i = 0
    while i + 1 < len(runs):
        if sigpath[runs[i][0]] == sigpath[runs[i + 1][0]]:
            runs[i][1] = runs[i + 1][1]
            runs.pop(i + 1)
        else:
            i += 1

    # Relabel each merged span from its aggregated histogram (most reliable label),
    # then coalesce any adjacent spans that land on the same signature.
    def label_run(a, b):
        agg = H[a:b].sum(axis=0)
        scores = np.array([
            _pearson(agg, np.roll(KK_MAJOR if ki < 12 else KK_MINOR, ki % 12))
            for ki in range(24)
        ])
        ki = int(np.argmax(scores))
        tonic, is_major = ki % 12, ki < 12
        sig_major = tonic if is_major else (tonic + 3) % 12  # relative major
        return {
            "w0": a, "w1": b,
            "t0": int(round(a * step)), "t1": int(round(b * step)),
            "sig_major_pc": sig_major,
            "key": MAJOR_NAME[sig_major],
            "label": (MAJOR_NAME if is_major else MINOR_NAME)[tonic],
            "corr": float(scores[ki]),
        }

    segments = [label_run(a, b) for a, b in runs]
    i = 0
    while i + 1 < len(segments):
        if segments[i]["sig_major_pc"] == segments[i + 1]["sig_major_pc"]:
            segments[i] = label_run(segments[i]["w0"], segments[i + 1]["w1"])
            segments.pop(i + 1)
        else:
            i += 1

    return {
        "segments": segments,
        "keys": [s["key"] for s in segments],
        "labels": [s["label"] for s in segments],
        "boundary_ticks": [s["t0"] for s in segments[1:]],
        "step": step,
    }


def summary(result, tpb, tempo):
    lines = []
    for i, s in enumerate(result["segments"]):
        t0 = mido.tick2second(s["t0"], tpb, tempo)
        t1 = mido.tick2second(s["t1"], tpb, tempo)
        arrow = ""
        if i > 0:
            prev = result["segments"][i - 1]["sig_major_pc"]
            d = (s["sig_major_pc"] - prev) % 12
            d = d - 12 if d > 6 else d
            arrow = f"  ({'+' if d > 0 else ''}{d} st {'up' if d > 0 else 'down'})"
        lines.append(
            f"  section {i+1}: {s['label']:<4} (sig {s['key']}) "
            f"[{t0:6.1f}s -> {t1:6.1f}s]  r={s['corr']:.2f}{arrow}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("--step-beats", type=float, default=1.0)
    ap.add_argument("--context-beats", type=float, default=16.0)
    ap.add_argument("--penalty", type=float, default=1.5)
    ap.add_argument("--min-segment-beats", type=float, default=24.0)
    args = ap.parse_args()

    tpb, tempo, notes = parse_notes(args.midi)
    res = detect_segments(notes, tpb, args.step_beats, args.context_beats,
                          args.penalty, args.min_segment_beats)
    print(f"{len(res['segments'])} section(s); keys={res['keys']}")
    print(summary(res, tpb, tempo))


if __name__ == "__main__":
    main()
