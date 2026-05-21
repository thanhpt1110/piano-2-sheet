#!/usr/bin/env python3
"""Key-aware, readable engraving for transcribed piano.

MuseScore imports a MIDI with one global key and a guessed meter, so modulated
sections drown in accidentals. This module detects sections + keys, re-spells each
section in its real key (a lower-to-base / transpose-back trip through MuseScore),
writes per-section key signatures + courtesy double barlines, merges each hand to a
single voice of chords, optionally re-bars to a forced meter, adds per-measure chord
symbols, and renders to PDF. Sounding pitch is unchanged.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import mido
from music21 import (bar, clef, converter, harmony, instrument, interval, key,
                     layout, metadata, meter, pitch, stream, tempo)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keysig

# Chord templates (pitch-class intervals from the root) -> music21 ChordSymbol kind.
CHORD_KINDS = [
    ("major", frozenset({0, 4, 7})),
    ("minor", frozenset({0, 3, 7})),
    ("dominant-seventh", frozenset({0, 4, 7, 10})),
    ("major-seventh", frozenset({0, 4, 7, 11})),
    ("minor-seventh", frozenset({0, 3, 7, 10})),
    ("major-sixth", frozenset({0, 4, 7, 9})),
    ("minor-sixth", frozenset({0, 3, 7, 9})),
    ("half-diminished", frozenset({0, 3, 6, 10})),
    ("diminished", frozenset({0, 3, 6})),
    ("diminished-seventh", frozenset({0, 3, 6, 9})),
    ("augmented", frozenset({0, 4, 8})),
    ("suspended-fourth", frozenset({0, 5, 7})),
    ("suspended-second", frozenset({0, 2, 7})),
]
ROOT_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_FLAT = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def _find_any(names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _musescore_base():
    exe = _find_any(["musescore", "mscore", "musescore4", "mscore4", "musescore3", "mscore3"])
    if not exe:
        raise RuntimeError("no MuseScore binary found")
    xvfb = _find_any(["xvfb-run"])
    return [xvfb, "-a", exe] if xvfb else [exe]


def _ms_run(base_cmd, args, timeout_s=120, retries=2):
    """Run MuseScore with a hard timeout + retry; xvfb-run can hang transiently."""
    cmd = ["timeout", str(timeout_s)] + base_cmd + [str(a) for a in args]
    last = None
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode == 0:
            return
        last = p.returncode
        if last == 124:
            print(f"  MuseScore timed out (attempt {attempt + 1}); retrying", file=sys.stderr)
    raise RuntimeError(f"MuseScore failed (exit={last}): {' '.join(map(str, args))}")


def _set_scaling(xml_path, spatium_mm):
    """Set staff size via the MusicXML <scaling> (overrides the style's Spatium)."""
    mm = round(spatium_mm * 4, 4)
    text = Path(xml_path).read_text(encoding="utf-8")
    text = re.sub(r"<millimeters>[\d.]+</millimeters>",
                  f"<millimeters>{mm}</millimeters>", text, count=1)
    Path(xml_path).write_text(text, encoding="utf-8")


def _m21_name(major_name):
    """'Gb' -> 'G-', 'F#' -> 'F#' (music21 uses '-' for flat)."""
    return major_name[0] + major_name[1:].replace("b", "-")


def minimal_interval(base_major, target_major):
    """Spelling-correct interval base->target, reduced to within +/- a tritone."""
    a = pitch.Pitch(_m21_name(base_major) + "4")
    b = pitch.Pitch(_m21_name(target_major) + "4")
    while b.ps - a.ps > 6:
        b.octave -= 1
    while b.ps - a.ps <= -6:
        b.octave += 1
    return interval.Interval(noteStart=a, noteEnd=b), int(round(b.ps - a.ps))


def lower_sections(in_mid, out_mid, boundary_ticks, drops):
    """Shift section i by -drops[i] semitones so the whole file sounds in the base key."""
    mid = mido.MidiFile(in_mid)

    def drop_at(t):
        sec = sum(1 for b in boundary_ticks if t >= b)
        return drops[min(sec, len(drops) - 1)]

    for track in mid.tracks:
        pending, t = {}, 0
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                d = drop_at(t)
                pending.setdefault(msg.note, []).append(d)
                msg.note -= d
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                st = pending.get(msg.note)
                d = st.pop(0) if st else drop_at(t)
                msg.note -= d
    mid.save(out_mid)


def measure_start_ticks(xml_path, tpb):
    score = converter.parse(xml_path)
    return [int(round(m.offset * tpb))
            for m in score.parts[0].getElementsByClass("Measure")]


def _snap(tick, barlines):
    return min(barlines, key=lambda b: abs(b - tick))


def clarify_score(score):
    """Merge each hand to one voice of chords, rebuild the grand-staff brace."""
    new = stream.Score()
    if score.metadata is not None:
        new.insert(0, score.metadata)
    parts = []
    for part in score.parts:
        ch = part.chordify(removeRedundantPitches=True)
        for inst in ch.recurse().getElementsByClass(instrument.Instrument):
            inst.partName = inst.partAbbreviation = inst.instrumentName = None
        ch.partName = ch.partAbbreviation = None
        parts.append(ch)
        new.insert(0, ch)
    new.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    return new


def rebar(score, ts_str):
    """Re-bar to a forced meter (MuseScore ignores MIDI meter). Offsets are kept; only
    barlines/ties change. Clef + tempo preserved; grand-staff brace rebuilt."""
    new = stream.Score()
    if score.metadata is not None:
        new.insert(0, score.metadata)
    parts = []
    for part in score.parts:
        flat = part.flatten()
        np_ = stream.Part()
        for cl in flat.getElementsByClass(clef.Clef)[:1]:
            np_.insert(0, cl)
        np_.insert(0, meter.TimeSignature(ts_str))
        for mm in flat.getElementsByClass(tempo.MetronomeMark):
            np_.insert(mm.offset, mm)
        for n in flat.notesAndRests:
            np_.insert(n.offset, n)
        np_.makeMeasures(inPlace=True)
        np_.makeTies(inPlace=True)
        for inst in np_.recurse().getElementsByClass(instrument.Instrument):
            inst.partName = inst.partAbbreviation = inst.instrumentName = None
        np_.partName = np_.partAbbreviation = None
        parts.append(np_)
        new.insert(0, np_)
    new.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    return new


def _chord_for_measure(measure, sharps_pref):
    """Best-fit chord symbol for a measure via duration-weighted template matching."""
    w = [0.0] * 12
    total = 0.0
    bass_pc, bass_midi = None, 999
    for n in measure.recurse().notes:
        ql = float(n.quarterLength) or 0.25
        for p in n.pitches:
            w[p.pitchClass] += ql
            total += ql
            if p.midi < bass_midi:
                bass_midi, bass_pc = p.midi, p.pitchClass
    if total <= 0:
        return None

    best = None
    for root in range(12):
        for kind, tmpl in CHORD_KINDS:
            inset = sum(w[(root + i) % 12] for i in tmpl) / total
            # Prefer simple triads: a 7th/6th must really be present to be worth its
            # extra note, and augmented/diminished need strong evidence.
            score = inset - 0.45 * (1 - inset) - 0.10 * (len(tmpl) - 3)
            if kind in ("augmented", "diminished", "diminished-seventh", "half-diminished"):
                score -= 0.08
            if root == bass_pc:
                score += 0.10
            if best is None or score > best[0]:
                best = (score, root, kind, tmpl, inset)

    _, root, kind, tmpl, inset = best
    if inset < 0.55:
        return None  # too ambiguous to label
    name = (ROOT_FLAT if sharps_pref < 0 else ROOT_SHARP)[root]
    try:
        cs = harmony.ChordSymbol(root=name, kind=kind)
        cs.writeAsChord = False
        return cs
    except Exception:
        return None


def _decorate(score, boundary_q, sharps, chords):
    """Per-section key signatures + double barlines, accidentals, and chord symbols."""
    def section_of(off):
        return sum(1 for b in boundary_q if off >= b - 1e-6)

    for pi, part in enumerate(score.parts):
        measures = list(part.getElementsByClass("Measure"))
        prev_sec, active = None, None
        for idx, m in enumerate(measures):
            sec = section_of(float(m.offset))
            if sec != prev_sec:
                for ks in list(m.getElementsByClass(key.KeySignature)):
                    m.remove(ks)
                active = key.KeySignature(sharps[min(sec, len(sharps) - 1)])
                m.insert(0, active)
                if prev_sec is not None and idx > 0:
                    measures[idx - 1].rightBarline = bar.Barline("double")
                prev_sec = sec
            m.makeAccidentals(useKeySignature=active, inPlace=True, overrideStatus=True)
            if chords and pi == 0:
                cs = _chord_for_measure(m, active.sharps if active else 0)
                if cs is not None:
                    m.insert(0.0, cs)


def _finalize(score, title=None, composer="", bars_per_system=0):
    """Set metadata and (optionally) force a fixed number of bars per system."""
    if score.metadata is None:
        score.insert(0, metadata.Metadata())
    if title:
        score.metadata.title = title
        score.metadata.movementName = title
    score.metadata.composer = composer or ""
    if bars_per_system and bars_per_system > 0 and score.parts:
        for part in score.parts:
            for m in part.getElementsByClass("Measure"):
                for el in list(m.getElementsByClass((layout.SystemLayout, layout.PageLayout))):
                    m.remove(el)
        for i, m in enumerate(score.parts[0].getElementsByClass("Measure")):
            if i > 0 and i % bars_per_system == 0:
                m.insert(0, layout.SystemLayout(isNew=True))


def engrave(twohand_mid, out_xml, out_pdf, *, style=None, title=None, composer="",
            bars_per_system=6, clarify=True, chords=True, time_sig=None,
            spatium_mm=1.75, detect_kwargs=None, manual_keys=None,
            manual_bar_ticks=None, workdir=None):
    twohand_mid = Path(twohand_mid)
    workdir = Path(workdir or twohand_mid.parent)
    ms = _musescore_base()
    tpb, tempo_us, notes = keysig.parse_notes(str(twohand_mid))

    global_xml = workdir / "06_global.musicxml"
    _ms_run(ms, [twohand_mid, "-o", global_xml])
    barlines = measure_start_ticks(global_xml, tpb)

    if manual_keys:
        keys, boundary_ticks = manual_keys, (manual_bar_ticks or [])
        print(f"keys {keys}")
    else:
        res = keysig.detect_segments(notes, tpb, **(detect_kwargs or {}))
        keys, boundary_ticks = res["keys"], res["boundary_ticks"]
        print(f"detected {len(keys)} section(s): {res['labels']} -> sig {keys}")
        print(keysig.summary(res, tpb, tempo_us))

    bar_q = None
    if time_sig:
        num, den = (int(x) for x in time_sig.split("/"))
        bar_q = num * 4.0 / den

    boundary_q = []
    for bt in boundary_ticks:
        bq = round((bt / tpb) / bar_q) * bar_q if time_sig else _snap(bt, barlines) / tpb
        if bq > 0:
            boundary_q.append(float(bq))
    boundary_q = sorted(set(boundary_q))
    snapped_ticks = [int(round(bq * tpb)) for bq in boundary_q]

    # Spelling: lower each section to the base key so MuseScore re-spells cleanly.
    drops = [minimal_interval(keys[0], k)[1] for k in keys]
    base_mid = workdir / "05c_base_key.mid"
    base_xml = workdir / "06a_base.musicxml"
    lower_sections(str(twohand_mid), str(base_mid), snapped_ticks, drops)
    _ms_run(ms, [base_mid, "-o", base_xml])

    score = converter.parse(base_xml)
    ivs = [minimal_interval(keys[0], k)[0] for k in keys]

    def sec_of(q):
        return sum(1 for b in boundary_q if q >= b - 1e-6)

    for part in score.parts:
        for m in part.getElementsByClass("Measure"):
            for n in m.recurse().notes:
                sec = sec_of(float(m.offset) + float(n.offset))
                if sec >= 1:
                    n.transpose(ivs[sec], inPlace=True)

    work = clarify_score(score) if clarify else score
    if time_sig:
        work = rebar(work, time_sig)
    sharps = [key.Key(_m21_name(k)).sharps for k in keys]
    _decorate(work, boundary_q, sharps, chords)
    _finalize(work, title, composer, bars_per_system)
    work.write("musicxml", fp=out_xml)
    if boundary_q:
        print(f"key changes at quarters {boundary_q}")

    _set_scaling(out_xml, spatium_mm)
    _ms_run(ms, (["-S", style] if style else []) + [out_xml, "-o", out_pdf])
    print(f"-> {out_pdf}")
    return out_xml, out_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("twohand_mid")
    ap.add_argument("--out-xml", required=True)
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--style", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--composer", default="")
    ap.add_argument("--bars-per-system", type=int, default=6)
    ap.add_argument("--staff-mm", type=float, default=1.75,
                    help="staff-space size in mm (smaller = more systems per page)")
    ap.add_argument("--time-sig", default=None, help="force a meter, e.g. 4/4 (re-bars)")
    ap.add_argument("--no-chords", action="store_true", help="do not add chord symbols")
    ap.add_argument("--no-chord-merge", action="store_true",
                    help="keep MuseScore's multi-voice notation instead of merging to chords")
    ap.add_argument("--context-beats", type=float, default=16.0)
    ap.add_argument("--penalty", type=float, default=1.5)
    ap.add_argument("--min-segment-beats", type=float, default=24.0)
    ap.add_argument("--keys", default=None, help="manual major-key sigs, e.g. Gb,Ab,Bb")
    ap.add_argument("--bars", default=None, help="manual boundary offsets in quarters")
    args = ap.parse_args()

    manual_keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    tpb = keysig.parse_notes(args.twohand_mid)[0]
    manual_bars = (
        [int(round(float(b) * tpb)) for b in args.bars.split(",")] if args.bars else None
    )
    engrave(
        args.twohand_mid, args.out_xml, args.out_pdf, style=args.style, title=args.title,
        composer=args.composer, bars_per_system=args.bars_per_system,
        clarify=not args.no_chord_merge, chords=not args.no_chords, time_sig=args.time_sig,
        spatium_mm=args.staff_mm,
        detect_kwargs=dict(context_beats=args.context_beats, penalty=args.penalty,
                           min_segment_beats=args.min_segment_beats),
        manual_keys=manual_keys, manual_bar_ticks=manual_bars,
    )


if __name__ == "__main__":
    main()
