#!/usr/bin/env python3
"""End-to-end CLI: piano audio (YouTube/file) -> MIDI -> MusicXML -> PDF.

download (yt-dlp) -> normalize (ffmpeg) -> transcribe (Transkun, CUDA) -> hand-split
-> key-aware engrave (MuseScore). Title + composer are required and are auto-extracted
from the video metadata when not supplied.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notation import split_hands

DEFAULT_STYLE = Path(__file__).resolve().parent.parent / "style" / "piano.mss"


def run(cmd, log_file, env=None, check=True):
    cmd = [str(x) for x in cmd]
    line = "+ " + " ".join(cmd)
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with exit={p.returncode}: {' '.join(cmd)}")
    return p.returncode


def find_any(names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if requested == "cuda":
        if not has_cuda:
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    # auto
    return "cuda" if has_cuda else "cpu"


def download_youtube(url, workdir, log_file, cookies=None):
    # ba/b: best audio container; --write-info-json: metadata for title/composer.
    out_template = str(workdir / "01_raw_audio.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist", "--write-info-json", "-f", "ba/b", "-o", out_template]
    if cookies:
        # Logged-in cookies clear YouTube's bot check on datacenter IPs.
        cmd += ["--cookies", str(cookies)]
    if not shutil.which("deno") and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    cmd.append(url)
    run(cmd, log_file)
    candidates = sorted(p for p in workdir.glob("01_raw_audio.*")
                        if not p.name.endswith(".info.json"))
    if not candidates:
        raise FileNotFoundError("yt-dlp did not create 01_raw_audio.*")
    return candidates[0]


def _clean_title(text):
    """Drop common video-title noise like '(Official MV)', '[Lyrics]', 'piano cover'."""
    if not text:
        return None
    text = re.sub(
        r"[\(\[\{][^\)\]\}]*"
        r"(official|mv|m/v|lyric|audio|video|hd|4k|live|cover|piano|instrumental|"
        r"karaoke|visualizer|teaser|full)[^\)\]\}]*[\)\]\}]",
        "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" -|·\t") or None


def resolve_title_composer(title_arg, composer_arg, workdir, is_url):
    """Title + composer are mandatory. Prefer the caller's values, else extract them
    from the downloaded YouTube metadata (best-effort). Raise if still unknown."""
    info = {}
    if is_url:
        for j in workdir.glob("01_raw_audio.info.json"):
            try:
                info = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            break
    auto_title = _clean_title(info.get("track") or info.get("title"))
    artist = (info.get("artist") or info.get("creator") or info.get("uploader")
              or info.get("channel") or "")
    auto_composer = re.sub(r"\s*-\s*Topic$", "", artist).strip() or None

    title = title_arg or auto_title
    composer = composer_arg or auto_composer
    if not title or not composer:
        missing = ", ".join(n for n, v in (("title", title), ("composer", composer)) if not v)
        raise RuntimeError(
            f"Title and composer are required but {missing} is unknown. "
            "Pass --title/--composer explicitly"
            + (" (could not extract from the video metadata)." if is_url
               else " (no YouTube metadata available for a local-file input).")
        )
    return title, composer


def normalize_audio(input_path, wav_path, log_file):
    # 44.1 kHz mono; no denoise/time-stretch (it damages note onsets).
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-af",
            "aresample=async=1:first_pts=0",
            wav_path,
        ],
        log_file,
    )


def run_transkun(wav_path, midi_path, log_file, gpu_id, device):
    env = os.environ.copy()
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    run(["transkun", wav_path, midi_path, "--device", device], log_file, env=env)


def run_hpt_optional(wav_path, midi_path, log_file, gpu_id, device):
    code = f"""
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate
audio, _ = librosa.load(r'{wav_path}', sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device='{device}', checkpoint_path=None)
transcriptor.transcribe(audio, r'{midi_path}')
"""
    env = os.environ.copy()
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("LD_LIBRARY_PATH", None)  # use torch's bundled cuDNN, not the system one
    run([sys.executable, "-c", code], log_file, env=env)


def export_with_musescore(midi_path, musicxml_path, pdf_path, log_file):
    exe = find_any(["musescore", "mscore", "musescore4", "mscore4", "musescore3", "mscore3"])
    if not exe:
        return False
    xvfb = find_any(["xvfb-run"])  # headless Qt rendering on servers
    base_cmd = [exe]
    if xvfb:
        base_cmd = [xvfb, "-a", exe]

    ok = True
    try:
        run(base_cmd + [str(midi_path), "-o", str(musicxml_path)], log_file)
        run(base_cmd + [str(midi_path), "-o", str(pdf_path)], log_file)
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"MuseScore export failed: {e}\n")
        ok = False
    return ok and musicxml_path.exists() and pdf_path.exists()


def export_with_music21_lilypond(midi_path, musicxml_path, pdf_path, log_file):
    # Fallback: MIDI -> MusicXML via music21, then MusicXML -> LilyPond -> PDF.
    code = f"""
from music21 import converter
from pathlib import Path
midi = Path(r'{midi_path}')
xml = Path(r'{musicxml_path}')
s = converter.parse(str(midi))
try:
    s = s.quantize()
except Exception:
    pass
s.write('musicxml', fp=str(xml))
print(xml)
"""
    run([sys.executable, "-c", code], log_file)

    ly_path = musicxml_path.with_suffix(".ly")
    run(["musicxml2ly", "-o", str(ly_path), str(musicxml_path)], log_file)

    out_base = pdf_path.with_suffix("")
    run(["lilypond", "-o", str(out_base), str(ly_path)], log_file)

    if not pdf_path.exists():
        candidates = list(pdf_path.parent.glob("*.pdf"))
        if candidates:
            shutil.copy2(candidates[0], pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not created: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube/video URL")
    src.add_argument("--audio", help="Local audio file")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--gpu", default="0", help="Physical GPU id to expose to this job")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--engine", default="transkun", choices=["transkun", "hpt", "both"])
    parser.add_argument("--cookies", default=None, help="Path to a Netscape cookies.txt for YouTube")
    parser.add_argument(
        "--no-split-hands",
        action="store_true",
        help="Engrave the raw single-staff MIDI instead of splitting into RH/LH",
    )
    parser.add_argument(
        "--split-point",
        type=int,
        default=60,
        help="MIDI pitch dividing right hand (>=) from left hand (<). Default 60 = C4",
    )
    # Key-aware engraving (detect modulations -> per-section key signatures).
    parser.add_argument(
        "--no-key-detect",
        action="store_true",
        help="Skip key-change detection; engrave with MuseScore's single global key",
    )
    parser.add_argument("--title", default=None, help="Score title shown on page 1")
    parser.add_argument("--composer", default="", help="Composer/credit text")
    parser.add_argument(
        "--style",
        default=str(DEFAULT_STYLE),
        help="MuseScore .mss style (readable layout + footer page numbers current/total)",
    )
    parser.add_argument(
        "--bars-per-system",
        type=int,
        default=6,
        help="Bars per system for readability (0 = let MuseScore decide)",
    )
    parser.add_argument(
        "--no-chord-merge",
        action="store_true",
        help="Keep dense multi-voice notation instead of merging each hand to chords",
    )
    parser.add_argument(
        "--staff-mm",
        type=float,
        default=1.75,
        help="Staff-space size in mm (smaller fits more systems/page; 1.675 = ideal)",
    )
    parser.add_argument(
        "--time-sig",
        default=None,
        help="Force a time signature, e.g. 4/4 (re-bars; MuseScore meter detection is overridden)",
    )
    parser.add_argument(
        "--no-chords",
        action="store_true",
        help="Do not add a chord symbol at the start of each measure",
    )
    parser.add_argument("--key-penalty", type=float, default=1.5,
                        help="Viterbi key-switch penalty; higher = fewer key changes")
    parser.add_argument("--key-context-beats", type=float, default=16.0)
    parser.add_argument("--key-min-segment-beats", type=float, default=24.0)
    parser.add_argument("--keys", default=None,
                        help="Manual per-section major-key sigs, e.g. Gb,Ab,Bb (skips detection)")
    parser.add_argument("--key-bars", default=None,
                        help="Manual key-change boundaries as quarter-note offsets, e.g. 342,399")
    args = parser.parse_args()

    device = resolve_device(args.device)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "run.log"
    log_file.write_text("", encoding="utf-8")

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": args.url,
        "audio": args.audio,
        "gpu": args.gpu,
        "device": device,
        "engine": args.engine,
    }

    if args.url:
        (workdir / "00_source.url").write_text(args.url + "\n", encoding="utf-8")
        raw_audio = download_youtube(args.url, workdir, log_file, cookies=args.cookies)
    else:
        raw_audio = workdir / ("01_raw_audio" + Path(args.audio).suffix)
        shutil.copy2(args.audio, raw_audio)

    # Title + composer are mandatory; auto-extract from the video when not supplied.
    title, composer = resolve_title_composer(args.title, args.composer, workdir, bool(args.url))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"title={title!r} composer={composer!r}\n")

    wav_path = workdir / "02_audio_44100_mono.wav"
    normalize_audio(raw_audio, wav_path, log_file)

    transkun_midi = workdir / "03_transkun.mid"
    hpt_midi = workdir / "04_hpt.mid"

    if args.engine in ("transkun", "both"):
        run_transkun(wav_path, transkun_midi, log_file, args.gpu, device)

    if args.engine in ("hpt", "both"):
        try:
            run_hpt_optional(wav_path, hpt_midi, log_file, args.gpu, device)
        except Exception as e:  # HPT is optional; never abort a good Transkun run
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"HPT engine failed (continuing): {e}\n")
            if args.engine == "hpt" and not transkun_midi.exists():
                raise

    chosen_midi = transkun_midi if transkun_midi.exists() else hpt_midi
    if not chosen_midi.exists():
        raise FileNotFoundError("No MIDI output produced")

    # Split into RH/LH so the engraver lays out a proper grand staff (treble + bass).
    if args.no_split_hands:
        engrave_midi = chosen_midi
    else:
        engrave_midi = workdir / "05_2hand.mid"
        rh, lh = split_hands(str(chosen_midi), str(engrave_midi), args.split_point)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"hand-split @ {args.split_point}: right={rh} left={lh} -> {engrave_midi}\n")

    musicxml_path = workdir / "06_score.musicxml"
    pdf_path = workdir / "07_score.pdf"

    have_musescore = find_any(
        ["musescore", "mscore", "musescore4", "mscore4", "musescore3", "mscore3"]
    )
    key_info = {"key_detect": False, "keys": None, "key_change_quarters": None}
    engraved = False
    if not args.no_key_detect and have_musescore:
        try:
            import keysig
            from engrave_keys import engrave as engrave_modulated

            tpb, tempo, notes = keysig.parse_notes(str(engrave_midi))
            if args.keys:  # manual override
                keys = [k.strip() for k in args.keys.split(",")]
                boundary_ticks = (
                    [int(round(float(b) * tpb)) for b in args.key_bars.split(",")]
                    if args.key_bars else []
                )
                labels = keys
            else:
                det = keysig.detect_segments(
                    notes, tpb,
                    context_beats=args.key_context_beats,
                    penalty=args.key_penalty,
                    min_segment_beats=args.key_min_segment_beats,
                )
                keys, boundary_ticks, labels = det["keys"], det["boundary_ticks"], det["labels"]
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"key detection: {labels} (sig {keys})\n")
                    f.write(keysig.summary(det, tpb, tempo) + "\n")

            style = args.style if args.style and Path(args.style).exists() else None
            engrave_modulated(
                engrave_midi, musicxml_path, pdf_path,
                style=style, title=title, composer=composer,
                bars_per_system=args.bars_per_system, clarify=not args.no_chord_merge,
                chords=not args.no_chords, time_sig=args.time_sig, spatium_mm=args.staff_mm,
                manual_keys=keys, manual_bar_ticks=boundary_ticks, workdir=workdir,
            )
            engraved = musicxml_path.exists() and pdf_path.exists()
            key_info = {
                "key_detect": not args.keys,
                "keys": keys,
                "key_labels": labels,
                "key_change_quarters": [round(b / tpb, 2) for b in boundary_ticks],
                "style": style,
                "bars_per_system": args.bars_per_system,
                "time_sig": args.time_sig,
                "chords": not args.no_chords,
            }
        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"key-aware engrave failed, falling back to plain engrave: {e}\n")

    if not engraved:
        if not export_with_musescore(engrave_midi, musicxml_path, pdf_path, log_file):
            export_with_music21_lilypond(engrave_midi, musicxml_path, pdf_path, log_file)

    metadata.update(
        {
            "raw_audio": str(raw_audio),
            "wav": str(wav_path),
            "chosen_midi": str(chosen_midi),
            "engrave_midi": str(engrave_midi),
            "split_hands": (not args.no_split_hands),
            "split_point": args.split_point,
            "musicxml": str(musicxml_path),
            "pdf": str(pdf_path),
            "title": title,
            "composer": composer,
            **key_info,
        }
    )
    (workdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("DONE")
    print("device:", device)
    print("MIDI:", chosen_midi)
    print("MusicXML:", musicxml_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
