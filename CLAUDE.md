# CLAUDE.md

Guidance for Claude / Claude Code working in this repo. **[AGENTS.md](AGENTS.md) is the full
contract** — read it. This file is the quick reference.

## What it is
`piano2sheet`: a local, GPU (CUDA) CLI that turns piano audio (YouTube link or local file) into
engraved sheet music (MusicXML + PDF). Flow: `yt-dlp → ffmpeg → Transkun (CUDA) → hand-split →
key/modulation detection → MuseScore engrave`.

## Non-negotiable rules
1. **Title + composer are mandatory on every score.** Resolution order (`pipeline.resolve_title_composer`):
   explicit `--title`/`--composer` → else auto-extract from the YouTube video metadata
   (`title`/`track`, `artist`/`uploader`) → else fail with a clear error. A local file with neither
   provided is an error. If the user tells you the song/artist, pass them explicitly.
2. **Never commit `cache/`** (YouTube session cookies = secrets) or any token. It is gitignored.
3. **Never commit** `.venv/` or the large `runs/**/01_raw_audio.*` / `runs/**/02_audio_*.wav` intermediates.

## Run
```bash
source .venv/bin/activate
python src/pipeline.py --url "https://youtu.be/<id>" --workdir runs/<name> \
  --device cuda --cookies cache/youtube_cookies.txt
python src/pipeline.py --audio song.mp3 --workdir runs/<name> --title "T" --composer "C"
```
Iterate on engraving without re-downloading via `python src/engrave_keys.py 05_2hand.mid --out-xml ... --out-pdf ...`.

## Gotchas (save yourself debugging time)
- MuseScore **ignores MIDI key & time-signature meta** — per-section key signatures and forced meters
  are done in **music21** (`src/engrave_keys.py`), not via MIDI.
- Modulated sections are respelled by **lower-to-base → let MuseScore spell → transpose back**.
- MusicXML `<scaling>` overrides the style `Spatium` (`_set_scaling`).
- Wrap MuseScore calls in `timeout` + retry (`_ms_run`); `xvfb-run` hangs occasionally.

## Style
Python, 4-space, ~100 cols, format with `black`. Comments explain *why*, not *what* — keep them sparse.

## Verify
`07_score.pdf` exists; title/composer on page 1; `grep -c "<fifths>" 06_score.musicxml` (key sigs) and
`grep -c "<harmony" 06_score.musicxml` (chords) are non-zero.
