# AGENTS.md — working agreement for AI coding agents

This file is the canonical contract for AI agents (Cursor, Claude Code, Copilot, etc.)
working in this repository. Read it fully before editing or running anything.
`CLAUDE.md` and `.cursor/rules/` point back here.

---

## 1. What this project is

`piano2sheet` is a **local, GPU-accelerated CLI** that converts a piano performance (a YouTube
link or a local audio file) into engraved sheet music (MusicXML + PDF).

Pipeline (see `src/pipeline.py`):

```
download (yt-dlp) ─▶ normalize (ffmpeg, 44.1 kHz mono) ─▶ transcribe (Transkun, CUDA)
   ─▶ hand-split RH/LH (notation.py) ─▶ key/modulation detection (keysig.py)
   ─▶ engrave: key sigs · chords · meter · layout (engrave_keys.py, MuseScore) ─▶ MusicXML + PDF
```

## 2. The golden rule: title + composer are MANDATORY

Every engraved score must carry a **title** and a **composer**. This is enforced in
`pipeline.resolve_title_composer()`:

1. If the caller passes `--title` / `--composer`, those win.
2. Otherwise, for a **YouTube URL**, extract them from the downloaded metadata
   (`01_raw_audio.info.json`, written by `yt-dlp --write-info-json`):
   - **title** ← `track` or `title` (lightly cleaned of "(Official MV)", "[Lyrics]", "piano cover", …).
   - **composer** ← `artist` → `creator` → `uploader` → `channel` (with a trailing `" - Topic"` stripped).
3. If either is still unknown, the run **fails fast** with a clear error.

Implications for agents:
- For a **local-file** input you must supply `--title` and `--composer` (no metadata to extract from).
- Auto-extraction is best-effort. If the user names the song/artist in chat, **prefer passing them
  explicitly** via `--title`/`--composer` rather than relying on extraction.
- Never silently drop the title/composer or hard-code one song's values into the pipeline. Song-specific
  values belong on the command line, not in the code.
- When you change the resolution logic, keep the "flags override, else auto-extract, else error" order.

## 3. How to run

Always use the project virtualenv:

```bash
cd piano2sheet
source .venv/bin/activate
python src/pipeline.py --url "https://youtu.be/<id>" --workdir runs/<name> \
  --device cuda --cookies cache/youtube_cookies.txt          # title/composer auto-extracted
python src/pipeline.py --audio song.mp3 --workdir runs/<name> \
  --title "Song" --composer "Artist"                          # required for a local file
```

- `src/engrave_keys.py` can be run standalone on an existing `05_2hand.mid` to re-engrave without
  re-downloading/transcribing (useful while iterating on layout/keys/chords). Each run folder stores
  the exact reproducible command in `engrave_cmd.txt`.
- Long MuseScore calls are wrapped in `timeout` + retry (`engrave_keys._ms_run`): `xvfb-run` can hang
  transiently. Never remove that guard.

## 4. Environment & external tools

- **GPU:** NVIDIA RTX PRO 6000 Blackwell (sm_120), CUDA 12.8, `torch==2.11.0+cu128`. CPU fallback: `--device cpu`.
- **apt:** `ffmpeg lilypond musescore3 fluidsynth fluid-soundfont-gm timidity xvfb`.
- **deno** on `PATH` — yt-dlp's JS challenge solver for YouTube.
- **Cookies:** logged-in YouTube cookies in `cache/youtube_cookies.txt` clear the datacenter bot-check.
  Convert a browser `Cookie:` header with `python src/cookies_to_netscape.py header.txt cache/youtube_cookies.txt`.

## 5. Safety rules (do not break these)

- **NEVER commit `cache/`** — it holds YouTube **session cookies (secrets)**. It is gitignored; keep it so.
- **NEVER commit secrets/tokens** in code, docs, or run artifacts.
- `.venv/` (multi-GB), the large `runs/**/01_raw_audio.*` and `runs/**/02_audio_*.wav` intermediates,
  and `*.info.json` are gitignored. Keep MIDI / MusicXML / PDF / previews / `metadata.json` / `run.log`.
- Don't add narrating comments; keep them for non-obvious intent only (see §7).

## 6. Code map

| File | Responsibility |
|------|----------------|
| `src/pipeline.py` | CLI + orchestration; `download_youtube`, `resolve_title_composer`, transcription, fallback engraving |
| `src/notation.py` | `split_hands` — RH/LH split at the MIDI-event level (preserves timing/meta) |
| `src/keysig.py` | `detect_segments` — KK profiles + Viterbi key tracking; returns sections, keys, boundaries |
| `src/engrave_keys.py` | `engrave` — lower→base / transpose-back respelling, key sigs, chords, `rebar`, `clarify_score`, scaling, MuseScore render |
| `src/cookies_to_netscape.py` | browser Cookie header → Netscape `cookies.txt` |
| `src/make_test_audio.py` | synth a short clip for an offline self-test |
| `style/piano.mss`, `style/compact.mss` | MuseScore styles (layout + `$p/$n` footer) |

## 7. Conventions

- Python, 4-space indent, ~100-col lines (see `.editorconfig`); format with `black` if available.
- **Comments explain *why*, not *what*.** Prefer a one-line docstring over a paragraph.
- Keep functions pure where practical; pass paths explicitly; write artifacts under the job `workdir`.
- Engraving facts worth remembering (they cost real debugging time):
  - MuseScore **ignores MIDI key-signature and time-signature meta** on import — it re-detects. So
    per-section key signatures and a forced meter are applied in **music21** (`engrave_keys`), not via MIDI meta.
  - To respell a modulated section cleanly we **lower it to the base key, let MuseScore spell it, then
    transpose it back** with a named interval. Don't replace this with naive enharmonic guessing.
  - The MusicXML `<scaling>` overrides the style's `Spatium`; staff size is set via `_set_scaling`.

## 8. Verifying output

```bash
N=$(gs -q -dNOSAFER -dNODISPLAY -c "(runs/<name>/07_score.pdf) (r) file runpdfbegin pdfpagecount = quit")
echo "$N pages"
grep -c "<fifths>" runs/<name>/06_score.musicxml     # key-signature elements
grep -c "<harmony"  runs/<name>/06_score.musicxml     # chord symbols
```

Render a page to PNG for visual review with `gs -dNOSAFER -sDEVICE=png16m -r120 -dFirstPage=1 -dLastPage=1 ...`.

## 9. Definition of done for a change

- The pipeline still produces `06_score.musicxml` + `07_score.pdf` for a sample input.
- Title and composer appear on page 1; per-section key signatures and chord symbols render.
- No secrets added; `.gitignore` still excludes `cache/` and the large audio intermediates.
- Comments trimmed to intent-only; `python -c "import ast; ast.parse(open(f).read())"` passes for edited files.
