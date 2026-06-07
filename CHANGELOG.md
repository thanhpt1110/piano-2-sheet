# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project uses date-based milestones.

## [0.7.0] - 2026-06-19
### Added
- **Mandatory title + composer** on every score, with **automatic extraction** from YouTube video
  metadata (`track`/`title`, `artist`/`uploader`) when not passed explicitly (`--title` / `--composer`).
- Enterprise-grade README with hero banner, workflow diagram (Mermaid), and badges.
- Agent documentation: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`, plus `Dockerfile`, `Makefile`,
  `CONTRIBUTING.md`, `LICENSE`.
### Changed
- Trimmed redundant comments across the source tree.

## [0.6.0] - 2026-06-17
### Added
- Chord-symbol detection printed at the start of each measure.
- `--time-sig` to force a meter (re-bars in music21, since MuseScore ignores MIDI meter meta).

## [0.5.0] - 2026-06-14
### Added
- Readable layout controls: fixed bars-per-system, staff scaling (`--staff-mm`), and a `compact.mss`
  style to pack more systems per page.

## [0.4.0] - 2026-06-09
### Added
- Per-section key signatures with correct enharmonic spelling (lower-to-base / transpose-back), a
  courtesy double barline at each change, and single-voice-of-chords clarity (grand-staff rebuild).

## [0.3.0] - 2026-06-04
### Added
- Data-driven key / key-change detection (`keysig.py`): Krumhansl-Kessler profiles + Viterbi tracking.

## [0.2.0] - 2026-05-30
### Added
- A4 MuseScore style with measure numbers and a `current/total` page-number footer.
- music21 + LilyPond fallback when MuseScore is unavailable.

## [0.1.0] - 2026-05-19
### Added
- End-to-end pipeline: yt-dlp download → ffmpeg normalize → Transkun (CUDA) → hand-split → engrave.
- Offline self-test audio synthesis and a browser-cookie → `cookies.txt` helper.
