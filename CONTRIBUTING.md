# Contributing

Thanks for your interest in piano2sheet! Contributions and PRs are welcome.

## Before you start
- Read **[AGENTS.md](AGENTS.md)** — it documents the pipeline contract, the mandatory title/composer
  rule, and the safety rules (never commit `cache/` cookies or any secret).

## Development setup
```bash
make setup          # creates .venv and installs deps (CUDA 12.8 wheels)
source .venv/bin/activate
```

## Conventions
- Python, 4-space indentation, ~100-column lines (`.editorconfig`). Format with `black` (`make fmt`).
- Comments explain *why*, not *what*; keep them sparse.
- Keep each job's artifacts under its `runs/<name>/` folder.

## Pull requests
- Keep PRs small and focused; use conventional-commit messages (`feat:`, `fix:`, `docs:`, `refactor:`…).
- Verify a sample run still produces `06_score.musicxml` + `07_score.pdf` with a title, composer,
  per-section key signatures, and chord symbols.
- Never commit secrets, `.venv/`, or the large `runs/**/*.wav` / `01_raw_audio.*` intermediates.
