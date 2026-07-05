# Claude Code project notes

## Git conventions

- **Never add Claude/AI as co-author on commits.** No `Co-Authored-By: Claude ...`
  trailer, no "Generated with Claude Code" lines — in commit messages or PR bodies.
  Plain commit messages only.
- Commit and push only when Felix explicitly asks.
- Never commit weights (`*.pt`), `data/`, `exp/`, or `.hf_publish.env` (all
  gitignored — keep it that way).

## Project shape

- Fork of X-VC (upstream `Jerrister/X-VC`, MIT) adding an accent fine-tuning
  pipeline; work happens on the `accent-finetuning` branch, remote `origin` is
  `fault9/X-VC-accent-finetuning`.
- Method/rationale: `CHANGES.md`. Operator guide: `docs/finetuning.md`.
- Speaker roster lives in `configs/data_groups.yaml` (never hardcode speakers).
- Training runs on a Linux GPU container, never on this Windows machine
  (DeepSpeed). Checkpoints are published to a private HF model repo via
  `scripts/publish_checkpoint.py`, not git.
