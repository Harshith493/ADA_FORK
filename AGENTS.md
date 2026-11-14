# FORK project rules

- Read `HANDOVER.md` first, then use `docs/onepager_summary.md` as the project brief.
- Follow the user-defined phase order. At phase end, summarize evidence, plots/metrics, learning notes, and three questions, then wait for approval.
- Never make a research/design choice silently. Present 2-3 options, trade-offs, and a recommendation; record the user's choice and rationale in `DECISIONS.md`.
- Regime/transition labels are scoring-only. Keep them outside detector, branch, and calibration inputs; add a test that enforces this.
- Use strict chronological splits; calibration must precede test. Online conformal updates may use only outcomes already observed.
- Fix and record seeds and configs. Cache expensive forecasts and processed data. Keep `EXPERIMENTS.md`, `CREDITS.md`, and `HANDOVER.md` current.
- Verify existence, access, and license of references, model weights, data, and baseline code before building on them. Ask before reusing external code. Implement FORK from scratch.
- Report measured results honestly; do not tune to the one-pager's targets. Do not proceed to stretch datasets/models/baselines without user approval.
