# SFU Quality-Level Allocation — Algorithms

Research code for a master's thesis (AGH, written in Polish): fair allocation of
SVC quality levels in a WebRTC SFU. This repo implements and compares allocation
algorithms for the single-epoch optimization problem defined in `docs/MODEL.md`.

The user writes prompts in Polish — answer in Polish. Code identifiers, comments,
and docstrings are in English.

## Source of truth

- `docs/MODEL.md` — the formal problem specification. Implement exactly this.
- `docs/PLAN.md` — milestone plan. Work on ONE milestone at a time, in order.
- `docs/model.tex` — the thesis chapter (Polish, LaTeX). Reference only. If
  MODEL.md and model.tex ever seem to disagree, STOP and ask the user; never
  silently pick one interpretation.

Never change formulas, tolerances, or the public API of `evaluator.py` without
asking first.

## Commands

- `uv sync` — install dependencies
- `uv run pytest -x -q` — run tests (must be green before a milestone is done)
- `uv run ruff format . && uv run ruff check --fix .` — format and lint

## Conventions

- Python 3.12+, managed by `uv`.
- Map thesis symbols to code names exactly per the table in `docs/MODEL.md`
  (e.g. `q_corr` for q̃, `Q_corr` for Q̃, `tau`, `lambda_prev`, `T_stab`, `b_hat`).
- Frozen dataclasses for problem data; numpy arrays internally; type hints on all
  public functions.
- Determinism: every random component takes an explicit seed or
  `numpy.random.Generator`.
- Float comparisons of quality values always go through `TOL = 1e-6` defined once
  in `sfu_alloc/constants.py`. Never compare quality floats with `==`.
- Tests: pytest + hypothesis. When hypothesis finds a failing instance, save it
  as a JSON fixture in `tests/fixtures/` so it becomes a regression test.
- Every algorithm re-evaluates its own output with `evaluator` before returning;
  never trust a solver's reported objective value.

## Workflow

1. Read the milestone the user points you to in `docs/PLAN.md`.
2. If anything in the spec or milestone is ambiguous, ask before coding.
3. Implement with tests. Run pytest and ruff before declaring done.
4. Summarize what changed and which acceptance criteria are met.
5. Commit as `M<k>: <short description>` when the user confirms.

Do not implement future milestones ahead of time. Do not add dependencies beyond
those pinned in `docs/PLAN.md` without asking.
